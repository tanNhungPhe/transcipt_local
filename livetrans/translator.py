"""Cached machine translation plus a background worker that coalesces live partials."""
import logging
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Protocol

import requests

log = logging.getLogger(__name__)

_CACHE_SIZE = 512


class TranslationBackend(Protocol):
    def __call__(self, text: str, target: str) -> str: ...


_GTX_URL = "https://translate.googleapis.com/translate_a/single"
_TIMEOUT_SEC = 5
_local = threading.local()


def _session() -> requests.Session:
    """One keep-alive session per thread so each request skips the TLS handshake."""
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0"
        _local.session = session
    return session


def parse_gtx_response(data: object) -> str:
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError("unexpected translation response")
    return "".join(part[0] for part in data[0] if isinstance(part, list) and part and isinstance(part[0], str))


def google_backend(text: str, target: str) -> str:
    params = {"client": "gtx", "sl": "auto", "tl": target, "dt": "t", "q": text}
    resp = _session().get(_GTX_URL, params=params, timeout=_TIMEOUT_SEC)
    resp.raise_for_status()
    return parse_gtx_response(resp.json())


class Translator:
    def __init__(self, target: str, backend: TranslationBackend = google_backend):
        self.target = target
        self._backend = backend
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        result = self._backend(text, self.target)
        with self._lock:
            self._cache[text] = result
            if len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return result


@dataclass(frozen=True)
class TranslationJob:
    utterance_id: int
    text: str
    final: bool


TranslationCallback = Callable[[TranslationJob, str], None]
ErrorCallback = Callable[[str], None]


class TranslationWorker:
    """Finals are translated in order; only the newest pending partial is kept."""

    def __init__(self, translator: Translator, on_result: TranslationCallback, on_error: ErrorCallback):
        self._translator = translator
        self._on_result = on_result
        self._on_error = on_error
        self._finals: queue.Queue[TranslationJob] = queue.Queue()
        self._partial: TranslationJob | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="translator", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def submit(self, job: TranslationJob) -> None:
        if job.final:
            self._finals.put(job)
            with self._lock:
                if self._partial and self._partial.utterance_id == job.utterance_id:
                    self._partial = None
        else:
            with self._lock:
                self._partial = job
        self._wake.set()

    def _next_job(self) -> TranslationJob | None:
        try:
            return self._finals.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            job, self._partial = self._partial, None
        return job

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._next_job()
            if job is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            try:
                self._on_result(job, self._translator.translate(job.text))
            except Exception as exc:  # noqa: BLE001 - network errors must not kill the worker
                log.warning("Translation failed: %s", exc)
                if job.final:
                    self._on_error(f"Lỗi dịch: {exc}")
