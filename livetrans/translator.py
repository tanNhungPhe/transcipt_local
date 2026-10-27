"""Cached offline translation plus a background worker that coalesces live partials."""
import logging
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Protocol

from .nllb import normalize_lang

log = logging.getLogger(__name__)

_CACHE_SIZE = 512


class TranslationBackend(Protocol):
    def __call__(self, text: str, target: str, source: str | None) -> str: ...


def _default_backend() -> TranslationBackend:
    from .nllb import get_nllb_backend

    return get_nllb_backend()


class Translator:
    """Everything runs on this machine; nothing is sent over the network."""

    def __init__(self, target: str, backend: TranslationBackend | None = None):
        self.target = target
        self._backend = backend if backend is not None else _default_backend()
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()

    def _same_language(self, source: str | None) -> bool:
        return bool(source) and normalize_lang(source) == normalize_lang(self.target)

    def translate(self, text: str, source: str | None = None) -> str:
        text = text.strip()
        if not text:
            return ""
        if self._same_language(source):
            return text
        key = (normalize_lang(source), text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        result = self._backend(text, self.target, source)
        with self._lock:
            self._cache[key] = result
            if len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return result

    def translate_many(self, texts: list[str], source: str | None = None) -> list[str]:
        """Batch path for files; falls back to one call per line for simple backends."""
        stripped = [t.strip() for t in texts]
        if self._same_language(source):
            return stripped
        batch = getattr(self._backend, "translate_batch", None)
        if batch is None:
            return [self.translate(t, source) for t in stripped]
        todo = [t for t in stripped if t]
        done = iter(batch(todo, self.target, source) if todo else [])
        return [next(done) if t else "" for t in stripped]


@dataclass(frozen=True)
class TranslationJob:
    utterance_id: int
    text: str
    final: bool
    source: str | None = None


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
            self._wake.clear()  # before polling, so a submit() racing with us is never lost
            job = self._next_job()
            if job is None:
                self._wake.wait(0.5)
                continue
            try:
                self._on_result(job, self._translator.translate(job.text, job.source))
            except Exception as exc:  # noqa: BLE001 - a bad line must not kill the worker
                log.warning("Translation failed: %s", exc)
                if job.final:
                    self._on_error(f"Lỗi dịch: {exc}")
