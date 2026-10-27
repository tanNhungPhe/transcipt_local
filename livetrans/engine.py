"""Real-time pipeline: capture -> VAD segmenter -> Whisper -> translation -> UI events."""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .audio_capture import AudioCapture
from .config import Settings
from .segmenter import SegmentEvent, StreamingSegmenter
from .transcriber import Transcriber, get_transcriber
from .translator import TranslationJob, TranslationWorker, Translator
from .vad import SileroScorer

log = logging.getLogger(__name__)

_PARTIAL_TRANSLATE_INTERVAL = 0.8
_PROMPT_CHARS = 200
_LANG_CONFIDENT = 0.7


@dataclass(frozen=True)
class UiEvent:
    kind: str                 # status | error | partial | final | translation
    utterance_id: int = -1
    text: str = ""
    final: bool = False
    start_sec: float = 0.0
    latency_ms: int = 0


def select_events(events: list[SegmentEvent]) -> list[SegmentEvent]:
    """Keep every final, but only a trailing partial (older partials are stale)."""
    finals = [e for e in events if e.kind == "final"]
    if events and events[-1].kind == "partial":
        return finals + [events[-1]]
    return finals


class LiveEngine:
    def __init__(self, settings: Settings, emit: Callable[[UiEvent], None]):
        self._settings = settings
        self._emit = emit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._transcriber: Transcriber | None = None
        self._capture: AudioCapture | None = None
        self._worker: TranslationWorker | None = None
        self._last_final_text = ""
        self._detected_lang: str | None = None
        self._last_partial_tr = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, settings: Settings) -> None:
        if self.running:
            return
        self._settings = settings
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_safe, name="engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_safe(self) -> None:
        try:
            self._setup()
            self._loop()
        except Exception as exc:  # noqa: BLE001 - surface every failure to the UI
            log.exception("Engine crashed")
            self._emit(UiEvent("error", text=str(exc)))
        finally:
            self._teardown()
            self._emit(UiEvent("status", text="Đã dừng"))

    def _setup(self) -> None:
        s = self._settings
        self._emit(UiEvent("status", text=f"Đang tải model {s.model}…"))
        self._transcriber = get_transcriber(s.model, s.device)
        self._transcriber.language = s.source_language
        self._detected_lang = None
        self._last_final_text = ""
        if s.translate:
            self._worker = TranslationWorker(Translator(s.target_language), self._on_translation, self._on_tr_error)
            self._worker.start()
        self._capture = AudioCapture(s.source)
        self._capture.start()
        device = self._transcriber.device.upper()
        self._emit(UiEvent("status", text=f"Đang nghe: {self._capture.device_name} · {device}"))

    def _teardown(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def _loop(self) -> None:
        segmenter = StreamingSegmenter(SileroScorer())
        chunks = self._capture.chunks
        while not self._stop.is_set():
            try:
                batch = [chunks.get(timeout=0.2)]
            except queue.Empty:
                continue
            while True:
                try:
                    batch.append(chunks.get_nowait())
                except queue.Empty:
                    break
            events = [ev for chunk in batch for ev in segmenter.feed(chunk)]
            for ev in select_events(events):
                self._handle(ev)
        for ev in segmenter.flush():
            self._handle(ev)

    def _handle(self, ev: SegmentEvent) -> None:
        final = ev.kind == "final"
        t0 = time.perf_counter()
        result = self._transcriber.transcribe(
            ev.audio,
            final=final,
            prompt=self._last_final_text[-_PROMPT_CHARS:] if final else None,
            language_hint=None if final else self._detected_lang,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        self._emit(UiEvent(ev.kind, ev.utterance_id, result.text, final, ev.start_sec, latency))
        if result.text and result.language_probability >= _LANG_CONFIDENT:
            # reusing the language skips Whisper's detection pass on later partials (~2x faster)
            self._detected_lang = result.language
        if final and result.text:
            self._last_final_text = result.text
        self._maybe_translate(ev.utterance_id, result.text, final)

    def _maybe_translate(self, utterance_id: int, text: str, final: bool) -> None:
        if self._worker is None or not text:
            return
        if not final:
            now = time.monotonic()
            if not self._settings.translate_partials or now - self._last_partial_tr < _PARTIAL_TRANSLATE_INTERVAL:
                return
            self._last_partial_tr = now
        self._worker.submit(TranslationJob(utterance_id, text, final))

    def _on_translation(self, job: TranslationJob, translated: str) -> None:
        self._emit(UiEvent("translation", job.utterance_id, translated, job.final))

    def _on_tr_error(self, message: str) -> None:
        self._emit(UiEvent("error", text=message))
