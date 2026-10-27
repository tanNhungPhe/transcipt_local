"""faster-whisper wrapper tuned for low-latency streaming."""
import logging
import threading
from dataclasses import dataclass

import numpy as np

from .cuda_setup import register_cuda_dlls
from .text_filters import collapse_repeats, is_hallucination, normalize_text

log = logging.getLogger(__name__)

_NO_SPEECH_DROP = 0.6
_LOGPROB_DROP = -1.0


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str
    language_probability: float = 1.0


def _pick_device(device: str) -> tuple[str, str]:
    if device != "auto":
        return device, ("float16" if device == "cuda" else "int8")
    register_cuda_dlls()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as exc:  # noqa: BLE001 - any CUDA probe failure means CPU
        log.warning("CUDA probe failed, using CPU: %s", exc)
    return "cpu", "int8"


_cache: dict[tuple[str, str], "Transcriber"] = {}
_cache_lock = threading.Lock()


def get_transcriber(model_size: str, device: str = "auto") -> "Transcriber":
    """Load each model once and share it between live and file modes (VRAM is tight)."""
    with _cache_lock:
        key = (model_size, device)
        if key not in _cache:
            _cache.clear()  # free the previous model before loading another
            _cache[key] = Transcriber(model_size, device)
        return _cache[key]


class Transcriber:
    def __init__(self, model_size: str, device: str = "auto", language: str | None = None):
        from faster_whisper import WhisperModel

        register_cuda_dlls()
        self.device, compute_type = _pick_device(device)
        self.language = language or None
        self.lock = threading.Lock()  # one GPU inference at a time
        log.info("Loading Whisper '%s' on %s (%s)", model_size, self.device, compute_type)
        self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type)
        self._warm_up()

    def _warm_up(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32), final=False)

    def transcribe(
        self,
        audio: np.ndarray,
        final: bool,
        prompt: str | None = None,
        language_hint: str | None = None,
    ) -> TranscriptResult:
        """`language_hint` is used only when no fixed source language was configured."""
        with self.lock:
            segments, info = self.model.transcribe(
                audio,
                language=self.language or language_hint,
                beam_size=5 if final else 1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=prompt or None,
                vad_filter=False,
                without_timestamps=True,
            )
            kept = [
                s.text for s in segments
                if not (s.no_speech_prob > _NO_SPEECH_DROP and s.avg_logprob < _LOGPROB_DROP)
            ]
        text = collapse_repeats(normalize_text(" ".join(kept)))
        if is_hallucination(text):
            text = ""
        return TranscriptResult(text=text, language=info.language, language_probability=info.language_probability)
