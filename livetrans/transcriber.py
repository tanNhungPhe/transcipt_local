"""Speech-to-text front door: shared result types and device-aware model loading."""
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .devices import CTRANSLATE2, DevicePlan, cuda_available, openvino_devices, plan_candidates
from .model_store import ModelMissingError, ct2_whisper_dir, has_ct2_model, has_ov_whisper, missing, ov_whisper_dir
from .text_filters import collapse_repeats, is_hallucination, normalize_text

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str
    language_probability: float = 1.0


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    text: str


class Transcriber(Protocol):
    device: str
    language: str | None

    def transcribe(self, audio: np.ndarray, final: bool, prompt: str | None = None,
                   language_hint: str | None = None) -> TranscriptResult: ...

    def transcribe_long(self, audio: np.ndarray, language: str | None,
                        progress: Progress) -> tuple[list[SpeechSegment], str]: ...


def clean_text(raw: str) -> str:
    text = collapse_repeats(normalize_text(raw))
    return "" if is_hallucination(text) else text


def _load(plan: DevicePlan, model_size: str) -> Transcriber:
    if plan.backend == CTRANSLATE2:
        from .ct2_backend import FasterWhisperTranscriber

        local = ct2_whisper_dir(model_size)
        # without a local copy, faster-whisper resolves the name from the HuggingFace cache (offline mode)
        source = str(local) if has_ct2_model(local) else model_size
        return FasterWhisperTranscriber(source, plan.device, plan.label)
    from .ov_backend import OpenVinoTranscriber

    path = ov_whisper_dir(model_size)
    if not has_ov_whisper(path):
        raise missing(f"Whisper {model_size} (OpenVINO)", path)
    return OpenVinoTranscriber(path, plan.device, plan.label)


def load_first_working(model_size: str, plans: list[DevicePlan],
                       loader: Callable[[DevicePlan, str], Transcriber] = _load) -> Transcriber:
    # keep only messages: a stored exception's traceback would pin a half-loaded model in memory
    errors: list[str] = []
    model_missing = False
    for plan in plans:
        try:
            log.info("Loading Whisper '%s' on %s (%s)", model_size, plan.label, plan.backend)
            return loader(plan, model_size)
        except Exception as exc:  # noqa: BLE001 - try the next device, report all failures at the end
            log.warning("Whisper on %s failed: %s", plan.label, exc)
            errors.append(f"{plan.label}: {exc}")
            model_missing |= isinstance(exc, ModelMissingError)
    message = "Không tải được Whisper trên thiết bị nào — " + " | ".join(errors)
    raise ModelMissingError(message) if model_missing else RuntimeError(message)


_cache: dict[tuple[str, str], Transcriber] = {}
_cache_lock = threading.Lock()


def get_transcriber(model_size: str, device: str = "auto") -> Transcriber:
    """Load each model once and share it between live and file modes (memory is tight)."""
    with _cache_lock:
        key = (model_size, device)
        if key not in _cache:
            _cache.clear()  # free the previous model before loading another
            plans = plan_candidates(device, device in ("auto", "cuda") and cuda_available(), openvino_devices())
            _cache[key] = load_first_working(model_size, plans)
        return _cache[key]
