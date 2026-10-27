"""OpenVINO GenAI Whisper backend: Intel NPU, Intel iGPU or CPU."""
import logging
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from .audio_utils import TARGET_RATE
from .model_store import OV_CACHE_DIR
from .transcriber import Progress, SpeechSegment, TranscriptResult, clean_text

log = logging.getLogger(__name__)

_FINAL_BEAMS = 5
_WINDOW_SEC = 29.0          # Whisper sees 30 s at a time; keep a margin
_MERGE_GAP_SEC = 1.0
# Features some accelerators reject (beam search fails on Intel iGPU); tried one at a time, least useful first.
_OPTIONAL_KWARGS = ("initial_prompt", "num_beams", "return_timestamps", "language")


def strip_lang_token(lang: str | None) -> str:
    return (lang or "").strip("<|>").split("-")[0].lower()


def merge_regions(regions: list[tuple[int, int]], max_len: int, max_gap: int) -> list[tuple[int, int]]:
    """Join nearby speech regions into windows no longer than `max_len` samples."""
    windows: list[tuple[int, int]] = []
    for start, end in regions:
        if windows and start - windows[-1][1] <= max_gap and end - windows[-1][0] <= max_len:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
    return windows


def speech_windows(audio: np.ndarray) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(audio, VadOptions(max_speech_duration_s=_WINDOW_SEC))
    regions = [(int(t["start"]), int(t["end"])) for t in stamps]
    return merge_regions(regions, int(_WINDOW_SEC * TARGET_RATE), int(_MERGE_GAP_SEC * TARGET_RATE))


class OpenVinoTranscriber:
    def __init__(self, model_dir: Path, device: str, label: str, language: str | None = None, pipeline=None):
        if pipeline is None:
            import openvino_genai

            OV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # the compiled-model cache turns a minutes-long NPU/GPU compile into a few seconds after the first run
            pipeline = openvino_genai.WhisperPipeline(str(model_dir), device, CACHE_DIR=str(OV_CACHE_DIR))
        self._pipe = pipeline
        self.device = label
        self.language = language
        self.lock = threading.Lock()
        self._unsupported: set[str] = set()
        self._generate(np.zeros(TARGET_RATE, dtype=np.float32))

    def _generate(self, audio: np.ndarray, **kwargs):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with self.lock:
            kwargs = {k: v for k, v in kwargs.items() if k not in self._unsupported}
            try:
                return self._pipe.generate(audio, **kwargs)
            except RuntimeError as exc:
                first_error = exc
            # drop one optional feature at a time and remember only the one that was rejected
            for key in (k for k in _OPTIONAL_KWARGS if k in kwargs):
                trial = {k: v for k, v in kwargs.items() if k != key}
                try:
                    result = self._pipe.generate(audio, **trial)
                except RuntimeError:
                    continue
                log.warning("%s does not support '%s' (%s); disabled", self.device, key, first_error)
                self._unsupported.add(key)
                return result
            raise first_error

    @staticmethod
    def _options(language: str | None, prompt: str | None = None, beams: int = 1,
                 timestamps: bool = False) -> dict:
        opts: dict = {}
        if timestamps:
            opts["return_timestamps"] = True
        if language:
            opts["language"] = f"<|{strip_lang_token(language)}|>"
        if prompt:
            opts["initial_prompt"] = prompt
        if beams > 1:
            opts["num_beams"] = beams
        return opts

    def transcribe(
        self,
        audio: np.ndarray,
        final: bool,
        prompt: str | None = None,
        language_hint: str | None = None,
    ) -> TranscriptResult:
        language = self.language or language_hint
        result = self._generate(audio, **self._options(language, prompt, _FINAL_BEAMS if final else 1))
        text = clean_text(" ".join(result.texts))
        # OpenVINO reports no language probability; finals re-detect, so a wrong guess self-corrects
        return TranscriptResult(text, strip_lang_token(result.language) or (language or ""), 1.0)

    def transcribe_long(self, audio: np.ndarray, language: str | None,
                        progress: Progress) -> tuple[list[SpeechSegment], str]:
        segments: list[SpeechSegment] = []
        votes: Counter[str] = Counter()
        for start, end in speech_windows(audio):
            # without a fixed language each window detects its own, so one intro/music window can't mislabel the file
            result = self._generate(audio[start:end], **self._options(language, beams=_FINAL_BEAMS, timestamps=True))
            found = _to_segments(result, start / TARGET_RATE, end / TARGET_RATE)
            if found and result.language:
                votes[strip_lang_token(result.language)] += end - start
            segments.extend(found)
            last = segments[-1].text if segments else ""
            progress(f"Phiên âm {end / max(len(audio), 1):.0%} · {last[:60]}")
        detected = language or (votes.most_common(1)[0][0] if votes else "unknown")
        return segments, detected


def _to_segments(result, offset: float, window_end: float) -> list[SpeechSegment]:
    chunks = list(result.chunks or [])
    if not chunks:  # device without timestamp support: one cue per window
        text = clean_text(" ".join(result.texts))
        return [SpeechSegment(offset, window_end, text)] if text else []
    out = []
    for chunk in chunks:
        text = clean_text(chunk.text)
        if text:
            # an unterminated last chunk has no usable end stamp; close it at the window end
            end = offset + chunk.end_ts if chunk.end_ts > chunk.start_ts else window_end
            out.append(SpeechSegment(offset + chunk.start_ts, end, text))
    return out
