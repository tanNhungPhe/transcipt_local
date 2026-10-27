"""faster-whisper (CTranslate2) backend: NVIDIA CUDA or plain CPU."""
import threading

import numpy as np

from .cuda_setup import register_cuda_dlls
from .transcriber import Progress, SpeechSegment, TranscriptResult, clean_text

_NO_SPEECH_DROP = 0.6
_LOGPROB_DROP = -1.0


class FasterWhisperTranscriber:
    def __init__(self, model: str, device: str, label: str, language: str | None = None):
        from faster_whisper import WhisperModel

        register_cuda_dlls()
        self.device = label
        self.language = language
        self.lock = threading.Lock()  # one inference at a time
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
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
        return TranscriptResult(clean_text(" ".join(kept)), info.language, info.language_probability)

    def transcribe_long(self, audio: np.ndarray, language: str | None,
                        progress: Progress) -> tuple[list[SpeechSegment], str]:
        with self.lock:
            segments, info = self.model.transcribe(audio, language=language, beam_size=5, vad_filter=True)
            out = []
            for seg in segments:
                text = clean_text(seg.text)
                if text:
                    out.append(SpeechSegment(seg.start, seg.end, text))
                progress(f"Phiên âm {seg.end / max(info.duration, 1e-6):.0%} · {text[:60]}")
        return out, info.language
