"""Speech-probability scorers for streaming chunks."""
from collections import deque

import numpy as np

from .audio_utils import rms

SILERO_WINDOW = 512
_CONTEXT_WINDOWS = 16  # ~0.5 s of history so the stateless model sees context


class SileroScorer:
    """Wraps faster-whisper's bundled Silero VAD. Robust to music/background noise."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()
        self._history: deque[np.ndarray] = deque(maxlen=_CONTEXT_WINDOWS)

    def __call__(self, chunk: np.ndarray) -> float:
        if len(chunk) % SILERO_WINDOW != 0:
            raise ValueError(f"chunk length must be a multiple of {SILERO_WINDOW}")
        new_windows = len(chunk) // SILERO_WINDOW
        for i in range(new_windows):
            self._history.append(chunk[i * SILERO_WINDOW:(i + 1) * SILERO_WINDOW])
        audio = np.concatenate(list(self._history)).astype(np.float32)
        probs = np.asarray(self._model(audio)).ravel()
        return float(probs[-new_windows:].max())


class EnergyScorer:
    """Fallback scorer: maps RMS loudness to a pseudo-probability."""

    def __init__(self, threshold: float = 0.01):
        self._threshold = threshold

    def __call__(self, chunk: np.ndarray) -> float:
        return min(1.0, 0.5 * rms(chunk) / self._threshold)
