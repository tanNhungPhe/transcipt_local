"""Pure numpy helpers for audio conversion and fixed-size chunking."""
import numpy as np

TARGET_RATE = 16000


def to_mono(frames: np.ndarray, channels: int) -> np.ndarray:
    """Interleaved float32 frames -> mono float32."""
    if channels <= 1:
        return frames.astype(np.float32, copy=False)
    usable = len(frames) - (len(frames) % channels)
    return frames[:usable].reshape(-1, channels).mean(axis=1).astype(np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    """Linear-interpolation resampler; adequate for speech recognition input."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    dst_len = int(round(len(audio) * dst_rate / src_rate))
    src_pos = np.linspace(0, len(audio) - 1, num=dst_len)
    return np.interp(src_pos, np.arange(len(audio)), audio).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


class ChunkRebuffer:
    """Turns arbitrarily sized arrays into chunks of exactly `size` samples."""

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be positive")
        self._size = size
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, audio: np.ndarray) -> list[np.ndarray]:
        data = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        count = len(data) // self._size
        chunks = [data[i * self._size:(i + 1) * self._size] for i in range(count)]
        self._pending = data[count * self._size:]
        return chunks
