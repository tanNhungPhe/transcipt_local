"""Speaker diarization: voice embeddings (WeSpeaker ResNet34 via sherpa-onnx) + online clustering."""
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from .audio_utils import TARGET_RATE

log = logging.getLogger(__name__)

MODEL_NAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{MODEL_NAME}"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / MODEL_NAME
_DOWNLOAD_TIMEOUT_SEC = 60


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log.info("Downloading speaker model to %s", path)
    with requests.get(MODEL_URL, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
    tmp.replace(path)
    return path


class SpeakerEmbedder:
    """Runs on CPU in its own thread so it overlaps with Whisper on the GPU."""

    def __init__(self, model_path: Path = MODEL_PATH, num_threads: int = 4):
        import sherpa_onnx

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(ensure_model(model_path)), num_threads=num_threads, provider="cpu",
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._lock = threading.Lock()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(TARGET_RATE, audio.astype(np.float32, copy=False))
            stream.input_finished()
            vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return _normalize(vec)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


@dataclass(frozen=True)
class TrackerConfig:
    threshold: float = 0.6       # cosine similarity needed to match a known speaker
    min_new_sec: float = 1.0     # shorter clips never create a new speaker
    short_margin: float = 0.1    # looser match allowed for short clips
    max_speakers: int = 12


class SpeakerTracker:
    """Online clustering: each speaker is the normalized sum of its embeddings."""

    def __init__(self, config: TrackerConfig = TrackerConfig(), prefix: str = "S"):
        self._cfg = config
        self._prefix = prefix
        self._sums: list[np.ndarray] = []
        self._last: str | None = None

    @property
    def speaker_count(self) -> int:
        return len(self._sums)

    def _label(self, index: int) -> str:
        return f"{self._prefix}{index + 1}"

    def nearest(self, emb: np.ndarray) -> tuple[int, float]:
        sims = [float(np.dot(_normalize(s), emb)) for s in self._sums]
        best = int(np.argmax(sims))
        return best, sims[best]

    def assign(self, emb: np.ndarray, duration_sec: float) -> str:
        if not self._sums:
            return self._remember(self._create(emb))
        best, sim = self.nearest(emb)
        long_enough = duration_sec >= self._cfg.min_new_sec
        if sim >= self._cfg.threshold:
            if long_enough:
                self._sums[best] = self._sums[best] + emb
            return self._remember(best)
        if not long_enough:
            if sim >= self._cfg.threshold - self._cfg.short_margin or self._last is None:
                return self._remember(best)
            return self._last
        if len(self._sums) >= self._cfg.max_speakers:
            return self._remember(best)
        return self._remember(self._create(emb))

    def _create(self, emb: np.ndarray) -> int:
        self._sums.append(emb.copy())
        return len(self._sums) - 1

    def _remember(self, index: int) -> str:
        self._last = self._label(index)
        return self._last

    def relabel(self, emb: np.ndarray) -> str:
        """Second pass for offline files: snap to the closest final centroid."""
        return self._label(self.nearest(emb)[0]) if self._sums else self._label(0)
