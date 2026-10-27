"""Offline transcription of a video/audio file into .srt (original + translated)."""
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_utils import TARGET_RATE
from .srt import Cue, build_srt
from .transcriber import Progress, Transcriber
from .translator import Translator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileResult:
    original_srt: Path
    translated_srt: Path | None
    language: str
    cue_count: int


def transcribe_file(
    path: Path,
    transcriber: Transcriber,
    translator: Translator | None,
    source_language: str | None = None,
    progress: Progress = lambda _msg: None,
) -> FileResult:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    progress("Đang đọc âm thanh…")
    audio = load_media(path)
    segments, language = transcriber.transcribe_long(audio, source_language, progress)
    cues = [Cue(s.start, s.end, s.text) for s in segments]

    original = path.with_suffix(f".{language}.srt")
    original.write_text(build_srt(cues), encoding="utf-8")
    translated_path = None
    if translator is not None and cues:
        progress(f"Đang dịch {len(cues)} câu…")
        texts = _safe_translate_all(translator, [c.text for c in cues], language)
        translated = [Cue(c.start, c.end, t) for c, t in zip(cues, texts, strict=True)]
        translated_path = path.with_suffix(f".{translator.target}.srt")
        translated_path.write_text(build_srt(translated), encoding="utf-8")
    progress("Xong")
    return FileResult(original, translated_path, language, len(cues))


def load_media(path: Path) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32, preferring the system ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return _decode_with_pyav(path)
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(TARGET_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603 - fixed argv, no shell
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg không đọc được file: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _decode_with_pyav(path: Path) -> np.ndarray:
    """PyAV (bundled with faster-whisper) fallback; faster_whisper.decode_audio breaks on PyAV >= 15."""
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_RATE)
    parts: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
            parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    except av.error.FFmpegError as exc:
        raise RuntimeError(f"Không đọc được âm thanh trong file: {exc}") from exc
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)


def _safe_translate_all(translator: Translator, texts: list[str], source: str) -> list[str]:
    try:
        return translator.translate_many(texts, source)
    except Exception as exc:  # noqa: BLE001 - keep the original lines on failure
        log.warning("Batch translation failed (%s); keeping original text", exc)
        return texts
