"""Offline transcription of a video/audio file into .srt (original + translated)."""
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_utils import TARGET_RATE
from .srt import Cue, build_srt
from .text_filters import is_hallucination, normalize_text
from .transcriber import Transcriber
from .translator import Translator

log = logging.getLogger(__name__)

Progress = Callable[[str], None]
_TRANSLATE_THREADS = 4


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
    with transcriber.lock:
        segments, info = transcriber.model.transcribe(
            audio, language=source_language, beam_size=5, vad_filter=True,
        )
        cues = []
        for seg in segments:
            text = normalize_text(seg.text)
            if not is_hallucination(text):
                cues.append(Cue(seg.start, seg.end, text))
            progress(f"Phiên âm {seg.end / max(info.duration, 1e-6):.0%} · {text[:60]}")

    original = path.with_suffix(f".{info.language}.srt")
    original.write_text(build_srt(cues), encoding="utf-8")
    translated_path = None
    if translator is not None and cues:
        progress(f"Đang dịch {len(cues)} câu…")
        with ThreadPoolExecutor(_TRANSLATE_THREADS) as pool:
            texts = list(pool.map(_safe_translate(translator), [c.text for c in cues]))
        translated = [Cue(c.start, c.end, t) for c, t in zip(cues, texts)]
        translated_path = path.with_suffix(f".{translator.target}.srt")
        translated_path.write_text(build_srt(translated), encoding="utf-8")
    progress("Xong")
    return FileResult(original, translated_path, info.language, len(cues))


def load_media(path: Path) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32, preferring the system ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        from faster_whisper import decode_audio

        return decode_audio(str(path), sampling_rate=TARGET_RATE)
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(TARGET_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603 - fixed argv, no shell
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg không đọc được file: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _safe_translate(translator: Translator) -> Callable[[str], str]:
    def run(text: str) -> str:
        try:
            return translator.translate(text)
        except Exception as exc:  # noqa: BLE001 - keep the original line on failure
            log.warning("Translation failed for %r: %s", text[:40], exc)
            return text
    return run
