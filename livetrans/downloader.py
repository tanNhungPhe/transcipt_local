"""One-time model download (needs Internet). Afterwards the app runs fully offline."""
import logging
from pathlib import Path
from typing import Callable

from .devices import cuda_available, openvino_devices
from .model_store import (
    NLLB_REPO,
    OV_WHISPER_REPOS,
    ct2_whisper_dir,
    has_ct2_model,
    has_nllb,
    has_ov_whisper,
    nllb_dir,
    ov_whisper_dir,
)

log = logging.getLogger(__name__)

BACKENDS = ("auto", "openvino", "ctranslate2", "both")


def wanted_backends(backend: str, has_openvino: bool, has_cuda: bool) -> tuple[bool, bool]:
    """(download OpenVINO Whisper, download CTranslate2 Whisper)."""
    if backend == "both":
        return True, True
    if backend == "openvino":
        return True, False
    if backend == "ctranslate2":
        return False, True
    return has_openvino, has_cuda or not has_openvino


def _snapshot(repo: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=target)


def download_models(sizes: list[str], backend: str = "auto", say: Callable[[str], None] = print) -> None:
    want_ov, want_ct2 = wanted_backends(backend, openvino_devices() is not None, cuda_available())
    for size in sizes:
        if want_ov:
            target = ov_whisper_dir(size)
            if has_ov_whisper(target):
                say(f"✓ Whisper {size} (OpenVINO) đã có")
            else:
                say(f"↓ Whisper {size} (OpenVINO) → {target}")
                _snapshot(OV_WHISPER_REPOS[size], target)
        if want_ct2:
            target = ct2_whisper_dir(size)
            if has_ct2_model(target):
                say(f"✓ Whisper {size} (faster-whisper) đã có")
            else:
                from faster_whisper import download_model

                say(f"↓ Whisper {size} (faster-whisper) → {target}")
                download_model(size, output_dir=str(target))
    target = nllb_dir()
    if has_nllb(target):
        say("✓ Model dịch NLLB-200 đã có")
    else:
        say(f"↓ Model dịch NLLB-200 1.3B (~1.4 GB) → {target}")
        _snapshot(NLLB_REPO, target)


def warm_up(size: str, device: str, say: Callable[[str], None] = print) -> None:
    """First NPU/GPU compile is slow; doing it now fills the cache so the app opens fast."""
    from .transcriber import get_transcriber

    say(f"Đang biên dịch Whisper {size} cho thiết bị '{device}' (lần đầu có thể mất vài phút)…")
    transcriber = get_transcriber(size, device)
    say(f"✓ Whisper chạy trên: {transcriber.device}")
