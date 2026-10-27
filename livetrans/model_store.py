"""Where every model lives on disk, and offline-mode switches."""
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OV_CACHE_DIR = MODELS_DIR / "ov_cache"

OV_WHISPER_REPOS: dict[str, str] = {
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "base": "OpenVINO/whisper-base-int8-ov",
    "tiny": "OpenVINO/whisper-tiny-int8-ov",
}
NLLB_REPO = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"


class ModelMissingError(RuntimeError):
    pass


def ov_whisper_dir(size: str) -> Path:
    return MODELS_DIR / "whisper-ov" / size


def ct2_whisper_dir(size: str) -> Path:
    return MODELS_DIR / "whisper-ct2" / size


def nllb_dir() -> Path:
    return MODELS_DIR / "nllb-1.3B"


def has_ov_whisper(path: Path) -> bool:
    return (path / "openvino_encoder_model.xml").is_file() and (path / "openvino_decoder_model.xml").is_file()


def has_ct2_model(path: Path) -> bool:
    return (path / "model.bin").is_file()


def has_nllb(path: Path) -> bool:
    return has_ct2_model(path) and (path / "tokenizer.json").is_file()


def missing(what: str, path: Path) -> ModelMissingError:
    return ModelMissingError(
        f"Chưa có model {what} tại {path}. Chạy 'setup_offline.bat' (cần Internet một lần) rồi mở lại."
    )


def openvino_consent_file() -> Path | None:
    """Location used by the openvino_telemetry package for its opt-in/opt-out flag."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "Intel Corporation" / "openvino_telemetry" if base else None
    return Path.home() / "intel" / "openvino_telemetry"


def opt_out_openvino_telemetry(path: Path | None = None) -> None:
    """`import openvino` sends usage stats to Google Analytics unless this file holds "0".

    Writing "0" is what Intel's own `opt_in_out --opt_out` command does.
    """
    path = path or openvino_consent_file()
    if path is None:
        return
    try:
        if path.is_file() and path.read_text(encoding="utf-8").strip() == "0":
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0", encoding="utf-8")
        log.info("Disabled OpenVINO telemetry (%s)", path)
    except OSError as exc:
        log.warning("Could not disable OpenVINO telemetry at %s: %s", path, exc)


def enable_offline_mode() -> None:
    """Block HuggingFace and OpenVINO network calls; must run before either library is imported."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    opt_out_openvino_telemetry()
