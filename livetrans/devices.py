"""Pick where Whisper runs: NVIDIA (CTranslate2) or Intel NPU / iGPU / CPU (OpenVINO)."""
import logging
from dataclasses import dataclass

from .cuda_setup import register_cuda_dlls

log = logging.getLogger(__name__)

DEVICES = ("auto", "npu", "igpu", "cuda", "cpu")
DEVICE_LABELS: dict[str, str] = {
    "Tự động": "auto",
    "Intel NPU": "npu",
    "Intel GPU (iGPU)": "igpu",
    "NVIDIA (CUDA)": "cuda",
    "CPU": "cpu",
}

OPENVINO = "openvino"
CTRANSLATE2 = "ctranslate2"


@dataclass(frozen=True)
class DevicePlan:
    backend: str   # OPENVINO | CTRANSLATE2
    device: str    # OpenVINO id ("NPU", "GPU.0", "CPU") or CTranslate2 device ("cuda", "cpu")
    label: str     # shown in the status bar


def cuda_available() -> bool:
    register_cuda_dlls()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception as exc:  # noqa: BLE001 - any CUDA probe failure means "no CUDA"
        log.info("CUDA unavailable: %s", exc)
        return False


def openvino_devices() -> dict[str, str] | None:
    """OpenVINO device id -> full name, or None when OpenVINO GenAI is not installed."""
    try:
        # openvino_genai must be imported before the first ov.Core(): the reverse order makes the
        # process hang at exit once a Whisper pipeline has run on an Intel GPU (OpenVINO 2026.4)
        import openvino_genai  # noqa: F401
        import openvino as ov
    except ImportError:
        return None
    core = ov.Core()
    devices: dict[str, str] = {}
    for d in core.available_devices:
        try:
            devices[d] = str(core.get_property(d, "FULL_DEVICE_NAME"))
        except RuntimeError as exc:  # a broken driver must not hide the other devices
            log.warning("OpenVINO device %s unusable: %s", d, exc)
    return devices


def intel_gpu_id(devices: dict[str, str]) -> str | None:
    """OpenVINO also lists NVIDIA cards as GPU.N via OpenCL; only Intel GPUs are supported."""
    return next((d for d, name in devices.items() if d.startswith("GPU") and "intel" in name.lower()), None)


def plan_candidates(requested: str, has_cuda: bool, ov_devices: dict[str, str] | None) -> list[DevicePlan]:
    """Ordered list of devices to try; later entries are fallbacks."""
    if requested not in DEVICES:
        raise ValueError(f"unknown device {requested!r}, expected one of {DEVICES}")
    ov = ov_devices or {}
    cuda = DevicePlan(CTRANSLATE2, "cuda", "NVIDIA CUDA")
    npu = DevicePlan(OPENVINO, "NPU", "Intel NPU") if "NPU" in ov else None
    gpu_id = intel_gpu_id(ov)
    igpu = DevicePlan(OPENVINO, gpu_id, "Intel iGPU") if gpu_id else None
    cpu = [DevicePlan(OPENVINO, "CPU", "CPU (OpenVINO)")] if ov_devices is not None else []
    cpu.append(DevicePlan(CTRANSLATE2, "cpu", "CPU"))

    if requested == "cuda":
        order = [cuda] + cpu
    elif requested == "cpu":
        order = cpu
    elif requested == "npu":
        order = [npu, igpu] + cpu
    elif requested == "igpu":
        order = [igpu, npu] + cpu
    else:
        order = ([cuda] if has_cuda else []) + [npu, igpu] + cpu
    return [p for p in order if p is not None]
