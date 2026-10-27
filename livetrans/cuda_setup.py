"""Make the pip-installed NVIDIA cuBLAS/cuDNN DLLs visible to CTranslate2 on Windows."""
import os
import site
import sys
from pathlib import Path

_NVIDIA_SUBDIRS = ("cublas", "cudnn", "cuda_nvrtc")


def register_cuda_dlls() -> list[str]:
    """Add nvidia/*/bin folders to the DLL search path. Returns the folders added."""
    if sys.platform != "win32":
        return []
    added = []
    roots = [Path(p) for p in site.getsitepackages()] + [Path(sys.prefix) / "Lib" / "site-packages"]
    for root in roots:
        for sub in _NVIDIA_SUBDIRS:
            bin_dir = root / "nvidia" / sub / "bin"
            if bin_dir.is_dir() and str(bin_dir) not in added:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                added.append(str(bin_dir))
    return added
