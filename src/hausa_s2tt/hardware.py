"""Deterministic seeding and runtime-aware precision selection."""

from __future__ import annotations

import os
import platform
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PrecisionSelection:
    device: str
    dtype: str
    bf16: bool
    fp16: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_precision(torch_module: Any | None = None, requested: str = "auto") -> PrecisionSelection:
    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("precision must be auto, bf16, fp16, or fp32")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return PrecisionSelection("cpu", "float32", False, False, "PyTorch unavailable")

    torch = torch_module
    has_cuda = bool(torch.cuda.is_available())
    has_mps = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )
    device = "cuda" if has_cuda else "mps" if has_mps else "cpu"
    bf16_supported = bool(
        has_cuda
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    if requested == "bf16":
        if not bf16_supported:
            raise RuntimeError("BF16 was requested but torch.cuda.is_bf16_supported() is false")
        return PrecisionSelection("cuda", "bfloat16", True, False, "explicit BF16")
    if requested == "fp16":
        if not has_cuda:
            raise RuntimeError("FP16 training was requested without CUDA")
        return PrecisionSelection("cuda", "float16", False, True, "explicit FP16")
    if requested == "fp32":
        return PrecisionSelection(device, "float32", False, False, "explicit FP32")
    if bf16_supported:
        return PrecisionSelection("cuda", "bfloat16", True, False, "CUDA BF16 supported")
    if has_cuda:
        return PrecisionSelection("cuda", "float16", False, True, "CUDA FP16 fallback")
    return PrecisionSelection(device, "float32", False, False, "CPU/MPS FP32 fallback")


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def hardware_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            result["torch_available"] = False
            return result
    torch = torch_module
    result.update(
        {
            "torch_available": True,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        }
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "vram_bytes": int(props.total_memory),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return result
