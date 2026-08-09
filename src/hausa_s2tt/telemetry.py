"""Run manifests, timing, throughput, VRAM, RTF, and pilot estimates."""

from __future__ import annotations

import importlib.metadata
import json
import math
import statistics
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from .hardware import hardware_snapshot


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parameter_counts(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": trainable / total if total else 0.0,
    }


def directory_size_bytes(path: str | Path) -> int:
    directory = Path(path)
    if not directory.exists():
        return 0
    return sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())


@dataclass
class RuntimeMeasurement:
    wall_seconds: float
    examples: int
    audio_seconds: float
    gpu_count: int
    peak_vram_bytes: int | None = None

    @property
    def examples_per_second(self) -> float:
        return self.examples / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def audio_seconds_per_second(self) -> float:
        return self.audio_seconds / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def real_time_factor(self) -> float | None:
        return self.wall_seconds / self.audio_seconds if self.audio_seconds else None

    @property
    def gpu_hours(self) -> float:
        return self.wall_seconds * self.gpu_count / 3600

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "examples_per_second": self.examples_per_second,
                "audio_seconds_per_second": self.audio_seconds_per_second,
                "real_time_factor": self.real_time_factor,
                "gpu_hours": self.gpu_hours,
            }
        )
        return value


class RuntimeTimer:
    def __init__(self, torch_module: Any | None = None) -> None:
        self.torch = torch_module
        self.started = 0.0

    def __enter__(self) -> Self:
        if self.torch is None:
            try:
                import torch

                self.torch = torch
            except ImportError:
                pass
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.wall_seconds = time.perf_counter() - self.started

    def measurement(self, examples: int, audio_seconds: float) -> RuntimeMeasurement:
        torch = self.torch
        gpu_count = (
            int(torch.cuda.device_count())
            if torch is not None and torch.cuda.is_available()
            else 0
        )
        peak = int(torch.cuda.max_memory_allocated()) if gpu_count else None
        return RuntimeMeasurement(
            self.wall_seconds, examples, audio_seconds, gpu_count, peak
        )


def estimate_full_run(
    pilot: RuntimeMeasurement,
    *,
    pilot_examples: int,
    full_examples_per_epoch: int,
    epochs: float,
    checkpoint_bytes_per_save: int = 0,
    expected_saves: int = 0,
    gpu_usd_per_hour: float | None = None,
) -> dict[str, Any]:
    if pilot_examples <= 0 or full_examples_per_epoch <= 0 or epochs <= 0:
        raise ValueError("Pilot and projected workload sizes must be positive")
    scale = (full_examples_per_epoch * epochs) / pilot_examples
    projected_seconds = pilot.wall_seconds * scale
    projected_gpu_hours = projected_seconds * pilot.gpu_count / 3600
    return {
        "measurement_basis": pilot.to_dict(),
        "assumptions": {
            "linear_scaling": True,
            "full_examples_per_epoch": full_examples_per_epoch,
            "epochs": epochs,
            "checkpoint_bytes_per_save": checkpoint_bytes_per_save,
            "expected_saves": expected_saves,
        },
        "projected_wall_seconds": projected_seconds,
        "projected_gpu_hours": projected_gpu_hours,
        "projected_checkpoint_storage_bytes": checkpoint_bytes_per_save
        * expected_saves,
        "projected_gpu_cost_usd": (
            projected_gpu_hours * gpu_usd_per_hour
            if gpu_usd_per_hour is not None
            else None
        ),
    }


def estimate_error_percent(predicted_seconds: float, actual_seconds: float) -> float:
    if actual_seconds <= 0:
        raise ValueError("actual_seconds must be positive")
    return abs(predicted_seconds - actual_seconds) / actual_seconds * 100


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def estimate_training_from_pilot_phases(
    steady_step_seconds: Iterable[float],
    *,
    optimizer_steps_per_epoch: int,
    epochs: float,
    startup_seconds: float,
    validation_seconds_per_run: float,
    validation_runs: int,
    checkpoint_seconds_per_save: float,
    checkpoint_saves: int,
    gpu_count: int,
    checkpoint_bytes_per_save: int = 0,
    examples_per_step: int | None = None,
    audio_seconds_per_step: float | None = None,
    gpu_usd_per_hour: float | None = None,
) -> dict[str, Any]:
    """Project a run from hardware-matched steady-state samples and fixed phases."""
    samples = [float(value) for value in steady_step_seconds]
    if not samples or any(value <= 0 or not math.isfinite(value) for value in samples):
        raise ValueError(
            "steady_step_seconds must contain positive finite measurements"
        )
    if optimizer_steps_per_epoch <= 0 or epochs <= 0:
        raise ValueError("Projected optimizer steps and epochs must be positive")
    if any(
        value < 0
        for value in (
            startup_seconds,
            validation_seconds_per_run,
            validation_runs,
            checkpoint_seconds_per_save,
            checkpoint_saves,
            gpu_count,
            checkpoint_bytes_per_save,
        )
    ):
        raise ValueError("Phase durations, counts, and storage must be non-negative")
    projected_steps = math.ceil(optimizer_steps_per_epoch * epochs)
    fixed_seconds = (
        startup_seconds
        + validation_seconds_per_run * validation_runs
        + checkpoint_seconds_per_save * checkpoint_saves
    )
    step_median = statistics.median(samples)
    step_mean = statistics.fmean(samples)
    step_p10 = _quantile(samples, 0.10)
    step_p90 = _quantile(samples, 0.90)
    projected_seconds = fixed_seconds + projected_steps * step_median
    lower_seconds = fixed_seconds + projected_steps * step_p10
    upper_seconds = fixed_seconds + projected_steps * step_p90
    gpu_hours = projected_seconds * gpu_count / 3600
    return {
        "status": "estimated_from_hardware_matched_pilot",
        "measurement_basis": {
            "steady_step_samples": len(samples),
            "steady_step_seconds_mean": step_mean,
            "steady_step_seconds_median": step_median,
            "steady_step_seconds_p10": step_p10,
            "steady_step_seconds_p90": step_p90,
            "startup_seconds": startup_seconds,
            "validation_seconds_per_run": validation_seconds_per_run,
            "checkpoint_seconds_per_save": checkpoint_seconds_per_save,
        },
        "assumptions": {
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "epochs": epochs,
            "projected_optimizer_steps": projected_steps,
            "validation_runs": validation_runs,
            "checkpoint_saves": checkpoint_saves,
            "gpu_count": gpu_count,
            "uncertainty_method": (
                "observed pilot p10-p90 steady-step duration; fixed startup, validation, "
                "and checkpoint phases"
            ),
        },
        "projected_wall_seconds": projected_seconds,
        "projected_wall_seconds_interval": [lower_seconds, upper_seconds],
        "projected_gpu_hours": gpu_hours,
        "projected_checkpoint_storage_bytes": checkpoint_bytes_per_save
        * checkpoint_saves,
        "projected_gpu_cost_usd": (
            gpu_hours * gpu_usd_per_hour if gpu_usd_per_hour is not None else None
        ),
        "projected_examples_per_second": (
            examples_per_step / step_median if examples_per_step is not None else None
        ),
        "projected_audio_seconds_per_second": (
            audio_seconds_per_step / step_median
            if audio_seconds_per_step is not None
            else None
        ),
    }


def build_run_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_unix": time.time(),
        "git_commit": git_commit(),
        "config": config,
        "hardware": hardware_snapshot(),
        "packages": package_versions(
            [
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "peft",
                "sacrebleu",
                "jiwer",
                "soundfile",
                "numpy",
                "scipy",
                "sentencepiece",
            ]
        ),
    }


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    artifact = Path(path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
