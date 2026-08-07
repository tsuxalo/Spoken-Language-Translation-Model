"""Audio loading, validation, channel conversion, resampling, and chunking."""

from __future__ import annotations

import io
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


class AudioValidationError(ValueError):
    """Raised when an audio sample is unsafe or unusable."""


def to_mono(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise AudioValidationError(f"Expected mono or stereo audio, got shape {array.shape}")
    # soundfile returns [frames, channels]. Accept [channels, frames] when obvious.
    channel_axis = 0 if array.shape[0] <= 8 and array.shape[1] > array.shape[0] else 1
    return array.mean(axis=channel_axis, dtype=np.float32)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise AudioValidationError("Sample rates must be positive")
    array = to_mono(samples)
    if source_rate == target_rate:
        return array.astype(np.float32, copy=False)
    if array.size == 0:
        raise AudioValidationError("Cannot resample empty audio")
    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, target_rate)
    result = resample_poly(
        array,
        up=target_rate // divisor,
        down=source_rate // divisor,
    ).astype(np.float32, copy=False)
    target_length = max(1, round(array.size * target_rate / source_rate))
    if result.size > target_length:
        result = result[:target_length]
    elif result.size < target_length:
        result = np.pad(result, (0, target_length - result.size))
    return result


def validate_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    max_duration_seconds: float | None = None,
) -> np.ndarray:
    array = to_mono(samples)
    if sample_rate <= 0:
        raise AudioValidationError("Sample rate must be positive")
    if array.size == 0:
        raise AudioValidationError("Audio is empty")
    if not np.isfinite(array).all():
        raise AudioValidationError("Audio contains NaN or infinite samples")
    duration = array.size / sample_rate
    if max_duration_seconds is not None and duration > max_duration_seconds:
        raise AudioValidationError(
            f"Audio duration {duration:.3f}s exceeds {max_duration_seconds:.3f}s"
        )
    return array.astype(np.float32, copy=False)


def load_audio(
    value: str | Path | bytes | dict[str, Any],
    *,
    target_rate: int = 16_000,
    max_duration_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load common Hugging Face or filesystem audio representations."""
    if isinstance(value, dict) and value.get("array") is not None:
        source_rate = int(value.get("sampling_rate") or target_rate)
        samples = np.asarray(value["array"])
    else:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("soundfile is required to decode audio") from exc
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                source = io.BytesIO(value["bytes"])
            elif value.get("path"):
                source = value["path"]
            else:
                raise AudioValidationError("Audio mapping has no array, bytes, or path")
        elif isinstance(value, bytes):
            source = io.BytesIO(value)
        else:
            source = str(value)
        try:
            samples, source_rate = sf.read(source, always_2d=False)
        except Exception as exc:
            raise AudioValidationError(f"Unable to decode audio: {exc}") from exc
    mono = validate_audio(np.asarray(samples), int(source_rate))
    resampled = resample_audio(mono, int(source_rate), target_rate)
    validated = validate_audio(
        resampled,
        target_rate,
        max_duration_seconds=max_duration_seconds,
    )
    return validated, target_rate


def iter_audio_chunks(
    samples: np.ndarray,
    sample_rate: int,
    *,
    chunk_length_seconds: float = 29.0,
    stride_seconds: float = 0.0,
) -> Iterator[tuple[np.ndarray, float, float]]:
    array = validate_audio(samples, sample_rate)
    if not 0 < chunk_length_seconds <= 30:
        raise ValueError("chunk_length_seconds must be in (0, 30]")
    if not 0 <= stride_seconds < chunk_length_seconds:
        raise ValueError("stride_seconds must be in [0, chunk_length_seconds)")
    chunk_samples = max(1, round(chunk_length_seconds * sample_rate))
    hop_samples = max(1, chunk_samples - round(stride_seconds * sample_rate))
    for start in range(0, array.size, hop_samples):
        end = min(array.size, start + chunk_samples)
        chunk = array[start:end]
        if chunk.size == 0:
            continue
        yield chunk, start / sample_rate, end / sample_rate
        if end == array.size:
            break


def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return math.fsum([float(np.asarray(samples).shape[0])]) / sample_rate
