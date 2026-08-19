"""Pinned C1 Hausa-to-English direct speech-translation inference.

The C1 graph is a Transformers ``SpeechEncoderDecoderModel`` with a
Wav2Vec2/XLS-R speech encoder and an mBART text decoder. It is independent of
the repository's historical Whisper/PEFT direct pilot.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from experiments.revisions import C1_MODEL_ID as MODEL_ID
from experiments.revisions import C1_MODEL_REVISION as MODEL_REVISION

SAMPLING_RATE = 16_000
MAX_SCORED_DURATION_SECONDS = 30.0

FROZEN_GENERATION = {
    "num_beams": 5,
    "max_new_tokens": 128,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
}


class AudioValidationError(ValueError):
    """Raised when decoded audio violates the C1 input contract."""


class EvaluationDurationError(AudioValidationError):
    """Raised when a clip is too long for the frozen scored-evaluation scope."""


class ModelContractError(RuntimeError):
    """Raised when the pinned Hub package no longer matches the verified contract."""


@dataclass(frozen=True)
class PreparedAudio:
    samples: np.ndarray
    sampling_rate: int
    duration_seconds: float


@dataclass(frozen=True)
class InferenceTelemetry:
    system: str
    model_id: str
    model_revision: str
    device: str
    audio_duration_seconds: float
    load_time_seconds: float
    inference_time_seconds: float
    real_time_factor: float
    peak_cuda_memory_mb: float | None
    parameter_count: int
    prediction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_audio(
    audio: np.ndarray | list[float],
    sampling_rate: int,
    *,
    scored_evaluation: bool = True,
) -> PreparedAudio:
    """Validate, downmix, and resample decoded audio without normalization.

    Two-dimensional input follows SoundFile's ``(frames, channels)`` layout.
    Audio is never truncated or chunked. Duration is measured from the decoded
    waveform before resampling.
    """

    if not isinstance(sampling_rate, (int, np.integer)) or sampling_rate <= 0:
        raise AudioValidationError("sampling_rate must be a positive integer")

    samples = np.asarray(audio)
    if samples.size == 0:
        raise AudioValidationError("audio is empty")
    if samples.ndim not in (1, 2):
        raise AudioValidationError("audio must have shape (frames,) or (frames, channels)")
    if samples.ndim == 2:
        if samples.shape[0] == 0 or samples.shape[1] == 0:
            raise AudioValidationError("audio is empty")
        samples = samples.astype(np.float64, copy=False).mean(axis=1)

    samples = np.asarray(samples, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise AudioValidationError("audio contains NaN or infinite values")

    duration_seconds = float(samples.shape[0] / int(sampling_rate))
    if scored_evaluation and duration_seconds > MAX_SCORED_DURATION_SECONDS:
        raise EvaluationDurationError(
            "C1's frozen evaluation contract covers clips no longer than 30 seconds. "
            "Longer clips are excluded from scored evaluation even if the underlying "
            "architecture might technically accept them."
        )

    if int(sampling_rate) != SAMPLING_RATE:
        divisor = math.gcd(int(sampling_rate), SAMPLING_RATE)
        samples = resample_poly(
            samples,
            up=SAMPLING_RATE // divisor,
            down=int(sampling_rate) // divisor,
        ).astype(np.float32, copy=False)

    if samples.size == 0 or not np.isfinite(samples).all():
        raise AudioValidationError("resampling produced invalid audio")

    return PreparedAudio(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sampling_rate=SAMPLING_RATE,
        duration_seconds=duration_seconds,
    )


def read_audio(
    wav_path: str | Path,
    *,
    scored_evaluation: bool = True,
) -> PreparedAudio:
    """Decode a SoundFile-supported audio file as float32 and prepare it for C1."""

    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    samples, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    return prepare_audio(
        samples,
        int(sampling_rate),
        scored_evaluation=scored_evaluation,
    )


def validate_generation_contract(generation_config: Any) -> None:
    """Reject any change to C1's frozen scored-generation settings."""

    mismatches = {
        key: {"expected": expected, "actual": getattr(generation_config, key, None)}
        for key, expected in FROZEN_GENERATION.items()
        if getattr(generation_config, key, None) != expected
    }
    if mismatches:
        raise ModelContractError(f"frozen generation mismatch: {mismatches}")


class C1Runtime:
    """Lazy, float32 runtime for the immutable C1 Hub package."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device: str | None = None,
        processor_cls: type[Any] | None = None,
        model_cls: type[Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self._processor_cls = processor_cls
        self._model_cls = model_cls
        self.processor: Any | None = None
        self.model: Any | None = None
        self.parameter_count = 0
        self.load_time_seconds = 0.0
        self.last_telemetry: InferenceTelemetry | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return

        from transformers import SpeechEncoderDecoderModel, Wav2Vec2Processor

        processor_cls = self._processor_cls or Wav2Vec2Processor
        model_cls = self._model_cls or SpeechEncoderDecoderModel

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        started = time.perf_counter()
        processor = processor_cls.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
        )
        model = model_cls.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        model = model.to(self.device).eval()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.load_time_seconds = time.perf_counter() - started

        config = model.config
        if list(getattr(config, "architectures", [])) != ["SpeechEncoderDecoderModel"]:
            raise ModelContractError(f"unexpected architectures: {config.architectures!r}")
        if getattr(config.encoder, "model_type", None) != "wav2vec2":
            raise ModelContractError("C1 encoder is not Wav2Vec2/XLS-R")
        if getattr(config.decoder, "model_type", None) != "mbart":
            raise ModelContractError("C1 decoder is not mBART")
        if getattr(processor.feature_extractor, "sampling_rate", None) != SAMPLING_RATE:
            raise ModelContractError("C1 processor sampling rate is not 16 kHz")
        if not getattr(processor.feature_extractor, "return_attention_mask", False):
            raise ModelContractError("C1 processor does not return an attention mask")
        validate_generation_contract(model.generation_config)

        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.processor = processor
        self.model = model

    def _generate(self, prepared: PreparedAudio) -> str:
        self._ensure_loaded()
        assert self.processor is not None and self.model is not None

        inputs = self.processor(
            prepared.samples,
            sampling_rate=prepared.sampling_rate,
            return_tensors="pt",
        )
        if "attention_mask" not in inputs:
            raise ModelContractError("processor output omitted attention_mask")
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        inference_seconds = time.perf_counter() - started

        prediction = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()
        peak_mb = None
        if self.device.startswith("cuda"):
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        self.last_telemetry = InferenceTelemetry(
            system="direct_c1",
            model_id=self.model_id,
            model_revision=self.revision,
            device=self.device,
            audio_duration_seconds=prepared.duration_seconds,
            load_time_seconds=self.load_time_seconds,
            inference_time_seconds=inference_seconds,
            real_time_factor=inference_seconds / prepared.duration_seconds,
            peak_cuda_memory_mb=peak_mb,
            parameter_count=self.parameter_count,
            prediction=prediction,
        )
        return prediction

    def translate_array(
        self,
        audio: np.ndarray | list[float],
        sampling_rate: int,
        *,
        scored_evaluation: bool = True,
    ) -> str:
        prepared = prepare_audio(
            audio,
            sampling_rate,
            scored_evaluation=scored_evaluation,
        )
        return self._generate(prepared)

    def translate_file(
        self,
        wav_path: str | Path,
        *,
        scored_evaluation: bool = True,
    ) -> str:
        prepared = read_audio(wav_path, scored_evaluation=scored_evaluation)
        return self._generate(prepared)

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.last_telemetry = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_DEFAULT_RUNTIME: C1Runtime | None = None


def _runtime() -> C1Runtime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = C1Runtime()
    return _DEFAULT_RUNTIME


def translate_file(wav_path: str | Path) -> str:
    return _runtime().translate_file(wav_path)


def translate_array(audio: np.ndarray | list[float], sampling_rate: int) -> str:
    return _runtime().translate_array(audio, sampling_rate)


def close() -> None:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is not None:
        _DEFAULT_RUNTIME.close()
        _DEFAULT_RUNTIME = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Hausa audio file")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--unscored-allow-long-audio",
        action="store_true",
        help="Allow >30 s input only for an explicitly unscored experiment",
    )
    args = parser.parse_args(argv)

    runtime = C1Runtime(device=args.device)
    try:
        prediction = runtime.translate_file(
            args.audio,
            scored_evaluation=not args.unscored_allow_long_audio,
        )
        telemetry = runtime.last_telemetry
        assert telemetry is not None
        print("System: C1 direct Hausa-to-English S2TT")
        print(f"Model ID: {telemetry.model_id}")
        print(f"Model revision: {telemetry.model_revision}")
        print(f"Device: {telemetry.device}")
        print(f"Audio duration: {telemetry.audio_duration_seconds:.3f} s")
        print(f"Model parameters: {telemetry.parameter_count:,}")
        print(f"English prediction: {prediction}")
        print(f"Model load time: {telemetry.load_time_seconds:.3f} s")
        print(f"Elapsed inference time: {telemetry.inference_time_seconds:.3f} s")
        print(f"Real-time factor: {telemetry.real_time_factor:.4f}")
        if telemetry.peak_cuda_memory_mb is not None:
            print(f"Peak CUDA memory: {telemetry.peak_cuda_memory_mb:.1f} MiB")
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
