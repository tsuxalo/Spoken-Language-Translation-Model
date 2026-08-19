from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

import direct_c1


def test_float32_conversion_and_mono_downmix() -> None:
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)
    prepared = direct_c1.prepare_audio(stereo, 16_000)
    assert prepared.samples.dtype == np.float32
    np.testing.assert_array_equal(prepared.samples, np.array([2.0, 3.0], dtype=np.float32))


def test_resampling_to_exact_contract_rate() -> None:
    prepared = direct_c1.prepare_audio(np.ones(8_000), 8_000)
    assert prepared.sampling_rate == 16_000
    assert len(prepared.samples) == 16_000
    assert prepared.duration_seconds == 1.0


@pytest.mark.parametrize("audio", [np.array([]), np.empty((0, 2))])
def test_empty_audio_rejected(audio: np.ndarray) -> None:
    with pytest.raises(direct_c1.AudioValidationError, match="empty"):
        direct_c1.prepare_audio(audio, 16_000)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_audio_rejected(value: float) -> None:
    with pytest.raises(direct_c1.AudioValidationError, match="NaN or infinite"):
        direct_c1.prepare_audio(np.array([0.0, value]), 16_000)


def test_30_second_evaluation_limit() -> None:
    direct_c1.prepare_audio(np.zeros(30 * 16_000), 16_000)
    with pytest.raises(direct_c1.EvaluationDurationError, match="no longer than 30 seconds"):
        direct_c1.prepare_audio(np.zeros(30 * 16_000 + 1), 16_000)


class _FakeFeatureExtractor:
    sampling_rate = 16_000
    return_attention_mask = True


class _FakeProcessor:
    feature_extractor = _FakeFeatureExtractor()
    calls: ClassVar[list[tuple[tuple[object, ...], dict[str, object]]]] = []

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> _FakeProcessor:
        cls.calls.append((args, kwargs))
        return cls()


class _FakeParameter:
    def numel(self) -> int:
        return 7


class _FakeGeneration:
    num_beams = 5
    max_new_tokens = 128
    early_stopping = True
    no_repeat_ngram_size = 3
    repetition_penalty = 1.0
    length_penalty = 1.0


class _FakeModel:
    calls: ClassVar[list[tuple[tuple[object, ...], dict[str, object]]]] = []

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            architectures=["SpeechEncoderDecoderModel"],
            encoder=SimpleNamespace(model_type="wav2vec2"),
            decoder=SimpleNamespace(model_type="mbart"),
        )
        self.generation_config = _FakeGeneration()

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> _FakeModel:
        cls.calls.append((args, kwargs))
        return cls()

    def to(self, _device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def parameters(self) -> list[_FakeParameter]:
        return [_FakeParameter()]


def test_loader_uses_pinned_concrete_class_contract() -> None:
    _FakeProcessor.calls.clear()
    _FakeModel.calls.clear()
    runtime = direct_c1.C1Runtime(
        device="cpu", processor_cls=_FakeProcessor, model_cls=_FakeModel
    )
    runtime._ensure_loaded()
    assert _FakeProcessor.calls[0][0] == (direct_c1.MODEL_ID,)
    assert _FakeModel.calls[0][0] == (direct_c1.MODEL_ID,)
    for _, kwargs in (_FakeProcessor.calls[0], _FakeModel.calls[0]):
        assert kwargs["revision"] == direct_c1.MODEL_REVISION
        assert kwargs["trust_remote_code"] is False
    assert _FakeModel.calls[0][1]["dtype"] is torch.float32
    assert runtime.parameter_count == 7


def test_frozen_generation_contract() -> None:
    direct_c1.validate_generation_contract(_FakeGeneration())
    broken = _FakeGeneration()
    broken.num_beams = 1
    with pytest.raises(direct_c1.ModelContractError, match="num_beams"):
        direct_c1.validate_generation_contract(broken)


def test_c1_generate_has_no_whisper_only_arguments() -> None:
    source = inspect.getsource(direct_c1.C1Runtime._generate)
    for forbidden in ("language=", "task=", "input_features", "PeftModel", "Whisper"):
        assert forbidden not in source
