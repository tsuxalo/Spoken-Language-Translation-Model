"""Whisper ASR/direct inference with batching-safe loading and long-audio chunks."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import iter_audio_chunks, load_audio
from .hardware import select_precision


@dataclass
class InferenceResult:
    text: str
    audio_duration_seconds: float
    inference_seconds: float
    real_time_factor: float
    segments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_text_segments(segments: list[str], max_overlap_words: int = 20) -> str:
    merged: list[str] = []
    for segment in segments:
        words = segment.strip().split()
        if not words:
            continue
        overlap = 0
        for size in range(min(len(merged), len(words), max_overlap_words), 0, -1):
            if [word.casefold() for word in merged[-size:]] == [
                word.casefold() for word in words[:size]
            ]:
                overlap = size
                break
        merged.extend(words[overlap:])
    return " ".join(merged)


class WhisperRuntime:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        task: str,
        language: str = "Hausa",
        precision: str = "auto",
        chunk_length_seconds: float = 29.0,
        stride_seconds: float = 0.0,
        max_new_tokens: int = 225,
        num_beams: int = 1,
        batch_size: int = 4,
    ) -> None:
        if task not in {"transcribe", "translate"}:
            raise ValueError("Whisper task must be transcribe or translate")
        self.model_id = model_id
        self.revision = revision
        self.task = task
        self.language = language
        self.precision_name = precision
        self.chunk_length_seconds = chunk_length_seconds
        self.stride_seconds = stride_seconds
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"

    def load(self) -> WhisperRuntime:
        if self.model is not None:
            return self
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        selected = select_precision(torch, self.precision_name)
        self.device = selected.device
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[selected.dtype]
        local_adapter = Path(self.model_id) / "adapter_config.json"
        self.processor = WhisperProcessor.from_pretrained(
            self.model_id, revision=self.revision
        )
        if local_adapter.is_file():
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:  # pragma: no cover - dependency message
                raise RuntimeError("peft is required to load a LoRA checkpoint") from exc
            adapter_config = PeftConfig.from_pretrained(self.model_id)
            base_id = adapter_config.base_model_name_or_path
            if not base_id:
                raise ValueError("LoRA adapter_config.json has no base model")
            base_model = WhisperForConditionalGeneration.from_pretrained(
                base_id,
                revision=adapter_config.revision,
                dtype=dtype,
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_id).to(
                self.device
            )
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id, revision=self.revision, dtype=dtype
            ).to(self.device)
        self.model.eval()
        return self

    def _generate(self, arrays: list[np.ndarray]) -> list[str]:
        self.load()
        import torch

        batch = self.processor.feature_extractor(
            arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            max_length=480_000,
            truncation=True,
            return_attention_mask=True,
        )
        inputs = batch["input_features"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            predicted = self.model.generate(
                input_features=inputs,
                attention_mask=attention_mask,
                language=self.language,
                task=self.task,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
        return [
            text.strip()
            for text in self.processor.tokenizer.batch_decode(
                predicted, skip_special_tokens=True
            )
        ]

    def _generate_batched(self, arrays: list[np.ndarray]) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(arrays), self.batch_size):
            texts.extend(self._generate(arrays[start : start + self.batch_size]))
        return texts

    def process(self, audio: str | Path | bytes | dict[str, Any]) -> InferenceResult:
        samples, sample_rate = load_audio(audio, target_rate=16_000)
        chunks = list(
            iter_audio_chunks(
                samples,
                sample_rate,
                chunk_length_seconds=self.chunk_length_seconds,
                stride_seconds=self.stride_seconds,
            )
        )
        started = time.perf_counter()
        texts = self._generate_batched([chunk for chunk, _, _ in chunks])
        elapsed = time.perf_counter() - started
        segment_records = [
            {"start_seconds": start, "end_seconds": end, "text": text}
            for (_, start, end), text in zip(chunks, texts)
        ]
        duration = samples.size / sample_rate
        return InferenceResult(
            text=_merge_text_segments(texts),
            audio_duration_seconds=duration,
            inference_seconds=elapsed,
            real_time_factor=elapsed / duration if duration else float("inf"),
            segments=segment_records,
        )

    def process_many(
        self, audio_items: Iterable[str | Path | bytes | dict[str, Any]]
    ) -> list[InferenceResult]:
        return [self.process(item) for item in audio_items]


def create_asr_runtime(model_id: str = "nahomazmach/whisper-small-ha", **kwargs: Any) -> WhisperRuntime:
    return WhisperRuntime(model_id, task="transcribe", **kwargs)


def create_direct_runtime(model_id: str, **kwargs: Any) -> WhisperRuntime:
    return WhisperRuntime(model_id, task="translate", **kwargs)


def create_zero_shot_runtime(model_id: str = "openai/whisper-small", **kwargs: Any) -> WhisperRuntime:
    return WhisperRuntime(model_id, task="translate", **kwargs)
