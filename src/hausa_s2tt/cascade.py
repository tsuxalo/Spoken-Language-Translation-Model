"""Composable Hausa ASR -> NLLB English cascade."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .inference import InferenceResult, WhisperRuntime, create_asr_runtime
from .mt import NLLBTranslator


@dataclass
class CascadeResult:
    hausa_text: str
    english_text: str
    audio_duration_seconds: float
    asr_seconds: float
    mt_seconds: float
    total_seconds: float
    real_time_factor: float
    timing_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CascadeTranslator:
    def __init__(
        self,
        asr: WhisperRuntime | None = None,
        mt: NLLBTranslator | None = None,
    ) -> None:
        self.asr = asr or create_asr_runtime()
        self.mt = mt or NLLBTranslator()

    def translate(self, audio: str | Path | bytes | dict[str, Any]) -> CascadeResult:
        started = time.perf_counter()
        asr_result = self.asr.process(audio)
        mt_started = time.perf_counter()
        english = self.mt.translate(asr_result.text)
        mt_seconds = time.perf_counter() - mt_started
        total = time.perf_counter() - started
        return CascadeResult(
            hausa_text=asr_result.text,
            english_text=english,
            audio_duration_seconds=asr_result.audio_duration_seconds,
            asr_seconds=asr_result.inference_seconds,
            mt_seconds=mt_seconds,
            total_seconds=total,
            real_time_factor=total / asr_result.audio_duration_seconds,
            timing_scope="per-example measured wall time",
        )

    def translate_many(
        self, audio_items: Iterable[str | Path | bytes | dict[str, Any]]
    ) -> list[CascadeResult]:
        asr_results: list[InferenceResult] = self.asr.process_many(audio_items)
        started = time.perf_counter()
        translations = self.mt.translate_batch(result.text for result in asr_results)
        mt_total = time.perf_counter() - started
        per_example_mt = mt_total / len(asr_results) if asr_results else 0.0
        return [
            CascadeResult(
                hausa_text=result.text,
                english_text=translation,
                audio_duration_seconds=result.audio_duration_seconds,
                asr_seconds=result.inference_seconds,
                mt_seconds=per_example_mt,
                total_seconds=result.inference_seconds + per_example_mt,
                real_time_factor=(result.inference_seconds + per_example_mt)
                / result.audio_duration_seconds,
                timing_scope="ASR per-example plus equal share of measured batched MT wall time",
            )
            for result, translation in zip(asr_results, translations)
        ]


def cascade_translate(audio: str | Path, *, asr_model_id: str = "nahomazmach/whisper-small-ha") -> str:
    return CascadeTranslator(asr=create_asr_runtime(asr_model_id)).translate(audio).english_text
