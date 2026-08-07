"""Artifact-backed evaluation and a guard against repeated final-test inspection."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .hardware import hardware_snapshot
from .metrics import compute_asr_metrics, compute_translation_metrics
from .telemetry import package_versions


class FinalTestGuard:
    def __init__(self, artifacts_dir: str | Path, run_name: str) -> None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_")
        if not safe_name:
            raise ValueError("run_name must contain at least one safe character")
        self.path = Path(artifacts_dir) / "final_test_runs" / f"{safe_name}.json"

    def ensure_unused(self, *, force: bool = False) -> None:
        if self.path.exists() and not force:
            previous = json.loads(self.path.read_text(encoding="utf-8"))
            raise RuntimeError(
                "Final test was already evaluated for this run. Use a new run name; "
                f"previous artifact: {self.path} ({previous.get('created_unix')})"
            )

    def seal(self, summary: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"created_unix": time.time(), "summary": dict(summary)}
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(self.path, flags)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_prediction_artifacts(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    payload = dict(summary)
    payload["packages"] = package_versions(
        [
            "torch",
            "transformers",
            "datasets",
            "sacrebleu",
            "sentencepiece",
            "numpy",
            "scipy",
            "soundfile",
        ]
    )
    payload["hardware"] = hardware_snapshot()
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def evaluate_asr_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    references = [str(row["reference"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    return {"examples": len(rows), **compute_asr_metrics(references, predictions)}


def evaluate_translation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    references = [str(row["reference"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    return {"examples": len(rows), **compute_translation_metrics(references, predictions)}


def analysis_flags(source: str, reference: str, duration: float | None = None) -> list[str]:
    combined = f"{source} {reference}"
    flags: list[str] = []
    if re.search(r"\b\d+[\d,.:/-]*\b", combined):
        flags.append("numbers_or_dates")
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", reference):
        flags.append("names_or_locations")
    if duration is not None and duration >= 20:
        flags.append("long_audio")
    return flags
