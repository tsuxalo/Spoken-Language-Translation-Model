"""Direct Hausa-audio to English-text Whisper training entrypoint."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .evaluation import evaluate_translation_rows, write_prediction_artifacts
from .hardware import hardware_snapshot
from .inference import create_direct_runtime
from .telemetry import parameter_counts, write_json_artifact
from .training import SpeechSeq2SeqCollator, load_training_data, train_experiment

MATCHED_PILOT_ALLOWED_DIFFERENCES = frozenset(
    {
        "model.id",
        "model.revision",
        "training.output_dir",
        "tracking.run_name",
    }
)


def _flatten_config(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            result.update(_flatten_config(child, path))
        else:
            result[path] = child
    return result


def matched_config_differences(
    left: ExperimentConfig,
    right: ExperimentConfig,
    *,
    allowed_differences: Sequence[str] = tuple(MATCHED_PILOT_ALLOWED_DIFFERENCES),
) -> dict[str, tuple[Any, Any]]:
    """Return unintended differences between supposedly matched pilot configs."""
    left.validate()
    right.validate()
    left_flat = _flatten_config(left.to_dict())
    right_flat = _flatten_config(right.to_dict())
    allowed = set(allowed_differences)
    return {
        name: (left_flat.get(name), right_flat.get(name))
        for name in sorted(set(left_flat) | set(right_flat))
        if left_flat.get(name) != right_flat.get(name) and name not in allowed
    }


def assert_matched_direct_configs(
    left: ExperimentConfig,
    right: ExperimentConfig,
) -> dict[str, tuple[Any, Any]]:
    differences = matched_config_differences(left, right)
    if differences:
        raise ValueError(f"Matched direct-S2TT configs drifted: {differences}")
    return differences


def select_direct_architecture(
    candidate_runs: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze a validation-only architecture decision as a machine-readable artifact."""
    if len(candidate_runs) < 2:
        raise ValueError(
            "Architecture selection requires at least two matched candidate runs"
        )
    checked: list[dict[str, Any]] = []
    required_review_fields = {
        "status",
        "source_language_leakage",
        "hallucination_repetition",
        "omissions_additions",
        "names_numbers_dates_negation",
        "eligible_for_selection",
    }
    for run in candidate_runs:
        candidate = dict(run)
        if candidate.get("official_dev_evaluated") is not False:
            raise ValueError(
                "Architecture selection requires official_dev_evaluated=false"
            )
        if candidate.get("split") != "validation":
            raise ValueError(
                "Architecture selection may use only the project validation split"
            )
        for metric in ("chrf_pp", "sacrebleu", "validation_loss"):
            if not isinstance(candidate.get(metric), (int, float)):
                raise TypeError(f"Candidate run lacks numeric {metric}")
        review = candidate.get("qualitative_review")
        if not isinstance(review, Mapping):
            raise TypeError("Candidate run lacks a qualitative_review mapping")
        missing_review = sorted(required_review_fields - set(review))
        if missing_review or review.get("status") != "complete":
            raise ValueError(
                f"Candidate qualitative review is incomplete; missing={missing_review}"
            )
        checked.append(candidate)
    eligible = [
        candidate
        for candidate in checked
        if candidate["qualitative_review"]["eligible_for_selection"] is True
    ]
    if not eligible:
        raise ValueError("No candidate passed qualitative eligibility review")
    ranked = sorted(
        eligible,
        key=lambda item: (
            float(item["chrf_pp"]),
            float(item["sacrebleu"]),
            -float(item["validation_loss"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    artifact = {
        "artifact_schema_version": "1.0",
        "selection_rule": (
            "qualitative eligibility gate, then highest speaker-disjoint validation chrF++; "
            "SacreBLEU then lower validation loss break exact ties"
        ),
        "primary_metric": "chrf_pp",
        "candidate_runs": checked,
        "selected_run": selected.get("run_name"),
        "selected_model_id": selected.get("model_id"),
        "selected_model_revision": selected.get("model_revision"),
        "reason": (
            f"Selected {selected.get('run_name')} by the predeclared validation-only rule."
        ),
        "official_dev_evaluated": False,
    }
    write_json_artifact(output_path, artifact)
    return artifact


def evaluate_initialization_baseline(
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate an untouched initialization on project validation, never official dev."""
    import time

    import torch
    from torch.utils.data import DataLoader

    config.validate()
    if config.kind != "direct_s2tt":
        raise ValueError("Initialization baselines require a direct_s2tt config")
    _, validation, target_column, data_manifest = load_training_data(config)
    if data_manifest.get("official_dev_evaluated") is not False:
        raise RuntimeError("Direct baseline data manifest lacks the official-dev guard")
    runtime = create_direct_runtime(
        config.model.id,
        revision=config.model.revision,
        precision=config.training.precision,
        chunk_length_seconds=config.generation.chunk_length_seconds,
        stride_seconds=config.generation.stride_seconds,
        max_new_tokens=config.generation.max_new_tokens,
        num_beams=config.generation.num_beams,
        batch_size=config.generation.batch_size,
    ).load()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    collator = SpeechSeq2SeqCollator(
        runtime.processor,
        target_column=target_column,
        audio_column=config.dataset.audio_column,
        sampling_rate=config.dataset.sampling_rate,
        max_duration_seconds=config.dataset.max_duration_seconds,
    )
    loader = DataLoader(
        validation,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    loss_total = 0.0
    loss_examples = 0
    loss_started = time.perf_counter()
    for batch in loader:
        batch_size = int(batch["labels"].shape[0])
        prepared = {
            name: value.to(runtime.device)
            if name == "labels" or not value.is_floating_point()
            else value.to(device=runtime.device, dtype=runtime.torch_dtype)
            for name, value in batch.items()
        }
        with torch.inference_mode():
            loss = runtime.model(**prepared).loss
        loss_total += float(loss.item()) * batch_size
        loss_examples += batch_size
    loss_seconds = time.perf_counter() - loss_started
    rows: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    for index in range(len(validation)):
        item = validation[index]
        result = runtime.process(item[config.dataset.audio_column])
        rows.append(
            {
                "source_id": item.get("source_text_id", index),
                "alignment_key": item.get("alignment_key"),
                "dataset_row_index": item.get("dataset_row_index"),
                "speaker_id": item.get("speaker_id"),
                "source_transcript": item.get("source_text"),
                "reference": item[target_column],
                "prediction": result.text,
                "duration": item.get("duration"),
                "inference_seconds": result.inference_seconds,
                "real_time_factor": result.real_time_factor,
                "project_split": "validation",
            }
        )
    generation_seconds = time.perf_counter() - generation_started
    metrics = evaluate_translation_rows(rows)
    metrics.update(
        {
            "run_name": config.tracking.run_name,
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "split": "validation",
            "validation_loss": loss_total / loss_examples,
            "loss_seconds": loss_seconds,
            "generation_seconds": generation_seconds,
            "mean_rtf": sum(row["real_time_factor"] for row in rows) / len(rows),
            "generation_settings": config.generation.__dict__,
            "data": data_manifest,
            "hardware": hardware_snapshot(),
            "parameters": parameter_counts(runtime.model),
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
            "official_dev_evaluated": False,
        }
    )
    write_prediction_artifacts(rows, metrics, output_dir)
    return metrics


def write_notebook03_handoff(
    config: ExperimentConfig,
    training_summary: Mapping[str, Any],
    *,
    architecture_selection_path: str | Path,
    output_path: str | Path,
    training_status: str,
) -> dict[str, Any]:
    """Freeze the direct-model contract consumed later by protected evaluation."""
    config.validate()
    summary = dict(training_summary)
    if summary.get("official_dev_evaluated") is not False:
        raise ValueError("Notebook 03 handoff requires official_dev_evaluated=false")
    selection_path = Path(architecture_selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("official_dev_evaluated") is not False:
        raise ValueError("Architecture selection used or failed to guard official dev")
    if (
        selection.get("selected_model_id") != config.model.id
        or selection.get("selected_model_revision") != config.model.revision
    ):
        raise ValueError("Notebook 03 handoff config is not the frozen selected model")
    checkpoint_dir = Path(config.training.output_dir)
    manifest_path = checkpoint_dir / "run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("official_dev_evaluated") is not False:
        raise ValueError("Training manifest lacks the official-dev guard")
    handoff = {
        "artifact_schema_version": "1.0",
        "selected_model_initialization": config.model.id,
        "resolved_config": config.to_dict(),
        "model_revision": config.model.revision,
        "dataset_revision": config.dataset.revision,
        "project_split_provenance": run_manifest["data"],
        "adapter_or_checkpoint_path": str(checkpoint_dir),
        "processor_path": str(checkpoint_dir),
        "generation_config": summary["generation_settings"],
        "validation_metrics": summary["validation_metrics"],
        "validation_predictions": summary["validation_predictions"],
        "training_history": summary["trainer_history"],
        "runtime_telemetry": summary["runtime"],
        "parameter_counts": summary["parameters"],
        "pilot_or_full_status": training_status,
        "architecture_selection_artifact": str(selection_path),
        "official_dev_evaluated": False,
    }
    write_json_artifact(output_path, handoff)
    return handoff


def train_direct_s2tt(config: ExperimentConfig):
    if config.kind != "direct_s2tt" or config.model.task != "translate":
        raise ValueError("Direct training requires kind=direct_s2tt and task=translate")
    if config.dataset.target_language.lower() != "english":
        raise ValueError("This pipeline requires genuine English translation labels")
    if config.dataset.target_column != "target_text":
        raise ValueError("This pipeline requires genuine aligned target_text labels")
    return train_experiment(config)
