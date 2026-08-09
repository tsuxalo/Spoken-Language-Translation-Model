"""Shared Whisper training code for ASR and direct speech translation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audio import load_audio
from .config import ExperimentConfig
from .datasets import (
    load_fleurs_splits,
    load_naija_split,
    load_pairing_artifacts,
    load_revision_matched_audit,
    pair_naija_dataset,
    resolve_pairing_records,
    split_dataset_by_speaker,
    split_membership_sha256,
)
from .evaluation import write_prediction_artifacts
from .hardware import select_precision, set_reproducible_seed
from .metrics import compute_asr_metrics, compute_translation_metrics
from .telemetry import (
    RuntimeTimer,
    build_run_manifest,
    directory_size_bytes,
    parameter_counts,
    write_json_artifact,
)


@dataclass
class SpeechSeq2SeqCollator:
    processor: Any
    target_column: str
    audio_column: str = "audio"
    sampling_rate: int = 16_000
    max_duration_seconds: float = 30.0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        arrays = [
            load_audio(
                feature[self.audio_column],
                target_rate=self.sampling_rate,
                max_duration_seconds=self.max_duration_seconds,
            )[0]
            for feature in features
        ]
        audio_batch = self.processor.feature_extractor(
            arrays,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=round(self.max_duration_seconds * self.sampling_rate),
            truncation=True,
            return_attention_mask=True,
        )
        targets = [str(feature[self.target_column]) for feature in features]
        labels_batch = self.processor.tokenizer(
            targets, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        decoder_start = self.processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        if labels.shape[1] and torch.all(labels[:, 0] == decoder_start):
            labels = labels[:, 1:]
        result = {"input_features": audio_batch["input_features"], "labels": labels}
        if "attention_mask" in audio_batch:
            result["attention_mask"] = audio_batch["attention_mask"]
        return result


def configure_whisper(config: ExperimentConfig) -> tuple[Any, Any]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    config.validate()
    processor = WhisperProcessor.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        language=config.model.language,
        task=config.model.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model.id, revision=config.model.revision
    )
    model.generation_config.language = config.model.language.lower()
    model.generation_config.task = config.model.task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_new_tokens = config.generation.max_new_tokens
    model.generation_config.num_beams = config.generation.num_beams
    model.config.forced_decoder_ids = None
    model = apply_efficiency_strategy(model, config)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if config.model.efficiency_strategy == "lora" and hasattr(
            model, "enable_input_require_grads"
        ):
            model.enable_input_require_grads()
    counts = parameter_counts(model)
    if not counts["trainable_parameters"]:
        raise RuntimeError(
            "Configured training strategy produced no trainable parameters"
        )
    return processor, model


def discover_lora_target_modules(
    model: Any, configured_targets: list[str]
) -> dict[str, list[str]]:
    """Resolve configured LoRA leaf names and fail before an expensive run if absent."""
    if not configured_targets or any(
        not str(target).strip() for target in configured_targets
    ):
        raise ValueError("At least one nonempty LoRA target module is required")
    names = [name for name, _ in model.named_modules()]
    resolved = {
        target: sorted(
            name for name in names if name == target or name.endswith(f".{target}")
        )
        for target in configured_targets
    }
    missing = sorted(target for target, matches in resolved.items() if not matches)
    if missing:
        raise ValueError(f"Configured LoRA target modules do not exist: {missing}")
    return resolved


def trainable_gradient_summary(model: Any) -> dict[str, Any]:
    """Report gradient coverage after backward without exposing tensor contents."""
    trainable = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    with_gradient = [name for name, value in trainable if value.grad is not None]
    finite = [
        name
        for name, value in trainable
        if value.grad is not None and bool(value.grad.detach().isfinite().all().item())
    ]
    return {
        "trainable_parameter_tensors": len(trainable),
        "tensors_with_gradients": len(with_gradient),
        "tensors_with_finite_gradients": len(finite),
        "all_observed_gradients_finite": bool(with_gradient)
        and len(finite) == len(with_gradient),
    }


def assert_trainable_gradients(model: Any) -> dict[str, Any]:
    summary = trainable_gradient_summary(model)
    if not summary["trainable_parameter_tensors"]:
        raise RuntimeError("Model has no trainable parameter tensors")
    if not summary["tensors_with_gradients"]:
        raise RuntimeError(
            "Backward pass produced no gradients on trainable parameters"
        )
    if not summary["all_observed_gradients_finite"]:
        raise RuntimeError("Backward pass produced non-finite trainable gradients")
    return summary


def apply_efficiency_strategy(model: Any, config: ExperimentConfig) -> Any:
    strategy = config.model.efficiency_strategy
    if strategy == "full":
        return model
    if strategy == "freeze_encoder":
        model.freeze_encoder()
        return model
    if strategy == "partial_freeze":
        count = config.model.partial_freeze_layers
        layers = model.model.encoder.layers
        if not 0 < count <= len(layers):
            raise ValueError(f"partial_freeze_layers must be in [1, {len(layers)}]")
        for layer in layers[:count]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        return model
    if strategy == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("peft is required for the LoRA strategy") from exc
        discover_lora_target_modules(model, config.model.lora_targets)
        lora = LoraConfig(
            base_model_name_or_path=config.model.id,
            revision=config.model.revision,
            inference_mode=False,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.lora_targets,
        )
        adapted = get_peft_model(model, lora)
        if not parameter_counts(adapted)["trainable_parameters"]:
            raise RuntimeError("LoRA attached no trainable parameters")
        return adapted
    raise ValueError(f"Unknown efficiency strategy: {strategy}")


def make_compute_metrics(processor: Any, kind: str):
    def compute(prediction: Any) -> dict[str, float | int | str]:
        predictions = prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = prediction.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        pred_text = processor.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        ref_text = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        if kind == "asr":
            return compute_asr_metrics(ref_text, pred_text)
        return compute_translation_metrics(ref_text, pred_text)

    return compute


def _select(dataset: Any, maximum: int | None) -> Any:
    if maximum is None:
        return dataset
    return dataset.select(range(min(maximum, len(dataset))))


def _data_split_summary(dataset: Any) -> dict[str, Any]:
    speakers = sorted({str(value) for value in dataset["speaker_id"]})
    durations = [float(value or 0.0) for value in dataset["duration"]]
    records = [
        {
            "split": project_split,
            "audio_locator": locator,
            "dataset_row_index": row_index,
            "speaker_id": speaker_id,
        }
        for project_split, locator, row_index, speaker_id in zip(
            dataset["split"],
            dataset["audio_locator"],
            dataset["dataset_row_index"],
            dataset["speaker_id"],
            strict=True,
        )
    ]
    return {
        "examples": len(dataset),
        "speakers": len(speakers),
        "audio_hours": sum(durations) / 3600,
        "membership_sha256": split_membership_sha256(records),
    }


def _verify_reconstruction_against_tracked_audit(
    paired: Any,
    derived: dict[str, Any],
    *,
    audit_path: str | None,
    dataset_id: str,
    dataset_revision: str,
) -> dict[str, Any] | None:
    if not audit_path or not Path(audit_path).is_file():
        return None
    tracked = load_revision_matched_audit(
        audit_path,
        expected_dataset_id=dataset_id,
        expected_dataset_revision=dataset_revision,
    )
    expected_train = tracked["train"]
    expected_derived = tracked["derived_seed_42_split"]
    measured = {
        "accepted_pairs": len(paired),
        "speakers": len(set(paired["speaker_id"])),
        "training_pairs": len(derived["train"]),
        "validation_pairs": len(derived["validation"]),
        "training_speakers": len(set(derived["train"]["speaker_id"])),
        "validation_speakers": len(set(derived["validation"]["speaker_id"])),
    }
    expected = {
        "accepted_pairs": int(expected_train["accepted_pairs"]),
        "speakers": int(expected_train["hausa_speakers"]),
        "training_pairs": int(expected_derived["training"]["pairs"]),
        "validation_pairs": int(expected_derived["validation"]["pairs"]),
        "training_speakers": int(expected_derived["training"]["speakers"]),
        "validation_speakers": int(expected_derived["validation"]["speakers"]),
    }
    if measured != expected:
        raise ValueError(
            "Seed-42 reconstruction disagrees with the revision-matched tracked audit: "
            f"measured={measured}, expected={expected}"
        )
    return {"status": "matched", "path": str(audit_path), "expected": expected}


def summarize_training_workload(
    dataset_examples: int,
    dataset_audio_seconds: float,
    completed_epoch_equivalents: float,
) -> dict[str, Any]:
    """Describe the workload represented by a completed Trainer run.

    Complete epochs have exact example/audio totals. For a partial epoch, the
    example count follows Trainer's fractional epoch progress and audio duration
    is explicitly marked as an estimate because clips have unequal durations.
    """
    if dataset_examples < 0 or dataset_audio_seconds < 0:
        raise ValueError("Dataset workload totals must be non-negative")
    if completed_epoch_equivalents < 0:
        raise ValueError("Completed epoch equivalents must be non-negative")
    examples = round(dataset_examples * completed_epoch_equivalents)
    audio_seconds = dataset_audio_seconds * completed_epoch_equivalents
    partial_epoch = not completed_epoch_equivalents.is_integer()
    return {
        "examples": examples,
        "audio_seconds": audio_seconds,
        "completed_epoch_equivalents": completed_epoch_equivalents,
        "partial_epoch_audio_estimated": partial_epoch,
        "basis": (
            "dataset totals multiplied by Trainer state.epoch; partial-epoch "
            "audio duration is proportional rather than clip-exact"
        ),
    }


def write_trainer_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    """Persist Trainer logs separately so curves are reproducible without a checkpoint."""
    write_json_artifact(path, {"log_history": [dict(row) for row in history]})


def load_training_data(
    config: ExperimentConfig,
) -> tuple[Any, Any, str, dict[str, Any]]:
    if config.kind == "asr":
        splits = load_fleurs_splits(
            revision=config.dataset.revision,
            sampling_rate=config.dataset.sampling_rate,
            splits=(
                config.dataset.train_split,
                config.dataset.validation_split,
            ),
        )
        train = _select(
            splits[config.dataset.train_split], config.dataset.max_train_samples
        )
        validation = _select(
            splits[config.dataset.validation_split],
            config.dataset.max_validation_samples,
        )
        return (
            train,
            validation,
            config.dataset.target_column,
            {
                "split_method": "official FLEURS train/validation; test not loaded by Trainer",
                "train_examples": len(train),
                "validation_examples": len(validation),
            },
        )

    if config.kind != "direct_s2tt":
        raise ValueError("Training is implemented for asr and direct_s2tt experiments")
    if not config.dataset.derive_validation_from_train:
        raise ValueError(
            "Direct S2TT must derive speaker-safe validation from train so official dev "
            "can remain untouched as the final held-out set"
        )
    artifact_dir = (
        Path(config.dataset.pairing_artifacts_dir)
        if config.dataset.pairing_artifacts_dir
        else None
    )
    manifest_path = artifact_dir / "manifest.json" if artifact_dir else None
    if manifest_path and manifest_path.is_file():
        artifact_splits, pairing_manifest = load_pairing_artifacts(
            artifact_dir,
            expected_dataset_id=config.dataset.id,
            expected_dataset_revision=str(config.dataset.revision),
            expected_seed=config.training.seed,
        )
        resolved = {
            split: resolve_pairing_records(
                records,
                sampling_rate=config.dataset.sampling_rate,
                dataset_loader=load_naija_split,
            )
            for split, records in artifact_splits.items()
        }
        train_full, validation_full = resolved["train"], resolved["validation"]
        source = {
            "mode": "notebook_00_artifacts",
            "artifact_directory": str(artifact_dir),
            "artifact_schema_version": pairing_manifest["artifact_schema_version"],
            "generating_git_commit": pairing_manifest["git_commit"],
            "split_policy": pairing_manifest["split_policy"],
        }
    else:
        if config.dataset.require_pairing_artifacts:
            raise FileNotFoundError(
                f"Notebook 00 pairing manifest is required but absent: {manifest_path}"
            )
        raw_train = load_naija_split(
            config.dataset.train_split,
            revision=config.dataset.revision,
            sampling_rate=config.dataset.sampling_rate,
        )
        paired_train, audit = pair_naija_dataset(
            raw_train,
            split=config.dataset.train_split,
            max_duration_seconds=config.dataset.max_duration_seconds,
            dataset_id=config.dataset.id,
            dataset_revision=str(config.dataset.revision),
        )
        derived = split_dataset_by_speaker(
            paired_train,
            test_fraction=config.dataset.validation_fraction,
            seed=config.training.seed,
            train_name="train",
            test_name="validation",
        )
        verification = _verify_reconstruction_against_tracked_audit(
            paired_train,
            derived,
            audit_path=config.dataset.tracked_audit_path,
            dataset_id=config.dataset.id,
            dataset_revision=str(config.dataset.revision),
        )
        train_full, validation_full = derived["train"], derived["validation"]
        source = {
            "mode": "deterministic_seed_42_reconstruction",
            "split_policy": (
                "NaijaS2ST official train partitioned by Hausa speaker; official dev reserved"
            ),
            "pairing_audit": audit.to_dict(),
            "tracked_audit_verification": verification,
        }
    overlap = set(train_full["speaker_id"]) & set(validation_full["speaker_id"])
    if overlap:
        raise ValueError(
            f"Direct S2TT speaker leakage detected: {len(overlap)} speakers"
        )
    full_summary = {
        "train": _data_split_summary(train_full),
        "validation": _data_split_summary(validation_full),
        "speaker_overlap": 0,
    }
    train = _select(train_full, config.dataset.max_train_samples)
    validation = _select(validation_full, config.dataset.max_validation_samples)
    return (
        train,
        validation,
        "target_text",
        {
            **source,
            "dataset_id": config.dataset.id,
            "dataset_revision": config.dataset.revision,
            "seed": config.training.seed,
            "full_split_summary": full_summary,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "training_speakers": len(set(train["speaker_id"])),
            "validation_speakers": len(set(validation["speaker_id"])),
            "speaker_overlap": len(
                set(train["speaker_id"]) & set(validation["speaker_id"])
            ),
            "official_dev_evaluated": False,
        },
    )


def train_experiment(config: ExperimentConfig) -> dict[str, Any]:
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    class StepTimingCallback(TrainerCallback):
        """Measure optimizer-step wall time outside Trainer evaluation/save hooks."""

        def __init__(self) -> None:
            self.started = 0.0
            self.step_seconds: list[float] = []

        @staticmethod
        def _synchronize() -> None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:  # pragma: no cover
                return

        def on_step_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            self._synchronize()
            self.started = time.perf_counter()

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            self._synchronize()
            self.step_seconds.append(time.perf_counter() - self.started)

    config.validate()
    started_at = datetime.now(UTC).isoformat()
    phase_seconds: dict[str, float] = {}
    set_reproducible_seed(config.training.seed)
    precision = select_precision(requested=config.training.precision)
    phase_start = time.perf_counter()
    train_dataset, validation_dataset, target_column, data_manifest = (
        load_training_data(config)
    )
    phase_seconds["data_startup"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    processor, model = configure_whisper(config)
    phase_seconds["model_startup"] = time.perf_counter() - phase_start
    output_dir = Path(config.training.output_dir)
    manifest = build_run_manifest(config.to_dict())
    manifest["precision"] = precision.to_dict()
    manifest["data"] = data_manifest
    manifest["parameters"] = parameter_counts(model)
    manifest["artifact_schema_version"] = "1.0"
    manifest["official_dev_evaluated"] = False
    manifest["started_at"] = started_at
    manifest["generation_settings"] = config.generation.__dict__
    manifest["phase_seconds"] = phase_seconds
    write_json_artifact(output_dir / "run_manifest.json", manifest)

    metric_name = "normalized_wer" if config.kind == "asr" else "chrf_pp"
    greater_is_better = config.kind != "asr"
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        bf16=precision.bf16,
        fp16=precision.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater_is_better,
        predict_with_generate=True,
        generation_max_length=None,
        generation_num_beams=None,
        remove_unused_columns=False,
        dataloader_num_workers=config.training.dataloader_num_workers,
        seed=config.training.seed,
        data_seed=config.training.seed,
        report_to=[],
        push_to_hub=False,
    )
    step_timing = StepTimingCallback()
    callbacks = [step_timing]
    if config.training.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience
            )
        )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=SpeechSeq2SeqCollator(
            processor=processor,
            target_column=target_column,
            audio_column=config.dataset.audio_column,
            sampling_rate=config.dataset.sampling_rate,
            max_duration_seconds=config.dataset.max_duration_seconds,
        ),
        compute_metrics=make_compute_metrics(processor, config.kind),
        processing_class=processor,
        callbacks=callbacks,
    )
    audio_seconds = 0.0
    if "duration" in train_dataset.column_names:
        audio_seconds = sum(float(value or 0.0) for value in train_dataset["duration"])
    elif "num_samples" in train_dataset.column_names:
        audio_seconds = sum(
            float(value or 0) / config.dataset.sampling_rate
            for value in train_dataset["num_samples"]
        )
    with RuntimeTimer() as timer:
        result = trainer.train(
            resume_from_checkpoint=config.training.resume_from_checkpoint
        )
    phase_seconds["training"] = timer.wall_seconds
    warmup_count = min(3, len(step_timing.step_seconds))
    phase_seconds["warmup_optimizer_steps"] = sum(
        step_timing.step_seconds[:warmup_count]
    )
    phase_seconds["steady_state_optimizer_steps"] = sum(
        step_timing.step_seconds[warmup_count:]
    )
    write_json_artifact(
        output_dir / "step_timing.json",
        {
            "warmup_step_seconds": step_timing.step_seconds[:warmup_count],
            "steady_step_seconds": step_timing.step_seconds[warmup_count:],
            "all_step_seconds": step_timing.step_seconds,
        },
    )
    phase_start = time.perf_counter()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    phase_seconds["checkpoint_writing"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    prediction_output = trainer.predict(
        validation_dataset, metric_key_prefix="validation"
    )
    predictions = prediction_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = prediction_output.label_ids.copy()
    labels[labels == -100] = processor.tokenizer.pad_token_id
    prediction_text = processor.tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    decoded_reference_text = processor.tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )
    validation_rows = []
    exact_references = []
    for index, (prediction, decoded_reference) in enumerate(
        zip(prediction_text, decoded_reference_text, strict=True)
    ):
        row = validation_dataset[index]
        exact_reference = str(row[target_column])
        exact_references.append(exact_reference)
        validation_rows.append(
            {
                "source_id": row.get("source_text_id", index),
                "alignment_key": row.get("alignment_key"),
                "dataset_row_index": row.get("dataset_row_index"),
                "speaker_id": row.get("speaker_id"),
                "source_transcript": row.get("source_text"),
                "reference": exact_reference,
                "reference_tokenizer_roundtrip": decoded_reference,
                "prediction": prediction,
                "duration": row.get("duration"),
                "project_split": "validation",
            }
        )
    validation_metrics = dict(prediction_output.metrics)
    exact_metrics = (
        compute_asr_metrics(exact_references, prediction_text)
        if config.kind == "asr"
        else compute_translation_metrics(exact_references, prediction_text)
    )
    validation_metrics.update(
        {f"validation_{name}": value for name, value in exact_metrics.items()}
    )
    validation_metrics["official_dev_evaluated"] = False
    validation_artifact_dir = output_dir / "validation"
    write_prediction_artifacts(
        validation_rows, validation_metrics, validation_artifact_dir
    )
    phase_seconds["validation_prediction"] = time.perf_counter() - phase_start
    write_trainer_history(
        output_dir / "trainer_history.json", trainer.state.log_history
    )
    completed_epochs = float(trainer.state.epoch or result.metrics.get("epoch", 0.0))
    workload = summarize_training_workload(
        len(train_dataset), audio_seconds, completed_epochs
    )
    runtime = timer.measurement(workload["examples"], workload["audio_seconds"])
    runtime_payload = runtime.to_dict()
    runtime_payload["workload_basis"] = workload
    final = {
        "train_metrics": result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "runtime": runtime_payload,
        "checkpoint_size_bytes": directory_size_bytes(
            trainer.state.best_model_checkpoint or output_dir
        ),
        "parameters": parameter_counts(model),
        "validation_metrics": validation_metrics,
        "validation_predictions": str(validation_artifact_dir / "predictions.jsonl"),
        "trainer_history": str(output_dir / "trainer_history.json"),
        "step_timing": str(output_dir / "step_timing.json"),
        "generation_settings": config.generation.__dict__,
        "phase_seconds": phase_seconds,
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "official_dev_evaluated": False,
    }
    write_json_artifact(output_dir / "training_summary.json", final)
    manifest.update(
        {
            "ended_at": final["ended_at"],
            "phase_seconds": phase_seconds,
            "best_checkpoint": final["best_checkpoint"],
            "best_metric": final["best_metric"],
            "runtime": runtime_payload,
            "checkpoint_size_bytes": final["checkpoint_size_bytes"],
            "validation_metrics": validation_metrics,
            "validation_prediction_artifact": final["validation_predictions"],
            "trainer_history": trainer.state.log_history,
            "train_metrics": result.metrics,
            "training_loss": result.metrics.get("train_loss"),
        }
    )
    write_json_artifact(output_dir / "run_manifest.json", manifest)
    return final
