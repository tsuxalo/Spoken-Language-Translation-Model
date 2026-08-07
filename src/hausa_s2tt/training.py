"""Shared Whisper training code for ASR and direct speech translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import load_audio
from .config import ExperimentConfig
from .datasets import (
    load_fleurs_splits,
    load_naija_split,
    pair_naija_dataset,
    split_dataset_by_speaker,
)
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
    model.config.forced_decoder_ids = None
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return processor, apply_efficiency_strategy(model, config)


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
        lora = LoraConfig(
            base_model_name_or_path=config.model.id,
            revision=config.model.revision,
            inference_mode=False,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=config.model.lora_targets,
        )
        return get_peft_model(model, lora)
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


def load_training_data(config: ExperimentConfig) -> tuple[Any, Any, str, dict[str, Any]]:
    if config.kind == "asr":
        splits = load_fleurs_splits(
            revision=config.dataset.revision,
            sampling_rate=config.dataset.sampling_rate,
            splits=(
                config.dataset.train_split,
                config.dataset.validation_split,
            ),
        )
        train = _select(splits[config.dataset.train_split], config.dataset.max_train_samples)
        validation = _select(
            splits[config.dataset.validation_split],
            config.dataset.max_validation_samples,
        )
        return train, validation, config.dataset.target_column, {
            "split_method": "official FLEURS train/validation; test not loaded by Trainer",
            "train_examples": len(train),
            "validation_examples": len(validation),
        }

    if config.kind != "direct_s2tt":
        raise ValueError("Training is implemented for asr and direct_s2tt experiments")
    if not config.dataset.derive_validation_from_train:
        raise ValueError(
            "Direct S2TT must derive speaker-safe validation from train so official dev "
            "can remain untouched as the final held-out set"
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
    )
    derived = split_dataset_by_speaker(
        paired_train,
        test_fraction=config.dataset.validation_fraction,
        seed=config.training.seed,
        train_name="train",
        test_name="validation",
    )
    train = derived["train"]
    validation = derived["validation"]
    train = _select(train, config.dataset.max_train_samples)
    validation = _select(validation, config.dataset.max_validation_samples)
    return train, validation, "target_text", {
        "split_method": (
            "NaijaS2ST train split partitioned by Hausa speaker; official dev is untouched final"
        ),
        "pairing_audit": audit.to_dict(),
        "train_examples": len(train),
        "validation_examples": len(validation),
    }


def train_experiment(config: ExperimentConfig) -> dict[str, Any]:
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    config.validate()
    set_reproducible_seed(config.training.seed)
    precision = select_precision(requested=config.training.precision)
    train_dataset, validation_dataset, target_column, data_manifest = load_training_data(config)
    processor, model = configure_whisper(config)
    output_dir = Path(config.training.output_dir)
    manifest = build_run_manifest(config.to_dict())
    manifest["precision"] = precision.to_dict()
    manifest["data"] = data_manifest
    manifest["parameters"] = parameter_counts(model)
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
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater_is_better,
        predict_with_generate=True,
        generation_max_length=config.generation.max_new_tokens,
        generation_num_beams=config.generation.num_beams,
        remove_unused_columns=False,
        dataloader_num_workers=config.training.dataloader_num_workers,
        seed=config.training.seed,
        data_seed=config.training.seed,
        report_to=[],
        push_to_hub=False,
    )
    callbacks = []
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
        result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    runtime = timer.measurement(len(train_dataset), audio_seconds)
    final = {
        "train_metrics": result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "runtime": runtime.to_dict(),
        "checkpoint_size_bytes": directory_size_bytes(output_dir),
        "parameters": parameter_counts(model),
        "test_evaluated": False,
    }
    write_json_artifact(output_dir / "training_summary.json", final)
    return final
