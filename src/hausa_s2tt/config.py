"""Small, strict experiment configuration system.

YAML is deliberately the only optional dependency in this module. JSON configs are
also accepted, which keeps basic inspection and smoke tests usable without PyYAML.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    id: str = "google/fleurs"
    config_name: str = "ha_ng"
    revision: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str | None = "test"
    audio_column: str = "audio"
    target_column: str = "raw_transcription"
    source_language: str = "hausa"
    target_language: str = "hausa"
    sampling_rate: int = 16_000
    max_duration_seconds: float = 30.0
    num_proc: int | None = None
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    derive_validation_from_train: bool = False
    validation_fraction: float = 0.1
    pairing_artifacts_dir: str | None = "artifacts/data/naija_s2st"
    require_pairing_artifacts: bool = False
    tracked_audit_path: str | None = "reports/naija_s2st_audit_summary.json"


@dataclass
class ModelConfig:
    id: str = "openai/whisper-small"
    revision: str | None = None
    language: str = "Hausa"
    task: str = "transcribe"
    efficiency_strategy: str = "full"
    partial_freeze_layers: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    output_dir: str = "artifacts/checkpoints/experiment"
    seed: int = 42
    learning_rate: float = 1e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 25
    eval_steps: int | None = None
    save_steps: int | None = None
    save_total_limit: int = 2
    early_stopping_patience: int | None = 2
    dataloader_num_workers: int = 0
    precision: str = "auto"
    resume_from_checkpoint: str | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 225
    num_beams: int = 1
    chunk_length_seconds: float = 29.0
    stride_seconds: float = 0.0
    batch_size: int = 4


@dataclass
class MTConfig:
    id: str = "facebook/nllb-200-distilled-600M"
    revision: str | None = None
    source_language: str = "hau_Latn"
    target_language: str = "eng_Latn"
    batch_size: int = 8
    max_new_tokens: int = 256
    num_beams: int = 4


@dataclass
class TrackingConfig:
    run_name: str = "experiment"
    artifacts_dir: str = "artifacts"
    gpu_usd_per_hour: float | None = None
    cpu_usd_per_hour: float | None = None


@dataclass
class ExperimentConfig:
    kind: str = "asr"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    mt: MTConfig = field(default_factory=MTConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self) -> None:
        if self.kind not in {"asr", "direct_s2tt", "cascade", "zero_shot"}:
            raise ValueError(f"Unsupported experiment kind: {self.kind!r}")
        if self.model.task not in {"transcribe", "translate"}:
            raise ValueError("model.task must be 'transcribe' or 'translate'")
        if self.kind == "asr" and self.model.task != "transcribe":
            raise ValueError("ASR experiments must use model.task='transcribe'")
        if self.kind == "asr" and self.dataset.target_language.casefold() != "hausa":
            raise ValueError("ASR experiments must target Hausa text")
        if self.kind == "direct_s2tt":
            if self.model.task != "translate":
                raise ValueError(
                    "Direct S2TT experiments must use model.task='translate'"
                )
            if self.dataset.target_language.casefold() != "english":
                raise ValueError("Direct S2TT experiments must target English text")
            if self.dataset.target_column != "target_text":
                raise ValueError(
                    "Direct S2TT requires genuine aligned English target_text labels"
                )
            if not self.dataset.derive_validation_from_train:
                raise ValueError(
                    "Direct S2TT validation must be derived from train by speaker"
                )
            if self.dataset.train_split != "train":
                raise ValueError(
                    "Direct S2TT may train only from NaijaS2ST official train"
                )
            if self.dataset.test_split != "dev":
                raise ValueError("Direct S2TT must reserve NaijaS2ST official dev")
            if self.dataset.validation_split != "derived_from_train":
                raise ValueError(
                    "Direct S2TT validation must be derived from official train"
                )
            if self.training.seed != 42:
                raise ValueError(
                    "Direct S2TT project split membership is fixed at seed 42"
                )
            if not self.dataset.id or not self.dataset.revision:
                raise ValueError(
                    "Direct S2TT requires a pinned dataset ID and revision"
                )
            if not self.model.id or not self.model.revision:
                raise ValueError("Direct S2TT requires a pinned model ID and revision")
        if self.kind == "zero_shot" and self.model.task != "translate":
            raise ValueError(
                "Zero-shot Whisper translation must use model.task='translate'"
            )
        if self.kind == "cascade":
            if self.model.task != "transcribe":
                raise ValueError("A cascade ASR stage must use model.task='transcribe'")
            if (self.mt.source_language, self.mt.target_language) != (
                "hau_Latn",
                "eng_Latn",
            ):
                raise ValueError(
                    "The Hausa-English cascade requires hau_Latn -> eng_Latn"
                )
        if self.model.efficiency_strategy not in {
            "full",
            "freeze_encoder",
            "partial_freeze",
            "lora",
        }:
            raise ValueError("Unknown model.efficiency_strategy")
        if self.dataset.sampling_rate <= 0:
            raise ValueError("dataset.sampling_rate must be positive")
        if self.dataset.max_duration_seconds <= 0:
            raise ValueError("dataset.max_duration_seconds must be positive")
        if not 0 < self.dataset.validation_fraction < 1:
            raise ValueError("dataset.validation_fraction must be in (0, 1)")
        if self.training.seed < 0:
            raise ValueError("training.seed must be non-negative")
        if self.training.eval_strategy != self.training.save_strategy:
            raise ValueError(
                "eval_strategy and save_strategy must match when selecting the best model"
            )
        if self.training.eval_strategy == "steps":
            if not self.training.eval_steps or not self.training.save_steps:
                raise ValueError(
                    "Step-based evaluation requires positive eval_steps and save_steps"
                )
            if self.training.eval_steps != self.training.save_steps:
                raise ValueError(
                    "eval_steps and save_steps must match for best-checkpoint selection"
                )
        if self.generation.chunk_length_seconds > 30.0:
            raise ValueError(
                "Whisper chunks must not exceed its 30 second input window"
            )
        if self.generation.stride_seconds < 0:
            raise ValueError("generation.stride_seconds must be non-negative")
        if self.generation.stride_seconds >= self.generation.chunk_length_seconds:
            raise ValueError("stride_seconds must be smaller than chunk_length_seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping")
    return dict(value)


def _reject_unknown(cls: type, values: Mapping[str, Any], section: str) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {', '.join(unknown)}")


def experiment_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    allowed = {"kind", "dataset", "model", "training", "generation", "mt", "tracking"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {', '.join(unknown)}")

    dataset_values = _section(data, "dataset")
    model_values = _section(data, "model")
    training_values = _section(data, "training")
    generation_values = _section(data, "generation")
    mt_values = _section(data, "mt")
    tracking_values = _section(data, "tracking")
    for cls, values, name in (
        (DatasetConfig, dataset_values, "dataset"),
        (ModelConfig, model_values, "model"),
        (TrainingConfig, training_values, "training"),
        (GenerationConfig, generation_values, "generation"),
        (MTConfig, mt_values, "mt"),
        (TrackingConfig, tracking_values, "tracking"),
    ):
        _reject_unknown(cls, values, name)

    config = ExperimentConfig(
        kind=str(data.get("kind", "asr")),
        dataset=DatasetConfig(**dataset_values),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**training_values),
        generation=GenerationConfig(**generation_values),
        mt=MTConfig(**mt_values),
        tracking=TrackingConfig(**tracking_values),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError(
                "PyYAML is required for YAML configs; install requirements.txt or use JSON"
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise TypeError("The experiment configuration must contain a mapping")
    return experiment_from_dict(data)
