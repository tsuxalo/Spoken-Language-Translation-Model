"""Hausa ASR training entrypoint."""

from __future__ import annotations

from .config import ExperimentConfig
from .training import train_experiment


def train_asr(config: ExperimentConfig):
    if config.kind != "asr" or config.model.task != "transcribe":
        raise ValueError("ASR training requires kind=asr and task=transcribe")
    return train_experiment(config)
