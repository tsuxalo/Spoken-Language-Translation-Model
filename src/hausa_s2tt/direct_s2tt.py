"""Direct Hausa-audio to English-text Whisper training entrypoint."""

from __future__ import annotations

from .config import ExperimentConfig
from .training import train_experiment


def train_direct_s2tt(config: ExperimentConfig):
    if config.kind != "direct_s2tt" or config.model.task != "translate":
        raise ValueError("Direct training requires kind=direct_s2tt and task=translate")
    if config.dataset.target_language.lower() != "english":
        raise ValueError("This pipeline requires genuine English translation labels")
    return train_experiment(config)
