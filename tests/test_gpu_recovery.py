from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from analysis.analyze_predictions import (
    corpus_bleu,
    corpus_chrf,
    load_prediction_frame,
    resolve_comet_checkpoint,
)
from experiments.generate_asr_noise import (
    load_asr_runtime,
    load_resume_rows,
    processor_compat_kwargs,
)
from experiments.gpu_preflight import REQUIRED_SCRIPTS, path_status, required_paths_for
from experiments.prepare_naijas2st_pairs import (
    METADATA_COLUMNS,
    load_metadata_dataset,
)


def _prediction_row(prediction: str = "") -> dict[str, object]:
    return {
        "alignment_id": "a1",
        "utterance_id": "u1",
        "speaker_id": "synthetic-speaker",
        "hausa_gold": "rubutun gwaji",
        "hausa_asr": "rubutun gwaji",
        "english_ref": "test text",
        "wer": 0.0,
        "prediction_nllb": prediction,
    }


def test_prediction_loader_preserves_and_scores_empty_hypothesis(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame([_prediction_row("")]).to_csv(path, index=False)

    frame, columns = load_prediction_frame(path)

    assert columns == ["prediction_nllb"]
    assert frame.loc[0, "prediction_nllb"] == ""
    assert not frame[columns].isna().any().any()
    assert corpus_bleu(["test text"], frame["prediction_nllb"].tolist()) == 0.0
    assert corpus_chrf(["test text"], frame["prediction_nllb"].tolist()) == 0.0


def test_prediction_loader_rejects_duplicate_utterances(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame([_prediction_row("one"), _prediction_row("two")]).to_csv(
        path, index=False
    )
    with pytest.raises(RuntimeError, match="duplicate utterance_id"):
        load_prediction_frame(path)


def test_comet_snapshot_receives_exact_revision(tmp_path: Path) -> None:
    checkpoint = tmp_path / "snapshot" / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")

    with patch(
        "huggingface_hub.snapshot_download",
        return_value=str(checkpoint.parents[1]),
    ) as snapshot_download:
        resolved = resolve_comet_checkpoint("org/model", "deadbeef", str(tmp_path))

    assert resolved == str(checkpoint)
    snapshot_download.assert_called_once_with(
        repo_id="org/model", revision="deadbeef", cache_dir=str(tmp_path)
    )


def test_processor_compatibility_preserves_token_values(tmp_path: Path) -> None:
    config = tmp_path / "tokenizer_config.json"
    config.write_text(
        json.dumps({"extra_special_tokens": ["<one>", "<two>"]}),
        encoding="utf-8",
    )
    assert processor_compat_kwargs(str(tmp_path), "unused") == {
        "extra_special_tokens": {
            "extra_special_token_0": "<one>",
            "extra_special_token_1": "<two>",
        }
    }


def test_processor_compatibility_passes_revision_to_hub(tmp_path: Path) -> None:
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"extra_special_tokens": ["<one>"]}), encoding="utf-8")
    with patch(
        "experiments.generate_asr_noise.hf_hub_download", return_value=str(config)
    ) as download:
        processor_compat_kwargs("org/model", "cafebabe")
    download.assert_called_once_with(
        repo_id="org/model",
        filename="tokenizer_config.json",
        revision="cafebabe",
    )


def test_asr_processor_and_model_receive_the_exact_revision(tmp_path: Path) -> None:
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"extra_special_tokens": ["<one>"]}), encoding="utf-8")
    model = type(
        "SyntheticModel",
        (),
        {"to": lambda self, device: self, "eval": lambda self: self},
    )()
    with (
        patch(
            "experiments.generate_asr_noise.WhisperProcessor.from_pretrained",
            return_value="processor",
        ) as processor_loader,
        patch(
            "experiments.generate_asr_noise.WhisperForConditionalGeneration.from_pretrained",
            return_value=model,
        ) as model_loader,
    ):
        processor, loaded_model = load_asr_runtime(str(tmp_path), "cafebabe", "cpu")

    assert processor == "processor"
    assert loaded_model is model
    processor_loader.assert_called_once_with(
        str(tmp_path),
        revision="cafebabe",
        extra_special_tokens={"extra_special_token_0": "<one>"},
    )
    model_loader.assert_called_once_with(str(tmp_path), revision="cafebabe")


def test_processor_compatibility_rejects_ambiguous_duplicates(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<same>", "<same>"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        processor_compat_kwargs(str(tmp_path), "unused")


def test_resume_manifest_rejects_duplicate_utterances(tmp_path: Path) -> None:
    path = tmp_path / "resume.jsonl"
    row = {"utterance_id": "u1", "wer": 0, "hausa_gold": "a", "hausa_asr": "a"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate utterance_id"):
        load_resume_rows(path)


class _FakeDataset:
    def __init__(self) -> None:
        self.features = {"language": None, "text_id": None, "text": None, "audio": None}
        self.decoded = True
        self.removed: list[str] = []

    def decode(self, value: bool):
        self.decoded = value
        return self

    def remove_columns(self, columns: list[str]):
        self.removed.extend(columns)
        self.features = {k: v for k, v in self.features.items() if k not in columns}
        return self


def test_metadata_loader_projects_text_columns_and_revision() -> None:
    dataset = _FakeDataset()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def loader(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return dataset

    result = load_metadata_dataset("train", "abc123", loader=loader)
    assert result is dataset
    assert calls[0][1]["columns"] == METADATA_COLUMNS
    assert calls[0][1]["revision"] == "abc123"
    assert dataset.decoded is False
    assert dataset.removed == ["audio"]


def test_preflight_explicit_paths_replace_or_supplement_defaults() -> None:
    replaced = required_paths_for("evaluate", ["private/dev.jsonl"], "replace")
    supplemented = required_paths_for(
        "evaluate", ["private/dev.jsonl", "private/adapters"], "supplement"
    )
    assert replaced == REQUIRED_SCRIPTS + ["private/dev.jsonl"]
    assert "outputs/final/mixed" in supplemented
    assert supplemented[-2:] == ["private/dev.jsonl", "private/adapters"]


def test_preflight_reports_present_and_missing_paths(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.write_text("synthetic", encoding="utf-8")

    assert path_status(str(present))["exists"] is True
    assert path_status(str(tmp_path / "missing")) == {
        "path": str(tmp_path / "missing"),
        "exists": False,
        "type": None,
    }
