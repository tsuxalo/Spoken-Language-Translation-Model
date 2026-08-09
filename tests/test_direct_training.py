import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hausa_s2tt.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from hausa_s2tt.direct_s2tt import (
    select_direct_architecture,
    write_notebook03_handoff,
)
from hausa_s2tt.evaluation import write_prediction_artifacts
from hausa_s2tt.training import (
    SpeechSeq2SeqCollator,
    assert_trainable_gradients,
    discover_lora_target_modules,
    load_training_data,
    write_trainer_history,
)


class _Audit:
    @staticmethod
    def to_dict():
        return {"status": "mocked"}


class DirectTrainingTests(unittest.TestCase):
    def test_collator_masks_padding_with_minus_100(self):
        import torch

        class FeatureExtractor:
            def __call__(self, arrays, **kwargs):
                return {
                    "input_features": torch.zeros((len(arrays), 80, 4)),
                    "attention_mask": torch.ones((len(arrays), 4), dtype=torch.long),
                }

        class Tokenizer:
            pad_token_id = 0

            @staticmethod
            def __call__(targets, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 7, 0], [1, 8, 9]]),
                    "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                }

            @staticmethod
            def convert_tokens_to_ids(token):
                return 1

        processor = type(
            "Processor",
            (),
            {"feature_extractor": FeatureExtractor(), "tokenizer": Tokenizer()},
        )()
        collator = SpeechSeq2SeqCollator(processor, target_column="target_text")
        batch = collator(
            [
                {
                    "audio": {
                        "array": np.zeros(160, dtype=np.float32),
                        "sampling_rate": 16000,
                    },
                    "target_text": "English one",
                },
                {
                    "audio": {
                        "array": np.zeros(160, dtype=np.float32),
                        "sampling_rate": 16000,
                    },
                    "target_text": "English two",
                },
            ]
        )
        self.assertEqual(batch["labels"].tolist(), [[7, -100], [8, 9]])

    def test_lora_discovery_and_gradient_guard(self):
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        discovered = discover_lora_target_modules(model, ["0", "1"])
        self.assertEqual(discovered["0"], ["0"])
        with self.assertRaisesRegex(ValueError, "do not exist"):
            discover_lora_target_modules(model, ["q_proj"])
        model[0].weight.requires_grad = False
        frozen = model[0].weight.detach().clone()
        model(torch.ones(1, 2)).sum().backward()
        summary = assert_trainable_gradients(model)
        self.assertGreater(summary["tensors_with_gradients"], 0)
        self.assertTrue(model[0].weight.detach().equal(frozen))

    def test_direct_training_reconstruction_loads_only_official_train(self):
        from datasets import Dataset

        rows = Dataset.from_dict(
            {
                "audio": [
                    {"array": np.zeros(16, dtype=np.float32), "sampling_rate": 16000}
                    for _ in range(4)
                ],
                "target_text": ["one", "two", "three", "four"],
                "source_text": ["a", "b", "c", "d"],
                "speaker_id": ["s1", "s1", "s2", "s2"],
                "duration": [1.0] * 4,
                "split": ["train"] * 4,
                "audio_locator": [
                    {
                        "source_dataset": "McGill-NLP/NaijaS2ST",
                        "dataset_revision": "revision",
                        "split": "train",
                        "dataset_row_index": index,
                    }
                    for index in range(4)
                ],
                "dataset_row_index": list(range(4)),
            }
        )
        config = ExperimentConfig(
            kind="direct_s2tt",
            dataset=DatasetConfig(
                id="McGill-NLP/NaijaS2ST",
                revision="revision",
                validation_split="derived_from_train",
                test_split="dev",
                target_column="target_text",
                target_language="english",
                derive_validation_from_train=True,
                pairing_artifacts_dir=None,
                tracked_audit_path=None,
            ),
            model=ModelConfig(revision="model-revision", task="translate"),
            training=TrainingConfig(seed=42),
        )
        derived = {"train": rows.select([0, 1]), "validation": rows.select([2, 3])}
        with (
            patch(
                "hausa_s2tt.training.load_naija_split", return_value=object()
            ) as loader,
            patch(
                "hausa_s2tt.training.pair_naija_dataset", return_value=(rows, _Audit())
            ),
            patch("hausa_s2tt.training.split_dataset_by_speaker", return_value=derived),
        ):
            train, validation, target, manifest = load_training_data(config)
        self.assertEqual(loader.call_args.args[0], "train")
        self.assertEqual(target, "target_text")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 2)
        self.assertEqual(manifest["official_dev_evaluated"], False)
        self.assertEqual(manifest["speaker_overlap"], 0)

    def test_architecture_selection_rejects_dev_and_serializes_false_guard(self):
        candidates = [
            {
                "run_name": "base",
                "model_id": "base",
                "model_revision": "a",
                "split": "validation",
                "chrf_pp": 20.0,
                "sacrebleu": 8.0,
                "validation_loss": 2.0,
                "official_dev_evaluated": False,
                "qualitative_review": {
                    "status": "complete",
                    "source_language_leakage": "none in inspected sample",
                    "hallucination_repetition": "none in inspected sample",
                    "omissions_additions": "reviewed",
                    "names_numbers_dates_negation": "reviewed",
                    "eligible_for_selection": True,
                },
            },
            {
                "run_name": "asr",
                "model_id": "asr",
                "model_revision": "b",
                "split": "validation",
                "chrf_pp": 21.0,
                "sacrebleu": 7.0,
                "validation_loss": 2.1,
                "official_dev_evaluated": False,
                "qualitative_review": {
                    "status": "complete",
                    "source_language_leakage": "none in inspected sample",
                    "hallucination_repetition": "none in inspected sample",
                    "omissions_additions": "reviewed",
                    "names_numbers_dates_negation": "reviewed",
                    "eligible_for_selection": True,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            result = select_direct_architecture(candidates, path)
            self.assertEqual(result["selected_run"], "asr")
            self.assertTrue(path.is_file())
            self.assertFalse(result["official_dev_evaluated"])
        candidates[0]["official_dev_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "official_dev"):
            select_direct_architecture(candidates, "unused.json")

    def test_history_and_validation_predictions_are_separate_serializable_artifacts(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path = root / "trainer_history.json"
            write_trainer_history(history_path, [{"step": 1, "loss": 2.5}])
            write_prediction_artifacts(
                [{"source_id": "H1", "reference": "Hello", "prediction": "Hi"}],
                {"chrf_pp": 12.0, "official_dev_evaluated": False},
                root / "validation",
            )
            history = json.loads(history_path.read_text(encoding="utf-8"))
            metrics = json.loads(
                (root / "validation/metrics.json").read_text(encoding="utf-8")
            )
            predictions = (root / "validation/predictions.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(history["log_history"][0]["step"], 1)
            self.assertFalse(metrics["official_dev_evaluated"])
            self.assertIn('"source_id": "H1"', predictions)

    def test_notebook03_handoff_freezes_complete_protected_dev_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "selected-model"
            output_dir.mkdir()
            config = ExperimentConfig(
                kind="direct_s2tt",
                dataset=DatasetConfig(
                    id="McGill-NLP/NaijaS2ST",
                    revision="dataset-revision",
                    validation_split="derived_from_train",
                    test_split="dev",
                    target_column="target_text",
                    target_language="english",
                    derive_validation_from_train=True,
                ),
                model=ModelConfig(
                    id="openai/whisper-small",
                    revision="model-revision",
                    task="translate",
                    efficiency_strategy="lora",
                ),
                training=TrainingConfig(output_dir=str(output_dir), seed=42),
            )
            (output_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "official_dev_evaluated": False,
                        "data": {"speaker_overlap": 0, "membership_sha256": "abc"},
                    }
                ),
                encoding="utf-8",
            )
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "official_dev_evaluated": False,
                        "selected_model_id": config.model.id,
                        "selected_model_revision": config.model.revision,
                    }
                ),
                encoding="utf-8",
            )
            summary = {
                "official_dev_evaluated": False,
                "generation_settings": {"max_new_tokens": 8},
                "validation_metrics": {"validation_chrf_pp": 1.0},
                "validation_predictions": "validation/predictions.jsonl",
                "trainer_history": "trainer_history.json",
                "runtime": {"wall_seconds": 2.0},
                "parameters": {"trainable": 10},
            }
            handoff_path = root / "notebook03_handoff.json"
            handoff = write_notebook03_handoff(
                config,
                summary,
                architecture_selection_path=selection_path,
                output_path=handoff_path,
                training_status="matched_pilot_selected",
            )
            required = {
                "selected_model_initialization",
                "resolved_config",
                "model_revision",
                "dataset_revision",
                "project_split_provenance",
                "adapter_or_checkpoint_path",
                "processor_path",
                "generation_config",
                "validation_metrics",
                "validation_predictions",
                "training_history",
                "runtime_telemetry",
                "parameter_counts",
                "pilot_or_full_status",
                "architecture_selection_artifact",
                "official_dev_evaluated",
            }
            self.assertTrue(required.issubset(handoff))
            self.assertFalse(handoff["official_dev_evaluated"])
            self.assertTrue(handoff_path.is_file())
            summary["official_dev_evaluated"] = True
            with self.assertRaisesRegex(ValueError, "official_dev"):
                write_notebook03_handoff(
                    config,
                    summary,
                    architecture_selection_path=selection_path,
                    output_path=handoff_path,
                    training_status="invalid",
                )


if __name__ == "__main__":
    unittest.main()
