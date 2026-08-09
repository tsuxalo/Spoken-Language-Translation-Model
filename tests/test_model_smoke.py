"""Opt-in real checkpoint smoke tests; normal CI never downloads checkpoints."""

import os
import tempfile
import unittest

import numpy as np

from hausa_s2tt.training import SpeechSeq2SeqCollator


@unittest.skipUnless(os.getenv("RUN_MODEL_SMOKE") == "1", "set RUN_MODEL_SMOKE=1")
class ModelSmokeTests(unittest.TestCase):
    model_id = "openai/whisper-tiny"
    revision = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"

    def test_asr_and_direct_whisper_tensor_shapes_and_output(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(
            self.model_id, revision=self.revision
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision
        )
        collator = SpeechSeq2SeqCollator(processor, target_column="target_text")
        batch = collator(
            [
                {
                    "audio": {
                        "array": np.zeros(16000, dtype=np.float32),
                        "sampling_rate": 16000,
                    },
                    "target_text": "An English label.",
                }
            ]
        )
        self.assertEqual(batch["input_features"].shape[-1], 3000)
        features = processor.feature_extractor(
            np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors="pt"
        ).input_features
        self.assertEqual(features.ndim, 3)
        for task in ("transcribe", "translate"):
            with torch.inference_mode():
                tokens = model.generate(
                    features, language="Hausa", task=task, max_new_tokens=2
                )
            text = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            self.assertIsInstance(text[0], str)

    def test_lora_adapter_save_and_direct_reload(self):
        from pathlib import Path

        from hausa_s2tt.config import (
            DatasetConfig,
            ExperimentConfig,
            GenerationConfig,
            ModelConfig,
            TrainingConfig,
        )
        from hausa_s2tt.inference import create_direct_runtime
        from hausa_s2tt.training import configure_whisper

        config = ExperimentConfig(
            kind="direct_s2tt",
            dataset=DatasetConfig(
                id="McGill-NLP/NaijaS2ST",
                revision="898f51582750fe244693794f22e3f4b32c5baf95",
                validation_split="derived_from_train",
                test_split="dev",
                target_column="target_text",
                target_language="english",
                derive_validation_from_train=True,
            ),
            model=ModelConfig(
                id=self.model_id,
                revision=self.revision,
                task="translate",
                efficiency_strategy="lora",
            ),
            training=TrainingConfig(gradient_checkpointing=True),
            generation=GenerationConfig(max_new_tokens=7, num_beams=1),
        )
        processor, model = configure_whisper(config)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(model.generation_config.task, "translate")
        self.assertEqual(model.generation_config.language, "hausa")
        self.assertIsNone(model.generation_config.forced_decoder_ids)
        self.assertEqual(model.generation_config.max_new_tokens, 7)
        self.assertEqual(model.generation_config.num_beams, 1)
        batch = SpeechSeq2SeqCollator(processor, target_column="target_text")(
            [
                {
                    "audio": {
                        "array": np.zeros(16000, dtype=np.float32),
                        "sampling_rate": 16000,
                    },
                    "target_text": "An aligned English target.",
                }
            ]
        )
        loss = model(**batch).loss
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        import torch

        frozen_parameter = next(
            parameter for parameter in model.parameters() if not parameter.requires_grad
        )
        frozen_before = frozen_parameter.detach().clone()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-4,
        )
        optimizer.step()
        self.assertTrue(frozen_parameter.detach().equal(frozen_before))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            processor.save_pretrained(directory)
            self.assertTrue((Path(directory) / "adapter_config.json").is_file())
            self.assertTrue(
                any(
                    (Path(directory) / name).is_file()
                    for name in ("processor_config.json", "preprocessor_config.json")
                )
            )
            runtime = create_direct_runtime(
                directory, precision="fp32", max_new_tokens=2
            )
            result = runtime.process(
                {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}
            )
            self.assertIsInstance(result.text, str)


if __name__ == "__main__":
    unittest.main()
