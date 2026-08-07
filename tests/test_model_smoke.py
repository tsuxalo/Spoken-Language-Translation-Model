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
                tokens = model.generate(features, language="Hausa", task=task, max_new_tokens=2)
            text = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            self.assertIsInstance(text[0], str)

    def test_lora_adapter_save_and_direct_reload(self):
        from pathlib import Path

        from hausa_s2tt.config import ExperimentConfig, ModelConfig, TrainingConfig
        from hausa_s2tt.inference import create_direct_runtime
        from hausa_s2tt.training import configure_whisper

        config = ExperimentConfig(
            kind="direct_s2tt",
            model=ModelConfig(
                id=self.model_id,
                revision=self.revision,
                task="translate",
                efficiency_strategy="lora",
            ),
            training=TrainingConfig(gradient_checkpointing=False),
        )
        processor, model = configure_whisper(config)
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
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            processor.save_pretrained(directory)
            self.assertTrue((Path(directory) / "adapter_config.json").is_file())
            runtime = create_direct_runtime(directory, precision="fp32", max_new_tokens=2)
            result = runtime.process(
                {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}
            )
            self.assertIsInstance(result.text, str)


if __name__ == "__main__":
    unittest.main()
