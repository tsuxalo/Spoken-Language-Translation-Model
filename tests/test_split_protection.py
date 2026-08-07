import unittest
from unittest.mock import patch

from hausa_s2tt.config import ExperimentConfig
from hausa_s2tt.training import load_training_data


class FakeDataset(list):
    def select(self, indices):
        return FakeDataset(self[index] for index in indices)


class SplitProtectionTests(unittest.TestCase):
    def test_asr_trainer_requests_only_train_and_validation(self):
        config = ExperimentConfig(kind="asr")
        fake = {
            "train": FakeDataset([{"raw_transcription": "a"}]),
            "validation": FakeDataset([{"raw_transcription": "b"}]),
        }
        with patch("hausa_s2tt.training.load_fleurs_splits", return_value=fake) as loader:
            train, validation, target, manifest = load_training_data(config)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)
        self.assertEqual(target, "raw_transcription")
        self.assertNotIn("test", loader.call_args.kwargs["splits"])
        self.assertIn("test not loaded", manifest["split_method"])


if __name__ == "__main__":
    unittest.main()
