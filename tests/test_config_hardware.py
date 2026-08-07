import json
import tempfile
import unittest
from pathlib import Path

from hausa_s2tt.config import ExperimentConfig, experiment_from_dict, load_config
from hausa_s2tt.hardware import select_precision
from hausa_s2tt.training import apply_efficiency_strategy


class FakeCuda:
    def __init__(self, available, bf16):
        self.available = available
        self.bf16 = bf16

    def is_available(self):
        return self.available

    def is_bf16_supported(self):
        return self.bf16


class FakeMps:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    def __init__(self, cuda, bf16):
        self.cuda = FakeCuda(cuda, bf16)
        self.backends = type("Backends", (), {"mps": FakeMps()})()


class FakeParameter:
    requires_grad = True


class FakeLayer:
    def __init__(self):
        self.values = [FakeParameter()]

    def parameters(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.frozen = False
        self.model = type(
            "Inner",
            (),
            {"encoder": type("Encoder", (), {"layers": [FakeLayer(), FakeLayer()]})()},
        )()

    def freeze_encoder(self):
        self.frozen = True


class ConfigHardwareTests(unittest.TestCase):
    def test_json_configuration_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"kind": "asr"}), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.dataset.validation_split, "validation")
        self.assertEqual(config.dataset.test_split, "test")

    def test_unknown_config_key_is_rejected(self):
        with self.assertRaises(ValueError):
            experiment_from_dict({"kind": "asr", "training": {"typo": 1}})

    def test_bf16_then_fp16_then_fp32_selection(self):
        self.assertEqual(select_precision(FakeTorch(True, True)).dtype, "bfloat16")
        self.assertEqual(select_precision(FakeTorch(True, False)).dtype, "float16")
        self.assertEqual(select_precision(FakeTorch(False, False)).dtype, "float32")

    def test_unsupported_explicit_bf16_fails(self):
        with self.assertRaises(RuntimeError):
            select_precision(FakeTorch(True, False), "bf16")

    def test_repository_configs_are_pinned_and_scientifically_typed(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        configs = {path.stem: load_config(path) for path in config_dir.glob("*.yaml")}
        self.assertTrue(configs)
        for config in configs.values():
            self.assertIsNotNone(config.dataset.revision)
            self.assertIsNotNone(config.model.revision)
        for name, config in configs.items():
            if name.startswith("direct_s2tt"):
                self.assertEqual(config.kind, "direct_s2tt")
                self.assertEqual(config.model.task, "translate")
                self.assertEqual(config.dataset.target_column, "target_text")
                self.assertEqual(config.dataset.target_language, "english")
                self.assertTrue(config.dataset.derive_validation_from_train)
            elif name.startswith("asr"):
                self.assertEqual(config.kind, "asr")
                self.assertEqual(config.model.task, "transcribe")
        cascade = configs["cascade"]
        self.assertEqual(cascade.mt.source_language, "hau_Latn")
        self.assertEqual(cascade.mt.target_language, "eng_Latn")
        self.assertIsNotNone(cascade.mt.revision)

    def test_freeze_and_partial_freeze_strategies(self):
        freeze_config = ExperimentConfig()
        freeze_config.model.efficiency_strategy = "freeze_encoder"
        frozen = apply_efficiency_strategy(FakeModel(), freeze_config)
        self.assertTrue(frozen.frozen)

        partial_config = ExperimentConfig()
        partial_config.model.efficiency_strategy = "partial_freeze"
        partial_config.model.partial_freeze_layers = 1
        partial = apply_efficiency_strategy(FakeModel(), partial_config)
        self.assertFalse(partial.model.encoder.layers[0].values[0].requires_grad)
        self.assertTrue(partial.model.encoder.layers[1].values[0].requires_grad)


if __name__ == "__main__":
    unittest.main()
