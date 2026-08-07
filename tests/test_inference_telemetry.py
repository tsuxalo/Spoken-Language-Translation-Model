import tempfile
import unittest

from hausa_s2tt.evaluation import FinalTestGuard
from hausa_s2tt.inference import _merge_text_segments
from hausa_s2tt.telemetry import (
    RuntimeMeasurement,
    estimate_error_percent,
    estimate_full_run,
)


class InferenceTelemetryTests(unittest.TestCase):
    def test_overlapping_chunk_text_is_deduplicated(self):
        merged = _merge_text_segments(["ina kwana lafiya", "lafiya lau yau"])
        self.assertEqual(merged, "ina kwana lafiya lau yau")

    def test_pilot_projection_distinguishes_measurement_and_projection(self):
        pilot = RuntimeMeasurement(10.0, 5, 25.0, 1, 1234)
        estimate = estimate_full_run(
            pilot,
            pilot_examples=5,
            full_examples_per_epoch=100,
            epochs=2,
            gpu_usd_per_hour=2.0,
        )
        self.assertEqual(estimate["projected_wall_seconds"], 400.0)
        self.assertIn("measurement_basis", estimate)
        self.assertIn("assumptions", estimate)
        self.assertAlmostEqual(estimate_error_percent(90, 100), 10)

    def test_final_test_guard_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = FinalTestGuard(directory, "run")
            guard.ensure_unused()
            guard.seal({"score": 1})
            with self.assertRaises(RuntimeError):
                guard.ensure_unused()


if __name__ == "__main__":
    unittest.main()
