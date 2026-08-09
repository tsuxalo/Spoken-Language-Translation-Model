import tempfile
import unittest

from hausa_s2tt.evaluation import FinalTestGuard
from hausa_s2tt.inference import (
    _merge_text_segments,
    create_asr_runtime,
    create_zero_shot_runtime,
)
from hausa_s2tt.mt import NLLBTranslator
from hausa_s2tt.revisions import (
    HAUSA_ASR_REVISION,
    NLLB_REVISION,
    WHISPER_SMALL_REVISION,
)
from hausa_s2tt.telemetry import (
    RuntimeMeasurement,
    estimate_error_percent,
    estimate_full_run,
    estimate_training_from_pilot_phases,
)
from hausa_s2tt.training import summarize_training_workload


class InferenceTelemetryTests(unittest.TestCase):
    def test_default_public_runtimes_are_revision_pinned(self):
        self.assertEqual(create_asr_runtime().revision, HAUSA_ASR_REVISION)
        self.assertEqual(create_zero_shot_runtime().revision, WHISPER_SMALL_REVISION)
        self.assertEqual(NLLBTranslator().revision, NLLB_REVISION)

    def test_training_workload_counts_multiple_epochs(self):
        workload = summarize_training_workload(100, 3_600.0, 3.0)
        self.assertEqual(workload["examples"], 300)
        self.assertEqual(workload["audio_seconds"], 10_800.0)
        self.assertFalse(workload["partial_epoch_audio_estimated"])

    def test_partial_epoch_workload_is_labeled_estimated(self):
        workload = summarize_training_workload(4, 23.414, 0.75)
        self.assertEqual(workload["examples"], 3)
        self.assertAlmostEqual(workload["audio_seconds"], 17.5605)
        self.assertTrue(workload["partial_epoch_audio_estimated"])

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

    def test_phase_aware_pilot_projection_separates_fixed_costs_and_uncertainty(self):
        estimate = estimate_training_from_pilot_phases(
            [2.0, 2.5, 3.0],
            optimizer_steps_per_epoch=10,
            epochs=2,
            startup_seconds=10,
            validation_seconds_per_run=5,
            validation_runs=2,
            checkpoint_seconds_per_save=3,
            checkpoint_saves=2,
            gpu_count=1,
            checkpoint_bytes_per_save=100,
            examples_per_step=16,
            audio_seconds_per_step=120,
            gpu_usd_per_hour=None,
        )
        self.assertEqual(estimate["projected_wall_seconds"], 76)
        self.assertEqual(estimate["projected_checkpoint_storage_bytes"], 200)
        self.assertIsNone(estimate["projected_gpu_cost_usd"])
        lower, upper = estimate["projected_wall_seconds_interval"]
        self.assertLess(lower, estimate["projected_wall_seconds"])
        self.assertGreater(upper, estimate["projected_wall_seconds"])


if __name__ == "__main__":
    unittest.main()
