import unittest

from hausa_s2tt.metrics import (
    compute_asr_metrics,
    compute_translation_metrics,
    corpus_error_rate,
    normalize_hausa,
)


class MetricsTests(unittest.TestCase):
    def test_hausa_normalizer_preserves_language_letters_and_digits(self):
        self.assertEqual(
            normalize_hausa("  ƘAFA—ɓÉÉ! ɗaya, 12.  "),
            "ƙafa ɓéé ɗaya 12",
        )

    def test_raw_and_normalized_metrics(self):
        result = compute_asr_metrics(["Sannu, Duniya!"], ["sannu duniya"])
        self.assertGreater(result["raw_wer"], 0)
        self.assertEqual(result["normalized_wer"], 0)
        self.assertEqual(result["normalized_cer"], 0)

    def test_translation_metrics_include_reproducible_signatures(self):
        result = compute_translation_metrics(
            ["This is a translation."], ["This is a translation."]
        )
        self.assertEqual(result["chrf_pp"], 100.0)
        self.assertIn("tok:13a", result["sacrebleu_signature"])
        self.assertIn("nw:2", result["chrf_signature"])

    def test_wer_can_exceed_one_due_to_insertions(self):
        result = corpus_error_rate(["sannu"], ["sannu da zuwa sosai"], unit="word")
        self.assertGreater(result.rate, 1.0)

    def test_corpus_weighting_uses_total_reference_units(self):
        result = corpus_error_rate(["a", "a b c"], ["x", "a b c"], unit="word")
        self.assertAlmostEqual(result.rate, 0.25)


if __name__ == "__main__":
    unittest.main()
