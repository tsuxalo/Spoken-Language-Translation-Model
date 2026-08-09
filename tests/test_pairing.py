import unittest

from hausa_s2tt.datasets import (
    align_naija_rows,
    alignment_key,
    assert_no_speaker_leakage,
    speaker_leakage,
    split_by_speaker,
)


def row(language, text_id, user_id, text, duration=2.0, split="train"):
    return {
        "audio": {"path": f"{user_id}-{text_id}.wav"},
        "language": language,
        "text_id": text_id,
        "user_id": user_id,
        "text": text,
        "duration": duration,
        "recorded_at": f"{user_id}-{text_id}",
        "split": split,
    }


class PairingTests(unittest.TestCase):
    def test_live_language_prefixed_ids_share_an_alignment_key(self):
        self.assertEqual(alignment_key("ETE_0001", "english"), "TE_0001")
        self.assertEqual(alignment_key("HTE_0001", "hausa"), "TE_0001")

    def test_shared_text_id_builds_hausa_audio_english_text_pairs(self):
        rows = [
            row("english", "T1", "E1", "Good morning"),
            row("english", "T1", "E2", "Good morning"),
            row("hausa", "T1", "H1", "Ina kwana"),
            row("hausa", "T1", "H2", "Ina kwana"),
        ]
        pairs, audit = align_naija_rows(rows, split="train")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["target_text"] == "Good morning" for pair in pairs))
        self.assertTrue(all(pair["source_language"] == "hausa" for pair in pairs))
        self.assertTrue(all(pair["target_language"] == "english" for pair in pairs))
        self.assertTrue(all(pair["alignment_key"] == "T1" for pair in pairs))
        self.assertEqual(audit.paired_examples, 2)
        self.assertEqual(audit.duplicate_target_text_ids, 1)

    def test_missing_invalid_and_conflicting_pairs_are_rejected(self):
        rows = [
            row("english", "ET1", "E1", "One"),
            row("english", "ET1", "E2", "Different"),
            row("hausa", "HT1", "H1", "Daya"),
            row("hausa", "MISSING", "H2", "Babu"),
            row("hausa", "LONG", "H3", "Dogo", duration=31),
        ]
        pairs, audit = align_naija_rows(rows, split="train")
        self.assertEqual(pairs, [])
        self.assertIn("T1", audit.conflicting_target_ids)
        self.assertIn("MISSING", audit.missing_target_ids)
        self.assertIn("LONG", audit.invalid_source_ids)
        self.assertEqual(audit.rejected_records, 3)
        self.assertEqual(audit.rejection_reasons["conflicting_english_targets"], 1)
        self.assertEqual(audit.rejection_reasons["duration_over_limit"], 1)
        self.assertEqual(audit.rejection_reasons["missing_english_target"], 1)
        self.assertEqual(audit.rejection_reasons["missing_audio_reference"], 0)

    def test_speaker_split_is_deterministic_and_leak_free(self):
        rows = [
            {"speaker_id": f"H{i}", "text_id": f"T{i}-{j}"}
            for i in range(5)
            for j in range(3)
        ]
        first = split_by_speaker(rows, test_fraction=0.2, seed=7)
        second = split_by_speaker(rows, test_fraction=0.2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(speaker_leakage(first), {})
        assert_no_speaker_leakage(first)

    def test_leakage_is_reported(self):
        rows = {
            "train": [{"speaker_id": "H1"}],
            "test": [{"speaker_id": "H1"}],
        }
        with self.assertRaises(ValueError):
            assert_no_speaker_leakage(rows)


if __name__ == "__main__":
    unittest.main()
