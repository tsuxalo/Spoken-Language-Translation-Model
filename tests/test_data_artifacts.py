import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hausa_s2tt.datasets import (
    ARTIFACT_SCHEMA_VERSION,
    PAIRING_REJECTION_REASONS,
    align_naija_rows,
    alignment_key,
    build_pairing_manifest,
    load_revision_matched_audit,
    pairing_artifact_record,
    split_by_speaker,
    validate_pairing_artifact,
    write_pairing_artifacts,
    write_pairing_manifest,
)
from hausa_s2tt.revisions import NAIJA_DATASET_ID, NAIJA_REVISION


def row(language, text_id, user_id, text, *, duration=2.0, audio=None, recorded_at=None):
    return {
        "audio": audio if audio is not None else {"path": f"audio/{user_id}-{text_id}.wav"},
        "language": language,
        "text_id": text_id,
        "user_id": user_id,
        "text": text,
        "duration": duration,
        "recorded_at": recorded_at or f"{user_id}-{text_id}",
        "split": "train",
    }


class DataArtifactTests(unittest.TestCase):
    def test_invalid_language_prefix_is_not_removed(self):
        self.assertEqual(alignment_key("ETE_0001", "hausa"), "ETE_0001")
        self.assertEqual(alignment_key("HTE_0001", "english"), "HTE_0001")

    def test_every_rejection_category_is_reported_and_unicode_is_preserved(self):
        rows = [
            row("english", "EOK", "E1", "Good morning."),
            row("english", "EAUDIO", "E2", "Audio target"),
            row("english", "ESPEAKER", "E3", "Speaker target"),
            row("english", "EDURATION", "E4", "Duration target"),
            row("english", "EZERO", "E5", "Zero target"),
            row("english", "ELONG", "E6", "Long target"),
            row("english", "ECONFLICT", "E7", "First target"),
            row("english", "ECONFLICT", "E8", "Second target"),
            row("english", "EINVALID", "E9", "   "),
            row("hausa", "HOK", "H1", "  Ɗan ƙauye ya ce ɓera.  "),
            row("hausa", "HAUDIO", "H2", "Sauti", audio={"path": ""}),
            row("hausa", "HSPEAKER", "", "Mai magana"),
            row("hausa", "", "H3", "Babu maɓalli"),
            row("hausa", "HDURATION", "H4", "Lokaci", duration="bad"),
            row("hausa", "HZERO", "H5", "Sifili", duration=0),
            row("hausa", "HLONG", "H6", "Dogo", duration=31),
            row("hausa", "HMISSING", "H7", "Babu fassara"),
            row("hausa", "HCONFLICT", "H8", "Rikici"),
            row("hausa", "HOK", "H1", "  Ɗan ƙauye ya ce ɓera.  "),
        ]
        pairs, audit = align_naija_rows(rows, split="train")

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["source_text"], "  Ɗan ƙauye ya ce ɓera.  ")
        self.assertEqual(
            pairs[0]["source_text_whitespace_normalized"], "Ɗan ƙauye ya ce ɓera."
        )
        self.assertEqual(set(audit.rejection_reasons), set(PAIRING_REJECTION_REASONS))
        for reason in PAIRING_REJECTION_REASONS:
            self.assertGreaterEqual(audit.rejection_reasons[reason], 1, reason)

    def test_revision_survives_align_split_and_serialization(self):
        revision = "a" * 40
        rows = [
            row("english", "ET1", "E1", "First target"),
            row("english", "ET2", "E2", "Second target"),
            row("hausa", "HT1", "H1", "Na farko", audio={"path": "audio/one.wav"}),
            row(
                "hausa",
                "HT2",
                "H2",
                "Na biyu",
                audio={"src": "https://example.invalid/audio.wav?X-Amz-Signature=secret"},
            ),
        ]
        pairs, audit = align_naija_rows(
            rows,
            split="train",
            dataset_revision=revision,
        )
        split = split_by_speaker(
            pairs,
            test_fraction=0.5,
            seed=42,
            train_name="train",
            test_name="validation",
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            all_records = []
            for name, records in split.items():
                split_audit = type(audit)(
                    split=name,
                    paired_examples=len(records),
                    rejection_reasons={reason: 0 for reason in PAIRING_REJECTION_REASONS},
                )
                paths = write_pairing_artifacts(records, split_audit, output)
                all_records.extend(
                    json.loads(line)
                    for line in Path(paths["examples"]).read_text(encoding="utf-8").splitlines()
                )
            manifest = build_pairing_manifest(
                [audit],
                git_commit="b" * 40,
                dataset_revision=revision,
                split_policy="speaker-disjoint official train split",
                seed=42,
                max_duration_seconds=30.0,
                generated_at="2026-08-08T00:00:00+00:00",
            )
            manifest_path = write_pairing_manifest(manifest, output)

            self.assertEqual(len(all_records), 2)
            serialized = json.dumps(all_records)
            for record in all_records:
                self.assertEqual(record["dataset_revision"], revision)
                self.assertEqual(record["audio_locator"]["dataset_revision"], revision)
                self.assertNotIn("audio", record)
                self.assertEqual(record["artifact_schema_version"], ARTIFACT_SCHEMA_VERSION)
            self.assertNotIn("X-Amz-Signature", serialized)
            self.assertNotIn("secret", serialized)
            self.assertNotIn("bytes", serialized)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_revision"],
                revision,
            )

    def test_unsafe_locator_payloads_are_rejected(self):
        pairs, _ = align_naija_rows(
            [
                row("english", "ET1", "E1", "Target"),
                row("hausa", "HT1", "H1", "Tushe"),
            ],
            split="train",
        )
        signed = pairing_artifact_record(pairs[0])
        signed["audio_locator"]["url"] = "https://example.test/a.wav?token=secret"
        with self.assertRaises(ValueError):
            validate_pairing_artifact(signed)

        waveform = pairing_artifact_record(pairs[0])
        waveform["audio_locator"]["waveform"] = np.zeros(16_000, dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_pairing_artifact(waveform)

    def test_tracked_audit_must_match_pinned_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": NAIJA_DATASET_ID,
                        "dataset_revision": NAIJA_REVISION,
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_revision_matched_audit(
                path,
                expected_dataset_id=NAIJA_DATASET_ID,
                expected_dataset_revision=NAIJA_REVISION,
            )
            self.assertEqual(loaded["dataset_revision"], NAIJA_REVISION)
            with self.assertRaises(ValueError):
                load_revision_matched_audit(
                    path,
                    expected_dataset_id=NAIJA_DATASET_ID,
                    expected_dataset_revision="wrong",
                )


if __name__ == "__main__":
    unittest.main()
