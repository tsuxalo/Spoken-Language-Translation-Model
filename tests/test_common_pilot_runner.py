from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import common_pilot_runner as runner
import comparison_v2 as comparison


def _private_row(index: int) -> dict:
    return {
        "example_id": f"example-{index}",
        "alignment_id": f"alignment-{index // 2}",
        "reference": f"reference {index}",
        "hausa_transcription": f"hausa {index}",
        "audio_duration_seconds": 0.1,
        "audio_locator": {"split": "train", "row_index": index},
        "dataset_revision": "a" * 40,
        "official_split": "train",
        "project_split": "validation",
    }


def test_row_resolution_is_lazy_and_scope_gated(tmp_path: Path) -> None:
    rows = [_private_row(index) for index in range(10)]
    for index in range(10):
        sf.write(
            tmp_path / f"{index:06d}_sample.wav",
            np.zeros(1_600, dtype=np.float32),
            16_000,
        )
    manifest = {
        "scope": comparison.PILOT_SCOPE,
        "membership_sha256": comparison.canonical_manifest_hash(rows),
        "rows": rows,
    }
    resolved = runner._resolve_rows(manifest, tmp_path, full_development=False)
    assert len(resolved) == 10
    assert all(isinstance(audio_path, Path) for _, audio_path in resolved)
    with pytest.raises(ValueError, match="does not match"):
        runner._resolve_rows(manifest, tmp_path, full_development=True)


def test_predictions_must_stay_private() -> None:
    with pytest.raises(ValueError, match="private directory"):
        runner.main(
            [
                "missing-manifest.json",
                "missing-audio-cache",
                "--predictions",
                "public-predictions.json",
            ]
        )
