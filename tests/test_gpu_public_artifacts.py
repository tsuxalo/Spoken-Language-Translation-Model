import copy
import json
from pathlib import Path

import pytest

from scripts.build_gpu_handoff_artifacts import membership_hash
from scripts.validate_gpu_handoff_artifacts import EXPECTED_FILES, validate_package

PUBLIC_ROOT = Path(__file__).parents[1] / "artifacts" / "gpu-handoff"


def test_checked_gpu_aggregate_package_is_cross_consistent():
    result = validate_package(PUBLIC_ROOT)

    assert result == {
        "status": "valid",
        "files": 11,
        "systems": 5,
        "utterances": 1500,
        "alignment_clusters": 500,
    }


def test_membership_hash_is_sorted_and_rejects_duplicates():
    rows = [{"utterance_id": "synthetic-b"}, {"utterance_id": "synthetic-a"}]
    assert membership_hash(rows) == membership_hash(list(reversed(rows)))

    with pytest.raises(ValueError, match="Duplicate"):
        membership_hash([rows[0], copy.deepcopy(rows[0])])


def test_public_validator_rejects_private_row_fields(tmp_path):
    for filename in EXPECTED_FILES:
        source = PUBLIC_ROOT / filename
        (tmp_path / filename).write_bytes(source.read_bytes())
    target = tmp_path / "environment.json"
    document = json.loads(target.read_text("utf-8"))
    document["utterance_id"] = "invented-test-value"
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Private row-level field"):
        validate_package(tmp_path)


def test_public_validator_rejects_absolute_local_paths(tmp_path):
    for filename in EXPECTED_FILES:
        source = PUBLIC_ROOT / filename
        (tmp_path / filename).write_bytes(source.read_bytes())
    target = tmp_path / "environment.json"
    document = json.loads(target.read_text("utf-8"))
    document["unsafe"] = "C:\\Users\\example\\private"
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Forbidden private/local string"):
        validate_package(tmp_path)
