from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts" / "colab" / "notebook_smoke.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def test_colab_smoke_artifact_is_sanitized_and_scoped() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert set(artifact) == {
        "schema_version",
        "status",
        "scope",
        "evidence",
        "runtime",
        "claims",
    }
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "passed"
    assert artifact["scope"] == "end-to-end Colab CPU smoke test"

    evidence = artifact["evidence"]
    assert set(evidence) == {
        "canonical_output_free_notebook_sha256",
        "executed_notebook_sha256",
        "repository_ref",
        "checked_out_commit",
        "notebook",
    }
    assert SHA256.fullmatch(evidence["canonical_output_free_notebook_sha256"])
    assert SHA256.fullmatch(evidence["executed_notebook_sha256"])
    assert GIT_COMMIT.fullmatch(evidence["checked_out_commit"])
    assert evidence["repository_ref"] == "main"
    assert evidence["notebook"] == {
        "nbformat_major": 4,
        "nbformat_minor": 5,
        "cells": 26,
        "code_cells": 11,
        "execution_counts": list(range(1, 12)),
        "error_outputs": 0,
    }

    runtime = artifact["runtime"]
    assert runtime["platform"] == "Google Colab"
    assert runtime["python"] == "3.11.9"
    assert runtime["device"] == "cpu"
    assert runtime["removed_optional_torchao"] == "0.10.0"
    assert set(runtime["binary_packages"]) == {
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pyarrow",
    }

    assert artifact["claims"] == {
        "gpu_notebook_gate_passed": False,
        "gpu_handoff_results_affected": False,
    }

    serialized = ARTIFACT.read_text(encoding="utf-8").lower()
    for forbidden in (
        "prediction",
        "transcript",
        "reference",
        "audio_path",
        "speaker_id",
        "/content/",
        "c:\\\\users\\\\",
    ):
        assert forbidden not in serialized
