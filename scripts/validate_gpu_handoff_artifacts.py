"""Validate the public, privacy-safe GPU handoff aggregate package.

This validator intentionally uses only the Python standard library so it can
run in ordinary CI without access to the private experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SYSTEMS = {"nllb", "afrinllb", "clean", "noisy", "mixed"}
METRICS = {"bleu", "chrf_pp", "ssa_comet"}
WER_BINS = {"<=20", "20-40", "40-60", "60-80", ">80"}

RAW_ARTIFACT_HASHES = {
    "whisper_prediction_suite": (
        "61ebe4bc29a5a63689aec28ff93452ec505ee81600866822040a1fbe63efe1a0"
    ),
    "gold_hausa_oracle_suite": (
        "1a014576359a5fdb828edefbc589ce811309118fddcd8886e310339d016e8390"
    ),
    "system_metrics": (
        "fb2cc584b2edd806e158c0daeeb828d011cf8971036f07d7ad4b35c9483b07eb"
    ),
    "paired_differences": (
        "c345940af76571cc6669a3d7d90e268c8b3caccdd959300673732c6879209f2e"
    ),
    "train_asr_manifest": (
        "de7fc7bebfbf8d48e6dcae560e799a804b631b1e0378a3ece382f13768798a0e"
    ),
    "research_train_manifest": (
        "0d360ae91d81a48b1edbb5b37cabb16b451528462adc2042fc4e35607d90dbfb"
    ),
    "research_validation_manifest": (
        "def8ee94b95ced618cf010491fcb2eda20dc7c5e562a714b3c84cf1045cfcab0"
    ),
    "dev_asr_manifest": (
        "61bfccc0a597f73264c829f04875201faccb75887d74621d14569638a0dc2ebc"
    ),
}

MODEL_REVISIONS = {
    "naijas2st": "898f51582750fe244693794f22e3f4b32c5baf95",
    "whisper_hausa": "c4e2b47d88ae8b3ee0a605e09863b93aafca72e3",
    "nllb_600m": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
    "afrinllb": "53b1bf8d09454d092a474a8e78d5c95a32b53154",
    "nllb_3_3b": "1a07f7d195896b2114afcb79b7b57ab512e7b43e",
    "ssa_comet": "6e64e0a56ce69524c67f304b092725687a362ef8",
}

EXPECTED_FILES = {
    "provenance.json",
    "raw_artifact_hashes.json",
    "split_summary.json",
    "training_metrics.json",
    "evaluation_metrics.json",
    "bootstrap_confidence_intervals.json",
    "paired_deltas.json",
    "error_correlations.json",
    "wer_bins.json",
    "environment.json",
    "model_revisions.json",
}

FORBIDDEN_KEYS = {
    "utterance_id",
    "utterance_ids",
    "speaker_id",
    "speaker_ids",
    "alignment_ids",
    "hausa_gold",
    "hausa_asr",
    "english_ref",
    "prediction",
    "predictions",
    "audio_path",
}
FORBIDDEN_TEXT_PATTERNS = (
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"/(?:Users|home)/"),
    re.compile(
        r"(?:HF_TOKEN|Bearer\s+|api[_-]?key|huggingface_[A-Za-z0-9])", re.IGNORECASE
    ),
    re.compile("final_" + "error_aware_" + "8752d0d", re.IGNORECASE),
    re.compile(r"GPU-[0-9a-f-]{16,}", re.IGNORECASE),
)


def load_package(root: Path) -> dict[str, Any]:
    """Load the exact expected JSON package and reject unexpected JSON files."""

    actual = {path.name for path in root.glob("*.json")}
    missing = EXPECTED_FILES - actual
    unexpected = actual - EXPECTED_FILES
    if missing or unexpected:
        raise ValueError(
            f"GPU aggregate file set mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        name.removesuffix(".json"): json.loads((root / name).read_text("utf-8"))
        for name in sorted(EXPECTED_FILES)
    }


def _walk(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"Private row-level field {key!r} found at {path}.")
            _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]")
    elif isinstance(value, str):
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"Forbidden private/local string found at {path}.")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_package(root: Path) -> dict[str, Any]:
    """Validate schemas, cross-file contracts, hashes, and privacy invariants."""

    package = load_package(root)
    for name, document in package.items():
        _require(
            document.get("schema_version") == SCHEMA_VERSION,
            f"{name}.json has an unsupported schema_version.",
        )
        _walk(document)

    hashes = package["raw_artifact_hashes"]["artifacts"]
    observed_hashes = {key: value["sha256"] for key, value in hashes.items()}
    _require(observed_hashes == RAW_ARTIFACT_HASHES, "Raw artifact hashes changed.")
    _require(
        package["provenance"]["authoritative_raw_artifact_hashes"]
        == RAW_ARTIFACT_HASHES,
        "Provenance and raw_artifact_hashes.json disagree.",
    )

    provenance = package["provenance"]
    _require(
        provenance["base_git_commit"] == "8752d0d220b2ece8a81e572fd8134fb4e1a3b8df",
        "Unexpected GPU experiment base commit.",
    )
    _require(provenance["dirty_worktree"] is True, "Dirty-run provenance was lost.")
    _require(provenance["seed"] == 42, "Unexpected experiment seed.")
    _require(provenance["bootstrap_replicates"] == 1000, "Unexpected bootstrap count.")
    _require(
        provenance["bootstrap_cluster_key"] == "alignment_id",
        "Unexpected bootstrap cluster key.",
    )

    revisions = package["model_revisions"]["resources"]
    observed_revisions = {
        key: value["resolved_revision"] for key, value in revisions.items()
    }
    _require(observed_revisions == MODEL_REVISIONS, "Pinned model revisions changed.")
    for resource in revisions.values():
        _require(
            resource["enforced_during_completed_run"] in {True, False, "unknown"},
            "Invalid completed-run enforcement status.",
        )
        _require(
            resource["future_default_revision"] is True,
            "A future-facing revision pin is not enabled.",
        )

    evaluation = package["evaluation_metrics"]
    scale = evaluation["evaluation_scale"]
    _require(
        scale == {"utterances": 1500, "alignment_clusters": 500, "speakers": 6},
        "Evaluation scale changed.",
    )
    _require(set(evaluation["systems"]) == SYSTEMS, "Evaluation system set changed.")
    for metrics in evaluation["systems"].values():
        _require(set(metrics) == METRICS, "Evaluation metric set changed.")

    bootstrap = package["bootstrap_confidence_intervals"]
    _require(bootstrap["replicates"] == 1000, "Bootstrap count is inconsistent.")
    _require(
        bootstrap["cluster_key"] == "alignment_id", "Bootstrap key is inconsistent."
    )
    bootstrap_pairs = {(row["system"], row["metric"]) for row in bootstrap["intervals"]}
    _require(
        bootstrap_pairs
        == {(system, metric) for system in SYSTEMS for metric in METRICS},
        "Bootstrap intervals do not cover the evaluation system/metric grid.",
    )

    paired = package["paired_deltas"]
    _require(paired["baseline"] == "nllb", "Unexpected paired-analysis baseline.")
    paired_pairs = {(row["system"], row["metric"]) for row in paired["deltas"]}
    _require(
        paired_pairs
        == {(system, metric) for system in SYSTEMS - {"nllb"} for metric in METRICS},
        "Paired deltas do not cover the predeclared comparisons.",
    )
    _require(
        paired["direct_noisy_vs_mixed_saved"] is False,
        "The package must not invent a noisy-versus-mixed comparison.",
    )

    wer_rows = package["wer_bins"]["bins"]
    _require(
        {(row["system"], row["wer_bin"]) for row in wer_rows}
        == {(system, label) for system in SYSTEMS for label in WER_BINS},
        "WER-bin aggregate grid is incomplete.",
    )
    for system in SYSTEMS:
        count = sum(row["n"] for row in wer_rows if row["system"] == system)
        _require(count == 1500, f"WER-bin counts do not conserve rows for {system}.")

    split = package["split_summary"]
    _require(split["seed"] == 42, "Split seed is inconsistent.")
    for overlap_name in (
        "train_validation_alignment_overlap",
        "train_validation_exact_bilingual_text_overlap",
        "train_official_dev_alignment_overlap",
        "validation_official_dev_alignment_overlap",
    ):
        _require(split[overlap_name] == 0, f"Leakage detected: {overlap_name}.")

    return {
        "status": "valid",
        "files": len(EXPECTED_FILES),
        "systems": len(SYSTEMS),
        "utterances": scale["utterances"],
        "alignment_clusters": scale["alignment_clusters"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/gpu-handoff"),
    )
    args = parser.parse_args()
    print(json.dumps(validate_package(args.artifact_root), indent=2))


if __name__ == "__main__":
    main()
