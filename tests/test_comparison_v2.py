from __future__ import annotations

import copy
from pathlib import Path

import pytest

import comparison_v2 as comparison

REVISION = "a" * 40


def _membership(example: str, alignment: str, *, official_split: str = "train") -> dict:
    return {
        "manifest_id": example,
        "pair_id": alignment,
        "target_text": f"reference {example}",
        "source_transcript": f"hausa {example}",
        "duration_seconds": 1.0,
        "audio_locator": {"kind": "test"},
        "dataset_revision": REVISION,
        "official_split": official_split,
        "project_split": "validation",
    }


def _prediction(example: str, alignment: str, system: str) -> dict:
    return {
        "example_id": example,
        "alignment_id": alignment,
        "system": system,
        "model_revisions": {"model": REVISION},
        "reference": f"reference {example}",
        "prediction": f"prediction {system} {example}",
        "audio_duration_seconds": 1.0,
        "load_time_seconds": 2.0,
        "inference_time_seconds": 0.5,
        "real_time_factor": 0.5,
        "peak_gpu_memory_mb": 100.0,
        "success": True,
        "error_type": None,
    }


def test_manifest_hash_is_reproducible_and_order_stable() -> None:
    rows = [_membership("b", "g2"), _membership("a", "g1")]
    assert comparison.canonical_manifest_hash(rows) == comparison.canonical_manifest_hash(
        list(reversed(copy.deepcopy(rows)))
    )


def test_official_dev_is_rejected() -> None:
    with pytest.raises(comparison.ManifestPolicyError, match="official dev"):
        comparison.validate_c1_membership([_membership("a", "g1", official_split="dev")])


def test_grouped_selection_is_seeded_and_keeps_clusters_whole() -> None:
    rows = [
        _membership(f"e{group}-{speaker}", f"g{group}")
        for group in range(12)
        for speaker in range(2)
    ]
    first = comparison.select_grouped_pilot(rows)
    second = comparison.select_grouped_pilot(copy.deepcopy(rows))
    assert first == second
    assert 10 <= len(first) <= 20
    selected_ids = {row["alignment_id"] for row in first}
    assert all(sum(row["pair_id"] == group for row in rows) == 2 for group in selected_ids)


def test_full_development_membership_is_complete_and_frozen() -> None:
    rows = [
        _membership(f"e{index}", f"g{index // 2}")
        for index in range(comparison.FULL_DEVELOPMENT_EXAMPLES)
    ]
    manifest = comparison.build_frozen_manifest(
        rows,
        scope=comparison.FULL_DEVELOPMENT_SCOPE,
    )
    assert manifest["scope"] == comparison.FULL_DEVELOPMENT_SCOPE
    assert len(manifest["rows"]) == comparison.FULL_DEVELOPMENT_EXAMPLES
    assert manifest["seed"] is None
    assert manifest["membership_sha256"] == comparison.canonical_manifest_hash(
        manifest["rows"]
    )
    with pytest.raises(comparison.ManifestPolicyError, match="must contain"):
        comparison.freeze_full_development_membership(rows[:-1])


def test_full_prediction_filename_selects_full_scoring_paths() -> None:
    full = comparison.default_score_paths("full_development_predictions.json")
    pilot = comparison.default_score_paths("common_predictions.json")
    assert full["manifest"].name == "full_development_manifest.json"
    assert full["output"] == Path("artifacts/comparison-v2/full_development_metrics.json")
    assert pilot["manifest"].name == "common_manifest.json"


def test_duplicate_and_missing_predictions_are_rejected() -> None:
    systems = ("cascade_real_asr", "direct_pilot", "direct_c1")
    rows = [_prediction("e1", "g1", system) for system in systems]
    comparison.validate_prediction_matrix(rows, ["e1"])
    with pytest.raises(comparison.PredictionAlignmentError, match="duplicate"):
        comparison.validate_prediction_matrix(rows + [copy.deepcopy(rows[0])], ["e1"])
    with pytest.raises(comparison.PredictionAlignmentError, match="missing"):
        comparison.validate_prediction_matrix(rows[:-1], ["e1"])


def test_reference_alignment_must_be_one_to_one() -> None:
    rows = [
        _prediction("e1", "g1", "cascade_real_asr"),
        _prediction("e1", "g1", "direct_pilot"),
        _prediction("e1", "g1", "direct_c1"),
    ]
    rows[1]["reference"] = "different reference"
    with pytest.raises(comparison.PredictionAlignmentError, match="reference mismatch"):
        comparison.validate_prediction_matrix(rows, ["e1"])


def test_metric_artifact_schema_and_revision_recording() -> None:
    row = {
        "system": "direct_c1",
        "system_type": "direct",
        "evaluation_set": "internal validation",
        "dataset_revision": REVISION,
        "number_of_examples": 10,
        "model_revisions": {"model": REVISION},
        "BLEU": 1.0,
        "chrF++": 2.0,
        "WER": None,
        "evidence_scope": "common-manifest-development",
        "training_or_selection_influence": "yes",
        "source_artifact_or_document": "test",
    }
    comparison.validate_metric_artifact({"rows": [row]})
    row["model_revisions"] = {"model": "main"}
    with pytest.raises(ValueError, match="unresolved immutable revision"):
        comparison.validate_metric_artifact({"rows": [row]})


def test_bootstrap_uses_frozen_seed_and_alignment_cluster() -> None:
    assert comparison.BOOTSTRAP_SEED == 42
    assert comparison.BOOTSTRAP_REPLICATES == 1_000
    assert comparison.BOOTSTRAP_CLUSTER_KEY == "alignment_id"
    rows = [
        _prediction(example, alignment, system)
        for example, alignment in (("e1", "g1"), ("e2", "g2"))
        for system in comparison.DEPLOYABLE_SYSTEMS
    ]

    def score(sample):
        total = sum(int(row["example_id"][-1]) for row in sample)
        return {"BLEU": float(total), "chrF++": float(total * 2)}

    first = comparison.paired_cluster_bootstrap(rows, replicates=20, score_function=score)
    second = comparison.paired_cluster_bootstrap(rows, replicates=20, score_function=score)
    assert first == second
    assert first["seed"] == 42
    assert first["cluster_key"] == "alignment_id"


def test_optimized_bootstrap_matches_naive_corpus_scoring() -> None:
    rows = [
        _prediction(example, alignment, system)
        for example, alignment in (
            ("e1", "g1"),
            ("e2", "g1"),
            ("e3", "g2"),
            ("e4", "g3"),
        )
        for system in comparison.DEPLOYABLE_SYSTEMS
    ]
    for row in rows:
        row["reference"] = f"the reference sentence for {row['example_id']}"
        row["prediction"] = (
            row["reference"]
            if row["system"] == "direct_c1"
            else f"a different output from {row['system']}"
        )
    optimized = comparison.paired_cluster_bootstrap(rows, replicates=30)
    naive = comparison.paired_cluster_bootstrap(
        rows,
        replicates=30,
        score_function=comparison._corpus_scores,
    )
    for system in comparison.DEPLOYABLE_SYSTEMS:
        for metric in ("BLEU", "chrF++"):
            assert optimized["system_intervals"][system][metric] == pytest.approx(
                naive["system_intervals"][system][metric]
            )
    for pair, metrics in optimized["paired_deltas"].items():
        for metric, values in metrics.items():
            assert values["mean"] == pytest.approx(
                naive["paired_deltas"][pair][metric]["mean"]
            )
            assert values["ci95"] == pytest.approx(
                naive["paired_deltas"][pair][metric]["ci95"]
            )
