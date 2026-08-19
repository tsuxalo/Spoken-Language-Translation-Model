"""Regenerate privacy-safe GPU handoff aggregates from authoritative evidence.

The private artifact root is read-only input.  No row-level text, identifiers,
predictions, paths, or checkpoints are copied into the public output package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments import revisions
from scripts.validate_gpu_handoff_artifacts import (
    MODEL_REVISIONS,
    RAW_ARTIFACT_HASHES,
    SCHEMA_VERSION,
    validate_package,
)

RAW_PATHS = {
    "whisper_prediction_suite": "results/final/whisper_error_aware_suite.csv",
    "gold_hausa_oracle_suite": "results/final/gold_mt_oracle.csv",
    "system_metrics": "results/final/analysis_whisper/system_metrics.csv",
    "paired_differences": (
        "results/final/analysis_whisper/paired_bootstrap_differences.csv"
    ),
    "train_asr_manifest": "generated/naijas2st_train_whisper_all.jsonl",
    "research_train_manifest": "generated/research_train.jsonl",
    "research_validation_manifest": "generated/research_val.jsonl",
    "dev_asr_manifest": "generated/naijas2st_dev_whisper_all.jsonl",
}

RECOVERY_PATCH_SHA256 = (
    "5d3bede898da681fd33f1f6483dd17c4e9109a29600f589a361dbb8565171931"
)
RECOVERY_PATCH_FILES = [
    "analysis/analyze_predictions.py",
    "experiments/generate_asr_noise.py",
    "experiments/gpu_preflight.py",
    "experiments/make_research_splits.py",
    "experiments/prepare_naijas2st_pairs.py",
    "requirements-comet.txt",
    "run_full_experiment.ps1",
]

SYSTEM_LABELS = {
    "nllb": "NLLB",
    "afrinllb": "AfriNLLB",
    "clean": "Clean LoRA",
    "noisy": "Noisy LoRA",
    "mixed": "Mixed LoRA",
}

RESOURCE_IDS = {
    "naijas2st": revisions.NAIJAS2ST_ID,
    "whisper_hausa": revisions.WHISPER_HAUSA_ID,
    "nllb_600m": revisions.NLLB_600M_ID,
    "afrinllb": revisions.AFRINLLB_ID,
    "nllb_3_3b": revisions.NLLB_3_3B_ID,
    "ssa_comet": revisions.SSA_COMET_ID,
}

GOLD_ORACLE_CONTRACT = {
    "nllb": {"bleu": 28.8034237, "chrf_pp": 52.13984},
    "afrinllb": {"bleu": 31.5290522, "chrf_pp": 54.3423189},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}.") from exc
    return rows


def membership_hash(rows: list[dict[str, Any]]) -> str:
    identifiers = [str(row.get("utterance_id", "")).strip() for row in rows]
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("A private split row is missing utterance_id.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate utterance_id in private split.")
    payload = "".join(f"{value}\n" for value in sorted(identifiers)).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def verify_raw_hashes(root: Path) -> dict[str, Path]:
    paths = {key: root / relative for key, relative in RAW_PATHS.items()}
    missing = [
        relative for key, relative in RAW_PATHS.items() if not paths[key].is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing authoritative artifacts: {missing}")
    observed = {key: sha256_file(path) for key, path in paths.items()}
    if observed != RAW_ARTIFACT_HASHES:
        failures = {
            key: {"expected": RAW_ARTIFACT_HASHES[key], "observed": observed[key]}
            for key in RAW_ARTIFACT_HASHES
            if observed[key] != RAW_ARTIFACT_HASHES[key]
        }
        raise ValueError(f"Authoritative raw-artifact hash mismatch: {failures}")
    return paths


def recompute_text_metrics(
    references: list[str], predictions: list[str]
) -> dict[str, float]:
    try:
        import sacrebleu
    except ImportError as exc:  # pragma: no cover - exercised by private command only
        raise RuntimeError("Private regeneration requires sacrebleu==2.6.0.") from exc
    if sacrebleu.__version__ != "2.6.0":
        raise RuntimeError(
            f"Metric regeneration requires sacrebleu==2.6.0, got {sacrebleu.__version__}."
        )
    return {
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf_pp": sacrebleu.corpus_chrf(predictions, [references], word_order=2).score,
    }


def verified_evaluation(paths: dict[str, Path], root: Path) -> dict[str, Any]:
    required = {
        "alignment_id",
        "utterance_id",
        "speaker_id",
        "english_ref",
        *(f"prediction_{system}" for system in SYSTEM_LABELS),
    }
    fields, rows = read_csv(paths["whisper_prediction_suite"])
    if not required.issubset(fields):
        raise ValueError(
            f"Prediction suite lacks fields: {sorted(required - set(fields))}"
        )
    if len(rows) != 1500:
        raise ValueError(f"Expected 1500 prediction rows, got {len(rows)}.")
    utterance_ids = [row["utterance_id"] for row in rows]
    if any(not value for value in utterance_ids) or len(set(utterance_ids)) != len(
        rows
    ):
        raise ValueError("Prediction rows are missing or duplicating utterance IDs.")
    if any(row.get(column) is None for row in rows for column in required):
        raise ValueError("A prediction row was misaligned during CSV parsing.")

    cluster_count = len({row["alignment_id"] for row in rows})
    speaker_count = len({row["speaker_id"] for row in rows})
    if cluster_count != 500 or speaker_count != 6:
        raise ValueError(
            f"Unexpected evaluation scale: {cluster_count} clusters, {speaker_count} speakers."
        )

    malformed_rows = [
        row
        for row in rows
        if all(
            row[f"prediction_{system}"] == ""
            for system in SYSTEM_LABELS
            if system != "mixed"
        )
        and row["prediction_mixed"] == "What?"
    ]
    if len(malformed_rows) != 1:
        raise ValueError(
            "The authoritative malformed-ASR/empty-output contract changed."
        )

    _, metric_rows = read_csv(paths["system_metrics"])
    authoritative = {
        row["system"]: {
            "bleu": float(row["bleu"]),
            "chrf_pp": float(row["chrf_pp"]),
            "ssa_comet": float(row["ssa_comet"]),
        }
        for row in metric_rows
    }
    if set(authoritative) != set(SYSTEM_LABELS):
        raise ValueError("Authoritative system set changed.")

    references = [row["english_ref"] for row in rows]
    for system in SYSTEM_LABELS:
        predictions = [row[f"prediction_{system}"] for row in rows]
        recomputed = recompute_text_metrics(references, predictions)
        for metric, value in recomputed.items():
            if not math.isclose(
                value, authoritative[system][metric], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{system} {metric} differs: recomputed={value}, "
                    f"authoritative={authoritative[system][metric]}"
                )

    _, gold_rows = read_csv(paths["gold_hausa_oracle_suite"])
    if len(gold_rows) != 1500:
        raise ValueError("Gold-Hausa oracle suite does not contain 1500 rows.")
    gold_references = [row["english_ref"] for row in gold_rows]
    gold_metrics: dict[str, dict[str, float]] = {}
    for system in ("nllb", "afrinllb"):
        gold_metrics[system] = recompute_text_metrics(
            gold_references, [row[f"prediction_{system}"] for row in gold_rows]
        )
        for metric, expected in GOLD_ORACLE_CONTRACT[system].items():
            if not math.isclose(
                gold_metrics[system][metric], expected, rel_tol=0.0, abs_tol=5e-8
            ):
                raise ValueError(f"Gold oracle {system} {metric} changed.")

    analysis_metadata = json.loads(
        (root / "results/final/analysis_whisper/analysis_metadata.json").read_text(
            "utf-8"
        )
    )
    if analysis_metadata["bootstrap_replicates"] != 1000:
        raise ValueError("Authoritative bootstrap replicate count changed.")
    if analysis_metadata["bootstrap_cluster_key"] != "alignment_id":
        raise ValueError("Authoritative bootstrap cluster key changed.")

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "fixed Whisper Hausa ASR text -> MT -> English",
        "evaluation_scope": "official NaijaS2ST dev",
        "evaluation_input": "Whisper-produced Hausa text",
        "evaluation_scale": {
            "utterances": len(rows),
            "alignment_clusters": cluster_count,
            "speakers": speaker_count,
        },
        "systems": authoritative,
        "gold_hausa_oracle": gold_metrics,
        "row_alignment": {
            "wide_table_shared_row_order_verified": True,
            "unique_utterance_rows_verified": True,
        },
        "malformed_asr_disclosure": {
            "cases": 1,
            "empty_hypothesis_system_count": 4,
            "mixed_hypothesis": "What?",
            "empty_hypotheses_retained_and_scored": True,
        },
    }


def verified_split(paths: dict[str, Path], root: Path) -> dict[str, Any]:
    train = read_jsonl(paths["research_train_manifest"])
    validation = read_jsonl(paths["research_validation_manifest"])
    dev = read_jsonl(paths["dev_asr_manifest"])
    split_stats = json.loads(
        (root / "results/final/split_stats.json").read_text("utf-8")
    )

    train_alignments = {str(row["alignment_id"]) for row in train}
    validation_alignments = {str(row["alignment_id"]) for row in validation}
    dev_alignments = {str(row["alignment_id"]) for row in dev}
    train_pairs = {(str(row["hausa_gold"]), str(row["english_ref"])) for row in train}
    validation_pairs = {
        (str(row["hausa_gold"]), str(row["english_ref"])) for row in validation
    }

    overlaps = {
        "train_validation_alignment_overlap": len(
            train_alignments & validation_alignments
        ),
        "train_validation_exact_bilingual_text_overlap": len(
            train_pairs & validation_pairs
        ),
        "train_official_dev_alignment_overlap": len(train_alignments & dev_alignments),
        "validation_official_dev_alignment_overlap": len(
            validation_alignments & dev_alignments
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"Scientific-integrity leakage gate failed: {overlaps}")

    expected_counts = {
        "input_rows": 14253,
        "input_alignment_clusters": 4751,
        "leakage_components": 4749,
        "cross_alignment_text_pair_collisions": 2,
        "max_component_alignment_clusters": 2,
        "target_validation_alignment_clusters": 475,
    }
    observed_counts = {
        "input_rows": split_stats["input_rows"],
        "input_alignment_clusters": split_stats["input_alignment_ids"],
        "leakage_components": split_stats["leakage_components"],
        "cross_alignment_text_pair_collisions": split_stats[
            "cross_alignment_text_pair_collisions"
        ],
        "max_component_alignment_clusters": split_stats["max_component_alignment_ids"],
        "target_validation_alignment_clusters": round(
            split_stats["input_alignment_ids"] * split_stats["val_fraction"]
        ),
    }
    if observed_counts != expected_counts:
        raise ValueError(f"Authoritative split counts changed: {observed_counts}")

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": split_stats["seed"],
        "validation_fraction": split_stats["val_fraction"],
        "grouping_key": "alignment_id plus exact bilingual-text connected components",
        **observed_counts,
        "train": {
            "rows": len(train),
            "alignment_clusters": len(train_alignments),
            "speakers": split_stats["train"]["speakers"],
            "mean_wer": split_stats["train"]["mean_wer"],
            "membership_sha256": membership_hash(train),
        },
        "validation": {
            "rows": len(validation),
            "alignment_clusters": len(validation_alignments),
            "speakers": split_stats["validation"]["speakers"],
            "mean_wer": split_stats["validation"]["mean_wer"],
            "membership_sha256": membership_hash(validation),
        },
        "official_dev": {
            "rows": len(dev),
            "alignment_clusters": len(dev_alignments),
            "speakers": len({str(row["speaker_id"]) for row in dev}),
            "status": "observed by the completed benchmark",
        },
        **overlaps,
        "membership_hash_algorithm": (
            "sha256(newline-delimited sorted UTF-8 utterance_id values)"
        ),
    }


def training_metrics(root: Path) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for mode in ("clean", "noisy", "mixed"):
        source = json.loads(
            (root / f"outputs/final/{mode}/experiment_metadata.json").read_text("utf-8")
        )
        conditions[mode] = {
            "train_examples": source["train_examples"],
            "validation_examples": source["val_examples"],
            "train_source_counts": source["train_source_counts"],
            "validation_distribution": source["validation_distribution"],
            "epochs": source["epochs"],
            "effective_batch_size": source["effective_batch_size"],
            "learning_rate": source["learning_rate"],
            "max_length": source["max_length"],
            "lora": {
                "rank": source["lora_r"],
                "alpha": source["lora_alpha"],
                "dropout": source["lora_dropout"],
                "trainable_parameters": source["trainable_parameters"],
            },
            "train_loss": source["train_metrics"]["train_loss"],
            "validation_loss": source["eval_metrics"]["eval_loss"],
            "train_runtime_seconds": source["train_metrics"]["train_runtime"],
        }
        if source["seed"] != 42:
            raise ValueError(f"Unexpected training seed for {mode}.")

    if any(value["train_examples"] != 12828 for value in conditions.values()):
        raise ValueError("Clean/noisy/mixed training exposure is inconsistent.")
    if conditions["mixed"]["train_source_counts"] != {
        "asr_noise": 6414,
        "clean": 6414,
    }:
        raise ValueError("Mixed exposure is not the expected 50/50 allocation.")
    return {"schema_version": SCHEMA_VERSION, "seed": 42, "conditions": conditions}


def csv_records(
    path: Path, integer_fields: set[str] | None = None
) -> list[dict[str, Any]]:
    integer_fields = integer_fields or set()
    _, rows = read_csv(path)
    records: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in integer_fields:
                converted[key] = int(value)
            elif key in {
                "point_estimate",
                "ci_2.5",
                "ci_97.5",
                "mean_bootstrap_delta",
                "paired_bootstrap_p",
                "pearson_r",
                "mean_wer",
                "mean_sentence_chrf",
            }:
                converted[key] = float(value)
            else:
                converted[key] = value
        records.append(converted)
    return records


def sanitized_environment(root: Path) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for line in (
        (root / "results/final/environment_training.txt")
        .read_text("utf-8")
        .splitlines()
    ):
        if "==" in line:
            name, version = line.split("==", 1)
            versions[name.lower()] = version
    preflight = json.loads(
        (root / "results/final/preflight_evaluate.json").read_text("utf-8")
    )
    gpu_text = (root / "results/final/gpu_info.txt").read_text("utf-8")
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", gpu_text)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", gpu_text)
    comet_log = (root / "results/final/95_analysis_install_comet.log").read_text(
        "utf-8", errors="replace"
    )
    if "unbabel-comet-2.2.7" not in comet_log:
        raise ValueError("Could not verify the completed COMET environment version.")
    return {
        "schema_version": SCHEMA_VERSION,
        "operating_system_family": "Windows",
        "python": preflight["python"].split()[0],
        "training_environment": {
            "torch": versions["torch"],
            "torch_cuda_build": preflight["cuda"]["torch_cuda_build"],
            "transformers": versions["transformers"],
            "datasets": versions["datasets"],
            "peft": versions["peft"],
            "sacrebleu": versions["sacrebleu"],
            "cuda_available": preflight["cuda"]["available"],
        },
        "analysis_environment": {
            "unbabel_comet": "2.2.7",
            "setuptools_compatibility": "<81",
        },
        "gpu": {
            "model": preflight["cuda"]["devices"][0]["name"],
            "driver": driver_match.group(1) if driver_match else "unknown",
            "reported_cuda": cuda_match.group(1) if cuda_match else "unknown",
        },
    }


def model_revisions() -> dict[str, Any]:
    central_registry = {
        "naijas2st": revisions.NAIJAS2ST_REVISION,
        "whisper_hausa": revisions.WHISPER_HAUSA_REVISION,
        "nllb_600m": revisions.NLLB_600M_REVISION,
        "afrinllb": revisions.AFRINLLB_REVISION,
        "nllb_3_3b": revisions.NLLB_3_3B_REVISION,
        "ssa_comet": revisions.SSA_COMET_REVISION,
    }
    if central_registry != MODEL_REVISIONS:
        raise ValueError("The public revision contract and central registry drifted.")
    return {
        "schema_version": SCHEMA_VERSION,
        "status_note": (
            "Revisions were resolved and captured for the completed run. Original "
            "commands did not enforce immutable revisions; future repository defaults do."
        ),
        "resources": {
            key: {
                "hub_id": RESOURCE_IDS[key],
                "resolved_revision": revision,
                "enforced_during_completed_run": False,
                "future_default_revision": True,
            }
            for key, revision in central_registry.items()
        },
    }


def build(private_root: Path, output_root: Path) -> dict[str, Any]:
    paths = verify_raw_hashes(private_root)
    evaluation = verified_evaluation(paths, private_root)
    split = verified_split(paths, private_root)

    raw_hash_document = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256",
        "artifacts": {
            key: {"logical_role": RAW_PATHS[key], "sha256": digest}
            for key, digest in RAW_ARTIFACT_HASHES.items()
        },
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "GPU error-aware MT benchmark",
        "base_git_commit": "8752d0d220b2ece8a81e572fd8134fb4e1a3b8df",
        "dirty_worktree": True,
        "dirty_patch_sha256": RECOVERY_PATCH_SHA256,
        "dirty_patch_hash_algorithm": (
            "sha256 of LF-normalized full-index binary Git diff for six tracked "
            "recovery files plus the LF-normalized executed supervisor snapshot"
        ),
        "dirty_patch_files": RECOVERY_PATCH_FILES,
        "recovery_fix_summary": [
            "Transformers-4 tokenizer metadata compatibility",
            "repeatable explicit GPU preflight paths",
            "connected-component leakage-safe split",
            "metadata-only dataset projection",
            "empty-hypothesis preservation during CSV analysis",
            "COMET Setuptools compatibility pin",
            "portable staged experiment supervision",
        ],
        "seed": 42,
        "bootstrap_replicates": 1000,
        "bootstrap_cluster_key": "alignment_id",
        "authoritative_raw_artifact_hashes": RAW_ARTIFACT_HASHES,
        "completed_run_revision_status": (
            "resolved and captured; not enforced by the original load commands"
        ),
        "official_dev_status": "observed by the completed benchmark",
    }

    documents = {
        "provenance.json": provenance,
        "raw_artifact_hashes.json": raw_hash_document,
        "split_summary.json": split,
        "training_metrics.json": training_metrics(private_root),
        "evaluation_metrics.json": evaluation,
        "bootstrap_confidence_intervals.json": {
            "schema_version": SCHEMA_VERSION,
            "replicates": 1000,
            "cluster_key": "alignment_id",
            "uncertainty_scope": "evaluation-cluster sampling, not training seeds",
            "intervals": csv_records(
                private_root
                / "results/final/analysis_whisper/bootstrap_confidence_intervals.csv",
                {"bootstrap_replicates"},
            ),
        },
        "paired_deltas.json": {
            "schema_version": SCHEMA_VERSION,
            "analysis": "predeclared paired cluster bootstrap against NLLB",
            "baseline": "nllb",
            "replicates": 1000,
            "cluster_key": "alignment_id",
            "direct_noisy_vs_mixed_saved": False,
            "deltas": csv_records(paths["paired_differences"]),
        },
        "error_correlations.json": {
            "schema_version": SCHEMA_VERSION,
            "unit": "utterance-level association summarized as one coefficient",
            "correlations": csv_records(
                private_root
                / "results/final/analysis_whisper/asr_error_correlations.csv"
            ),
        },
        "wer_bins.json": {
            "schema_version": SCHEMA_VERSION,
            "privacy": "aggregate bins only; no row-level values",
            "bins": csv_records(
                private_root
                / "results/final/analysis_whisper/wer_binned_translation.csv",
                {"n"},
            ),
        },
        "environment.json": sanitized_environment(private_root),
        "model_revisions.json": model_revisions(),
    }
    for filename, document in documents.items():
        write_json(output_root / filename, document)

    validation = validate_package(output_root)
    return {
        "status": "regenerated_and_validated",
        "output_files": len(documents),
        "metric_recomputation": "exact within 1e-12 for BLEU/chrF++",
        "public_validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/gpu-handoff")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.private_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
