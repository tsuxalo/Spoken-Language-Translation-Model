"""Privacy-aware harness utilities for a future common-manifest pilot.

This module deliberately refuses the NaijaS2ST official ``dev`` split. It
prepares only project-validation rows recovered from the C1 training split,
stores row-level material under the ignored private artifact directory, and
produces aggregate metrics suitable for checking in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

PILOT_SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_CLUSTER_KEY = "alignment_id"
PILOT_SCOPE = "pilot"
FULL_DEVELOPMENT_SCOPE = "full-development"
FULL_DEVELOPMENT_EXAMPLES = 1_037

SCOPE_LABELS = {
    PILOT_SCOPE: "Common-manifest development pilot; not independent test evidence.",
    FULL_DEVELOPMENT_SCOPE: (
        "Full C1 internal-validation development evaluation; not independent test evidence."
    ),
}

DEPLOYABLE_SYSTEMS = ("cascade_real_asr", "direct_pilot", "direct_c1")
EVIDENCE_SCOPES = {
    "historical",
    "qualitative",
    "development-pilot",
    "common-manifest-development",
    "independent",
}
METRIC_ROW_FIELDS = {
    "system",
    "system_type",
    "evaluation_set",
    "dataset_revision",
    "number_of_examples",
    "model_revisions",
    "BLEU",
    "chrF++",
    "WER",
    "evidence_scope",
    "training_or_selection_influence",
    "source_artifact_or_document",
}
PREDICTION_FIELDS = {
    "example_id",
    "alignment_id",
    "system",
    "model_revisions",
    "reference",
    "prediction",
    "audio_duration_seconds",
    "load_time_seconds",
    "inference_time_seconds",
    "real_time_factor",
    "peak_gpu_memory_mb",
    "success",
    "error_type",
}

_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)*)(?!\w)")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_NEGATIONS = {"no", "not", "never", "neither", "nor", "nothing", "without", "n't"}


class ManifestPolicyError(ValueError):
    """Raised when membership violates the predeclared evaluation policy."""


class PredictionAlignmentError(ValueError):
    """Raised for duplicate, missing, or misaligned predictions."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_manifest_hash(rows: Sequence[dict[str, Any]]) -> str:
    """Return an order-stable SHA-256 over complete canonical row content."""

    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("example_id", row.get("manifest_id", ""))),
            str(row.get("system", "")),
        ),
    )
    return hashlib.sha256(_canonical_json(ordered)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON artifact in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ManifestPolicyError(f"line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def reject_official_dev(rows: Iterable[dict[str, Any]]) -> None:
    """Fail closed if any row identifies NaijaS2ST's official dev split."""

    for row in rows:
        values = {
            str(row.get(key, "")).strip().lower()
            for key in ("official_split", "dataset_split", "split")
        }
        if values & {"dev", "validation"}:
            raise ManifestPolicyError("NaijaS2ST official dev is prohibited for comparison-v2")


def validate_c1_membership(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ManifestPolicyError("membership is empty")
    reject_official_dev(rows)
    if {row.get("official_split") for row in rows} != {"train"}:
        raise ManifestPolicyError("C1 membership must originate only from official train")
    if {row.get("project_split") for row in rows} != {"validation"}:
        raise ManifestPolicyError("C1 membership must be project-validation only")
    if any(not str(row.get("target_text", "")).strip() for row in rows):
        raise ManifestPolicyError("every membership row must retain an English reference")
    if any(float(row.get("duration_seconds", math.inf)) > 30.0 for row in rows):
        raise ManifestPolicyError("membership contains audio outside the <=30 s contract")


def to_common_record(row: dict[str, Any]) -> dict[str, Any]:
    """Map the recovered C1 schema to comparison-v2's private schema."""

    alignment_id = row.get("alignment_id") or row.get("pair_id")
    example_id = row.get("example_id") or row.get("manifest_id")
    if not alignment_id or not example_id:
        raise ManifestPolicyError("membership row lacks an example or alignment ID")
    return {
        "example_id": str(example_id),
        "alignment_id": str(alignment_id),
        "reference": str(row["target_text"]),
        "hausa_transcription": str(row.get("source_transcript", "")),
        "audio_duration_seconds": float(row["duration_seconds"]),
        "audio_locator": row.get("audio_locator"),
        "dataset_revision": str(row.get("dataset_revision", "")),
        "official_split": str(row.get("official_split", "")),
        "project_split": str(row.get("project_split", "")),
    }


def select_grouped_pilot(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = PILOT_SEED,
    minimum_examples: int = 10,
    maximum_examples: int = 20,
) -> list[dict[str, Any]]:
    """Freeze 10-20 rows by deterministically sampling complete alignment groups."""

    if not 1 <= minimum_examples <= maximum_examples:
        raise ValueError("invalid pilot size bounds")
    validate_c1_membership(rows)
    records = [to_common_record(row) for row in rows]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record[BOOTSTRAP_CLUSTER_KEY]].append(record)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    for _, group in ordered_groups:
        group = sorted(group, key=lambda row: row["example_id"])
        if len(selected) + len(group) > maximum_examples:
            continue
        selected.extend(group)
        if len(selected) >= minimum_examples:
            break
    if not minimum_examples <= len(selected) <= maximum_examples:
        raise ManifestPolicyError("could not form a complete-group pilot within size bounds")
    return selected


def freeze_full_development_membership(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Freeze the complete recovered C1 development membership without sampling."""

    validate_c1_membership(rows)
    if len(rows) != FULL_DEVELOPMENT_EXAMPLES:
        raise ManifestPolicyError(
            f"full development membership must contain {FULL_DEVELOPMENT_EXAMPLES} rows"
        )
    records = [to_common_record(row) for row in rows]
    if len({record["example_id"] for record in records}) != len(records):
        raise ManifestPolicyError("full development membership has duplicate example IDs")
    return records


def build_frozen_manifest(
    source_rows: Sequence[dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    """Build a private, immutable evaluation manifest for an approved scope."""

    if scope == PILOT_SCOPE:
        selected = select_grouped_pilot(source_rows)
    elif scope == FULL_DEVELOPMENT_SCOPE:
        selected = freeze_full_development_membership(source_rows)
    else:
        raise ValueError(f"unsupported evaluation scope: {scope}")
    return {
        "schema_version": "1.0",
        "scope": scope,
        "label": SCOPE_LABELS[scope],
        "seed": PILOT_SEED if scope == PILOT_SCOPE else None,
        "cluster_key": BOOTSTRAP_CLUSTER_KEY,
        "source_number_of_examples": len(source_rows),
        "membership_sha256": canonical_manifest_hash(selected),
        "rows": selected,
    }


def validate_revision_map(revisions: dict[str, str]) -> None:
    if not revisions:
        raise ValueError("model_revisions is empty")
    for model_id, revision in revisions.items():
        if not model_id or not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError(f"unresolved immutable revision for {model_id!r}")


def validate_prediction_row(row: dict[str, Any]) -> None:
    missing = PREDICTION_FIELDS - set(row)
    if missing:
        raise PredictionAlignmentError(f"prediction row is missing fields: {sorted(missing)}")
    validate_revision_map(row["model_revisions"])
    if row["system"] not in DEPLOYABLE_SYSTEMS:
        raise PredictionAlignmentError(f"unexpected deployable system: {row['system']}")


def validate_prediction_matrix(
    rows: Sequence[dict[str, Any]],
    example_ids: Sequence[str],
    systems: Sequence[str] = DEPLOYABLE_SYSTEMS,
) -> None:
    expected = {(str(example_id), system) for example_id in example_ids for system in systems}
    actual: set[tuple[str, str]] = set()
    references: dict[str, str] = {}
    alignments: dict[str, str] = {}
    for row in rows:
        validate_prediction_row(row)
        key = (str(row["example_id"]), str(row["system"]))
        if key in actual:
            raise PredictionAlignmentError(f"duplicate prediction pair: {key}")
        actual.add(key)
        example_id = key[0]
        reference = str(row["reference"])
        alignment = str(row["alignment_id"])
        if example_id in references and references[example_id] != reference:
            raise PredictionAlignmentError(f"reference mismatch for {example_id}")
        if example_id in alignments and alignments[example_id] != alignment:
            raise PredictionAlignmentError(f"alignment mismatch for {example_id}")
        references[example_id] = reference
        alignments[example_id] = alignment
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PredictionAlignmentError(
            "prediction matrix mismatch; "
            f"missing_count={len(missing)}, missing_sample={missing[:10]}, "
            f"extra_count={len(extra)}, extra_sample={extra[:10]}"
        )


class PredictionStore:
    """Atomic, resume-safe private prediction store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise PredictionAlignmentError("prediction store is not a JSON list")
            self.rows = value

    def append(self, row: dict[str, Any], *, persist: bool = True) -> None:
        validate_prediction_row(row)
        key = (str(row["example_id"]), str(row["system"]))
        existing = {(str(item["example_id"]), str(item["system"])) for item in self.rows}
        if key in existing:
            raise PredictionAlignmentError(f"duplicate prediction pair: {key}")
        self.rows.append(row)
        if persist:
            self.checkpoint()

    def checkpoint(self) -> None:
        """Atomically persist all completed rows for safe resume."""

        atomic_write_json(self.path, self.rows)

    def finalize(
        self,
        example_ids: Sequence[str],
        systems: Sequence[str] = DEPLOYABLE_SYSTEMS,
    ) -> None:
        validate_prediction_matrix(self.rows, example_ids, systems)
        atomic_write_json(self.path, self.rows)


def validate_metric_artifact(value: dict[str, Any]) -> None:
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("metric artifact must contain nonempty rows")
    for index, row in enumerate(rows):
        missing = METRIC_ROW_FIELDS - set(row)
        if missing:
            raise ValueError(f"metric row {index} missing {sorted(missing)}")
        if row["evidence_scope"] not in EVIDENCE_SCOPES:
            raise ValueError(f"metric row {index} has invalid evidence_scope")
        validate_revision_map(row["model_revisions"])


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def _contains_repeated_trigram(text: str) -> bool:
    tokens = _normalize(text).split()
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return len(trigrams) != len(set(trigrams))


def _number_mismatch(reference: str, prediction: str) -> bool:
    return _NUMBER_RE.findall(reference.casefold()) != _NUMBER_RE.findall(prediction.casefold())


def _negation_omission(reference: str, prediction: str) -> bool:
    reference_tokens = set(_normalize(reference).split())
    prediction_tokens = set(_normalize(prediction).split())
    return bool(reference_tokens & _NEGATIONS) and not bool(prediction_tokens & _NEGATIONS)


def _corpus_scores(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    import sacrebleu

    predictions = [str(row["prediction"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    return {
        "BLEU": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrF++": sacrebleu.corpus_chrf(predictions, [references], word_order=2).score,
    }


def compute_system_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_prediction_row(row)
        by_system[str(row["system"])].append(row)

    output: dict[str, dict[str, Any]] = {}
    for system, system_rows in sorted(by_system.items()):
        successes = [row for row in system_rows if row["success"]]
        scores = _corpus_scores(successes) if successes else {"BLEU": 0.0, "chrF++": 0.0}
        normalized = [_normalize(str(row["prediction"])) for row in successes]
        nonempty = [value for value in normalized if value]
        one_per_alignment: dict[str, str] = {}
        for row, value in sorted(
            zip(successes, normalized), key=lambda item: str(item[0]["example_id"])
        ):
            one_per_alignment.setdefault(str(row["alignment_id"]), value)
        prediction_to_references: dict[str, set[str]] = defaultdict(set)
        for row, value in zip(successes, normalized):
            prediction_to_references[value].add(_normalize(str(row["reference"])))
        times = [float(row["inference_time_seconds"]) for row in successes]
        rtfs = [float(row["real_time_factor"]) for row in successes]
        output_lengths = [len(value.split()) for value in normalized]
        peaks = [
            float(row["peak_gpu_memory_mb"])
            for row in successes
            if row["peak_gpu_memory_mb"] is not None
        ]
        top_count = max(Counter(one_per_alignment.values()).values(), default=0)
        output[system] = {
            **scores,
            "number_of_examples": len(system_rows),
            "successes": len(successes),
            "failure_rate": 1 - len(successes) / len(system_rows),
            "empty_output_rate": sum(not value for value in normalized) / max(len(successes), 1),
            "raw_unique_output_rate": len(set(normalized)) / max(len(successes), 1),
            "alignment_adjusted_unique_output_rate": len(set(one_per_alignment.values()))
            / max(len(one_per_alignment), 1),
            "cross_reference_collisions": sum(
                len(references) > 1 for references in prediction_to_references.values()
            ),
            "top_hypothesis_frequency": top_count / max(len(one_per_alignment), 1),
            "repeated_3gram_rate": sum(_contains_repeated_trigram(str(row["prediction"])) for row in successes)
            / max(len(successes), 1),
            "number_mismatch_rate": sum(
                _number_mismatch(str(row["reference"]), str(row["prediction"])) for row in successes
            )
            / max(len(successes), 1),
            "negation_omission_rate": sum(
                _negation_omission(str(row["reference"]), str(row["prediction"]))
                for row in successes
            )
            / max(len(successes), 1),
            "output_tokens_mean": statistics.fmean(output_lengths) if output_lengths else 0.0,
            "output_tokens_median": statistics.median(output_lengths) if output_lengths else 0.0,
            "output_tokens_min": min(output_lengths, default=0),
            "output_tokens_max": max(output_lengths, default=0),
            "load_time_seconds": max((float(row["load_time_seconds"]) for row in successes), default=0.0),
            "inference_time_seconds_mean": statistics.fmean(times) if times else 0.0,
            "inference_time_seconds_median": statistics.median(times) if times else 0.0,
            "real_time_factor_mean": statistics.fmean(rtfs) if rtfs else 0.0,
            "peak_gpu_memory_mb": max(peaks, default=None),
            "nonempty_outputs": len(nonempty),
        }
    return output


def paired_cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    score_function: Callable[[Sequence[dict[str, Any]]], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Paired bootstrap that resamples identical alignment clusters for all systems.

    The default path precomputes SacreBLEU 2.6.0 segment statistics once and
    sums those sufficient statistics for every replicate. This is exactly
    equivalent to re-tokenizing each sampled corpus, including repeated
    clusters, but scales to the complete 1,037-row development membership.
    A custom score function retains the simple reference implementation used
    by lightweight tests.
    """

    import numpy as np

    systems = sorted({str(row["system"]) for row in rows})
    clusters = sorted({str(row[BOOTSTRAP_CLUSTER_KEY]) for row in rows})
    by_system_cluster: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system_cluster[(str(row["system"]), str(row[BOOTSTRAP_CLUSTER_KEY]))].append(row)
    if any(not by_system_cluster[(system, cluster)] for system in systems for cluster in clusters):
        raise PredictionAlignmentError("bootstrap inputs do not share identical alignment clusters")

    rng = np.random.default_rng(seed)
    cluster_indexes = {cluster: index for index, cluster in enumerate(clusters)}
    metric_objects: dict[str, Any] = {}
    statistics: dict[str, dict[str, Any]] = {}
    if score_function is None:
        from sacrebleu.metrics import BLEU, CHRF

        metric_objects = {"BLEU": BLEU(), "chrF++": CHRF(word_order=2)}
        for system in systems:
            statistics[system] = {}
            for metric, metric_object in metric_objects.items():
                cluster_statistics = []
                for cluster in clusters:
                    cluster_rows = by_system_cluster[(system, cluster)]
                    predictions = [str(row["prediction"]) for row in cluster_rows]
                    references = [str(row["reference"]) for row in cluster_rows]
                    segment_statistics = metric_object._extract_corpus_statistics(
                        predictions,
                        [references],
                    )
                    cluster_statistics.append(
                        np.asarray(segment_statistics, dtype=np.int64).sum(axis=0)
                    )
                statistics[system][metric] = np.stack(cluster_statistics)

    draws: dict[str, dict[str, list[float]]] = {
        system: {"BLEU": [], "chrF++": []} for system in systems
    }
    for _ in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        for system in systems:
            if score_function is None:
                sampled_indexes = [cluster_indexes[str(cluster)] for cluster in sampled]
                scores = {
                    metric: metric_objects[metric]
                    ._compute_score_from_stats(
                        statistics[system][metric][sampled_indexes]
                        .sum(axis=0)
                        .tolist()
                    )
                    .score
                    for metric in ("BLEU", "chrF++")
                }
            else:
                replicate_rows = [
                    row
                    for cluster in sampled
                    for row in by_system_cluster[(system, str(cluster))]
                ]
                scores = score_function(replicate_rows)
            for metric in ("BLEU", "chrF++"):
                draws[system][metric].append(float(scores[metric]))

    alpha = (1 - BOOTSTRAP_CONFIDENCE) / 2
    intervals: dict[str, dict[str, list[float]]] = {}
    for system in systems:
        intervals[system] = {}
        for metric in ("BLEU", "chrF++"):
            intervals[system][metric] = [
                float(np.quantile(draws[system][metric], alpha)),
                float(np.quantile(draws[system][metric], 1 - alpha)),
            ]

    pairs = (
        ("direct_c1", "direct_pilot"),
        ("direct_c1", "cascade_real_asr"),
        ("cascade_real_asr", "direct_pilot"),
    )
    deltas: dict[str, dict[str, Any]] = {}
    for left, right in pairs:
        if left not in draws or right not in draws:
            continue
        key = f"{left}_minus_{right}"
        deltas[key] = {}
        for metric in ("BLEU", "chrF++"):
            values = np.asarray(draws[left][metric]) - np.asarray(draws[right][metric])
            deltas[key][metric] = {
                "mean": float(values.mean()),
                "ci95": [float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))],
            }
    return {
        "method": "paired cluster bootstrap",
        "cluster_key": BOOTSTRAP_CLUSTER_KEY,
        "replicates": replicates,
        "seed": seed,
        "confidence_level": BOOTSTRAP_CONFIDENCE,
        "system_intervals": intervals,
        "paired_deltas": deltas,
    }


def estimate_full_runtime_hours(mean_seconds_per_example: float, examples: int = 1_037) -> float:
    if mean_seconds_per_example <= 0 or examples <= 0:
        raise ValueError("timing and example count must be positive")
    return mean_seconds_per_example * examples / 3_600


def default_score_paths(predictions: str | Path) -> dict[str, Path]:
    """Select safe aggregate paths for the canonical pilot or full prediction file."""

    full_development = Path(predictions).name == "full_development_predictions.json"
    if full_development:
        return {
            "manifest": Path(
                "artifacts/comparison-v2/private/full_development_manifest.json"
            ),
            "qualitative_queue": Path(
                "artifacts/comparison-v2/private/"
                "full_development_qualitative_review_queue.json"
            ),
            "output": Path("artifacts/comparison-v2/full_development_metrics.json"),
        }
    return {
        "manifest": Path("artifacts/comparison-v2/private/common_manifest.json"),
        "qualitative_queue": Path(
            "artifacts/comparison-v2/private/qualitative_review_queue.json"
        ),
        "output": Path("artifacts/comparison-v2/common_manifest_metrics.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument("source", type=Path)
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/comparison-v2/private/common_manifest.json"),
    )
    prepare_full = subparsers.add_parser("prepare-full-manifest")
    prepare_full.add_argument("source", type=Path)
    prepare_full.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/comparison-v2/private/full_development_manifest.json"),
    )
    metrics = subparsers.add_parser("score-private-predictions")
    metrics.add_argument("predictions", type=Path)
    metrics.add_argument(
        "--manifest",
        type=Path,
    )
    metrics.add_argument(
        "--qualitative-queue",
        type=Path,
    )
    metrics.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args(argv)

    if args.command in {"prepare-manifest", "prepare-full-manifest"}:
        source_rows = load_jsonl(args.source)
        scope = (
            PILOT_SCOPE
            if args.command == "prepare-manifest"
            else FULL_DEVELOPMENT_SCOPE
        )
        value = build_frozen_manifest(source_rows, scope=scope)
        atomic_write_json(args.output, value)
        print(f"Wrote {len(value['rows'])} private {scope} rows to {args.output}")
        return 0

    score_paths = default_score_paths(args.predictions)
    args.manifest = args.manifest or score_paths["manifest"]
    args.qualitative_queue = args.qualitative_queue or score_paths["qualitative_queue"]
    args.output = args.output or score_paths["output"]
    prediction_rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    membership = manifest.get("rows")
    if not isinstance(membership, list) or not membership:
        raise ManifestPolicyError("private evaluation manifest is empty")
    scope = str(manifest.get("scope", PILOT_SCOPE))
    if scope not in SCOPE_LABELS:
        raise ManifestPolicyError(f"unsupported private evaluation scope: {scope}")
    if scope == PILOT_SCOPE and not 10 <= len(membership) <= 20:
        raise ManifestPolicyError("pilot evaluation must contain 10-20 rows")
    if scope == FULL_DEVELOPMENT_SCOPE and len(membership) != FULL_DEVELOPMENT_EXAMPLES:
        raise ManifestPolicyError(
            f"full development evaluation must contain {FULL_DEVELOPMENT_EXAMPLES} rows"
        )
    reject_official_dev(membership)
    membership_hash = canonical_manifest_hash(membership)
    if membership_hash != manifest.get("membership_sha256"):
        raise ManifestPolicyError("private evaluation membership changed after it was frozen")
    example_ids = [str(row["example_id"]) for row in membership]
    validate_prediction_matrix(prediction_rows, example_ids)
    system_revisions = {
        system: next(
            row["model_revisions"]
            for row in prediction_rows
            if row["system"] == system
        )
        for system in DEPLOYABLE_SYSTEMS
    }
    computed_metrics = compute_system_metrics(prediction_rows)
    qualitative_queue = []
    for row in membership:
        reference = str(row["reference"])
        tokens = reference.split()
        reasons = []
        if _NUMBER_RE.search(reference):
            reasons.append("number_or_date_review")
        if any(token[:1].isupper() for token in tokens[1:]):
            reasons.append("possible_name_review")
        if reasons:
            qualitative_queue.append(
                {
                    "example_id": str(row["example_id"]),
                    "alignment_id": str(row["alignment_id"]),
                    "reasons": reasons,
                }
            )
    atomic_write_json(args.qualitative_queue, qualitative_queue)
    aggregate = {
        "schema_version": "1.0",
        "evaluation_scope": scope,
        "label": SCOPE_LABELS[scope],
        "dataset": {
            "id": "McGill-NLP/NaijaS2ST",
            "revision": str(membership[0]["dataset_revision"]),
            "official_source_split": "train",
            "project_split": "validation",
        },
        "membership": {
            "number_of_examples": len(membership),
            "alignment_clusters": len({str(row[BOOTSTRAP_CLUSTER_KEY]) for row in membership}),
            "sha256": membership_hash,
            "selection_seed": PILOT_SEED if scope == PILOT_SCOPE else None,
            "cluster_key": BOOTSTRAP_CLUSTER_KEY,
            "max_duration_seconds": max(float(row["audio_duration_seconds"]) for row in membership),
        },
        "model_revisions": system_revisions,
        "metrics": computed_metrics,
        "bootstrap": paired_cluster_bootstrap(prediction_rows),
        "observed_total_inference_hours": {
            system: sum(
                float(row["inference_time_seconds"])
                for row in prediction_rows
                if row["system"] == system
            )
            / 3_600
            for system in DEPLOYABLE_SYSTEMS
        },
        "qualitative_review_queue": {
            "private_path": (
                "artifacts/comparison-v2/private/"
                f"{args.qualitative_queue.name}"
            ),
            "number_of_examples": len(qualitative_queue),
            "note": "Names and dates are queued for human review, not assigned a formal score.",
        },
        "direct_pilot_overlap_audit": "UNVERIFIED",
        "limitation": (
            "Direct-pilot training overlap could not be fully audited. "
            "This evaluation must not be presented as leakage-free."
        ),
    }
    if scope == PILOT_SCOPE:
        aggregate["estimated_full_1037_inference_hours"] = {
            system: estimate_full_runtime_hours(values["inference_time_seconds_mean"])
            for system, values in computed_metrics.items()
        }
        aggregate["runtime_estimate_note"] = (
            "Linear extrapolation of mean batch-size-1 inference time only; excludes "
            "one-time model loading, dataset I/O, scoring, and operational margin."
        )
    atomic_write_json(args.output, aggregate)
    print(f"Wrote aggregate metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
