"""Create leakage-resistant train/validation splits for error-aware MT.

NaijaS2ST contains multiple recordings of the same underlying sentence.
Rows that share an alignment_id must stay in the same split; otherwise the
same Hausa/English text target can leak across train and validation through
different speakers.

The official NaijaS2ST dev split should NOT be passed through this script.
Keep it untouched for final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe(name: str, rows: list[dict]) -> dict:
    alignment_ids = {str(r["alignment_id"]) for r in rows}
    speaker_ids = {str(r.get("speaker_id", "")) for r in rows if r.get("speaker_id")}
    wers = [float(r["wer"]) for r in rows if r.get("wer") is not None]
    return {
        "split": name,
        "rows": len(rows),
        "alignment_ids": len(alignment_ids),
        "speakers": len(speaker_ids),
        "mean_wer": (sum(wers) / len(wers)) if wers else None,
    }


def canonical_membership_hash(rows: list[dict]) -> str:
    """Hash sorted private utterance IDs without publishing membership rows."""

    identifiers = [str(row.get("utterance_id", "")).strip() for row in rows]
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("Every row must contain a nonempty utterance_id.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("utterance_id values must be unique within a split.")
    payload = "".join(f"{value}\n" for value in sorted(identifiers)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_leakage_components(
    groups: dict[str, list[dict]],
) -> tuple[list[list[str]], int]:
    """Join alignment IDs that share one exact Hausa/English text pair.

    Component ordering and seeded shuffling intentionally match the completed
    authoritative run. Internal alignment inconsistencies are rejected before
    union-find so a corrupted alignment cannot silently bridge unrelated text.
    """

    alignment_pairs: dict[str, tuple[str, str]] = {}
    for alignment_id in sorted(groups):
        pairs = {
            (str(row["hausa_gold"]), str(row["english_ref"]))
            for row in groups[alignment_id]
        }
        if len(pairs) != 1:
            raise ValueError(
                f"alignment_id {alignment_id!r} maps to {len(pairs)} bilingual pairs"
            )
        alignment_pairs[alignment_id] = next(iter(pairs))

    parent = {alignment_id: alignment_id for alignment_id in groups}

    def find(alignment_id: str) -> str:
        while parent[alignment_id] != alignment_id:
            parent[alignment_id] = parent[parent[alignment_id]]
            alignment_id = parent[alignment_id]
        return alignment_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pair_owner: dict[tuple[str, str], str] = {}
    collision_pairs = 0
    for alignment_id in sorted(groups):
        pair = alignment_pairs[alignment_id]
        owner = pair_owner.get(pair)
        if owner is None:
            pair_owner[pair] = alignment_id
        elif find(owner) != find(alignment_id):
            collision_pairs += 1
            union(owner, alignment_id)

    components: dict[str, list[str]] = defaultdict(list)
    for alignment_id in sorted(groups):
        components[find(alignment_id)].append(alignment_id)

    ordered = [sorted(component) for component in components.values()]
    ordered.sort()
    return ordered, collision_pairs


def select_validation_ids(
    groups: dict[str, list[dict]],
    val_fraction: float,
    seed: int,
) -> tuple[set[str], dict[str, int]]:
    """Select whole leakage components using the authoritative seeded logic."""

    alignment_ids = sorted(groups)
    if len(alignment_ids) < 2:
        raise ValueError("At least two alignment IDs are required for a split.")
    components, collision_pairs = build_leakage_components(groups)
    rng = random.Random(seed)
    rng.shuffle(components)

    target = max(1, round(len(alignment_ids) * val_fraction))
    target = min(target, len(alignment_ids) - 1)
    validation_ids: set[str] = set()
    for component in components:
        if len(validation_ids) >= target:
            break
        validation_ids.update(component)

    return validation_ids, {
        "leakage_components": len(components),
        "cross_alignment_text_pair_collisions": collision_pairs,
        "max_component_alignment_ids": max(map(len, components)),
        "target_validation_alignment_ids": target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional JSON file containing split statistics.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")

    rows = read_jsonl(args.input)
    if not rows:
        raise RuntimeError("Input manifest is empty.")

    missing = [i for i, row in enumerate(rows) if "alignment_id" not in row]
    if missing:
        raise KeyError(f"{len(missing)} rows are missing alignment_id.")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["alignment_id"])].append(row)

    alignment_ids = sorted(groups)
    val_ids, component_stats = select_validation_ids(
        groups,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_ids = set(alignment_ids) - val_ids

    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(alignment_ids)

    train_rows = [row for row in rows if str(row["alignment_id"]) in train_ids]
    val_rows = [row for row in rows if str(row["alignment_id"]) in val_ids]

    train_text_pairs = {(r["hausa_gold"], r["english_ref"]) for r in train_rows}
    val_text_pairs = {(r["hausa_gold"], r["english_ref"]) for r in val_rows}
    overlap = train_text_pairs & val_text_pairs
    if overlap:
        raise RuntimeError(
            f"Detected {len(overlap)} Hausa/English text pairs in both train and val."
        )

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.val_output), val_rows)

    train_speakers = {r["speaker_id"] for r in train_rows if r.get("speaker_id")}
    val_speakers = {r["speaker_id"] for r in val_rows if r.get("speaker_id")}

    stats = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "grouping_key": "alignment_id+exact_text_pair_connected_components",
        "membership_hash_algorithm": (
            "sha256(newline-delimited sorted UTF-8 utterance_id values)"
        ),
        "input_rows": len(rows),
        "input_alignment_ids": len(alignment_ids),
        **component_stats,
        "train": describe("train", train_rows),
        "validation": describe("validation", val_rows),
        "train_membership_sha256": canonical_membership_hash(train_rows),
        "validation_membership_sha256": canonical_membership_hash(val_rows),
        "speaker_overlap": len(train_speakers & val_speakers),
        "alignment_overlap": len(train_ids & val_ids),
        "text_pair_overlap": len(overlap),
        "exact_bilingual_text_overlap": len(overlap),
    }

    print(json.dumps(stats, indent=2))

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
