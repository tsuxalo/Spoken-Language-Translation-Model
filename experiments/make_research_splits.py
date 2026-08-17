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
    rng = random.Random(args.seed)
    rng.shuffle(alignment_ids)

    n_val = max(1, round(len(alignment_ids) * args.val_fraction))
    n_val = min(n_val, len(alignment_ids) - 1)

    val_ids = set(alignment_ids[:n_val])
    train_ids = set(alignment_ids[n_val:])

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
        "grouping_key": "alignment_id",
        "input_rows": len(rows),
        "input_alignment_ids": len(alignment_ids),
        "train": describe("train", train_rows),
        "validation": describe("validation", val_rows),
        "speaker_overlap": len(train_speakers & val_speakers),
        "alignment_overlap": len(train_ids & val_ids),
        "text_pair_overlap": len(overlap),
    }

    print(json.dumps(stats, indent=2))

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
