"""Step 1 (separate process): metadata-only audit + pairing + file selection.
Writes results to JSON so step 2 can run in a fresh process, since running
the metadata scan and load_dataset() in the same process breaks load_dataset.
"""

import json
from collections import defaultdict

from hausa_s2tt.datasets import align_naija_rows, iter_dataset_parquet_metadata

NAIJA_DATASET_ID = "McGill-NLP/NaijaS2ST"
NAIJA_REVISION = "898f51582750fe244693794f22e3f4b32c5baf95"
TARGET_PAIRS = 600

print("Step 1: metadata-only audit across all train shards (no audio bytes)...")
metadata_rows = list(iter_dataset_parquet_metadata("train", workers=4))
print(f"Got {len(metadata_rows)} metadata rows")

pairs, audit = align_naija_rows(
    metadata_rows, split="train", max_duration_seconds=30.0,
    dataset_id=NAIJA_DATASET_ID, dataset_revision=NAIJA_REVISION,
)
print(f"Found {len(pairs)} valid Hausa/English pairs (official audit found 13871)")

by_file = defaultdict(list)
for p in pairs:
    by_file[p["dataset_parquet_file"]].append(p)

files_sorted = sorted(by_file.items(), key=lambda kv: -len(kv[1]))
chosen_files: list[str] = []
chosen_pairs: list[dict] = []
for fname, flist in files_sorted:
    chosen_files.append(fname)
    chosen_pairs.extend(flist)
    if len(chosen_pairs) >= TARGET_PAIRS:
        break
print(f"Selected {len(chosen_files)} shard files covering {len(chosen_pairs)} pairs: {chosen_files}")

with open("_pairs.json", "w", encoding="utf-8") as f:
    json.dump({"chosen_files": chosen_files, "chosen_pairs": chosen_pairs}, f)
print("Wrote _pairs.json")
