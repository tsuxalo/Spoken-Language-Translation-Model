"""Step 2 (separate, fresh process): load the pairs found by step 1, download
only the needed shard files, build the final small dataset, and hand off to
his actual LoRA training code unmodified. Not part of his committed branch.
"""

import io
import json
import sys

import soundfile as sf
from datasets import Audio, load_dataset

import hausa_s2tt.training as training_mod
from hausa_s2tt.datasets import split_dataset_by_speaker

PARQUET_REVISION = "refs/convert/parquet"
NAIJA_DATASET_ID = "McGill-NLP/NaijaS2ST"
MAX_TRAIN = 256
MAX_VAL = 128

with open("_pairs.json", encoding="utf-8") as f:
    data = json.load(f)
chosen_files = data["chosen_files"]
chosen_pairs = data["chosen_pairs"]
print(f"Loaded {len(chosen_pairs)} pairs across {len(chosen_files)} files from step 1")

# dataset_parquet_file stores bare filenames (e.g. "0030.parquet"); data_files
# needs the full repo-relative path.
FULL_PATHS = [f"default/train/{name}" for name in chosen_files]
print(f"Full paths requested: {FULL_PATHS}")

print("Step 2: downloading only those shard files...")
dataset = load_dataset(
    NAIJA_DATASET_ID, data_files={"train": FULL_PATHS}, revision=PARQUET_REVISION, split="train"
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
print(f"Loaded {len(dataset)} raw rows from {len(chosen_files)} files")

wanted_ids = {p["source_text_id"] for p in chosen_pairs}
hausa_only = dataset.filter(lambda row: row["language"] == "hausa" and row["text_id"] in wanted_ids)
print(f"Filtered to {len(hausa_only)} matching Hausa rows")


# The metadata "duration" field proved unreliable for at least one clip
# (reported under his 30s pairing cutoff, but the real decoded audio measured
# 30.613s and crashed training mid-run). Filter on the REAL decoded duration
# instead of trusting metadata.
def real_duration_ok(row):
    audio_array, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
    return (len(audio_array) / sr) <= 29.5


before_real = len(hausa_only)
hausa_only = hausa_only.filter(real_duration_ok)
print(f"Real-duration safety filter: kept {len(hausa_only)}/{before_real} rows (<=29.5s actual)")

pair_by_id = {p["source_text_id"]: p for p in chosen_pairs}


def attach(row):
    p = pair_by_id[row["text_id"]]
    return {"target_text": p["target_text"], "speaker_id": p["speaker_id"], "duration": p["duration"]}


paired_dataset = hausa_only.map(attach)
print(f"Final paired dataset: {len(paired_dataset)} rows")

derived = split_dataset_by_speaker(
    paired_dataset, test_fraction=0.1, seed=42, train_name="train", test_name="validation"
)
train_full, validation_full = derived["train"], derived["validation"]
print(f"Speaker-safe split -> train={len(train_full)}, validation={len(validation_full)}")

train = train_full.select(range(min(MAX_TRAIN, len(train_full))))
validation = validation_full.select(range(min(MAX_VAL, len(validation_full))))
print(f"Final selected -> train={len(train)}, validation={len(validation)}")


def patched_load_training_data(config):
    return (
        train,
        validation,
        "target_text",
        {
            "split_method": "targeted-shard pilot (local patch, not the full 115-shard download)",
            "train_examples": len(train),
            "validation_examples": len(validation),
            "source_parquet_files": chosen_files,
        },
    )


training_mod.load_training_data = patched_load_training_data

if __name__ == "__main__":
    from hausa_s2tt.cli import train_main

    sys.argv = ["hausa-s2tt-train", "--config", "configs/direct_s2tt_pilot_base.yaml"]
    train_main()
