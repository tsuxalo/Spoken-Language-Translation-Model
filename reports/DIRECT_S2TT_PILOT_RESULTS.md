# Direct S2TT Pilot: Results and Notes

Small-scale pilot run of the `direct_s2tt_pilot_base.yaml` config (LoRA fine-tune
of `openai/whisper-small` in `task=translate` mode, on NaijaS2ST), run locally
to get a first real, trained-model comparison point against the project's
existing Whisper-ha -> NLLB-200 cascade.

## Why a targeted subset instead of the full training split

`train_direct_s2tt` (via `load_naija_split` -> `load_dataset(..., split="train")`)
downloads the *entire* NaijaS2ST train split before any `max_train_samples`
filtering happens — all 115 parquet shards, ~69GB, even though this pilot
config only needs 256 train / 128 validation examples.

Instead, this run:
1. Used `iter_dataset_parquet_metadata` + `align_naija_rows` (already in
   `src/hausa_s2tt/datasets.py`) to audit **all** train-split metadata without
   downloading any audio (confirmed 13,871 valid pairs, matching the tracked
   audit in `reports/naija_s2st_audit_summary.json`).
2. Picked the 2 parquet shards (`0030.parquet`, `0019.parquet`) that together
   contain the most pairs (903), and downloaded only those — a few GB instead
   of ~69GB.
3. Ran his existing pairing/speaker-split/selection logic on that subset to
   get the final 256 train / 128 validation examples.

See `scripts/direct_s2tt_pilot_step1_find_pairs.py` (metadata audit + shard
selection) and `scripts/direct_s2tt_pilot_step2_train.py` (targeted download +
training handoff). Both are standalone scripts, not part of the installed
`hausa_s2tt` CLI — they monkeypatch `load_training_data` to inject the
targeted-subset dataset, then call the real `train_main()` unmodified.

## Bugs hit and fixed along the way

1. **`data_files` path bug (in the pilot scripts, not his code):** the pairing
   metadata's `dataset_parquet_file` field stores bare filenames
   (`"0030.parquet"`), but `datasets.load_dataset(..., data_files=...)` needs
   the full repo-relative path (`"default/train/0030.parquet"`). Fixed in
   `direct_s2tt_pilot_step2_train.py`.
2. **`Seq2SeqTrainingArguments` no longer accepts `warmup_ratio`** in the
   `transformers` version this environment resolved to (5.15.0) — only
   `warmup_steps` exists. Fixed in `src/hausa_s2tt/training.py` by converting
   `warmup_ratio * max_steps` to an explicit step count. This is a real
   version-compatibility bug, not specific to the targeted-subset approach —
   worth keeping regardless of how the data gets loaded.
3. **Metadata `duration` field is unreliable for at least one clip:** a clip
   passed the 30s pairing cutoff using its metadata-reported duration, but the
   real decoded audio measured 30.613s and crashed training mid-run (twice,
   identically). Fixed by filtering on the *real decoded* audio duration
   (`<=29.5s`) in step 2, rather than trusting the metadata field.

## Result

Training completed: 50 steps / 3.125 epochs, LoRA adapter (1.77M trainable
params, 0.7% of the 243M total), saved to
`artifacts/experiments/direct_s2tt/pilot_base/checkpoint-50`.

| Metric | Value |
|---|---|
| Train loss (avg) | 30.98 |
| Validation loss | 4.04 |
| Validation BLEU | **0.24** |
| Validation chrF++ | **14.39** |

### Compared to the existing cascade (Whisper-ha -> NLLB-200)

| System | BLEU | chrF++ |
|---|---|---|
| Cascade, gold Hausa text | ~22–25 | ~46–50 |
| Cascade, real ASR output | **~8–10** | **~33–35** |
| Direct pilot (this run) | **0.24** | **14.39** |

The direct pilot is far below the cascade, in both conditions. Given the tiny
training set (256 examples, 50 steps), this most plausibly reflects
insufficient training data rather than a fundamental weakness of the direct
paradigm — the cascade benefits from two independently, massively pretrained
models (Whisper's 680k hours, NLLB's large parallel-text corpus), while this
pilot has to learn the entire audio-to-English mapping from a very small
slice of data.

### Caveats

- **Not yet an apples-to-apples comparison.** This validation score is on a
  split *derived from the train set* (`official_dev_evaluated: false` in the
  training output), not the same official NaijaS2ST `dev` examples the
  cascade's BLEU 8–10 was measured on. For a rigorous side-by-side, this
  checkpoint should be re-evaluated on that same held-out set.
- **This is a pilot, not a final result.** 256 training examples is far below
  what a direct model would realistically need to be competitive. Treat this
  as a feasibility/first-signal run, not a verdict on cascade vs. direct as
  paradigms.

## Reproducing this run

```powershell
# From this branch, in a fresh venv with -e . installed:
python scripts/direct_s2tt_pilot_step1_find_pairs.py   # writes _pairs.json
python scripts/direct_s2tt_pilot_step2_train.py         # trains, ~10 min on an RTX 5050
```
