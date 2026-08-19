# Direct S2TT Pilot: Results and Notes

Small-scale pilot: a LoRA fine-tune of `openai/whisper-small` in `task=translate`
mode (Hausa audio -> English text, directly — no Hausa-text intermediate step),
trained on real Hausa/English pairs from
[McGill-NLP/NaijaS2ST](https://huggingface.co/datasets/McGill-NLP/NaijaS2ST).
This gives a first real, trained-model comparison point against this project's
main cascade (Whisper-ha -> NLLB-200 — see the root `README.md`).

The training config and package this pilot runs on (`hausa_s2tt`) live on a
teammate's branch, `feature/direct-s2tt`, not on `main` — `main` stays a
lightweight, standalone-script project. This folder documents what was run and
why, and keeps reference copies of the exact scripts used, but running them
requires checking out that branch (or
[`nahom/direct-s2tt-pilot-results`](https://github.com/tsuxalo/Spoken-Language-Translation-Model/tree/nahom/direct-s2tt-pilot-results),
which has the fixes described below already applied).

The trained pilot adapter is published on the Hugging Face Hub at
[nahomazmach/whisper-small-ha-en-direct-pilot](https://huggingface.co/nahomazmach/whisper-small-ha-en-direct-pilot)
— see the notebook's Section 6 for a runnable demo that doesn't need any of
the above.

## Why a targeted subset instead of the full training split

The branch's training code downloads the *entire* NaijaS2ST train split
before any sample-count filtering happens — all 115 parquet shards, ~69GB —
even though this pilot config only needs 256 train / 128 validation examples.

Instead, this run:
1. Used the branch's existing metadata-only audit tooling to check **all**
   train-split metadata without downloading any audio (confirmed 13,871 valid
   Hausa/English pairs, matching its own tracked audit).
2. Picked the 2 parquet shards that together contain the most pairs (903),
   and downloaded only those — a few GB instead of ~69GB.
3. Ran the branch's existing pairing/speaker-split/selection logic on that
   subset to get the final 256 train / 128 validation examples.

See `pilot_step1_find_pairs.py` (metadata audit + shard selection) and
`pilot_step2_train.py` (targeted download + training handoff) in this folder.
Both are standalone scripts that monkeypatch the branch's data-loading
function to inject the targeted-subset dataset, then call its real training
entrypoint unmodified — nothing about the model or training logic itself was
changed.

## Bugs hit and fixed along the way

1. **`data_files` path bug (in these pilot scripts, not the branch's code):**
   the pairing metadata's file-reference field stores bare filenames
   (`"0030.parquet"`), but `datasets.load_dataset(..., data_files=...)` needs
   the full repo-relative path (`"default/train/0030.parquet"`). Fixed in
   `pilot_step2_train.py`.
2. **`Seq2SeqTrainingArguments` no longer accepts `warmup_ratio`** in the
   `transformers` version this environment resolved to (5.15.0) — only
   `warmup_steps` exists. Fixed by converting `warmup_ratio * max_steps` to an
   explicit step count. A real version-compatibility bug in the branch's
   code, not specific to the targeted-subset approach.
3. **The dataset's metadata `duration` field is unreliable for at least one
   clip:** it passed the 30-second pairing cutoff using its metadata-reported
   duration, but the real decoded audio measured 30.613s and crashed training
   mid-run (twice, identically). Fixed by filtering on the *real decoded*
   audio duration (`<=29.5s`) instead of trusting the metadata field.

## Result

Training completed: 50 steps / 3.125 epochs, LoRA adapter (1.77M trainable
parameters, 0.7% of the 243M total).

| Metric | Value |
|---|---|
| Train loss (avg) | 30.98 |
| Validation loss | 4.04 |
| Validation BLEU | **0.24** |
| Validation chrF++ | **14.39** |

### Historical context beside the existing cascade (Whisper-ha -> NLLB-200)

| System | BLEU | chrF++ |
|---|---|---|
| Cascade, gold Hausa text | ~22–25 | ~46–50 |
| Cascade, real ASR output | **~8–10** | **~33–35** |
| Direct pilot (this run) | **0.24** | **14.39** |

A concrete example, same test clip used elsewhere in this README (ground
truth: *"An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong
da ginshiƙi mai walƙiya..."* — about Hong Kong's skyline and Victoria
Harbour). The direct pilot's output: *"The government has been working on the
development of the Hong Kong-based economic system in the region, which is
the only way to help the country win the war."* — fluent-sounding English,
but essentially unrelated to the actual content. That's what a BLEU of 0.24
looks like in practice: the model hasn't really learned to translate yet.

These values are historical context, not a common-manifest ranking. The pilot
validation came from NaijaS2ST `train`, while the cited cascade range came from
other memberships. The pilot result establishes that this 256-example run did
not learn reliable translation; it does not establish a statistical ordering
between direct and cascade architectures.

### Caveats

- **Not an apples-to-apples comparison.** This validation score is on a split
  derived from NaijaS2ST `train`. The C1/direct-pilot integration did not score
  official `dev`; it used the recovered 1,037-row C1 internal-validation
  membership. A separate completed fixed-ASR MT experiment has since observed
  official `dev`, so that split is no longer an untouched future tuning set.
- **Training overlap remains UNVERIFIED.** The exact historical `_pairs.json`
  membership was not preserved, so the common-membership evaluation states
  that direct-pilot training overlap could not be fully audited and must not be
  presented as leakage-free.
- **This is a pilot, not a final result.** 256 training examples is far below
  what a direct model would realistically need to be competitive. Treat this
  as a feasibility/first-signal run, not a verdict on cascade vs. direct as
  paradigms in general.

## Reproducing this run

```powershell
# Check out feature/direct-s2tt (or nahom/direct-s2tt-pilot-results), set up
# its venv with -e . installed, then from that branch's root:
python direct_pilot/pilot_step1_find_pairs.py   # writes _pairs.json
python direct_pilot/pilot_step2_train.py         # trains, ~10 min on an RTX 5050
```
