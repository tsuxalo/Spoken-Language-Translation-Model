# Notebook workflow

The project is decomposing the original `capstone_demo.ipynb` into four focused notebooks. Only Notebook 00 exists today; the remaining filenames below are planned interfaces, not placeholder files or completed results.

Run them in numerical order. Notebook 00 is inexpensive by default. Notebooks 02 and 03 contain the expensive training and protected final-evaluation stages.

| Order | Notebook | Responsibility | Required inputs | Produced outputs | Cost/data policy |
|---:|---|---|---|---|---|
| 00 | [`00_data_loading_preprocessing.ipynb`](00_data_loading_preprocessing.ipynb) | Load tracked audits, explain dataset roles, align Hausa audio to English text, reject invalid rows, derive a speaker-disjoint train/validation split, demonstrate audio/text preprocessing, and define artifacts | Pinned public resources and checked-in audit summaries | Optional ignored JSONL under `artifacts/data/naija_s2st/` plus a versioned manifest | Safe path downloads at most one short official-train clip; full audit/build/write are disabled |
| 01 | `01_asr_and_cascade.ipynb` (planned) | Evaluate Hausa ASR separately, then translate its Hausa hypotheses with NLLB `hau_Latn → eng_Latn`; measure cascade latency/error propagation | Notebook 00 `audio_locator`, `source_text`, project split, speaker and provenance fields | Hausa ASR predictions and WER/CER; English cascade predictions and translation metrics | Train/validation only during development; no NaijaS2ST official-dev targets |
| 02 | `02_direct_s2tt_training.ipynb` (planned) | Smoke-test and efficiently fine-tune Whisper on genuine English targets; save checkpoints and telemetry | Notebook 00 `audio_locator`, `target_text`, speaker-disjoint train/validation records, manifest | Direct-S2TT checkpoint, config, validation predictions, runtime/compute evidence | Expensive; run a pilot and estimate storage/runtime before the full job |
| 03 | `03_final_evaluation_submission.ipynb` (planned) | Compare frozen zero-shot, cascade, and trained direct systems; report ASR separately; analyze efficiency, errors, ethics, and limitations | Frozen checkpoints/configs/predictions plus official held-out data through the final guard | Traceable per-example predictions, aggregate metrics, final comparison | One-time protected access to NaijaS2ST official dev and FLEURS test; never tune on results |

## Artifact handoff

Notebook 00's JSONL contract has schema version `1.0`. Each example contains:

```text
artifact_schema_version
audio_locator
source_text
target_text
duration
speaker_id
split
source_language
target_language
source_text_id
target_text_ids
alignment_key
dataset_row_index
source_dataset
dataset_revision
```

`audio_locator` contains durable dataset coordinates, not raw audio or a temporary signed URL. Its source split remains the official dataset split; the record's top-level `split` is the project train/validation assignment. Consumers must validate the schema and pinned dataset revision before resolving audio.

Generated data, audits, predictions, and checkpoints remain under ignored `artifacts/` paths. Small evidence summaries suitable for version control remain under `reports/`.

## Held-out-data rule

Notebook 00 may show only the checked-in aggregate NaijaS2ST `dev` audit. Notebooks 00–02 must not inspect dev targets, generate dev training examples, score dev predictions, or choose settings from dev results. Notebook 03 must use the repository's final-evaluation guard and must not feed results back into model selection. The same principle keeps FLEURS test out of ASR training and checkpoint selection.

## Colab

After the branch is published, open Notebook 00 in Colab, select a Python runtime, and run from the top. Its setup cell clones or safely fast-forwards `feature/data-preprocessing-notebook`, performs an editable install, evicts stale imports, and verifies the package path. After merge, update `REPO_REF` to `main`.

Leave `RUN_FULL_METADATA_AUDIT`, `BUILD_FULL_TRAINING_DATASET`, and `WRITE_ARTIFACTS` false for the safe path. Enable them only after reviewing the storage/runtime note in Section 19.
