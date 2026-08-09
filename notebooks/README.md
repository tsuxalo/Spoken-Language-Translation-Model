# Notebook workflow

The project is decomposing the original `capstone_demo.ipynb` into four focused notebooks. Notebooks 00 and 02 exist and pass structural validation; Notebooks 01 and 03 remain planned interfaces, not placeholder files or completed results.

Run them in numerical order. Notebook 00 is inexpensive by default. Notebooks 02 and 03 contain the expensive training and protected final-evaluation stages.

| Order | Notebook | Responsibility | Required inputs | Produced outputs | Cost/data policy |
|---:|---|---|---|---|---|
| 00 | [`00_data_loading_preprocessing.ipynb`](00_data_loading_preprocessing.ipynb) | Load tracked audits, explain dataset roles, align Hausa audio to English text, reject invalid rows, derive a speaker-disjoint train/validation split, demonstrate audio/text preprocessing, and define artifacts | Pinned public resources and checked-in audit summaries | Optional ignored JSONL under `artifacts/data/naija_s2st/` plus a versioned manifest | Safe path downloads at most one short official-train clip; full audit/build/write are disabled |
| 01 | `01_asr_and_cascade.ipynb` (planned) | Evaluate Hausa ASR separately, then translate its Hausa hypotheses with NLLB `hau_Latn → eng_Latn`; measure cascade latency/error propagation | Notebook 00 `audio_locator`, `source_text`, project split, speaker and provenance fields | Hausa ASR predictions and WER/CER; English cascade predictions and translation metrics | Train/validation only during development; no NaijaS2ST official-dev targets |
| 02 | [`02_direct_s2tt_training.ipynb`](02_direct_s2tt_training.ipynb) | Screen architectures; validate exact Notebook 00 membership; smoke-test tiny LoRA; compare matched base/ASR initializations; train selected Whisper-small direct S2TT; save checkpoints, predictions, and phase-aware telemetry | Notebook 00 `audio_locator`, genuine English `target_text`, speaker-disjoint train/validation records, manifest (or labeled seed-42 reconstruction) | Adapter + processor, resumable Trainer checkpoints, validation predictions/metrics, history, telemetry, architecture-selection JSON | All expensive flags false; GPU pilots, frozen selection, and estimate gate full training |
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

After the branch is published, open the desired notebook in Colab, select an appropriate runtime, and run from the top. Notebook 00 uses `feature/direct-s2tt`; Notebook 02 uses `feature/direct-s2tt-training-notebook`. Their setup cells clone or safely fast-forward a clean checkout, stop on dirty work, perform an editable install, evict stale imports, and verify package/Git/version provenance. After merge, update `REPO_REF` to `main`.

Leave `RUN_FULL_METADATA_AUDIT`, `BUILD_FULL_TRAINING_DATASET`, and `WRITE_ARTIFACTS` false for the safe path. Enable them only after reviewing the storage/runtime note in Section 19.

In Notebook 02, leave `RUN_STRUCTURAL_SMOKE`, `RUN_BASELINE_EVALUATION`, `RUN_MATCHED_PILOTS`, `FREEZE_ARCHITECTURE_SELECTION`, and `RUN_FULL_TRAINING` false initially. The structural smoke is a mechanics check. Baseline/pilot artifacts and qualitative validation review must exist before freezing selection; a hardware-matched phase-aware projection is required before full training.
