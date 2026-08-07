---
language:
- ha
license: other
tags:
- automatic-speech-recognition
- whisper
- hausa
datasets:
- google/fleurs
metrics:
- wer
- cer
---

# Hausa Whisper ASR v2 — draft

This is a pre-publication template. **No ASR-v2 checkpoint or final metric has been produced in the current repository state.** Do not upload this card until every placeholder is replaced from saved run artifacts and the intended license is re-reviewed.

## Model description

- Task: Hausa automatic speech recognition (Hausa audio → Hausa text).
- It is not a Hausa→English translation model.
- Base checkpoint: `nahomazmach/whisper-small-ha` at revision `c4e2b47d88ae8b3ee0a605e09863b93aafca72e3` for the improved configuration; compare with `openai/whisper-small` in the corrected baseline.
- Architecture: Whisper encoder-decoder.

## Data and splits

- Dataset: `google/fleurs`, configuration `ha_ng`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC BY 4.0.
- Training: official train, measured 3,259 recordings / 13.618 hours.
- Selection: official validation, measured 296 recordings / 1.530 hours.
- Final evaluation: official test, measured 621 recordings / 3.340 hours; run only after model selection.
- The current schema has no public speaker identifier. FLEURS documentation states that train/development and test speakers are disjoint.

## Training

- Objective: teacher-forced Hausa token prediction with Whisper `task=transcribe`.
- Seed: 42.
- Hyperparameters: copy the final resolved YAML and run manifest here.
- Hardware/precision: not measured.
- Training duration/GPU-hours/peak VRAM/checkpoint size: not measured.

## Evaluation

| Metric | Validation | Test |
|---|---:|---:|
| Raw WER | not measured | not measured |
| Normalized WER | not measured | not measured |
| Raw CER | not measured | not measured |
| Normalized CER | not measured | not measured |

The normalizer uses NFKC, lowercasing, apostrophe unification, punctuation/symbol removal, and whitespace collapse while preserving Hausa letters, diacritics, and digits. Attach `predictions.jsonl`, `metrics.json`, run manifest, and exact metric/library versions before release.

The predecessor model card reports WER 44.68, but its original workflow repeatedly evaluated FLEURS test during training. That historical value is not a clean ASR-v2 result.

## Intended use and limitations

Intended for research and supervised prototyping. Nigerian Hausa coverage does not establish performance for all Hausa dialects, regions, accents, genders, domains, or recording conditions. Transcripts may omit or hallucinate words, names, numbers, or negation. Do not use this model as the sole basis for medical, legal, government, immigration, policing, or other consequential decisions.

Review dataset consent/attribution terms, conduct subgroup and human evaluation, and provide a failure-reporting channel before deployment.

## License and citation

The intended checkpoint license has not been finally determined. The base model card reports Apache-2.0 and FLEURS is CC BY 4.0; confirm inherited obligations before publication. Cite Whisper, FLEURS, the base checkpoint, and this repository's archived commit/run artifact.
