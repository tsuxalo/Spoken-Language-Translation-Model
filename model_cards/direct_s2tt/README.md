---
language:
- ha
- en
tags:
- speech-translation
- whisper
- hausa
- english
datasets:
- McGill-NLP/NaijaS2ST
metrics:
- bleu
- chrf
---

# Hausa→English direct Whisper S2TT — draft

This is a pre-publication template. **No full direct-S2TT checkpoint or held-out quality result exists in the current repository state.** The three-step tiny checkpoint is a mechanical smoke artifact and must not be published as a trained research model.

## Model description

- Task: direct speech-to-text translation, Hausa audio → English text.
- Base candidates: `openai/whisper-small` revision `973afd24965f72e36ca33b3055d56a652f456b4d`, or Hausa ASR initialization `nahomazmach/whisper-small-ha` revision `c4e2b47d88ae8b3ee0a605e09863b93aafca72e3`.
- Objective: teacher-forced English token prediction with Whisper `language=Hausa`, `task=translate`.
- Genuine labels: NaijaS2ST aligned English `target_text`; this is not an ASR checkpoint relabeled with `task=translate`.

## Data and splits

- Dataset: `McGill-NLP/NaijaS2ST`, revision `898f51582750fe244693794f22e3f4b32c5baf95`, CC BY 4.0.
- Accepted train pool: 13,871 Hausa recordings / 48.391 hours / 60 speakers after the measured 30-second and English-target filters.
- Training after the deterministic split: 12,385 pairs / 43.128 hours / 53 speakers.
- Selection: 1,486 pairs / 5.263 hours / 7 speakers, deterministically held out by speaker with seed 42 from accepted train pairs.
- Final evaluation: official dev, 1,500 accepted pairs / 3.888 hours / 6 speakers; no overlap with official train Hausa speakers in the complete metadata audit.
- No Hub test split exists. Full audio decode corruption was not audited.

## Training

- Efficiency strategy and hyperparameters: copy the selected pinned YAML and resolved run manifest here.
- Hardware/precision: not measured.
- Training duration/GPU-hours/peak VRAM/throughput/parameter counts/checkpoint size: not measured.

## Evaluation

| Metric | Train-derived validation | Official dev final |
|---|---:|---:|
| SacreBLEU | not measured | not measured |
| chrF++ | not measured | not measured |
| COMET | not run | not run |
| RTF | not measured | not measured |

Attach row-level predictions, `metrics.json`, SacreBLEU/chrF signatures, package versions, dataset/model revisions, run manifest, and final-test guard artifact before release. Include side-by-side zero-shot and cascade results on exactly the same final examples.

## Intended use and limitations

Research and supervised prototyping only. Hausa dialect, regional, accent, speaker, gender, topic, and recording-condition coverage is incomplete. Direct generation can be fluent while omitting, contradicting, or hallucinating meaning. Names, numbers, dates, and negation are especially consequential. Do not use as a sole translation source for medical, legal, immigration, policing, government, or other high-impact decisions.

Human review by fluent Hausa/English speakers and targeted subgroup/error analysis are required before broader use.

## License and citation

The final checkpoint license is not yet determined. NaijaS2ST is CC BY 4.0; review the selected base-model license and all inherited obligations before publication. Cite Whisper, NaijaS2ST and its paper (arXiv:2604.16287), any ASR initialization, and this repository's archived commit/run artifact.
