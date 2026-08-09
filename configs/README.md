# Experiment configuration guide

Every YAML file is strict, revision-pinned, and designed for a single scientific role. `hausa-s2tt-train` consumes ASR and direct-S2TT configs. The cascade file is a reviewable reference manifest; inference and evaluation currently receive its equivalent pinned values through CLI defaults and flags.

| Config | Base model | Output | Efficiency | Data scope | Intended use |
|---|---|---|---|---|---|
| `asr_baseline.yaml` | `openai/whisper-small` | Hausa | Full-parameter | Full FLEURS train/validation | Corrected ASR baseline |
| `asr_improved.yaml` | Published Hausa ASR | Hausa | Full-parameter | Full FLEURS train/validation | Continued ASR diagnostic |
| `asr_lora.yaml` | Published Hausa ASR | Hausa | LoRA | Full FLEURS train/validation | Lower-memory ASR alternative |
| `direct_s2tt_smoke.yaml` | `openai/whisper-tiny` | English | Frozen encoder | 16 train / 8 validation maximum | Dataset-backed three-step integration smoke |
| `direct_s2tt_pilot_base.yaml` | `openai/whisper-small` | English | LoRA | Same 256 train / 128 validation maximum | 50-step GPU pilot from multilingual base |
| `direct_s2tt_pilot_from_asr.yaml` | Published Hausa ASR | English | LoRA | Same 256 train / 128 validation maximum | Matched 50-step GPU pilot from Hausa-ASR initialization |
| `direct_s2tt_full.yaml` | `openai/whisper-small` | English | LoRA | Full accepted train pool | Full-data direct candidate from base Whisper |
| `direct_s2tt_from_asr.yaml` | Published Hausa ASR | English | LoRA | Full accepted train pool | Full-data direct candidate from ASR initialization |
| `cascade.yaml` | Published Hausa ASR + NLLB | English | No project training | Reserved NaijaS2ST dev | Reference manifest for cascade evaluation |

In `direct_s2tt_full.yaml`, **full means the full accepted dataset**, not full-parameter fine-tuning. Both full-data direct candidates use LoRA so they can be compared under a similar memory budget.

`assert_matched_direct_configs` permits the two pilot files to differ only in model ID, model revision, output directory, and run name. Any other difference—including subset, seed, optimization, LoRA, precision, or generation settings—fails tests.

## Field wiring

- Training consumes `dataset`, `model`, `training`, the generation length/beam fields, and `tracking.run_name` through the saved resolved config and manifest.
- Cascade inference/evaluation consumes the model IDs, revisions, language codes, beam count, and token limit through CLI/runtime construction; `cascade.yaml` documents the intended immutable combination but is not passed to `hausa-s2tt-train`.
- `generation.chunk_length_seconds`, `stride_seconds`, and `batch_size` are runtime inference controls; use the corresponding `hausa-s2tt-infer` flags.
- `dataset.num_proc`, `tracking.artifacts_dir`, and tracking cost fields are reserved schema fields. They do not currently change execution. Cost is supplied to `hausa-s2tt-estimate` explicitly.

Unknown fields and scientifically inconsistent task combinations are rejected. In particular, ASR must use `transcribe` with Hausa targets, and direct S2TT must use `translate` with aligned English `target_text` labels and a train-derived speaker-safe validation set.
