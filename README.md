# Hausa→English Speech-to-Text Translation

This repository is my reproducible research pipeline for comparing four distinct systems:

| System | Input → output | Role |
|---|---|---|
| Zero-shot Whisper | Hausa audio → English text | Direct baseline; no fine-tuning |
| Hausa Whisper ASR | Hausa audio → Hausa text | Diagnostic ASR system, not translation |
| ASR + NLLB | Hausa audio → Hausa text → English text | Cascaded S2TT |
| Direct S2TT | Hausa audio → English text | Whisper trained on genuine aligned English labels |

The reusable pipeline and direct-training notebook are implemented and CPU-safe tests pass. The research result is still **partial**: matched initialization pilots, full direct-S2TT training, and the one-time common held-out comparison have not run, so pilot/full chrF++, SacreBLEU, GPU-hours, and peak VRAM are explicitly unmeasured.

## Scientific protocol

- FLEURS `ha_ng` uses official train/validation/test partitions. Training and checkpoint selection use train/validation only; test requires an explicit, one-time guarded command.
- NaijaS2ST train yields training pairs. Validation is deterministically derived by Hausa speaker from train. Official dev is reserved for the final zero-shot/direct/cascade comparison.
- The complete metadata audit measured no Hausa-speaker overlap between NaijaS2ST train and dev.
- Direct training reads `target_text`, which is the aligned English reference. Switching Whisper to `task=translate` without these labels is only zero-shot translation, not a fine-tuned direct model.
- Every evaluation writes row-level JSONL predictions, aggregate metrics, model/dataset provenance, and library versions.

See [DIRECT_S2TT_ARCHITECTURE_DECISION.md](reports/DIRECT_S2TT_ARCHITECTURE_DECISION.md), [IMPLEMENTATION_REPORT.md](reports/IMPLEMENTATION_REPORT.md), [EXPERIMENT_COMPARISON.md](reports/EXPERIMENT_COMPARISON.md), and [SOURCE_VERIFICATION.md](reports/SOURCE_VERIFICATION.md) for evidence and limitations.

## Notebook workflow

The original monolithic [capstone demo](capstone_demo.ipynb) is being decomposed into four focused notebooks with explicit artifact handoffs and held-out-data boundaries. The complete execution order and interface contract are in [notebooks/README.md](notebooks/README.md).

| Notebook | Status | Responsibility |
|---|---|---|
| [00 — Data loading and preprocessing](notebooks/00_data_loading_preprocessing.ipynb) | Available | Revision checks, tracked audits, Hausa-English alignment, rejection accounting, speaker-safe splitting, audio/text preprocessing, and durable artifacts |
| 01 — ASR and cascade | Planned; file not yet created | Hausa ASR diagnostics and Hausa ASR → NLLB English translation |
| [02 — Direct S2TT training](notebooks/02_direct_s2tt_training.ipynb) | Available; expensive stages disabled | Architecture screen, exact Notebook 00 handoff, tiny structural LoRA smoke, matched initialization pilots, validation, checkpoints, and phase-aware compute telemetry |
| 03 — Final evaluation and submission | Planned; file not yet created | Protected held-out comparison of frozen zero-shot, cascade, and direct systems |

Notebook 00's default path reads checked-in audits, runs deterministic small examples, and downloads at most one short public NaijaS2ST **train** clip. Full metadata audit, full dataset construction, and artifact writing are explicit opt-ins. NaijaS2ST official dev targets remain reserved for Notebook 03.

Notebook 02 selects a 244M-parameter Whisper-small direct encoder-decoder with rank-16 LoRA as the provisional bounded architecture. It compares multilingual-base and Hausa-ASR initializations with configs that are mechanically matched except for model identity and output provenance. “Full” in the direct config means all accepted training examples, not full-parameter fine-tuning.

## Measured data audit

NaijaS2ST revision `898f51582750fe244693794f22e3f4b32c5baf95` was audited through metadata-only Parquet range reads on 2026-08-07. Audio bytes were not downloaded for the complete audit.

| Split | All rows | Hausa rows | Accepted Hausa→English pairs | Hausa speakers | Paired hours | Rejected Hausa rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 52,000 | 15,000 | 13,871 | 60 | 48.391 | 1,129 |
| dev (final) | 5,500 | 1,500 | 1,500 | 6 | 3.888 | 0 |

Train rejections were 395 clips over Whisper's 30-second input limit and 734 recordings without an English target (249 unique alignment keys). There were no conflicting English targets, duplicate source recordings, missing audio paths, or train/dev Hausa-speaker overlaps in the metadata audit. Full audio decode integrity remains unmeasured.

The current FLEURS `ha_ng` revision has 3,259 train, 296 validation, and 621 test recordings (13.618, 1.530, and 3.340 hours). It exposes no speaker identifier; the FLEURS documentation states that train/development and test speakers are disjoint, while this repository measured zero audio-path overlap across splits.

## Install

Python 3.10+ is required. Install the appropriate PyTorch build for the actual machine first; the project deliberately does not pin a CUDA-specific wheel.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e .
```

For tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
pytest -q
python scripts/validate_notebook.py capstone_demo.ipynb
```

The tested local environment used CPU PyTorch 2.13.0 and Transformers 5.14.1. Runtime precision selection is BF16 on supported CUDA, otherwise FP16 on CUDA, otherwise FP32 on CPU/MPS.

## Reproducible commands

Audit NaijaS2ST without downloading the 69.9 GB corpus:

```bash
hausa-s2tt-data naija --source parquet --workers 4 \
  --output-dir artifacts/audits/naija_s2st
```

Run the CPU-safe three-step trainer smoke:

```bash
hausa-s2tt-smoke-train --output-dir artifacts/smoke
```

The smoke uses Whisper-tiny plus LoRA, verifies gradients and frozen weights, writes a resumable Trainer checkpoint, reloads the local adapter/processor, and executes translation-mode generation. It makes no translation-quality claim.

Inspect the matched GPU pilot configs before enabling the Notebook 02 gate:

```bash
python -c "from hausa_s2tt.config import load_config; from hausa_s2tt.direct_s2tt import assert_matched_direct_configs; assert_matched_direct_configs(load_config('configs/direct_s2tt_pilot_base.yaml'), load_config('configs/direct_s2tt_pilot_from_asr.yaml'))"
```

Train the corrected ASR diagnostic and direct S2TT systems:

```bash
hausa-s2tt-train --config configs/asr_improved.yaml
hausa-s2tt-train --config configs/direct_s2tt_full.yaml
```

The direct job downloads the NaijaS2ST training audio and is an expensive full experiment. Run a hardware-matched pilot before authorizing the full job. The LoRA alternative initialized from the published Hausa ASR checkpoint is in `configs/direct_s2tt_from_asr.yaml`.

Inference keeps ASR, zero-shot, direct, and cascade outputs distinct:

```bash
# Default public models resolve the immutable revisions listed in
# reports/SOURCE_VERIFICATION.md.
hausa-s2tt-infer --task asr sample.wav
hausa-s2tt-infer --task zero_shot sample.wav
hausa-s2tt-infer --task direct --model-id artifacts/checkpoints/whisper-small-ha-en-s2tt sample.wav
hausa-s2tt-infer --task cascade sample.wav
```

For a different public model, provide both `--model-id` and `--model-revision`. The Python factory defaults for the published Hausa ASR, Whisper-small, and NLLB cascade are also revision-pinned.

After model selection is frozen, run the common held-out comparison exactly once with a unique run name:

```bash
hausa-s2tt-evaluate \
  --systems zero_shot direct cascade \
  --direct-model-id artifacts/checkpoints/whisper-small-ha-en-s2tt \
  --dataset-revision 898f51582750fe244693794f22e3f4b32c5baf95 \
  --zero-shot-model-revision 973afd24965f72e36ca33b3055d56a652f456b4d \
  --asr-model-revision c4e2b47d88ae8b3ee0a605e09863b93aafca72e3 \
  --mt-model-revision f8d333a098d19b4fd9a8b18f94170487ad3f821d \
  --run-name final-v1 --confirm-final-test
```

Do not use a final-test result to revise hyperparameters. Use a new experimental protocol—not a new run name—to justify any future repeat.

## Metrics and artifacts

ASR evaluation reports corpus raw and normalized WER/CER. Translation evaluation reports case-sensitive SacreBLEU with `13a` tokenization and chrF++ (`word_order=2`), including signatures and package versions. COMET is optional and has not been run.

Ignored runtime outputs live under `artifacts/`; checkpoints, model binaries, datasets, caches, and generated predictions are excluded from Git. Small evidence summaries are tracked in `reports/`. The canonical functional-smoke values, local artifact paths, and SHA-256 checksums are in [`reports/smoke_results.json`](reports/smoke_results.json).

Notebook 02's measured mechanics-only LoRA smoke is recorded separately in [`reports/direct_s2tt_structural_smoke.json`](reports/direct_s2tt_structural_smoke.json). It verifies gradients, frozen weights, adapter/processor persistence, resumable state, and local translation-mode generation; it does not measure translation improvement.

## Colab

[capstone_demo.ipynb](capstone_demo.ipynb) is the 17-section graduate-project workflow. Start with a fresh GPU runtime, choose an explicit source mode in the setup cell, and run hardware detection. Git mode requires `REPO_REF` to exist remotely; until this local branch is pushed or merged, use archive-upload mode with a source ZIP made from this checkout. Use the marked **FAST DEMO** cells first, and run **EXPENSIVE TRAINING** only after the pilot estimate is acceptable. Colab accelerator type and price are never assumed.

For the decomposed workflow, run [Notebook 00](notebooks/00_data_loading_preprocessing.ipynb) before [Notebook 02](notebooks/02_direct_s2tt_training.ipynb). Notebook 02's Colab setup uses `REPO_REF = "feature/direct-s2tt-training-notebook"` until merge; then change it to `main`. It clones a missing checkout, safely fast-forwards a clean checkout, stops on dirty work, installs editable, verifies the import path and versions, and leaves every download/training flag false. Review the measured structural smoke and hardware-matched estimate before separately enabling baseline, pilot, selection, or full-training gates.

Create the unpublished-source archive from a committed checkout with:

```bash
git archive --format=zip --prefix=Spoken-Language-Translation-Model/ HEAD -o hausa-s2tt-source.zip
```

## Ethics, use, and licenses

This is a research prototype, not a sole translation source for medical, legal, immigration, policing, benefits, education, or other consequential decisions. Hausa dialect, Nigerian/regional coverage, accent, gender, speaker, topic, and recording-condition biases have not been fully characterized. Names, numbers, and negation are high-risk errors; both direct and cascade systems can omit, distort, or hallucinate content.

FLEURS and NaijaS2ST are CC BY 4.0. The published Hausa ASR checkpoint reports Apache-2.0. NLLB-200 is CC BY-NC 4.0, so the provided cascade must not be described as cleared for commercial use. Model outputs and fine-tuned checkpoints may inherit additional obligations from data and base-model licenses. Review [ETHICS_AND_LIMITATIONS.md](reports/ETHICS_AND_LIMITATIONS.md) before deployment or publication.

## Repository map

- `src/hausa_s2tt/`: reusable data, training, inference, evaluation, hardware, and telemetry modules.
- [`configs/`](configs/README.md): pinned ASR, LoRA, direct, cascade, smoke, and matched-pilot configurations, including a drift guard.
- `tests/`: CPU-safe unit tests plus an opt-in real-checkpoint smoke.
- `scripts/`: replayable structural validation utilities.
- `reports/`: audit, comparison, ethics, source verification, and human-evaluation materials.
- `model_cards/`: honest drafts for future ASR-v2 and direct-S2TT checkpoints.
- `notebooks/`: decomposed, ordered research notebooks and their artifact interfaces.
- Root Python files: backward-compatible wrappers around installed commands.

The most valuable next experiment is a hardware-matched LoRA pilot comparing initialization from `openai/whisper-small` versus the Hausa ASR checkpoint on the same speaker-safe NaijaS2ST validation set.

## Group project building notes

The team plans separate notebooks for the Hausa ASR diagnostic, the NLLB cascade, and the direct speech-encoder-to-English-text system, followed by a final comparison notebook. The final submission should compare architectures and outputs without conflating ASR with translation. Hausa is the current focus because usable data are available; extending the approach to other spoken languages remains a longer-term goal. Hausa's tonal and dialectal variation should be considered when interpreting results.
