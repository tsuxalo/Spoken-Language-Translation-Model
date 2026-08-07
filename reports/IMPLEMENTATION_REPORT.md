# Implementation and experiment report

## Status

**Partial.** The complete installable pipeline, dataset pairing audit, CPU-safe tests, opt-in checkpoint smoke, genuine-label three-step training smoke, direct inference smoke, and published-checkpoint cascade smoke are complete. Full ASR-v2 training, full direct-S2TT training, and the common held-out comparison were not run; no final quality metric is claimed.

## Repository origin and preservation

- Repository: `https://github.com/tsuxalo/Spoken-Language-Translation-Model.git`
- Audited baseline/`origin/main`: `329254875059600f1b611aabe6ebb7c3e0e1d124`
- Working branch: `feature/direct-s2tt`
- No reset, checkout-based discard, history rewrite, push, checkpoint upload, or external publication was performed.

The original project was a working but scientifically compromised Hausa ASR prototype. Detailed findings are in [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md).

## Architecture

1. Zero-shot: base Whisper with `language=Hausa`, `task=translate` emits English. It receives no project fine-tuning.
2. ASR diagnostic: Hausa-fine-tuned Whisper with `task=transcribe` emits Hausa. It is never called a translation model.
3. Cascade: Hausa ASR emits Hausa; NLLB uses verified `hau_Latn` → `eng_Latn` and emits English.
4. Direct S2TT: Whisper with `task=translate` is optimized against NaijaS2ST English `target_text` labels aligned to Hausa audio.

All Whisper audio is mono 16 kHz, validated, polyphase-resampled, and divided into at most 29-second chunks. Chunk batches are padded/truncated to Whisper's fixed 30-second feature window; overlap text is merged. Full pairing and speaker splitting use Arrow dataset views and do not materialize audio bytes in Python lists.

## Data evidence

The exact NaijaS2ST and FLEURS measurements are stored in [naija_s2st_audit_summary.json](naija_s2st_audit_summary.json) and [fleurs_ha_ng_audit_summary.json](fleurs_ha_ng_audit_summary.json). The complete local alignment artifacts contain 13,871 train and 1,500 dev JSONL records.

Direct validation is derived from accepted train pairs by deterministically shuffling sorted speaker IDs with seed 42 and assigning complete speakers until at least 10% of rows are held out. This measured split has 12,385 training pairs (53 speakers, 43.128 hours) and 1,486 validation pairs (7 speakers, 5.263 hours), with zero speaker overlap. The official dev speaker set also has zero overlap with official train in the measured metadata. Official dev is not loaded by `train_experiment`.

The full audio corpus was not decoded. Missing paths, invalid durations, long clips, duplicates, target conflicts, missing targets, counts, hours, and speaker overlap were measured; byte-level corruption is unmeasured.

## Reproducibility and efficiency

- Configs pin all used dataset and model revisions.
- Python, NumPy, and PyTorch seeds are set; cuDNN deterministic mode is enabled.
- Run manifests capture Git SHA, hardware, configuration, dependency versions, data counts/audits, and parameter counts.
- Training summaries capture best validation checkpoint/metric, wall time, throughput, RTF, GPU-hours, peak allocated VRAM, and checkpoint size when a real run occurs.
- Precision selection is BF16 on compatible CUDA, FP16 on other CUDA, and FP32 otherwise. Explicit unsupported BF16/FP16 requests fail.
- Full, encoder-freeze, partial-freeze, and LoRA paths are configurable.

## Measured smokes

- Environment: Windows 11, Intel64 Family 6 Model 204, 16 logical CPUs, Python 3.12.13, PyTorch 2.13.0+cpu, FP32, no CUDA/MPS. Exact versions are in [smoke_environment.json](smoke_environment.json).
- Editable install and all seven console entrypoints load from an uninstalled checkout after `pip install -e .`.
- Public `openai/whisper-tiny` opt-in test generated with both `transcribe` and `translate` tasks.
- Three genuine-label direct optimizer steps passed on CPU; local checkpoint save/reload passed and emitted English.
- Zero-shot and exact published-checkpoint cascade inference passed on a genuine NaijaS2ST Hausa recording and emitted English.
- The tiny one-step estimator validation had 160.13% error because fixed overhead dominates; it is unsuitable as a full-job estimate.

See [EXPERIMENT_COMPARISON.md](EXPERIMENT_COMPARISON.md) for exact smoke outputs and honest unmeasured fields.

## Verification commands

The final handoff records exact outputs after the final Git state is committed. The verification suite is:

```powershell
git status --short --branch
git diff --check
python -m compileall -q src data_prep.py train.py inference.py evaluate_asr.py evaluate_s2tt.py estimate_experiment.py
python -m pip install -e . --no-deps
python -m pytest -q
$env:RUN_MODEL_SMOKE='1'; python -m pytest -q tests/test_model_smoke.py
ruff check .
python -m nbformat --validate capstone_demo.ipynb
hausa-s2tt-data --help
hausa-s2tt-train --help
hausa-s2tt-infer --help
hausa-s2tt-evaluate-asr --help
hausa-s2tt-evaluate --help
hausa-s2tt-estimate --help
hausa-s2tt-smoke-train --help
```

Credential, absolute-path, and large-file searches are also part of final verification.

## Colab

Open [capstone_demo.ipynb](../capstone_demo.ipynb), start a fresh runtime, and execute setup/hardware detection. First run the **FAST DEMO** audit and user-audio inference cells. The direct training and final comparison cells are marked **EXPENSIVE TRAINING** and require a pilot/authorization. The notebook imports package modules instead of duplicating the implementation.

## Publication commands—not executed

After a checkpoint and its model card have passed final review:

```bash
hf auth login
hf repos create YOUR_ACCOUNT/whisper-small-ha-asr-v2 --type model
hf upload YOUR_ACCOUNT/whisper-small-ha-asr-v2 artifacts/checkpoints/whisper-small-ha-asr-v2 . --type model --commit-message "Publish reviewed ASR v2"
hf repos create YOUR_ACCOUNT/whisper-small-ha-en-s2tt --type model
hf upload YOUR_ACCOUNT/whisper-small-ha-en-s2tt artifacts/checkpoints/whisper-small-ha-en-s2tt . --type model --commit-message "Publish reviewed direct S2TT"
```

Replace `YOUR_ACCOUNT`, copy the reviewed model card into each checkpoint directory, verify licenses, and obtain explicit authorization before running any command.

## Most valuable next experiment

Run matched 32-example, fixed-step LoRA pilots initialized from (a) `openai/whisper-small` and (b) `nahomazmach/whisper-small-ha`, using the identical speaker-safe train/validation subset. Select by validation chrF++ while recording time and peak VRAM. This directly tests whether Hausa ASR initialization improves translation under the project's low-resource constraint.
