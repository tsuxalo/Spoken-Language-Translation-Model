# Hausa→English S2TT execution plan

This plan reflects the implemented project rather than the superseded ASR-only prototype.

## Experimental contract

| Phase | System | Training labels | Selection data | Final data | State |
|---|---|---|---|---|---|
| 1 | Whisper Hausa ASR | Hausa transcripts | FLEURS validation | FLEURS test | Pipeline complete; full v2 run pending |
| 2 | Zero-shot Whisper | None | None | NaijaS2ST dev | Smoke complete; final comparison pending |
| 3 | Hausa ASR → NLLB | ASR/MT checkpoints | NaijaS2ST train-derived validation if tuned | NaijaS2ST dev | Smoke complete; final comparison pending |
| 4 | Direct Whisper S2TT | Aligned English `target_text` | Speaker-held-out subset of NaijaS2ST train | NaijaS2ST dev | Three-step smoke complete; full run pending |

NaijaS2ST dev remains sealed until all system choices are frozen. A metadata audit may verify pairing and leakage, but no dev prediction or metric is used for model selection. FLEURS test is likewise excluded from Trainer construction.

## Completed engineering

- Strict YAML configuration and pinned dataset/model revisions.
- Deterministic seeds, hardware-aware precision, gradient checkpointing, encoder freezing, partial freezing, and LoRA.
- Memory-safe NaijaS2ST Hausa-audio/English-text pairing with explicit rejection reasons.
- Official FLEURS split loading and raw/normalized WER/CER.
- Separate ASR, zero-shot direct, fine-tuned direct, and ASR→NLLB cascade runtimes.
- Long-audio chunking, stereo-to-mono conversion, polyphase resampling, invalid-audio checks, and batched chunk generation.
- Guarded final evaluation with row-level predictions and metric/provenance artifacts.
- Runtime, throughput, RTF, GPU-hours, peak VRAM, parameter-count, and checkpoint-size telemetry.
- Editable installation, console commands, CPU tests, real-checkpoint inference smoke, and three-step Trainer smoke.
- Revision-pinned library defaults, a replayable notebook validator, and a portable checksummed smoke summary.

## Remaining experiments

1. Run a 16–64 example hardware-matched direct-S2TT pilot for both base-Whisper and ASR-initialized LoRA variants.
2. Record wall time, storage, peak VRAM, and validation chrF++; compare against the estimator.
3. Authorize and run the selected full configuration only if the pilot stays within the agreed time/storage/cost budget.
4. Freeze the winning direct checkpoint and all generation settings.
5. Run ASR test and common NaijaS2ST dev comparisons once, using unique guarded run names.
6. Populate the comparison report and model cards only from saved artifacts.
7. Conduct optional blinded human evaluation with fluent Hausa/English reviewers.

## Exact commands

```bash
hausa-s2tt-data naija --source parquet --workers 4 --output-dir artifacts/audits/naija_s2st
hausa-s2tt-smoke-train --output-dir artifacts/smoke
hausa-s2tt-train --config configs/direct_s2tt_smoke.yaml
hausa-s2tt-train --config configs/direct_s2tt_full.yaml
hausa-s2tt-train --config configs/direct_s2tt_from_asr.yaml
```

The configured dataset smoke still downloads NaijaS2ST audio; the `hausa-s2tt-smoke-train` command is the lightweight API check. Any job expected to exceed 30 minutes, 20 GB, or incur cost requires a representative pilot and authorization first.

## Success criteria

- No ASR checkpoint is described as a translation model.
- Direct training labels are verified English references.
- Train/validation/final partitions follow the experimental contract.
- Full held-out metrics, if run, map to predictions and provenance artifacts.
- Reported values distinguish measured, projected, historical, and unmeasured fields.
- Checkpoints and large data remain ignored and are never published without explicit authorization.
