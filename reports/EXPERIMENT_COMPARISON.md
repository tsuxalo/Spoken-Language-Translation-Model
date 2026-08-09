# Experimental comparison

Status: **partial** as of 2026-08-09. No common held-out English evaluation has been run. “Not measured” is intentional.

Notebook 02 now defines matched development pilots for base-Whisper and Hausa-ASR initialization. Neither pilot has been run, so no initialization winner or measured training improvement is available yet.

The Notebook 02 structural smoke measured only software mechanics: Whisper-tiny LoRA exposed 73,728 trainable of 37,834,368 total parameters; all 48 trainable tensors received finite gradients; a checked frozen tensor was unchanged; adapter, processor, optimizer/scheduler state saved; local adapter reload and translation-mode generation completed. The CPU telemetry context took 1.735 seconds and wrote a 9,141,536-byte checkpoint. These values are not transferable runtime estimates or quality results. Exact provenance and hashes are in [direct_s2tt_structural_smoke.json](direct_s2tt_structural_smoke.json).

## Held-out results

| System | Fine-tuned for this output? | Architecture | Output | SacreBLEU | chrF++ | COMET | Training GPU-hours | Peak VRAM | RTF |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Whisper-small zero-shot | No | Direct | English | not measured | not measured | not run | 0 | not measured | not measured |
| Whisper-small-ha ASR | Yes, ASR only | Direct ASR | Hausa | N/A | N/A | N/A | historical/unverified | historical/unverified | not measured |
| Hausa ASR → NLLB-200 | Components pre-trained/fine-tuned separately | Cascade | English | not measured | not measured | not run | 0 in this project state | not measured | not measured |
| Whisper-small-ha-en | Yes, requires English labels | Direct | English | not measured | not measured | not run | not measured | not measured | not measured |

ASR v2 final raw/normalized WER and CER are also not measured. The historical model card reports WER 44.68, but that checkpoint's original workflow evaluated FLEURS test during training, so it is not entered as a clean result.

## Functional smoke evidence (not held-out metrics)

One public NaijaS2ST train example, HNX_0001, was used only to validate mechanics. The English reference is “Welsh AMs worried about 'looking like muppets'.” The tiny direct checkpoint was trained and tested on this same example; its output cannot estimate generalization.

| Path | Checkpoint(s) | Output | Measured inference | Interpretation |
|---|---|---|---:|---|
| Zero-shot direct | `openai/whisper-tiny` | “Well, say, I'm tired of the moment that I can't do it.” | 4.993 s; RTF 0.853 | English and nonempty, but inaccurate |
| Cascade | `nahomazmach/whisper-small-ha` → NLLB-200 600M | Hausa: “Welse Ems ta da mu game da ake zama bawaye.” English: “Welse Ems warned us about being crazy.” | 10.398 s; RTF 1.776 | Both stages function; meaning remains imperfect |
| Three-step direct smoke | `openai/whisper-tiny`, encoder frozen, genuine HNX_0001 English label | “Well, say, I'm tired of the moment that I can't do anything.” | 4.160 s; RTF 0.711 | Saved/reloaded direct checkpoint emits English; no quality claim |

Canonical cached-local verification artifacts:

- `artifacts/verification/zero_shot.json`
- `artifacts/verification/cascade.json`
- `artifacts/verification/training-20260807T161840Z/summary.json`
- `artifacts/verification/fine_tuned_direct.json`

These paths are intentionally ignored by Git because they accompany generated audio/checkpoints. Their measurements and SHA-256 checksums are tracked in [smoke_results.json](smoke_results.json). Earlier first-run measurements were 7.876 s zero-shot, 52.069 s cascade, and 4.178 s direct; the large cache-state variation is retained in that JSON and is why none of these single-example values is presented as production throughput.

## Efficiency and estimator accuracy

The canonical genuine-label three-step CPU verification measured 0.8807 seconds inside Trainer and 1.0162 seconds in the telemetry context, with 3.406 train steps/s, zero GPU-hours, and no VRAM measurement. Its saved full tiny checkpoint measured 155,007,480 bytes (147.827 MiB); checkpoint writing occurred outside the timed training block.

Estimator validation used the earlier three-step run so it remains a matched comparison: a one-step outer-wall pilot measured 0.8223 seconds, linear extrapolation predicted 2.4669 seconds, and that run's measured three-step value was 0.9483 seconds—an absolute percentage error of 160.13%. Fixed startup overhead dominates at this scale, so this estimator result must not be transferred to Whisper-small or GPU training.

Artifact: `artifacts/smoke/estimator_validation.json`.

## Required final command

After direct model selection is frozen, use the guarded command in the README to evaluate zero-shot, direct, and cascade systems against the same 1,500 dev references. That command writes every prediction before any result should be added to this table.
