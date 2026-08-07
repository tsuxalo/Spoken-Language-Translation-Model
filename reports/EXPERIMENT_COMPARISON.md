# Experimental comparison

Status: **partial** as of 2026-08-07. No common held-out English evaluation has been run. “Not measured” is intentional.

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
| Zero-shot direct | `openai/whisper-tiny` | “Well, say, I'm tired of the moment that I can't do it.” | 7.876 s; RTF 1.346 | English and nonempty, but inaccurate |
| Cascade | `nahomazmach/whisper-small-ha` → NLLB-200 600M | Hausa: “Welse Ems ta da mu game da ake zama bawaye.” English: “Welse Ems warned us about being crazy.” | 52.069 s; RTF 8.895 | Both stages function; meaning remains imperfect |
| Three-step direct smoke | `openai/whisper-tiny`, encoder frozen, genuine HNX_0001 English label | “Well, say, I'm tired of the moment that I can't do anything.” | 4.178 s; RTF 0.714 | Saved/reloaded direct checkpoint emits English; no quality claim |

Runtime artifacts:

- `artifacts/smoke/zero_shot_direct.json`
- `artifacts/smoke/cascade.json`
- `artifacts/smoke/training-20260807T155504Z/summary.json`
- `artifacts/smoke/fine_tuned_direct.json`

These paths are intentionally ignored by Git because they include generated audio/checkpoints. The tracked summary reports only measured values from them.

## Efficiency and estimator accuracy

The genuine-label three-step CPU smoke measured 0.8112 seconds inside Trainer and 0.9483 seconds in the telemetry context, with 3.698 train steps/s, zero GPU-hours, and no VRAM measurement. Its saved full tiny checkpoint measured 155,007,480 bytes (147.827 MiB); checkpoint writing occurred outside the timed training block.

A one-step outer-wall pilot measured 0.8223 seconds. Linear extrapolation predicted 2.4669 seconds for three steps; the measured three-step value was 0.9483 seconds, an absolute percentage error of 160.13%. Fixed startup overhead dominates at this scale, so this estimator result must not be transferred to Whisper-small or GPU training.

Artifact: `artifacts/smoke/estimator_validation.json`.

## Required final command

After direct model selection is frozen, use the guarded command in the README to evaluate zero-shot, direct, and cascade systems against the same 1,500 dev references. That command writes every prediction before any result should be added to this table.
