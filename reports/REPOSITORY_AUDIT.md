# Original repository audit

Audit date: 2026-08-07. Baseline commit: `329254875059600f1b611aabe6ebb7c3e0e1d124` (`origin/main`).

## Critical findings

- The repository title said translation, but the implemented model was Hausa ASR: audio → Hausa transcription.
- Training evaluated FLEURS test after every epoch and selected/communicated results from it. That test set is therefore not an untouched final partition for the historical checkpoint.
- The README interpreted 44.7% WER as “a bit under half the words right.” WER is edit errors divided by reference words and can exceed 100%; it is not word accuracy.
- The historical README/notebook embedded training time, loss, WER, and qualitative examples without local prediction or run-manifest artifacts that make those values independently traceable.
- The old data path precomputed Whisper features for entire splits, increasing storage and reducing configuration flexibility.
- Root scripts assumed `hausa_s2tt` was importable without installing the package.
- Dependencies included a machine-specific PyTorch/CUDA build and did not describe portable installation.
- The notebook repeatedly inspected test examples and covered ASR only.

## Important findings

- Only train and test were used; FLEURS validation was ignored.
- Metrics were raw WER only; normalization, CER, metric signatures, library versions, and per-example outputs were absent.
- Seeds, dataset/model revisions, hardware manifests, throughput, RTF, peak VRAM, parameter counts, and checkpoint size were not captured.
- Audio handling did not establish a general policy for stereo, resampling, invalid samples, or clips beyond Whisper's 30-second window.
- No genuine Hausa-audio/English-text pairing, direct S2TT trainer, NLLB cascade, common English evaluation, final-test guard, or model-card drafts existed.
- `.gitignore` did not comprehensively protect generated checkpoints, caches, and heavyweight model formats.

## Historical checkpoint classification

`nahomazmach/whisper-small-ha` is a Hausa ASR checkpoint initialized from `openai/whisper-small`. Its current model card reports WER 44.68 and Apache-2.0, but the original repository evaluated the FLEURS test partition during training. I therefore treat 44.68 as a historical, externally reported figure—not a new measurement and not a clean final-test result.

No history was rewritten and no original changes were discarded. Root scripts remain as compatibility wrappers around the installable package.
