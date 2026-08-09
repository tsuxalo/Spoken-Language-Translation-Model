# Direct Hausa-to-English S2TT architecture decision

Status: implementation decision, 2026-08-09. No matched pilot or full training result is claimed here.

## Decision

Implement Whisper-small with genuine Hausa-audio/English-`target_text` supervision, rank-16 LoRA on attention query/value projections, gradient checkpointing, and hardware-selected BF16/FP16/FP32. Compare two pinned initializations in a matched validation-only pilot:

1. multilingual `openai/whisper-small`;
2. Hausa-ASR-adapted `nahomazmach/whisper-small-ha`.

The selected initialization is intentionally unresolved until the matched pilot. Speaker-disjoint project-validation chrF++ is primary; SacreBLEU and validation loss are complementary. NaijaS2ST official dev is not part of this decision.

This is a direct model at inference: waveform → Whisper encoder → decoder cross-attention → English tokens. It neither produces nor consumes an intermediate Hausa transcript.

## Evidence reviewed

- OpenAI's [Whisper paper](https://arxiv.org/abs/2212.04356), [official model card](https://github.com/openai/whisper/blob/main/model-card.md), and pinned [Whisper-small model card](https://huggingface.co/openai/whisper-small) establish the 244M multilingual encoder-decoder, any-to-English translation pretraining, 30-second input design, and Apache-2.0 weights.
- The [LoRA paper](https://arxiv.org/abs/2106.09685) establishes frozen pretrained weights plus trainable low-rank updates. Current [PEFT checkpoint documentation](https://huggingface.co/docs/peft/developer_guides/checkpoint) establishes the adapter/config persistence contract; project-specific trainable counts remain measured at runtime.
- The [SeamlessM4T paper](https://arxiv.org/abs/2308.11596), [Seamless paper](https://arxiv.org/abs/2312.05187), and [v2 model card](https://huggingface.co/facebook/seamless-m4t-v2-large) establish direct S2TT, Hausa (`ha`) coverage, 2.3B parameters, and CC-BY-NC-4.0 licensing.
- The [OWSM v3.1 paper](https://arxiv.org/abs/2401.16658), [OWSM model card](https://huggingface.co/espnet/owsm_v3.1_ebf), and [OWSM-CTC ACL paper](https://aclanthology.org/2024.acl-long.549/) establish open ESPnet speech-recognition/translation models of 101M–1.02B parameters and CC-BY-4.0 licensing. The reviewed cards did not explicitly verify Hausa support.
- The [AfriHuBERT paper](https://arxiv.org/abs/2409.20201) and [model card](https://huggingface.co/ajesujoba/AfriHuBERT) establish a roughly 95M African-language HuBERT speech encoder. It has no translation decoder; the Hub license label is only `cc`, which is insufficiently specific for publication decisions.
- The [MMS paper](https://arxiv.org/abs/2305.13516) and [300M model card](https://huggingface.co/facebook/mms-300m) establish a massively multilingual speech encoder/ASR family, not a direct translation encoder-decoder; the 300M checkpoint is CC-BY-NC-4.0.
- The [W2v-BERT 2.0 model card](https://huggingface.co/facebook/w2v-bert-2.0) establishes a 600M MIT-licensed bare speech encoder trained across 143+ languages. A translation head/decoder must be designed and trained.
- The recent [MCAT paper](https://arxiv.org/abs/2512.01512) and [released repository](https://github.com/yxduir/m2m-70) describe 9B/27B speech-encoder/LLM systems. Its published 70-language list omits Hausa; the repository reports 24 GB BF16 VRAM for the 9B variant and 80 GB for the 27B variant.
- The 2026 [MSRT preprint](https://arxiv.org/abs/2608.04586) describes a 4B mixture-of-speech-encoders system over 45 languages. It is too new, much larger, and not demonstrated here for Hausa.

Published/model-card claims above are not project measurements. VRAM and Colab assessments below are project engineering inferences unless explicitly attributed.

## Candidate matrix

| Candidate | Direct at inference | Total parameters | Trainable parameters | Hausa evidence | Paired-data suitability | VRAM / Colab feasibility | Complexity (train / infer) | Open weights / license | Decision |
|---|---:|---:|---:|---|---|---|---|---|---|
| Whisper-small, full fine-tune | yes | 244M verified | 244M | Hausa tokenizer/language support; translation quality here unmeasured | excellent: native audio→English seq2seq objective | high training memory; uncertain on free Colab | medium / low | yes / Apache-2.0 | defer unless LoRA underfits |
| Whisper-small, LoRA q/v | yes | 244M verified | ≈1.77M estimated for rank 16; manifest measures exact | same | excellent | plausible on a suitable Colab GPU; pilot must measure | low / low | yes / Apache-2.0 | **implement** |
| Whisper-small, frozen encoder | yes | 244M verified | decoder-side parameters only; exact count runtime-measured if run | same | good, but acoustic adaptation is blocked | easier than full; heavier than LoRA optimizer state in decoder | low / low | yes / Apache-2.0 | ablation/fallback |
| Whisper-small, partial freezing | yes | 244M verified | layer choice dependent; must be runtime-measured | same | good but adds a tuning dimension | intermediate, not yet measured | medium / low | yes / Apache-2.0 | future ablation |
| Hausa-ASR Whisper-small + LoRA | yes after English-target fine-tuning | 244M verified | ≈1.77M estimated; exact count in manifest | checkpoint is explicitly Hausa ASR | excellent labels; initialization effect is ambiguous | same as base LoRA | low / low | yes; card reports Apache-2.0 | **matched pilot** |
| Whisper encoder + projection + NLLB-style decoder | yes | not a released fixed architecture; 244M and 600M constituent checkpoints are verified, composed total depends on discarded/shared modules | adapter plus selected decoder layers, design dependent | Whisper/NLLB have Hausa components, but no pinned coupled checkpoint was found | potentially strong, but 48.4 h is small for learning a new modality bridge | materially heavier and riskier than selected path | high / medium | components open; NLLB is CC-BY-NC-4.0 | future work |
| SeamlessM4T-v2 large | yes | 2.3B verified | full 2.3B or adapter-dependent | model card lists `ha` speech input/text languages | strong pretrained task match | likely beyond ordinary Colab fine-tuning; not measured | medium / medium | yes / CC-BY-NC-4.0 | serious future comparator |
| OWSM v3.1 encoder-decoder | yes | 101M, 367M, 1.02B variants verified | recipe dependent | explicit Hausa evidence not found in reviewed primary material | task match is good if Hausa coverage is established | smaller variants plausible, but requires a second ESPnet stack | high / medium | yes / CC-BY-4.0 | pilot only after Hausa verification |
| OWSM-CTC v3.1 | yes, encoder-only CTC translation | 1.01B verified | recipe dependent | explicit Hausa evidence not found | direct translation exists, but CTC fine-tuning path differs substantially | 8.05 GB Hub repository; training not measured | high / low | yes / CC-BY-4.0 | future comparator |
| AfriHuBERT + multilingual decoder | yes only after custom coupling | ≈95M encoder verified; decoder makes total design dependent | projection/decoder dependent | Africa-focused encoder; exact Hausa contribution not quantified in card | needs a new decoder bridge learned from limited paired data | plausible encoder, high integration risk | high / medium | weights yes / license label insufficiently specific | reject for primary path |
| MMS-300M + text decoder | yes only after custom coupling | 300M encoder verified; decoder additional | design dependent | massively multilingual speech family; no direct Hausa→English decoder | same bridge problem | likely feasible only with careful freezing; unmeasured | high / medium | yes / CC-BY-NC-4.0 | future work |
| W2v-BERT 2.0 + text decoder | yes only after custom coupling | 600M encoder verified; decoder additional | design dependent | 143-language pretraining; card does not provide task-specific Hausa S2TT evidence | bare encoder needs a complete translation head | poor for bounded Colab work | high / medium | yes / MIT | future work |
| MCAT | yes | 9B or 27B verified | paper reports ≈100M trainable | released language list omits Hausa | paper uses 10 h/language, but no Hausa route exists | repository reports 24/80 GB BF16 VRAM | high / high | released components; licenses must be resolved jointly | reject for this project |
| MSRT | yes | 4B verified in preprint | not independently measured | 45-language claim; Hausa not verified here | designed for low-resource S2TT | not ordinary Colab scale | very high / high | release claimed; exact checkpoint/license needs verification | future research only |

“Design dependent” is intentional: inventing a parameter count for an unspecified encoder-decoder composition would not be verification.

## Why the selected design fits this experiment

1. Whisper already has the exact causal interface—speech encoder plus English translation decoder—so all 48.391 accepted hours can supervise the desired task without learning a new cross-family bridge.
2. At 244M parameters, it is far smaller than SeamlessM4T-v2, MCAT, MSRT, and the 1B OWSM variants. LoRA further limits optimizer/gradient state while preserving the pretrained base.
3. The existing package, inference runtime, pinned revisions, and validation metrics already use Transformers/PEFT. Replacing them with ESPnet or a custom decoder would expand software and experimental risk before the primary hypothesis is tested.
4. Base versus Hausa-ASR initialization is scientifically meaningful and cheap to isolate. The Hausa-ASR checkpoint may offer better Hausa-adapted speech representations, but its prior objective adapted the system toward Hausa transcription. Only the matched translation pilot can determine whether that helps or hurts Hausa→English translation.

Newer or larger is not synonymous with more appropriate. The primary bottleneck is a defensible low-resource comparison on one protected split, not maximum parameter count.

## Locked pilot rule

`configs/direct_s2tt_pilot_base.yaml` and `configs/direct_s2tt_pilot_from_asr.yaml` must differ only in model ID, model revision, output directory, and run name. `assert_matched_direct_configs` rejects other drift. Both use the exact Notebook 00 train/validation assignment when present; otherwise the seed-42 reconstruction must match the tracked audit. Architecture selection writes `official_dev_evaluated: false`.

## Revisit conditions

Reconsider the architecture only if the genuine structural smoke fails for a non-repairable model-interface reason, both matched pilots show unusable validation behavior, or measured memory prevents the LoRA pilot on the authorized hardware. A next-tier experiment should first verify Hausa coverage and run a bounded OWSM 101M/367M or Seamless adapter pilot—not start an unbounded custom bridge.
