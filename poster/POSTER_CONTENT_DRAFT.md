# Poster Content Draft

## Title
**Error-Aware Machine Translation for Low-Resource Hausa→English Speech Translation**

### Subtitle
A controlled evaluation of ASR error propagation on NaijaS2ST

## Research question
> **Can adapting Hausa→English MT to the real error distribution of a fixed Hausa ASR system improve downstream speech-translation quality?**

## Motivation
Cascaded speech translation can reuse strong pretrained ASR and MT models, an important advantage in low-resource settings. However, the MT stage is typically trained on clean written text while receiving noisy machine-generated transcripts at inference time. A 25-example pilot in this project showed a large translation-quality drop when gold Hausa was replaced with Whisper ASR output, motivating a larger controlled test of this mismatch.

## Methods
**Fixed speech front end:** Hausa audio → fine-tuned Whisper-small → Hausa ASR transcript.

**MT conditions:** NLLB, AfriNLLB, Clean LoRA, Noisy LoRA, Mixed LoRA.

**Controlled adaptation:** 12,828 training examples; seed 42; 3 epochs; effective batch size 8; learning rate 2e-4; rank-16 LoRA. Internal validation uses 1,425 noisy examples.

**Final evaluation:** 1,500 official-dev renditions in 500 alignment clusters from six speakers. Metrics: BLEU, chrF++, SSA-COMET. Uncertainty: 1,000 paired cluster-bootstrap replicates.

## Key result
Mixed LoRA reaches **15.36 BLEU / 39.36 chrF++**, while Noisy LoRA reaches the highest **SSA-COMET (0.466)**. Both Noisy and Mixed improve over base NLLB on all three predeclared paired measures.

## Oracle result
With gold Hausa instead of ASR Hausa, NLLB rises from **13.33 → 28.80 BLEU** and **37.51 → 52.14 chrF++**. ASR remains the dominant bottleneck even after MT adaptation.

## Interpretation
Error-aware adaptation partially mitigates ASR-to-MT distribution mismatch. The gains are statistically supported against base NLLB, but they are much smaller than the gold-transcript oracle gap. The best next step is therefore to combine robust MT with stronger Hausa ASR rather than treating MT adaptation as a complete solution.

## Limitations
- one training seed
- official dev now observed
- no predeclared Noisy-vs-Mixed significance test
- fixed Whisper ASR remains high-error
- direct S2TT pilot is not a matched final comparison

## Exploratory direct S2TT pilot
A 256-example LoRA pilot produced 0.24 BLEU / 14.39 chrF++. This result is useful as exploratory evidence about extreme low-data direct translation but should not be shown in the main matched result table.

## Takeaway
> **Training NLLB on realistic Hausa ASR errors improves translation robustness, but upstream speech recognition still accounts for the largest remaining quality loss.**
