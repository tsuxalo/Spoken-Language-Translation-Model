# Final Research Story

## Working title
**Error-Aware Machine Translation for Low-Resource Hausa→English Speech Translation**

### Subtitle
A controlled evaluation of ASR-to-MT error propagation on NaijaS2ST

## Primary research question
> **Can adapting Hausa→English MT to the real error distribution of a fixed Hausa ASR system improve downstream speech-translation quality?**

## Motivation
Cascaded speech translation reuses strong pretrained ASR and MT systems, but it introduces an intermediate text bottleneck. A translation model trained primarily on clean written Hausa receives machine-generated Hausa at inference time. Our initial 25-example pilot showed a large clean-vs-ASR translation gap, motivating a larger controlled study of this distribution mismatch.

## Contribution
We conduct a controlled error-aware Hausa→English MT benchmark in which five translation systems receive the same fixed Whisper-generated Hausa transcripts. Using leakage-resistant NaijaS2ST splits and paired cluster-bootstrap analysis, we show that ASR-noise-aware adaptation improves translation over base NLLB, while a gold-transcript oracle demonstrates that upstream ASR remains the dominant bottleneck.

## Experimental design

### Fixed upstream ASR
Hausa audio → `nahomazmach/whisper-small-ha` → Hausa ASR transcript

### Translation systems
1. NLLB-200 distilled 600M
2. AfriNLLB
3. Clean LoRA — trained on gold Hausa
4. Noisy LoRA — trained on ASR Hausa
5. Mixed LoRA — equal clean/noisy exposure

### Controlled LoRA conditions
- same NLLB-600M base
- seed 42
- 12,828 training examples
- 1,425 noisy internal-validation examples
- three epochs
- effective batch size 8
- learning rate 2e-4
- rank-16 LoRA

### Leakage control
The split groups by bilingual alignment and connected duplicate bilingual pairs. Final train/validation membership has:
- zero alignment overlap
- zero exact bilingual-text overlap
- zero alignment overlap with official dev

### Final evaluation
Official NaijaS2ST dev:
- 1,500 speech renditions
- 500 alignment clusters
- 6 speakers
- BLEU, chrF++, SSA-COMET
- 1,000 paired cluster-bootstrap replicates

## Main results

| System | BLEU | chrF++ | SSA-COMET |
|---|---:|---:|---:|
| NLLB | 13.33 | 37.51 | 0.4446 |
| AfriNLLB | 14.21 | 38.27 | 0.4494 |
| Clean LoRA | 14.21 | 38.31 | 0.4469 |
| Noisy LoRA | 15.26 | 39.12 | **0.4663** |
| Mixed LoRA | **15.36** | **39.36** | 0.4630 |

### Predeclared paired improvements vs. NLLB
- Noisy LoRA: +1.93 BLEU, +1.63 chrF++, +0.0217 SSA-COMET
- Mixed LoRA: +2.04 BLEU, +1.87 chrF++, +0.0184 SSA-COMET
- all Noisy and Mixed 95% cluster-bootstrap intervals are above zero

### Gold-Hausa oracle
| System | ASR Hausa BLEU | Gold Hausa BLEU | ASR Hausa chrF++ | Gold Hausa chrF++ |
|---|---:|---:|---:|---:|
| NLLB | 13.33 | 28.80 | 37.51 | 52.14 |
| AfriNLLB | 14.21 | 31.53 | 38.27 | 54.34 |

## Interpretation
1. **ASR-aware MT adaptation works.** Noisy and Mixed LoRA improve over base NLLB across all three predeclared paired measures.
2. **Domain adaptation alone is not the full explanation.** Clean LoRA improves BLEU/chrF++, but the noise-aware conditions improve more.
3. **ASR remains the major bottleneck.** Gold Hausa produces far larger gains than any MT adaptation.
4. **Do not rank Noisy vs. Mixed statistically.** No direct paired Noisy-vs-Mixed comparison was predeclared; the point-estimate ordering is descriptive only.
5. **Substitutions appear especially damaging.** In final diagnostics, substitution rate has a more negative relationship with sentence chrF++ than deletion or insertion rate.

## Role of the earlier 25-example pilot
The 25-example study was exploratory. It established that replacing gold Hausa with Whisper-generated Hausa sharply degraded MT and produced a preliminary negative WER/translation-quality relationship. It motivated the final experiment but should not be presented as the main evidence now that the 1,500-rendition official-dev benchmark is complete.

## Role of the direct S2TT pilot
The 256-example direct speech-to-text translation experiment is a secondary exploratory comparison. It showed that an extremely low-data direct model was not yet competitive, but its evaluation was not matched to the completed error-aware benchmark. It should not appear in the main quantitative comparison table.

## Limitations
- one training seed; bootstrap intervals quantify evaluation-cluster uncertainty, not seed variability
- official dev has now been observed and cannot be reused as an untouched tuning set
- Noisy vs. Mixed was not a predeclared paired comparison
- the fixed Whisper ASR still has substantial error; stronger ASR could change downstream rankings
- direct S2TT was not evaluated in the same matched final protocol
- one malformed-ASR dev case produced empty outputs for several systems and is retained in scoring

## Poster one-sentence takeaway
> **Training the MT stage on realistic Hausa ASR noise produces small but statistically supported gains, while the much larger gold-transcript oracle gap shows that upstream ASR remains the dominant constraint.**
