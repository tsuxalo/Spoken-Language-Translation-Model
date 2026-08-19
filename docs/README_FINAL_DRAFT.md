# Error-Aware Hausa→English Speech Translation

A controlled study of **ASR error propagation and error-aware machine translation** in a low-resource Hausa→English speech-translation cascade.

## Research question

> **Can adapting Hausa→English MT to the real error distribution of a fixed Hausa ASR system improve downstream speech-translation quality?**

## Why this matters

Cascaded speech translation separates the problem into two reusable pretrained components:

```text
Hausa audio → Hausa ASR → Hausa text → MT → English
```

This modular design is data-efficient, but the MT model is normally trained on relatively clean written text while receiving imperfect machine-generated text at inference time. An earlier 25-example pilot in this project showed a large translation drop when gold Hausa was replaced with Whisper-generated Hausa. We therefore treat **ASR-to-MT distribution mismatch** as the main research problem and test whether the translation stage can be adapted to it.

## Final experiment

We hold the upstream Hausa ASR system fixed and compare five MT systems on the same NaijaS2ST examples:

| System | Adaptation |
|---|---|
| NLLB | none |
| AfriNLLB | Africa-specialized MT |
| Clean LoRA | gold Hausa only |
| Noisy LoRA | ASR-generated Hausa only |
| Mixed LoRA | equal clean/noisy exposure |

The three LoRA conditions use the same NLLB-600M base, seed, training population, training schedule, effective batch size, learning rate, and adapter capacity.

### Leakage-controlled data

Training and internal validation are built from NaijaS2ST `train` using alignment-aware connected-component grouping so duplicated bilingual targets cannot cross the split.

| Split | Rows | Alignment clusters | Speakers |
|---|---:|---:|---:|
| Train | 12,828 | 4,276 | 60 |
| Internal validation | 1,425 | 475 | 57 |
| Official dev | 1,500 | 500 | 6 |

The final internal split has **zero alignment overlap and zero exact bilingual-text overlap**. Official NaijaS2ST `dev` was reserved until final evaluation.

## Main results

All systems below receive the **same fixed Whisper-generated Hausa transcripts** on the 1,500-rendition official dev benchmark.

| System | BLEU ↑ | chrF++ ↑ | SSA-COMET ↑ |
|---|---:|---:|---:|
| NLLB | 13.33 | 37.51 | 0.4446 |
| AfriNLLB | 14.21 | 38.27 | 0.4494 |
| Clean LoRA | 14.21 | 38.31 | 0.4469 |
| Noisy LoRA | 15.26 | 39.12 | **0.4663** |
| Mixed LoRA | **15.36** | **39.36** | 0.4630 |

Using 1,000 paired cluster-bootstrap replicates, both **Noisy** and **Mixed** improve over base NLLB on BLEU, chrF++, and SSA-COMET. The predeclared analysis did not include a direct Noisy-vs-Mixed paired comparison, so their point estimates should not be interpreted as proof that one is statistically superior.

### Error-aware gains vs. NLLB

| System | Δ BLEU | 95% CI | Δ chrF++ | 95% CI |
|---|---:|---:|---:|---:|
| AfriNLLB | +0.88 | [+0.23, +1.51] | +0.77 | [+0.32, +1.25] |
| Clean LoRA | +0.88 | [+0.39, +1.35] | +0.81 | [+0.44, +1.20] |
| Noisy LoRA | +1.93 | [+1.25, +2.58] | +1.63 | [+1.14, +2.09] |
| Mixed LoRA | **+2.04** | **[+1.41, +2.70]** | **+1.87** | **[+1.40, +2.33]** |

## ASR remains the dominant bottleneck

A gold-transcript oracle replaces the fixed-ASR Hausa with the reference Hausa while holding MT constant:

| System | ASR Hausa BLEU | Gold Hausa BLEU | ASR Hausa chrF++ | Gold Hausa chrF++ |
|---|---:|---:|---:|---:|
| NLLB | 13.33 | 28.80 | 37.51 | 52.14 |
| AfriNLLB | 14.21 | 31.53 | 38.27 | 54.34 |

The oracle gap is much larger than the gain from MT adaptation. Error-aware MT helps, but **better Hausa ASR remains the highest-leverage direction**.

## Key findings

- **Realistic ASR-noise adaptation improves downstream translation.**
- **Clean Hausa adaptation alone does not explain the full gain.**
- **Mixed and Noisy LoRA outperform base NLLB on all three predeclared paired metrics.**
- **ASR substitutions are more strongly associated with translation degradation than deletions or insertions in the final diagnostics.**
- **The gold-transcript oracle shows that upstream ASR still dominates the remaining error budget.**

## What the project does not claim

- We do not claim Mixed is statistically superior to Noisy; that direct comparison was not predeclared.
- Bootstrap intervals do not capture training-seed variability because the LoRA study used one training seed.
- Official dev has now been observed and is no longer available as an untouched future tuning set.
- The 256-example direct S2TT pilot is exploratory and not a matched comparison with this final benchmark.

## Earlier project stages

### Hausa ASR
Whisper-small was fine-tuned on FLEURS Hausa. Held-out WER improved from 50.5% after epoch 1 to **44.7% after epoch 3**.

### Direct S2TT pilot
A LoRA-adapted Whisper-small direct Hausa-audio→English pilot was trained on 256 NaijaS2ST examples. It produced **0.24 BLEU / 14.39 chrF++** and demonstrated that extreme low-data direct translation was not yet competitive. Because the pilot used a different development protocol, it is treated as exploratory rather than part of the final matched benchmark.

## Reproducibility

The final experiment documents:
- immutable dataset/model revisions
- leakage-controlled membership hashes
- training and evaluation metadata
- exact LoRA hyperparameters
- 1,000 paired cluster-bootstrap replicates
- preserved empty hypotheses during scoring
- recorded recovery provenance for the completed GPU run

See:
- `docs/GPU_EXPERIMENT_RESULTS.md`
- `docs/FINAL_EXPERIMENTS.md`
- `docs/FINAL_RESEARCH_STORY.md`
- `poster/FIGURE_CAPTIONS.md`

## Generate poster figures

```bash
python poster/generate_figures.py
```

Outputs are written to `poster/figures/` as both PNG and SVG.

## Repository guide

- `train.py`, `data_prep.py`, `inference.py` — original Hausa ASR/cascade pipeline
- `experiments/` — error-aware MT experiments and evaluation
- `analysis/` — statistical/error analyses
- `direct_pilot/` — exploratory direct S2TT pilot
- `docs/` — experiment verification and research documentation
- `poster/` — final poster data, figures, and captions

## Final takeaway

> **Training the MT stage on realistic Hausa ASR noise produces statistically supported improvements, while the much larger gold-transcript oracle gap shows that upstream ASR remains the dominant constraint.**
