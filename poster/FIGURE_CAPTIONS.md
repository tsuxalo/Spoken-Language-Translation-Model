# Poster Figure Captions

## Figure 1 — BLEU system comparison
**Error-aware MT improves Hausa→English translation.** All five systems receive the same fixed Whisper-generated Hausa transcripts from the 1,500-rendition official NaijaS2ST dev evaluation. Mixed LoRA has the highest BLEU point estimate (15.36), followed closely by Noisy LoRA (15.26).

## Figure 2 — chrF++ system comparison
**Noise-aware adaptation improves character-level translation quality.** Mixed LoRA reaches 39.36 chrF++, compared with 37.51 for base NLLB.

## Figure 3 — SSA-COMET
**Semantic quality also improves with ASR-aware adaptation.** Noisy LoRA has the highest SSA-COMET point estimate (0.466), while Mixed LoRA reaches 0.463. The predeclared analysis did not test Noisy vs. Mixed directly, so do not claim statistical superiority between them.

## Figure 4 — Paired BLEU bootstrap
**Improvements over NLLB persist under paired cluster resampling.** Error bars show 95% cluster-bootstrap intervals over 500 evaluation alignment clusters. Noisy and Mixed LoRA have intervals entirely above zero.

## Figure 5 — Paired chrF++ bootstrap
**The same pattern appears for chrF++.** Noisy and Mixed LoRA show the largest improvements over base NLLB.

## Figures 6–7 — Gold-transcript oracle
**ASR remains the dominant bottleneck.** Replacing fixed-ASR Hausa with gold Hausa roughly doubles BLEU for both NLLB and AfriNLLB, showing that error-aware MT recovers only part of the quality lost upstream.

## Figure 8 — Data split
**Evaluation uses a leakage-controlled split.** Training and internal validation were constructed from NaijaS2ST train with zero alignment and exact bilingual-text overlap; official dev contains 1,500 renditions in 500 alignment clusters from six speakers.
