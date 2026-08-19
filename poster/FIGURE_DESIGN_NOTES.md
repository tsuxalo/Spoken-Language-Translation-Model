# Poster Figure Design Notes

## Design principle

The poster figures are organized around four questions rather than around every available metric:

1. **What is the absolute translation performance?**
   - `01_main_metrics`
   - Compact dot plots for BLEU, chrF++, and SSA-COMET.
   - Dot plots are used instead of bars because the relevant score ranges are narrow; position is the primary visual encoding.

2. **Are the gains over base NLLB supported by the predeclared uncertainty analysis?**
   - `02_bootstrap_improvements`
   - Forest-style plots show the mean paired delta and 95% cluster-bootstrap interval for BLEU, chrF++, and SSA-COMET.
   - This is the primary inferential figure.

3. **How much quality is still lost upstream because of ASR?**
   - `03_asr_oracle_gap`
   - A two-panel slopegraph compares ASR-transcript input against the gold-transcript oracle while holding MT fixed.
   - Delta annotations make the remaining ASR bottleneck visually explicit.

4. **Does the advantage persist as ASR quality gets worse?**
   - `04_wer_degradation`
   - Mean sentence chrF++ is shown across categorical WER bins for NLLB, Noisy LoRA, and Mixed LoRA.
   - Categorical bins are used instead of plotting the mean WER value on a continuous x-axis because WER can exceed 100% when insertions are numerous.
   - Per-bin sample counts are printed under the x-axis categories.

## Literature-informed choices

The structure follows conventions used in closely related speech-translation robustness work:

- **NaijaS2ST** presents aggregate benchmark comparisons primarily in compact tables, while reserving figures for diagnostic behavior and failure-mode analysis.
- **CLAD-ST (EMNLP 2023)** reports aggregate translation results separately and uses a dedicated curve showing translation quality as ASR token errors increase. It also uses paired bootstrap resampling for statistical significance.
- **Robustness of Multi-Source MT to Transcription Errors (Findings ACL 2023)** explicitly frames transcription error propagation as a robustness problem.
- **PiDA (2026)** separates ASR error categorization/impact analysis from the downstream robust-MT performance comparison.

This poster therefore avoids using several redundant bar charts for the same five systems. The goal is one visual claim per figure.

## Interpretation rules

- Do **not** claim Mixed LoRA is statistically better than Noisy LoRA. The predeclared paired analysis compares each system to base NLLB, not Noisy directly against Mixed.
- Do **not** interpret the bootstrap intervals as training-seed uncertainty. The LoRA study used one training seed; the intervals quantify evaluation-cluster sampling uncertainty.
- The gold-Hausa comparison is an **oracle**, not a deployable speech-translation system.
- The 256-example direct S2TT pilot should remain exploratory and outside the main matched quantitative figure set.

## Recommended poster hierarchy

Largest figure:
- `02_bootstrap_improvements.svg`

Secondary figures:
- `03_asr_oracle_gap.svg`
- `04_wer_degradation.svg`

Compact supporting figure or table replacement:
- `01_main_metrics.svg`

Use SVG in PowerPoint whenever possible for lossless scaling. Keep PNG copies for quick preview and sharing.
