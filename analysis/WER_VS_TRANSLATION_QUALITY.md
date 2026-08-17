# Per-utterance ASR error vs. translation quality

A closer look at the cascade's "gold Hausa vs. real ASR" gap (Part 3 of the
root `README.md`): instead of just the two aggregate BLEU numbers (~22–25 on
gold text, ~8–10 on real ASR output), this breaks the same 25-example
error-propagation reproduction down **per utterance** — plotting each
individual clip's ASR word error rate against its own translation quality.

![WER vs translation quality](../images/wer_vs_translation_quality.png)

## Result

| | Pearson r vs. WER |
|---|---|
| Sentence BLEU (NLLB-200) | **-0.35** |
| Sentence chrF++ (NLLB-200) | **-0.38** |

Both show a real negative relationship — as ASR word error rate goes up,
translation quality for that same utterance tends to go down — consistent
with the corpus-level finding in the README. The correlation is moderate,
not dramatic: sentence-level BLEU/chrF++ are inherently noisy for short
sentences (a single word match or miss swings the score a lot), and n=25 is
a small sample. This is corroborating evidence for the error-propagation
finding, not a replacement for the corpus-level numbers, which remain the
primary result.

## Reproducing this

```powershell
# From this project's own venv (sacrebleu, matplotlib, numpy already installed):
# 1. Run experiments/evaluate_mt.py --source-field hausa_asr on the
#    karun/error-aware-hausa-st branch to produce _results_asr.csv
# 2. Copy that CSV next to this script, then:
python analysis/build_wer_vs_translation_quality.py
```
