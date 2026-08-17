"""Builds analysis/wer_vs_translation_quality.png: a per-utterance scatter of
ASR word error rate against translation quality (sentence BLEU/chrF++),
computed from the 25-example error-propagation reproduction's predictions
(same source data behind the cascade's Part 3 "gold vs. real ASR" numbers
in the README, just analyzed per-utterance instead of aggregated).

Reads the predictions CSV produced by evaluate_mt.py (--source-field
hausa_asr) on the karun/error-aware-hausa-st branch's pipeline; that CSV
already contains per-utterance WER, both models' predictions, and the
English reference in one place.
"""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import sacrebleu

CSV_PATH = "_results_asr.csv"  # produced by experiments/evaluate_mt.py --source-field hausa_asr


def main():
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row["english_ref"]
            nllb_pred = row["prediction_nllb"]
            rows.append({
                "wer": float(row["wer"]),
                "bleu": sacrebleu.sentence_bleu(nllb_pred, [ref]).score,
                "chrf": sacrebleu.sentence_chrf(nllb_pred, [ref], word_order=2).score,
            })

    with open("wer_bleu_data.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    wer = np.array([r["wer"] for r in rows])
    bleu = np.array([r["bleu"] for r in rows])
    chrf = np.array([r["chrf"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, y, ylabel in [(axes[0], bleu, "Sentence BLEU"), (axes[1], chrf, "Sentence chrF++")]:
        r = np.corrcoef(wer, y)[0, 1]
        ax.scatter(wer, y, alpha=0.75, edgecolor="k", linewidth=0.5)
        m, b = np.polyfit(wer, y, 1)
        xs = np.linspace(wer.min(), wer.max(), 50)
        ax.plot(xs, m * xs + b, color="darkorange", linewidth=2, label=f"trend (r={r:.2f})")
        ax.set_xlabel("ASR Word Error Rate (%)")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.suptitle("Per-Utterance ASR Error vs. Translation Quality (Cascade, NLLB-200, n=25)")
    plt.tight_layout()
    plt.savefig("../images/wer_vs_translation_quality.png", dpi=150)
    print("Saved ../images/wer_vs_translation_quality.png")


if __name__ == "__main__":
    main()
