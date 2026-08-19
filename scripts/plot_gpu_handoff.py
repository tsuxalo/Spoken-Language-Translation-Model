"""Plot the checked privacy-safe WER-bin aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_ORDER = ("nllb", "afrinllb", "clean", "noisy", "mixed")
SYSTEM_LABELS = {
    "nllb": "NLLB",
    "afrinllb": "AfriNLLB",
    "clean": "Clean LoRA",
    "noisy": "Noisy LoRA",
    "mixed": "Mixed LoRA",
}
COLORS = {
    "nllb": "#6b7280",
    "afrinllb": "#2563eb",
    "clean": "#0d9488",
    "noisy": "#ea580c",
    "mixed": "#7c3aed",
}


def plot(input_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    document = json.loads(input_path.read_text("utf-8"))
    rows = document["bins"]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    figure, axis = plt.subplots(figsize=(8.2, 5.1))
    for system in SYSTEM_ORDER:
        subset = [row for row in rows if row["system"] == system]
        axis.plot(
            [row["mean_wer"] for row in subset],
            [row["mean_sentence_chrf"] for row in subset],
            marker="o",
            linewidth=2,
            markersize=5,
            color=COLORS[system],
            label=SYSTEM_LABELS[system],
        )
    axis.set_title("Translation quality declines as fixed-ASR error increases")
    axis.set_xlabel("Mean Whisper Hausa ASR WER within bin (%)")
    axis.set_ylabel("Mean sentence chrF++")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=180,
        metadata={"Software": "Spoken-Language-Translation-Model"},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/gpu-handoff/wer_bins.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/gpu_wer_vs_chrf_binned.png")
    )
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
