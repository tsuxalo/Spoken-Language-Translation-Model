"""Generate publication-style poster figures from verified capstone results.

Run from the repository root:
    python poster/generate_figures.py

Outputs PNG and SVG versions to poster/figures/.
The data files in poster/data/ are intentionally small, human-readable copies of
the canonical values reported in docs/GPU_EXPERIMENT_RESULTS.md.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: poster/figures/{stem}.png")
    print(f"saved: poster/figures/{stem}.svg")

def label_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width()/2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

def system_metric(metric, ylabel, title, stem, ylim=None):
    df = pd.read_csv(DATA / "final_metrics.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(df["system"], df[metric])
    label_bars(ax, bars, "{:.2f}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, stem)

def comet_metric():
    df = pd.read_csv(DATA / "final_metrics.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(df["system"], df["ssa_comet"])
    label_bars(ax, bars, "{:.3f}")
    ax.set_ylabel("SSA-COMET")
    ax.set_title("Semantic translation quality on official NaijaS2ST dev")
    ax.set_ylim(0.42, 0.48)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "03_ssa_comet_systems")

def bootstrap_forest(metric, xlabel, title, stem):
    df = pd.read_csv(DATA / "bootstrap_deltas.csv")
    df = df[df["metric"] == metric].copy()
    order = ["AfriNLLB", "Clean LoRA", "Noisy LoRA", "Mixed LoRA"]
    df["system"] = pd.Categorical(df["system"], categories=order, ordered=True)
    df = df.sort_values("system")
    y = np.arange(len(df))
    x = df["mean_delta"].to_numpy()
    low = df["ci_low"].to_numpy()
    high = df["ci_high"].to_numpy()
    xerr = np.vstack([x-low, high-x])

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=5)
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_yticks(y, df["system"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

    for xi, yi, lo, hi in zip(x, y, low, high):
        ax.text(
            hi + max(0.02, 0.02 * max(abs(high))),
            yi,
            f"{xi:+.2f} [{lo:+.2f}, {hi:+.2f}]",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    save(fig, stem)

def oracle_gap(metric, ylabel, title, stem):
    df = pd.read_csv(DATA / "oracle_metrics.csv")
    systems = ["NLLB", "AfriNLLB"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for system in systems:
        sub = df[df["system"] == system]
        x = [0, 1]
        y = [
            float(sub[sub["input"] == "ASR Hausa"][metric].iloc[0]),
            float(sub[sub["input"] == "Gold Hausa"][metric].iloc[0]),
        ]
        ax.plot(x, y, marker="o", linewidth=2, label=system)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks([0, 1], ["ASR transcript", "Gold transcript"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, stem)

def data_split():
    df = pd.read_csv(DATA / "split_summary.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bars = ax.bar(df["split"], df["rows"])
    label_bars(ax, bars, "{:.0f}")
    ax.set_ylabel("Utterance renditions")
    ax.set_title("Leakage-controlled experimental data")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "08_data_split")

def main():
    system_metric(
        "bleu",
        "BLEU ↑",
        "Error-aware MT improves Hausa→English translation",
        "01_bleu_systems",
        (12, 16),
    )
    system_metric(
        "chrf_pp",
        "chrF++ ↑",
        "Error-aware MT improves Hausa→English translation",
        "02_chrf_systems",
        (36, 40.5),
    )
    comet_metric()
    bootstrap_forest(
        "BLEU",
        "BLEU difference vs. base NLLB",
        "Paired cluster-bootstrap improvement over NLLB",
        "04_bootstrap_bleu",
    )
    bootstrap_forest(
        "chrF++",
        "chrF++ difference vs. base NLLB",
        "Paired cluster-bootstrap improvement over NLLB",
        "05_bootstrap_chrf",
    )
    oracle_gap(
        "bleu",
        "BLEU ↑",
        "ASR errors leave a large translation-quality gap",
        "06_oracle_bleu_gap",
    )
    oracle_gap(
        "chrf_pp",
        "chrF++ ↑",
        "ASR errors leave a large translation-quality gap",
        "07_oracle_chrf_gap",
    )
    data_split()

if __name__ == "__main__":
    main()
