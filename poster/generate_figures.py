"""
Poster-ready figures for the Hausa -> English error-aware MT capstone.

Run from the repository root:

    python poster/generate_figures.py

Inputs are the privacy-safe, versioned JSON aggregates in:
    artifacts/gpu-handoff/evaluation_metrics.json
    artifacts/gpu-handoff/paired_deltas.json
    artifacts/gpu-handoff/wer_bins.json

Outputs:
    poster/figures/01_main_metrics.png/.svg
    poster/figures/02_bootstrap_improvements.png/.svg
    poster/figures/03_asr_oracle_gap.png/.svg
    poster/figures/04_wer_degradation.png/.svg   (when WER-binned CSV exists)

Design goals:
- poster-readable at a distance
- no truncated bar charts
- direct numeric labels
- one claim per figure
- uncertainty shown explicitly
- redundant legends removed where possible
- SVG output for lossless placement in PowerPoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent / "artifacts" / "gpu-handoff"
DEFAULT_OUT = ROOT / "figures"

# Consistent system order across figures.
SYSTEM_ORDER = [
    "NLLB",
    "AfriNLLB",
    "Clean LoRA",
    "Noisy LoRA",
    "Mixed LoRA",
]

SYSTEM_MARKERS = {
    "NLLB": "o",
    "AfriNLLB": "s",
    "Clean LoRA": "^",
    "Noisy LoRA": "D",
    "Mixed LoRA": "*",
}

# Use Matplotlib's active default color cycle, but bind each system to the same
# cycle position across every figure. This keeps the visual identity consistent
# without hard-coding a custom palette into the analysis.
_DEFAULT_CYCLE = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
SYSTEM_COLORS = {
    system: _DEFAULT_CYCLE[i % len(_DEFAULT_CYCLE)]
    for i, system in enumerate(SYSTEM_ORDER)
} if _DEFAULT_CYCLE else {}

# Normalize labels from analysis outputs to poster-facing names.
SYSTEM_ALIASES = {
    "nllb": "NLLB",
    "afrinllb": "AfriNLLB",
    "clean": "Clean LoRA",
    "noisy": "Noisy LoRA",
    "mixed": "Mixed LoRA",
    "NLLB": "NLLB",
    "AfriNLLB": "AfriNLLB",
    "Clean LoRA": "Clean LoRA",
    "Noisy LoRA": "Noisy LoRA",
    "Mixed LoRA": "Mixed LoRA",
}

WER_BIN_ORDER = ["<=20", "20-40", "40-60", "60-80", ">80"]

# Typography is intentionally restrained: the poster section title will already
# provide visual hierarchy, so figures should not fight with the poster.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
    }
)


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run from a complete repository checkout and confirm "
            "artifacts/gpu-handoff exists."
        )
    return path


def load_artifact(name: str) -> dict:
    path = require(ARTIFACTS / name)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def normalize_system(value: str) -> str:
    value = str(value).strip()
    return SYSTEM_ALIASES.get(value, value)


def save(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for path in (png, svg):
        try:
            display_path = path.relative_to(ROOT.parent)
        except ValueError:
            display_path = path
        print(f"saved: {display_path}")


def poster_footer(fig: plt.Figure, text: str, y: float = 0.01) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def padded_limits(values, frac=0.18, min_pad=None):
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = hi - lo
    if span == 0:
        span = abs(hi) if hi != 0 else 1.0
    pad = span * frac
    if min_pad is not None:
        pad = max(pad, min_pad)
    return lo - pad, hi + pad


def ordered_metrics_df() -> pd.DataFrame:
    payload = load_artifact("evaluation_metrics.json")
    systems = payload.get("systems")
    if not isinstance(systems, dict):
        raise TypeError("evaluation_metrics.json missing systems object")
    df = pd.DataFrame(
        {"system": system, **metrics}
        for system, metrics in systems.items()
    )
    df["system"] = df["system"].map(normalize_system)
    missing = set(SYSTEM_ORDER) - set(df["system"])
    if missing:
        raise ValueError(
            f"evaluation_metrics.json missing systems: {sorted(missing)}"
        )
    df["system"] = pd.Categorical(
        df["system"], categories=SYSTEM_ORDER, ordered=True
    )
    return df.sort_values("system").reset_index(drop=True)


def ordered_bootstrap_df() -> pd.DataFrame:
    payload = load_artifact("paired_deltas.json")
    deltas = payload.get("deltas")
    if not isinstance(deltas, list):
        raise TypeError("paired_deltas.json missing deltas array")
    df = pd.DataFrame(deltas).rename(
        columns={
            "mean_bootstrap_delta": "mean_delta",
            "ci_2.5": "ci_low",
            "ci_97.5": "ci_high",
        }
    )
    df["system"] = df["system"].map(normalize_system)
    order = ["AfriNLLB", "Clean LoRA", "Noisy LoRA", "Mixed LoRA"]
    df["system"] = pd.Categorical(df["system"], categories=order, ordered=True)
    return df.sort_values(["metric", "system"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1: absolute metrics
# ---------------------------------------------------------------------------
def figure_main_metrics(output_dir: Path) -> None:
    """
    Three-panel dot plot for absolute official-dev scores.

    Why dots rather than bars:
    these metrics occupy narrow numeric ranges. Position is the visual encoding,
    so a non-zero axis does not exaggerate effect magnitude the way a truncated
    bar chart can.
    """
    df = ordered_metrics_df()

    panels = [
        ("bleu", "BLEU ↑", 2),
        ("chrf_pp", "chrF++ ↑", 2),
        ("ssa_comet", "SSA-COMET ↑", 3),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.7),
        sharey=True,
        gridspec_kw={"wspace": 0.22},
    )

    y_positions = np.arange(len(SYSTEM_ORDER))[::-1]

    for ax, (column, title, decimals) in zip(axes, panels):
        values = df[column].astype(float).to_numpy()

        # Individual scatter calls let the default Matplotlib cycle distinguish
        # systems without hard-coding a palette.
        for idx, (system, value, y) in enumerate(
            zip(df["system"].astype(str), values, y_positions)
        ):
            size = 120 if system == "Mixed LoRA" else 78
            if system == "Noisy LoRA":
                size = 92
            scatter_kwargs = {}
            if system in SYSTEM_COLORS:
                scatter_kwargs["color"] = SYSTEM_COLORS[system]
            ax.scatter(
                value,
                y,
                s=size,
                marker=SYSTEM_MARKERS.get(system, "o"),
                zorder=3,
                **scatter_kwargs,
            )

            fmt = f"{{:.{decimals}f}}"
            xlo, xhi = padded_limits(values, frac=0.28)
            text_offset = (xhi - xlo) * 0.018
            ax.text(
                value + text_offset,
                y,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=9.5,
            )

        xlo, xhi = padded_limits(
            values,
            frac=0.32,
            min_pad=0.006 if column == "ssa_comet" else None,
        )
        ax.set_xlim(xlo, xhi)
        ax.set_title(title, pad=10)
        ax.grid(axis="x")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions, SYSTEM_ORDER)
    for label in axes[0].get_yticklabels():
        if label.get_text() in {"Noisy LoRA", "Mixed LoRA"}:
            label.set_fontweight("semibold")

    # Suppress repeated y labels on the remaining panels.
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.suptitle(
        "Official NaijaS2ST dev: translation quality by MT condition",
        fontsize=16,
        fontweight="semibold",
        y=1.02,
    )
    poster_footer(
        fig,
        "All systems receive the same fixed Whisper Hausa ASR transcripts • "
        "1,500 renditions / 500 alignment clusters • higher is better",
        y=-0.015,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save(fig, "01_main_metrics", output_dir)


# ---------------------------------------------------------------------------
# Figure 2: paired bootstrap improvements
# ---------------------------------------------------------------------------
def figure_bootstrap_improvements(output_dir: Path) -> None:
    """
    Forest plots are the main inferential figure.

    Each point is the mean paired bootstrap difference relative to base NLLB.
    Horizontal intervals show the 95% cluster-bootstrap interval.
    """
    df = ordered_bootstrap_df()

    metric_specs = [
        ("BLEU", "BLEU Δ"),
        ("chrF++", "chrF++ Δ"),
        ("SSA-COMET", "SSA-COMET Δ"),
    ]

    # Support either human-readable or internal metric labels.
    metric_aliases = {
        "BLEU": {"BLEU", "bleu"},
        "chrF++": {"chrF++", "chrf_pp"},
        "SSA-COMET": {"SSA-COMET", "ssa_comet"},
    }

    systems = ["AfriNLLB", "Clean LoRA", "Noisy LoRA", "Mixed LoRA"]
    y_positions = np.arange(len(systems))[::-1]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 4.8),
        sharey=True,
        gridspec_kw={"wspace": 0.30},
    )

    for ax, (metric_name, panel_title) in zip(axes, metric_specs):
        subset = df[df["metric"].astype(str).isin(metric_aliases[metric_name])].copy()
        subset["system"] = pd.Categorical(
            subset["system"], categories=systems, ordered=True
        )
        subset = subset.sort_values("system")

        if len(subset) != len(systems):
            raise ValueError(
                "paired_deltas.json does not contain all four systems for "
                f"{metric_name}"
            )

        means = subset["mean_delta"].astype(float).to_numpy()
        lows = subset["ci_low"].astype(float).to_numpy()
        highs = subset["ci_high"].astype(float).to_numpy()

        # Add visual breathing room while always retaining x=0.
        min_val = min(0.0, float(lows.min()))
        max_val = max(0.0, float(highs.max()))
        span = max_val - min_val
        if span == 0:
            span = 1.0
        ax.set_xlim(min_val - 0.06 * span, max_val + 0.25 * span)

        for system, mean, low, high, y in zip(
            systems, means, lows, highs, y_positions
        ):
            left = mean - low
            right = high - mean
            size = 9.5 if system == "Mixed LoRA" else 7.5
            errorbar_kwargs = {}
            if system in SYSTEM_COLORS:
                errorbar_kwargs["color"] = SYSTEM_COLORS[system]
            ax.errorbar(
                mean,
                y,
                xerr=np.array([[left], [right]]),
                fmt=SYSTEM_MARKERS.get(system, "o"),
                markersize=size,
                capsize=4,
                linewidth=1.8,
                zorder=3,
                **errorbar_kwargs,
            )

            if metric_name == "SSA-COMET":
                label = f"{mean:+.3f}"
            else:
                label = f"{mean:+.2f}"

            ax.text(
                high + 0.025 * span,
                y,
                label,
                va="center",
                ha="left",
                fontsize=9,
            )

        ax.axvline(0, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_title(panel_title, pad=10)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    axes[0].set_yticks(y_positions, systems)
    for label in axes[0].get_yticklabels():
        if label.get_text() in {"Noisy LoRA", "Mixed LoRA"}:
            label.set_fontweight("semibold")

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.suptitle(
        "Paired improvement over base NLLB",
        fontsize=16,
        fontweight="semibold",
        y=1.02,
    )
    poster_footer(
        fig,
        "Mean paired difference with 95% cluster-bootstrap interval • "
        "1,000 replicates over 500 alignment clusters • dashed line = no improvement",
        y=-0.015,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save(fig, "02_bootstrap_improvements", output_dir)


# ---------------------------------------------------------------------------
# Figure 3: gold transcript oracle
# ---------------------------------------------------------------------------
def oracle_metrics_df() -> pd.DataFrame:
    payload = load_artifact("evaluation_metrics.json")
    systems = payload.get("systems")
    oracle = payload.get("gold_hausa_oracle")
    if not isinstance(systems, dict) or not isinstance(oracle, dict):
        raise TypeError(
            "evaluation_metrics.json must contain systems and gold_hausa_oracle"
        )

    rows = []
    for system in ("nllb", "afrinllb"):
        if system not in systems or system not in oracle:
            raise ValueError(
                f"evaluation_metrics.json missing oracle data for {system}"
            )
        rows.append({"system": system, "input": "ASR transcript", **systems[system]})
        rows.append({"system": system, "input": "Gold transcript", **oracle[system]})
    return pd.DataFrame(rows)


def figure_oracle_gap(output_dir: Path) -> None:
    """
    Two-panel slopegraph.

    This deliberately compares ASR transcript vs gold transcript while keeping
    the MT system fixed. The visual slope makes the remaining upstream ASR
    bottleneck immediately interpretable.
    """
    df = oracle_metrics_df()
    df["system"] = df["system"].map(normalize_system)

    systems = ["NLLB", "AfriNLLB"]
    panels = [
        ("bleu", "BLEU ↑", 2),
        ("chrf_pp", "chrF++ ↑", 2),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.9),
        gridspec_kw={"wspace": 0.28},
    )

    x = np.array([0, 1])

    for ax, (metric, title, decimals) in zip(axes, panels):
        all_values = []

        for system in systems:
            sub = df[df["system"] == system]
            if len(sub) != 2:
                raise ValueError(
                    "evaluation_metrics.json should contain exactly ASR and Gold "
                    f"rows for {system}"
                )

            asr_row = sub[sub["input"].astype(str).str.contains("ASR", case=False)]
            gold_row = sub[sub["input"].astype(str).str.contains("Gold", case=False)]
            if asr_row.empty or gold_row.empty:
                raise ValueError(
                    f"Could not identify ASR/Gold oracle rows for {system}"
                )

            asr = float(asr_row[metric].iloc[0])
            gold = float(gold_row[metric].iloc[0])
            delta = gold - asr
            all_values.extend([asr, gold])

            marker = SYSTEM_MARKERS.get(system, "o")
            plot_kwargs = {}
            if system in SYSTEM_COLORS:
                plot_kwargs["color"] = SYSTEM_COLORS[system]
            ax.plot(
                x,
                [asr, gold],
                marker=marker,
                markersize=8,
                linewidth=2.1,
                label=system,
                **plot_kwargs,
            )

            # Separate endpoint labels vertically so the two systems remain
            # readable even when their scores are close.
            endpoint_offset = -9 if system == "NLLB" else 9
            ax.annotate(
                f"{asr:.{decimals}f}",
                (0, asr),
                xytext=(8, endpoint_offset),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9.3,
            )
            ax.annotate(
                f"{gold:.{decimals}f}",
                (1, gold),
                xytext=(-8, endpoint_offset),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=9.3,
            )

            # Offset the delta labels in opposite directions so the two
            # slope annotations never sit on top of each other.
            delta_offset = -14 if system == "NLLB" else 14
            ax.annotate(
                f"{system}: +{delta:.2f}",
                (0.50, (asr + gold) / 2),
                xytext=(0, delta_offset),
                textcoords="offset points",
                va="center",
                ha="center",
                fontsize=9.2,
                fontweight="semibold",
            )

        lo, hi = padded_limits(all_values, frac=0.12)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.08, 1.08)
        ax.set_xticks(
            [0, 1],
            ["ASR transcript", "Gold transcript\n(oracle)"],
        )
        ax.set_title(title, pad=10)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Translation score")
    axes[0].legend(frameon=False, loc="upper left")

    fig.suptitle(
        "Gold transcripts reveal a large remaining ASR bottleneck",
        fontsize=16,
        fontweight="semibold",
        y=1.02,
    )
    poster_footer(
        fig,
        "Oracle experiment: MT system is held fixed; only the Hausa source transcript changes.",
        y=-0.015,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save(fig, "03_asr_oracle_gap", output_dir)


# ---------------------------------------------------------------------------
# Figure 4: robustness vs ASR error
# ---------------------------------------------------------------------------
def wer_metrics_df() -> pd.DataFrame:
    payload = load_artifact("wer_bins.json")
    bins = payload.get("bins")
    if not isinstance(bins, list):
        raise TypeError("wer_bins.json missing bins array")
    return pd.DataFrame(bins)


def figure_wer_degradation(output_dir: Path) -> None:
    """
    Categorical WER-bin robustness plot.

    We intentionally plot bins, rather than the mean WER value as a continuous
    x-axis. This avoids the visually distracting >100% WER means that can occur
    when insertion-heavy hypotheses produce WER above 100%.
    """
    df = wer_metrics_df()

    required = {
        "wer_bin",
        "mean_wer",
        "mean_sentence_chrf",
        "n",
        "system",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"wer_bins.json is missing required fields: {sorted(missing)}"
        )

    df["system"] = df["system"].map(normalize_system)
    df["wer_bin"] = df["wer_bin"].astype(str)

    systems = ["NLLB", "Noisy LoRA", "Mixed LoRA"]
    x = np.arange(len(WER_BIN_ORDER))

    fig, ax = plt.subplots(figsize=(8.5, 5.1))

    for system in systems:
        sub = df[df["system"] == system].copy()
        if sub.empty:
            raise ValueError(
                f"wer_bins.json does not contain required system: {system}"
            )

        sub["wer_bin"] = pd.Categorical(
            sub["wer_bin"],
            categories=WER_BIN_ORDER,
            ordered=True,
        )
        sub = sub.sort_values("wer_bin")

        # Some bins could theoretically be absent. Reindex to make that explicit.
        sub = sub.set_index("wer_bin").reindex(WER_BIN_ORDER)
        y = sub["mean_sentence_chrf"].astype(float).to_numpy()

        plot_kwargs = {}
        if system in SYSTEM_COLORS:
            plot_kwargs["color"] = SYSTEM_COLORS[system]
        ax.plot(
            x,
            y,
            marker=SYSTEM_MARKERS.get(system, "o"),
            markersize=7.5 if system != "Mixed LoRA" else 9,
            linewidth=2.0,
            label=system,
            **plot_kwargs,
        )

    # n is the utterance count within a WER bin and is independent of the MT system.
    baseline = df[df["system"] == "NLLB"].copy()
    baseline["wer_bin"] = pd.Categorical(
        baseline["wer_bin"],
        categories=WER_BIN_ORDER,
        ordered=True,
    )
    baseline = baseline.sort_values("wer_bin").set_index("wer_bin").reindex(WER_BIN_ORDER)

    tick_labels = []
    for bin_label in WER_BIN_ORDER:
        n_value = baseline.loc[bin_label, "n"]
        if pd.isna(n_value):
            tick_labels.append(bin_label)
        else:
            tick_labels.append(f"{bin_label}\n(n={int(n_value)})")

    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("Whisper Hausa ASR WER bin (%)")
    ax.set_ylabel("Mean sentence chrF++ ↑")
    ax.set_title(
        "Translation quality degrades as ASR error increases",
        pad=10,
    )
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper right")

    # Leave a little room at the bottom for two-line categorical tick labels.
    fig.text(
        0.5,
        0.005,
        "WER may exceed 100% because insertion errors contribute to the numerator.",
        ha="center",
        va="bottom",
        fontsize=8.8,
    )

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "04_wer_degradation", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate poster figures from checked public GPU aggregates."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: poster/figures)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    print("Generating poster-ready figures...")
    figure_main_metrics(output_dir)
    figure_bootstrap_improvements(output_dir)
    figure_oracle_gap(output_dir)
    figure_wer_degradation(output_dir)
    print("\nDone.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
