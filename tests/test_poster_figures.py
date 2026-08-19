from __future__ import annotations

from pathlib import Path

import pytest

from poster import generate_figures as figures


def test_poster_tables_are_built_from_public_aggregates() -> None:
    metrics = figures.ordered_metrics_df()
    assert metrics["system"].astype(str).tolist() == figures.SYSTEM_ORDER
    assert metrics.loc[metrics["system"] == "Mixed LoRA", "bleu"].iloc[0] == pytest.approx(
        15.362530633712403
    )

    deltas = figures.ordered_bootstrap_df()
    mixed_bleu = deltas[
        (deltas["system"] == "Mixed LoRA") & (deltas["metric"] == "bleu")
    ].iloc[0]
    assert mixed_bleu["mean_delta"] == pytest.approx(2.043242882296044)
    assert mixed_bleu["ci_low"] == pytest.approx(1.4133748996396147)
    assert mixed_bleu["ci_high"] == pytest.approx(2.698542884872617)

    oracle = figures.oracle_metrics_df()
    assert len(oracle) == 4
    assert set(oracle["input"]) == {"ASR transcript", "Gold transcript"}

    wer = figures.wer_metrics_df()
    assert len(wer) == 25
    assert set(wer["system"]) == {"nllb", "afrinllb", "clean", "noisy", "mixed"}


def test_all_poster_figures_render_to_an_empty_directory(tmp_path: Path) -> None:
    figures.figure_main_metrics(tmp_path)
    figures.figure_bootstrap_improvements(tmp_path)
    figures.figure_oracle_gap(tmp_path)
    figures.figure_wer_degradation(tmp_path)

    expected = {
        f"{stem}.{extension}"
        for stem in (
            "01_main_metrics",
            "02_bootstrap_improvements",
            "03_asr_oracle_gap",
            "04_wer_degradation",
        )
        for extension in ("png", "svg")
    }
    rendered = {path.name for path in tmp_path.iterdir() if path.is_file()}
    assert rendered == expected
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)
