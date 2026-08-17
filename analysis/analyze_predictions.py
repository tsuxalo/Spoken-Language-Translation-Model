"""Final statistical analysis for Hausa speech-translation experiments.

Outputs:
- system_metrics.csv
- bootstrap_confidence_intervals.csv
- paired_bootstrap_differences.csv
- sentence_metrics.csv
- wer_binned_translation.csv
- asr_error_correlations.csv
- qualitative_examples.csv
- wer_vs_chrf.png

Bootstrap resampling is clustered by alignment_id so repeated recordings of the
same sentence are not incorrectly treated as independent samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacrebleu


def corpus_bleu(refs: list[str], preds: list[str]) -> float:
    return sacrebleu.corpus_bleu(preds, [refs]).score


def corpus_chrf(refs: list[str], preds: list[str]) -> float:
    return sacrebleu.corpus_chrf(preds, [refs], word_order=2).score


def sentence_chrf(ref: str, pred: str) -> float:
    return sacrebleu.sentence_chrf(pred, [ref], word_order=2).score


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask].astype(float).to_numpy()
    yv = y[mask].astype(float).to_numpy()
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def add_asr_error_counts(df: pd.DataFrame) -> pd.DataFrame:
    substitutions = []
    deletions = []
    insertions = []
    ref_words = []

    for ref, hyp in zip(df["hausa_gold"].astype(str), df["hausa_asr"].astype(str)):
        result = jiwer.process_words(ref, hyp)
        substitutions.append(result.substitutions)
        deletions.append(result.deletions)
        insertions.append(result.insertions)
        ref_len = result.hits + result.substitutions + result.deletions
        ref_words.append(max(ref_len, 1))

    df = df.copy()
    df["asr_substitutions"] = substitutions
    df["asr_deletions"] = deletions
    df["asr_insertions"] = insertions
    df["asr_ref_words"] = ref_words
    df["substitution_rate"] = 100.0 * df["asr_substitutions"] / df["asr_ref_words"]
    df["deletion_rate"] = 100.0 * df["asr_deletions"] / df["asr_ref_words"]
    df["insertion_rate"] = 100.0 * df["asr_insertions"] / df["asr_ref_words"]
    return df


def score_comet(df, prediction_columns, model_name, batch_size):
    try:
        import torch
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "COMET requested but unbabel-comet is not installed."
        ) from exc

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    gpus = 1 if torch.cuda.is_available() else 0

    system_scores = {}
    sentence_scores = {}

    for column in prediction_columns:
        label = column.removeprefix("prediction_")
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(
                df["hausa_gold"].astype(str),
                df[column].astype(str),
                df["english_ref"].astype(str),
            )
        ]
        output = model.predict(data, batch_size=batch_size, gpus=gpus)

        scores = getattr(output, "scores", None)
        system_score = getattr(output, "system_score", None)

        if scores is None and isinstance(output, (tuple, list)) and len(output) >= 1:
            scores = output[0]
        if scores is None:
            raise RuntimeError("Could not extract sentence scores from COMET output.")

        scores = [float(x) for x in scores]
        if system_score is None:
            system_score = float(np.mean(scores))

        system_scores[label] = float(system_score)
        sentence_scores[label] = scores

    return system_scores, sentence_scores


def build_cluster_indices(df):
    groups = {}
    for alignment_id, group in df.groupby("alignment_id", sort=True):
        groups[str(alignment_id)] = group.index.to_numpy()
    return groups


def bootstrap_samples(
    df,
    prediction_columns,
    n_bootstrap,
    seed,
    comet_sentence_columns=None,
):
    rng = np.random.default_rng(seed)
    clusters = build_cluster_indices(df)
    cluster_ids = np.array(sorted(clusters), dtype=object)

    distributions = {}
    for column in prediction_columns:
        label = column.removeprefix("prediction_")
        distributions[label] = {"bleu": [], "chrf_pp": []}
        if comet_sentence_columns and label in comet_sentence_columns:
            distributions[label]["ssa_comet"] = []

    for i in range(n_bootstrap):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_indices = np.concatenate([clusters[str(cid)] for cid in sampled_ids])
        sample = df.loc[sampled_indices]

        refs = sample["english_ref"].astype(str).tolist()
        for column in prediction_columns:
            label = column.removeprefix("prediction_")
            preds = sample[column].astype(str).tolist()
            distributions[label]["bleu"].append(corpus_bleu(refs, preds))
            distributions[label]["chrf_pp"].append(corpus_chrf(refs, preds))

            if comet_sentence_columns and label in comet_sentence_columns:
                comet_col = comet_sentence_columns[label]
                distributions[label]["ssa_comet"].append(
                    float(sample[comet_col].mean())
                )

        if (i + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"Bootstrap {i + 1}/{n_bootstrap}")

    return distributions


def ci(values):
    arr = np.asarray(values, dtype=float)
    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", default="nllb")
    parser.add_argument("--candidate", default="mixed")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qualitative-n", type=int, default=40)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--comet-model", default="McGill-NLP/ssa-comet-mtl")
    parser.add_argument("--comet-batch-size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if df.empty:
        raise RuntimeError("Prediction CSV is empty.")

    required = {
        "alignment_id",
        "speaker_id",
        "hausa_gold",
        "hausa_asr",
        "english_ref",
        "wer",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    prediction_columns = [
        column for column in df.columns if column.startswith("prediction_")
    ]
    if not prediction_columns:
        raise RuntimeError("No prediction_* columns found.")

    labels = [c.removeprefix("prediction_") for c in prediction_columns]
    if args.baseline not in labels:
        raise KeyError(f"Baseline {args.baseline!r} not found. Available: {labels}")
    if args.candidate not in labels:
        raise KeyError(f"Candidate {args.candidate!r} not found. Available: {labels}")

    df = add_asr_error_counts(df)

    for column in prediction_columns:
        label = column.removeprefix("prediction_")
        df[f"sentence_chrf_{label}"] = [
            sentence_chrf(ref, pred)
            for ref, pred in zip(
                df["english_ref"].astype(str),
                df[column].astype(str),
            )
        ]

    comet_system_scores = {}
    comet_sentence_columns = {}

    if args.comet:
        comet_system_scores, comet_sentence_scores = score_comet(
            df=df,
            prediction_columns=prediction_columns,
            model_name=args.comet_model,
            batch_size=args.comet_batch_size,
        )
        for label, values in comet_sentence_scores.items():
            column = f"sentence_comet_{label}"
            df[column] = values
            comet_sentence_columns[label] = column

    metric_rows = []
    refs = df["english_ref"].astype(str).tolist()

    for column in prediction_columns:
        label = column.removeprefix("prediction_")
        preds = df[column].astype(str).tolist()
        row = {
            "system": label,
            "n_utterances": len(df),
            "n_alignment_ids": df["alignment_id"].nunique(),
            "n_speakers": df["speaker_id"].nunique(),
            "bleu": corpus_bleu(refs, preds),
            "chrf_pp": corpus_chrf(refs, preds),
        }
        if args.comet:
            row["ssa_comet"] = comet_system_scores[label]
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "system_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    distributions = bootstrap_samples(
        df=df,
        prediction_columns=prediction_columns,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        comet_sentence_columns=comet_sentence_columns if args.comet else None,
    )

    ci_rows = []
    for system, metric_dict in distributions.items():
        point_row = metrics_df.loc[metrics_df["system"] == system].iloc[0]
        for metric, values in metric_dict.items():
            low, high = ci(values)
            ci_rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "point_estimate": float(point_row[metric]),
                    "ci_2.5": low,
                    "ci_97.5": high,
                    "bootstrap_replicates": args.bootstrap,
                    "cluster_key": "alignment_id",
                }
            )

    pd.DataFrame(ci_rows).to_csv(
        output_dir / "bootstrap_confidence_intervals.csv",
        index=False,
    )

    diff_rows = []
    baseline_dist = distributions[args.baseline]
    for system, metric_dict in distributions.items():
        if system == args.baseline:
            continue
        for metric, values in metric_dict.items():
            base_values = np.asarray(baseline_dist[metric], dtype=float)
            system_values = np.asarray(values, dtype=float)
            diff = system_values - base_values
            low, high = ci(diff.tolist())
            p_two_sided = min(
                1.0,
                2.0 * min(
                    float(np.mean(diff <= 0.0)),
                    float(np.mean(diff >= 0.0)),
                ),
            )
            diff_rows.append(
                {
                    "baseline": args.baseline,
                    "system": system,
                    "metric": metric,
                    "mean_bootstrap_delta": float(np.mean(diff)),
                    "ci_2.5": low,
                    "ci_97.5": high,
                    "paired_bootstrap_p": p_two_sided,
                }
            )

    pd.DataFrame(diff_rows).to_csv(
        output_dir / "paired_bootstrap_differences.csv",
        index=False,
    )

    error_features = ["wer", "substitution_rate", "deletion_rate", "insertion_rate"]
    corr_rows = []
    for label in labels:
        metric_col = f"sentence_chrf_{label}"
        for feature in error_features:
            corr_rows.append(
                {
                    "system": label,
                    "asr_error_feature": feature,
                    "translation_metric": "sentence_chrf_pp",
                    "pearson_r": safe_corr(df[feature], df[metric_col]),
                }
            )
    pd.DataFrame(corr_rows).to_csv(
        output_dir / "asr_error_correlations.csv",
        index=False,
    )

    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    labels_bins = ["<=20", "20-40", "40-60", "60-80", ">80"]
    df["wer_bin"] = pd.cut(
        df["wer"].astype(float),
        bins=bins,
        labels=labels_bins,
        right=True,
    )

    binned_rows = []
    for label in labels:
        metric_col = f"sentence_chrf_{label}"
        grouped = (
            df.groupby("wer_bin", observed=True)
            .agg(
                mean_wer=("wer", "mean"),
                mean_sentence_chrf=(metric_col, "mean"),
                n=("alignment_id", "size"),
            )
            .reset_index()
        )
        grouped["system"] = label
        binned_rows.append(grouped)

    wer_binned = pd.concat(binned_rows, ignore_index=True)
    wer_binned.to_csv(output_dir / "wer_binned_translation.csv", index=False)

    plt.figure(figsize=(8, 5))
    for label in labels:
        subset = wer_binned[wer_binned["system"] == label]
        plt.plot(
            subset["mean_wer"],
            subset["mean_sentence_chrf"],
            marker="o",
            label=label,
        )
    plt.xlabel("Mean ASR WER in bin (%)")
    plt.ylabel("Mean sentence chrF++")
    plt.title("Translation quality as ASR error increases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "wer_vs_chrf.png", dpi=200)
    plt.close()

    baseline_col = f"sentence_chrf_{args.baseline}"
    candidate_col = f"sentence_chrf_{args.candidate}"
    df["candidate_minus_baseline_chrf"] = df[candidate_col] - df[baseline_col]

    half = max(1, args.qualitative_n // 2)
    gains = (
        df.sort_values("candidate_minus_baseline_chrf", ascending=False)
        .drop_duplicates("alignment_id")
        .head(half)
        .assign(example_type="largest_gain")
    )
    regressions = (
        df.sort_values("candidate_minus_baseline_chrf", ascending=True)
        .drop_duplicates("alignment_id")
        .head(half)
        .assign(example_type="largest_regression")
    )
    qualitative = pd.concat([gains, regressions], ignore_index=True)

    keep_columns = [
        "example_type",
        "alignment_id",
        "utterance_id" if "utterance_id" in df.columns else "speaker_id",
        "speaker_id",
        "wer",
        "substitution_rate",
        "deletion_rate",
        "insertion_rate",
        "hausa_gold",
        "hausa_asr",
        "english_ref",
        f"prediction_{args.baseline}",
        f"prediction_{args.candidate}",
        baseline_col,
        candidate_col,
        "candidate_minus_baseline_chrf",
    ]
    keep_columns = list(dict.fromkeys(keep_columns))
    qualitative[keep_columns].to_csv(
        output_dir / "qualitative_examples.csv",
        index=False,
    )

    df.to_csv(output_dir / "sentence_metrics.csv", index=False)

    run_metadata = {
        "input": args.input,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_cluster_key": "alignment_id",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "comet_enabled": args.comet,
        "comet_model": args.comet_model if args.comet else None,
        "prediction_systems": labels,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
