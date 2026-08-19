"""Evaluate NLLB/AfriNLLB and multiple LoRA adapters on one manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .evaluate_mt import (
        BASELINE_REVISIONS,
        BASELINES,
        TOKENIZER_ID,
        load_jsonl,
        score_predictions,
        translate,
    )
    from .revisions import NLLB_600M_REVISION, revision_for
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_mt import (
        BASELINE_REVISIONS,
        BASELINES,
        TOKENIZER_ID,
        load_jsonl,
        score_predictions,
        translate,
    )
    from revisions import NLLB_600M_REVISION, revision_for


def parse_key_value(spec: str, noun: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"{noun} must be written as LABEL=VALUE."
        )
    label, value = spec.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise argparse.ArgumentTypeError(f"{noun} label/value cannot be empty.")
    return label, value


def parse_adapter(spec: str) -> tuple[str, str]:
    return parse_key_value(spec, "Adapter")


def parse_model(spec: str) -> tuple[str, str]:
    return parse_key_value(spec, "Model")


def parse_revision(spec: str) -> tuple[str, str]:
    return parse_key_value(spec, "Revision")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--source-field",
        choices=["hausa_gold", "hausa_asr"],
        default="hausa_asr",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        choices=sorted(BASELINES),
        help=(
            "Baseline to run. Repeat for multiple baselines. "
            "If omitted, nllb and afrinllb are both run."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        default=[],
        metavar="LABEL=MODEL_ID",
        help=(
            "Evaluate an additional Hugging Face seq2seq model. Repeat as needed. "
            "Example: nllb3b=facebook/nllb-200-3.3B"
        ),
    )
    parser.add_argument(
        "--model-revision",
        action="append",
        type=parse_revision,
        default=[],
        metavar="LABEL=REVISION",
    )
    parser.add_argument(
        "--baseline-revision",
        action="append",
        type=parse_revision,
        default=[],
        metavar="LABEL=REVISION",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_adapter,
        default=[],
        metavar="LABEL=PATH",
        help="Repeat to evaluate multiple PEFT adapters.",
    )
    parser.add_argument("--adapter-base-model", default=TOKENIZER_ID)
    parser.add_argument("--adapter-base-revision", default=NLLB_600M_REVISION)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise RuntimeError("Input manifest contains no rows.")

    sources = [str(row[args.source_field]) for row in rows]
    references = [str(row["english_ref"]) for row in rows]
    predictions_by_model: dict[str, list[str]] = {}

    baseline_labels = args.baseline or ["nllb", "afrinllb"]
    baseline_revisions = dict(args.baseline_revision)
    model_revisions = dict(args.model_revision)

    for label in baseline_labels:
        predictions = translate(
            model_id=BASELINES[label],
            source_texts=sources,
            batch_size=args.batch_size,
            model_revision=baseline_revisions.get(
                label,
                BASELINE_REVISIONS[label],
            ),
        )
        predictions_by_model[label] = predictions
        scores = score_predictions(predictions, references)
        print(
            f"{label}: BLEU={scores['bleu']:.2f}, "
            f"chrF++={scores['chrf++']:.2f}"
        )

    for label, model_id in args.model:
        model_revision = model_revisions.get(label)
        if model_revision is None:
            model_revision = revision_for(model_id)
        predictions = translate(
            model_id=model_id,
            source_texts=sources,
            batch_size=args.batch_size,
            model_revision=model_revision,
        )
        predictions_by_model[label] = predictions
        scores = score_predictions(predictions, references)
        print(
            f"{label}: BLEU={scores['bleu']:.2f}, "
            f"chrF++={scores['chrf++']:.2f}"
        )

    for label, adapter_path in args.adapter:
        predictions = translate(
            model_id=args.adapter_base_model,
            source_texts=sources,
            batch_size=args.batch_size,
            model_revision=args.adapter_base_revision,
            adapter_path=adapter_path,
        )
        predictions_by_model[label] = predictions
        scores = score_predictions(predictions, references)
        print(
            f"{label}: BLEU={scores['bleu']:.2f}, "
            f"chrF++={scores['chrf++']:.2f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    fieldnames.extend(f"prediction_{label}" for label in predictions_by_model)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            for label, predictions in predictions_by_model.items():
                out[f"prediction_{label}"] = predictions[i]
            writer.writerow(out)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
