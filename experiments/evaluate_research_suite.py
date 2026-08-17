"""Evaluate NLLB/AfriNLLB and multiple LoRA adapters on one manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate_mt import BASELINES, TOKENIZER_ID, load_jsonl, score_predictions, translate


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
        "--adapter",
        action="append",
        type=parse_adapter,
        default=[],
        metavar="LABEL=PATH",
        help="Repeat to evaluate multiple PEFT adapters.",
    )
    parser.add_argument("--adapter-base-model", default=TOKENIZER_ID)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise RuntimeError("Input manifest contains no rows.")

    sources = [str(row[args.source_field]) for row in rows]
    references = [str(row["english_ref"]) for row in rows]
    predictions_by_model: dict[str, list[str]] = {}

    baseline_labels = args.baseline or ["nllb", "afrinllb"]

    for label in baseline_labels:
        predictions = translate(
            model_id=BASELINES[label],
            source_texts=sources,
            batch_size=args.batch_size,
        )
        predictions_by_model[label] = predictions
        scores = score_predictions(predictions, references)
        print(
            f"{label}: BLEU={scores['bleu']:.2f}, "
            f"chrF++={scores['chrf++']:.2f}"
        )

    for label, model_id in args.model:
        predictions = translate(
            model_id=model_id,
            source_texts=sources,
            batch_size=args.batch_size,
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
