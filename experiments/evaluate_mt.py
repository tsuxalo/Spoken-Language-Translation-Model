"""Evaluate Hausa -> English MT models on aligned NaijaS2ST text.

This lets us answer two important questions:

1. Does Africa-specialized AfriNLLB beat the generic NLLB model already
   used by our repository?

2. Later, how much translation quality is lost when the MT model receives
   noisy ASR output instead of the gold Hausa transcript?

Metrics:
    SacreBLEU
    chrF++

chrF++ is particularly useful for morphologically rich / low-resource
translation because it compares character n-grams as well as word n-grams.
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import sacrebleu
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizerFast,
)

try:
    from .revisions import (
        AFRINLLB_ID,
        AFRINLLB_REVISION,
        NLLB_600M_ID,
        NLLB_600M_REVISION,
    )
except ImportError:  # pragma: no cover - direct script execution
    from revisions import (
        AFRINLLB_ID,
        AFRINLLB_REVISION,
        NLLB_600M_ID,
        NLLB_600M_REVISION,
    )

TOKENIZER_ID = NLLB_600M_ID

BASELINES = {
    "nllb": NLLB_600M_ID,
    "afrinllb": AFRINLLB_ID,
}
BASELINE_REVISIONS = {
    "nllb": NLLB_600M_REVISION,
    "afrinllb": AFRINLLB_REVISION,
}

SOURCE_LANG = "hau_Latn"
TARGET_LANG = "eng_Latn"


def load_jsonl(path: str) -> list[dict]:
    """Read our experiment manifest into memory."""

    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def translate(
    model_id: str,
    source_texts: list[str],
    batch_size: int,
    model_revision: str,
    tokenizer_revision: str = NLLB_600M_REVISION,
    adapter_path: str | None = None,
) -> list[str]:
    """Translate a batch of Hausa sentences into English."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print(f"Loading model: {model_id}")
    print(f"Device: {device}")

    # AfriNLLB is derived from NLLB-200, so using the original NLLB
    # tokenizer ensures the Hausa/English language tokens are identical
    # across our baseline comparison.
    tokenizer = NllbTokenizerFast.from_pretrained(
        TOKENIZER_ID,
        revision=tokenizer_revision,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    # `dtype` is the current Transformers argument.
    # Older versions used `torch_dtype`, which now produces a deprecation warning.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        revision=model_revision,
        dtype=dtype,
    )

    # If we trained a LoRA adapter, attach it to the frozen base model.
    if adapter_path is not None:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

    model = model.to(device)
    model.eval()

    english_token_id = tokenizer.convert_tokens_to_ids(
        TARGET_LANG
    )

    predictions = []

    for start in range(0, len(source_texts), batch_size):
        batch = source_texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=english_token_id,
                num_beams=4,
                max_new_tokens=256,
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        predictions.extend(text.strip() for text in decoded)

        completed = min(
            start + batch_size,
            len(source_texts),
        )

        print(
            f"\rTranslated {completed}/{len(source_texts)}",
            end="",
            flush=True,
        )

    print()

    # Explicitly release each 600M model before loading the next one.
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions


def score_predictions(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Calculate corpus-level translation metrics."""

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    ).score

    # word_order=2 gives chrF++, rather than character-only chrF.
    chrf_pp = sacrebleu.corpus_chrf(
        predictions,
        [references],
        word_order=2,
    ).score

    return {
        "bleu": bleu,
        "chrf++": chrf_pp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="JSONL manifest produced by our experiment scripts.",
    )

    parser.add_argument(
        "--source-field",
        default="hausa_gold",
        choices=["hausa_gold", "hausa_asr"],
        help="Evaluate clean Hausa or noisy ASR-produced Hausa.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output",
        default="results/mt_predictions.csv",
    )

    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT/LoRA adapter directory.",
    )

    parser.add_argument(
        "--adapter-base-model",
        default=TOKENIZER_ID,
        help="Foundation model used underneath the LoRA adapter.",
    )
    parser.add_argument("--nllb-revision", default=NLLB_600M_REVISION)
    parser.add_argument("--afrinllb-revision", default=AFRINLLB_REVISION)
    parser.add_argument("--adapter-base-revision", default=NLLB_600M_REVISION)

    args = parser.parse_args()

    rows = load_jsonl(args.input)

    if not rows:
        raise RuntimeError("Input manifest contains no rows.")

    if args.source_field not in rows[0]:
        raise KeyError(
            f"{args.source_field!r} is not in the input manifest."
        )

    sources = [row[args.source_field] for row in rows]
    references = [row["english_ref"] for row in rows]

    predictions_by_model = {}

    # First evaluate the two fixed baselines.
    requested_revisions = {
        "nllb": args.nllb_revision,
        "afrinllb": args.afrinllb_revision,
    }
    for label, model_id in BASELINES.items():
        predictions = translate(
            model_id=model_id,
            source_texts=sources,
            batch_size=args.batch_size,
            model_revision=requested_revisions[label],
        )

        predictions_by_model[label] = predictions

        scores = score_predictions(
            predictions,
            references,
        )

        print()
        print(f"{label} using {args.source_field}")
        print("--------------------------------")
        print(f"BLEU:   {scores['bleu']:.2f}")
        print(f"chrF++: {scores['chrf++']:.2f}")

    # Optionally evaluate our own trained adapter.
    if args.adapter:
        predictions = translate(
            model_id=args.adapter_base_model,
            source_texts=sources,
            batch_size=args.batch_size,
            model_revision=args.adapter_base_revision,
            adapter_path=args.adapter,
        )

        predictions_by_model["noise_aware_lora"] = predictions

        scores = score_predictions(
            predictions,
            references,
        )

        print()
        print("noise_aware_lora")
        print("----------------")
        print(f"BLEU:   {scores['bleu']:.2f}")
        print(f"chrF++: {scores['chrf++']:.2f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    for label in predictions_by_model:
        fieldnames.append(f"prediction_{label}")

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, row in enumerate(rows):
            output_row = dict(row)

            for label, predictions in predictions_by_model.items():
                output_row[f"prediction_{label}"] = predictions[index]

            writer.writerow(output_row)

    print()
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
