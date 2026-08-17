"""Create aligned Hausa -> English text pairs from NaijaS2ST.

This script constructs a lightweight parallel-text manifest for later
speech-translation experiments.

It intentionally DOES NOT decode audio. We only need:
- language
- text_id
- text

The resulting JSONL file can later be used for:
1. MT-only evaluation.
2. English-reference lookup during ASR evaluation.
3. Building clean/noisy Hausa -> English training data.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "McGill-NLP/NaijaS2ST"


def alignment_key(text_id: str, language: str) -> str:
    """Convert language-specific sentence IDs into a shared alignment key.

    NaijaS2ST may use language-specific prefixes on sentence IDs.
    For example, conceptually:

        E1234 -> 1234   # English
        H1234 -> 1234   # Hausa

    Removing the expected language prefix lets us match translations
    that correspond to the same underlying sentence.
    """

    text_id = str(text_id).strip()

    expected_prefix = {
        "english": "E",
        "hausa": "H",
    }.get(language.lower())

    if (
        expected_prefix
        and len(text_id) > 1
        and text_id[0].upper() == expected_prefix
    ):
        return text_id[1:]

    # If the ID does not follow the expected pattern, leave it unchanged.
    return text_id


def main() -> None:
    """Load NaijaS2ST and create aligned Hausa-English sentence pairs."""

    # ------------------------------------------------------------
    # Parse command-line arguments FIRST.
    #
    # This is important because args.split, args.output, etc. do not
    # exist until parser.parse_args() has executed.
    # ------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev"],
        help="NaijaS2ST split to process.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help=(
            "Maximum number of aligned sentence pairs to write. "
            "Use 0 to save all available pairs."
        ),
    )

    parser.add_argument(
        "--output",
        default="experiments/generated/naijas2st_dev_pairs.jsonl",
        help="Destination JSONL file.",
    )

    # args is created HERE.
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load the dataset only AFTER args exists.
    # ------------------------------------------------------------

    print(
        f"Loading {DATASET_ID} "
        f"split={args.split} in streaming mode..."
    )

    dataset = load_dataset(
        DATASET_ID,
        split=args.split,
        streaming=True,
    )

    # ------------------------------------------------------------
    # Disable media decoding.
    #
    # Newer Hugging Face Datasets versions may automatically try to
    # decode Audio columns using torchcodec during iteration.
    #
    # We do not need audio for this script, so decode(False) prevents
    # that dependency from being invoked.
    # ------------------------------------------------------------

    dataset = dataset.decode(False)

    # Remove the audio column completely because this script only needs
    # textual metadata.
    if "audio" in dataset.features:
        dataset = dataset.remove_columns(["audio"])

    # Store unique English and Hausa sentences by alignment key.
    english = {}
    hausa = {}

    # These counters help detect strange cases where one supposedly
    # shared sentence ID maps to multiple different texts.
    english_conflicts = 0
    hausa_conflicts = 0

    # ------------------------------------------------------------
    # Walk through the streaming dataset.
    # ------------------------------------------------------------

    for row in dataset:
        language = str(row["language"]).lower().strip()
        text_id = str(row["text_id"]).strip()
        text = str(row["text"]).strip()

        if not text:
            continue

        if language == "english":
            key = alignment_key(
                text_id=text_id,
                language="english",
            )

            if key not in english:
                english[key] = {
                    "text_id": text_id,
                    "text": text,
                }

            elif english[key]["text"] != text:
                english_conflicts += 1

        elif language == "hausa":
            key = alignment_key(
                text_id=text_id,
                language="hausa",
            )

            if key not in hausa:
                hausa[key] = {
                    "text_id": text_id,
                    "text": text,
                }

            elif hausa[key]["text"] != text:
                hausa_conflicts += 1

    # ------------------------------------------------------------
    # Find sentence IDs that exist in BOTH languages.
    # ------------------------------------------------------------

    all_shared_keys = sorted(
        set(english.keys()) & set(hausa.keys())
    )

    if not all_shared_keys:
        raise RuntimeError(
            "No Hausa-English aligned sentence IDs were found.\n"
            "This probably means our alignment_key() assumption does "
            "not match NaijaS2ST's actual text_id format.\n"
            "Inspect several English and Hausa text_id values before "
            "continuing."
        )

    # Limit the smoke-test dataset if requested.
    if args.max_samples > 0:
        selected_keys = all_shared_keys[: args.max_samples]
    else:
        selected_keys = all_shared_keys

    # ------------------------------------------------------------
    # Write aligned pairs.
    # ------------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for key in selected_keys:
            record = {
                "alignment_id": key,
                "split": args.split,
                "hausa_text_id": hausa[key]["text_id"],
                "english_text_id": english[key]["text_id"],
                "hausa_gold": hausa[key]["text"],
                "english_ref": english[key]["text"],
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ------------------------------------------------------------
    # Print validation statistics.
    #
    # We care about these numbers before doing any model evaluation.
    # ------------------------------------------------------------

    print()
    print("Alignment summary")
    print("-----------------")
    print(
        f"Unique Hausa texts:   {len(hausa):,}"
    )
    print(
        f"Unique English texts: {len(english):,}"
    )
    print(
        f"Aligned pairs found:  {len(all_shared_keys):,}"
    )
    print(
        f"Pairs written:        {len(selected_keys):,}"
    )
    print(
        f"English conflicts:    {english_conflicts:,}"
    )
    print(
        f"Hausa conflicts:      {hausa_conflicts:,}"
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()