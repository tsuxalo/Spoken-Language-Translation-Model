"""Generate realistic Hausa ASR errors from NaijaS2ST speech.

The central experiment is NOT to inject random spelling mistakes.

Instead, we let our real Hausa Whisper model make its real mistakes:

    Hausa speech
        ->
    fine-tuned Whisper
        ->
    noisy Hausa transcript

We pair that noisy transcript with the SAME English reference.

The resulting dataset lets us measure error propagation and later train an
MT model specifically to translate the type of Hausa text that our ASR
system actually produces.
"""

import argparse
import io
import json
from pathlib import Path
from statistics import mean

import jiwer
import librosa
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from prepare_naijas2st_pairs import alignment_key


DATASET_ID = "McGill-NLP/NaijaS2ST"
ASR_MODEL_ID = "nahomazmach/whisper-small-ha"
SAMPLING_RATE = 16_000


def load_pairs(path: str) -> dict[str, dict]:
    """Index English references by their shared alignment ID."""

    pairs = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)

            pairs[row["alignment_id"]] = row

    return pairs


def decode_audio(audio_field: dict) -> tuple:
    """Decode Hugging Face's raw audio dictionary using soundfile.

    We intentionally use decode=False in datasets because the parent
    repository already avoids torchcodec/FFmpeg compatibility problems.
    """

    audio_bytes = audio_field.get("bytes")

    if audio_bytes is not None:
        return sf.read(io.BytesIO(audio_bytes))

    audio_path = audio_field.get("path")

    if audio_path:
        return sf.read(audio_path)

    raise RuntimeError("Audio example has neither bytes nor a path.")


def transcribe_array(
    audio_array,
    sample_rate: int,
    processor,
    model,
    device: str,
) -> str:
    """Run our published Hausa Whisper model on one waveform."""

    if getattr(audio_array, "ndim", 1) > 1:
        audio_array = audio_array.mean(axis=1)

    if sample_rate != SAMPLING_RATE:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sample_rate,
            target_sr=SAMPLING_RATE,
        )

    features = processor.feature_extractor(
        audio_array,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
    ).input_features.to(device)

    with torch.inference_mode():
        predicted_ids = model.generate(
            features,
            language="hausa",
            task="transcribe",
        )

    prediction = processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    return prediction.strip()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling streaming examples.",
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=2000,
        help=(
        "Streaming shuffle buffer. Larger values provide better "
        "mixing across speakers while keeping memory bounded."
        ),
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="Aligned pair manifest from prepare_naijas2st_pairs.py.",
    )

    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev"],
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--output",
        default="experiments/generated/naijas2st_dev_asr_noise.jsonl",
    )

    parser.add_argument(
        "--asr-model",
        default=ASR_MODEL_ID,
    )

    args = parser.parse_args()

    references = load_pairs(args.pairs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"ASR model: {args.asr_model}")
    print(f"Reference sentences: {len(references):,}")

    processor = WhisperProcessor.from_pretrained(
        args.asr_model
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        args.asr_model
    ).to(device)

    model.eval()

    # Stream audio instead of downloading the entire NaijaS2ST corpus.
    dataset = load_dataset(
        DATASET_ID,
        split=args.split,
        streaming=True,
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False),
    )

    # NaijaS2ST rows may be grouped by speaker in storage order.
# Shuffle the streaming dataset so a small evaluation sample does not
# accidentally evaluate only the first speaker in the dataset.
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(
        seed=args.seed,
        buffer_size=args.shuffle_buffer,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Individual utterance WERs are useful for error analysis.
    wer_scores = []

    # These are also kept so we can calculate standard corpus-level WER.
    gold_transcripts = []
    asr_transcripts = []

    processed = 0

    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            if str(row["language"]).lower() != "hausa":
                continue

            key = alignment_key(
                row["text_id"],
                "hausa",
            )

            # Only evaluate utterances whose English reference was
            # successfully aligned by our previous script.
            if key not in references:
                continue

            reference = references[key]

            audio_array, sample_rate = decode_audio(
                row["audio"]
            )

            asr_text = transcribe_array(
                audio_array=audio_array,
                sample_rate=sample_rate,
                processor=processor,
                model=model,
                device=device,
            )

            gold_hausa = str(row["text"]).strip()

            # jiwer returns a proportion, so multiply by 100 for readability.
            sample_wer = 100.0 * jiwer.wer(
                gold_hausa,
                asr_text,
            )

            result = {
                "alignment_id": key,
                "utterance_id": (
                    f"{row['user_id']}::{row['text_id']}"
                ),
                "speaker_id": row["user_id"],
                "split": args.split,
                "hausa_gold": gold_hausa,
                "hausa_asr": asr_text,
                "english_ref": reference["english_ref"],
                "wer": sample_wer,
                "duration": row.get("duration"),
                "snr_db": row.get("snr_db"),
                "speech_rate": row.get("speech_rate"),
            }

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            wer_scores.append(sample_wer)
            gold_transcripts.append(gold_hausa)
            asr_transcripts.append(asr_text)
            processed += 1

            print(
                f"[{processed}/{args.max_samples}] "
                f"WER={sample_wer:.1f}% | "
                f"{asr_text[:80]}"
            )

            if (
                args.max_samples > 0
                and processed >= args.max_samples
            ):
                break

    if not processed:
        raise RuntimeError(
            "No matching Hausa audio examples were processed."
        )

    print()
    print("ASR benchmark summary")
    print("---------------------")
    print(f"Examples: {processed}")
    # Corpus WER weights errors by the number of words in the references,
    # rather than giving every utterance identical weight.
    corpus_wer = 100.0 * jiwer.wer(
        gold_transcripts,
        asr_transcripts,
    )

    print(f"Corpus WER:          {corpus_wer:.2f}%")
    print(f"Mean utterance WER:  {mean(wer_scores):.2f}%")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()