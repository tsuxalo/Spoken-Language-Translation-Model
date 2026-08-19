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
from huggingface_hub import hf_hub_download
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

try:
    from .prepare_naijas2st_pairs import alignment_key
    from .revisions import (
        NAIJAS2ST_ID,
        NAIJAS2ST_REVISION,
        WHISPER_HAUSA_ID,
        WHISPER_HAUSA_REVISION,
    )
except ImportError:  # pragma: no cover - direct script execution
    from prepare_naijas2st_pairs import alignment_key
    from revisions import (
        NAIJAS2ST_ID,
        NAIJAS2ST_REVISION,
        WHISPER_HAUSA_ID,
        WHISPER_HAUSA_REVISION,
    )

DATASET_ID = NAIJAS2ST_ID
ASR_MODEL_ID = WHISPER_HAUSA_ID
SAMPLING_RATE = 16_000


def processor_compat_kwargs(model_id: str, revision: str) -> dict:
    """Adapt unambiguous Transformers-5 token metadata for Transformers 4."""

    local_config = Path(model_id) / "tokenizer_config.json"
    if local_config.is_file():
        config_path = local_config
    else:
        config_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="tokenizer_config.json",
                revision=revision,
            )
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    extra_tokens = config.get("extra_special_tokens")
    if not isinstance(extra_tokens, list):
        return {}
    if not extra_tokens or any(not isinstance(token, str) for token in extra_tokens):
        raise ValueError("extra_special_tokens must be a nonempty list of strings")
    if len(set(extra_tokens)) != len(extra_tokens):
        raise ValueError("extra_special_tokens contains ambiguous duplicate values")
    return {
        "extra_special_tokens": {
            f"extra_special_token_{index}": token
            for index, token in enumerate(extra_tokens)
        }
    }


def load_asr_runtime(model_id: str, revision: str, device: str):
    """Load the processor/model pair with one immutable revision."""

    processor = WhisperProcessor.from_pretrained(
        model_id,
        revision=revision,
        **processor_compat_kwargs(model_id, revision),
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
    ).to(device)
    model.eval()
    return processor, model


def load_resume_rows(path: Path) -> list[dict]:
    """Load and validate an existing append-only ASR manifest."""

    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid resume JSON on line {line_number}") from exc
    ids = [str(row.get("utterance_id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Resume manifest has missing or duplicate utterance_id values")
    return rows


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
    parser.add_argument("--asr-model-revision", default=WHISPER_HAUSA_REVISION)
    parser.add_argument("--dataset-revision", default=NAIJAS2ST_REVISION)
    parser.add_argument(
        "--target-matched-count",
        type=int,
        default=0,
        help="Stop after this many total matched rows, including resumed rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only rows whose utterance_id is absent from an existing output.",
    )

    args = parser.parse_args()

    references = load_pairs(args.pairs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"ASR model: {args.asr_model}")
    print(f"Reference sentences: {len(references):,}")

    processor, model = load_asr_runtime(
        args.asr_model,
        args.asr_model_revision,
        device,
    )

    # Stream audio instead of downloading the entire NaijaS2ST corpus.
    dataset = load_dataset(
        DATASET_ID,
        split=args.split,
        streaming=True,
        revision=args.dataset_revision,
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

    existing_rows = load_resume_rows(output_path) if args.resume else []
    completed_ids = {str(row["utterance_id"]) for row in existing_rows}
    target_count = args.target_matched_count or args.max_samples
    if target_count < 0:
        raise ValueError("Matched-count limits cannot be negative.")

    # Individual utterance WERs are useful for error analysis.
    wer_scores = [float(row["wer"]) for row in existing_rows]

    # These are also kept so we can calculate standard corpus-level WER.
    gold_transcripts = [str(row["hausa_gold"]) for row in existing_rows]
    asr_transcripts = [str(row["hausa_asr"]) for row in existing_rows]

    processed = len(existing_rows)

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as file:
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

            utterance_id = f"{row['user_id']}::{row['text_id']}"
            if utterance_id in completed_ids:
                continue
            if target_count > 0 and processed >= target_count:
                break

            reference = references[key]

            audio_array, sample_rate = decode_audio(row["audio"])

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
                "utterance_id": utterance_id,
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
            completed_ids.add(utterance_id)

            print(
                f"[{processed}/{target_count or 'all'}] "
                f"WER={sample_wer:.1f}% | "
                f"{asr_text[:80]}"
            )

            if target_count > 0 and processed >= target_count:
                break

    if not processed:
        raise RuntimeError("No matching Hausa audio examples were processed.")

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
