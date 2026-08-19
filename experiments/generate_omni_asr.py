"""Generate NaijaS2ST Hausa ASR hypotheses with Meta Omnilingual ASR.

This produces the same manifest schema as generate_asr_noise.py, allowing
the existing NLLB evaluator to reproduce a strong Omnilingual-ASR + NLLB
cascade baseline on the same examples as our Whisper-based cascade.

Recommended: run this on Linux with an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
from pathlib import Path
from statistics import mean

import jiwer
import soundfile as sf
from datasets import Audio, load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

try:
    from .prepare_naijas2st_pairs import alignment_key
    from .revisions import NAIJAS2ST_ID, NAIJAS2ST_REVISION
except ImportError:  # pragma: no cover - direct script execution
    from prepare_naijas2st_pairs import alignment_key
    from revisions import NAIJAS2ST_ID, NAIJAS2ST_REVISION

DATASET_ID = NAIJAS2ST_ID
DEFAULT_MODEL_CARD = "omniASR_LLM_1B"
LANG_CODE = "hau_Latn"


def load_pairs(path: str) -> dict[str, dict]:
    pairs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pairs[str(row["alignment_id"])] = row
    return pairs


def decode_audio(audio_field: dict):
    audio_bytes = audio_field.get("bytes")
    if audio_bytes is not None:
        return sf.read(io.BytesIO(audio_bytes))
    audio_path = audio_field.get("path")
    if audio_path:
        return sf.read(audio_path)
    raise RuntimeError("Audio example has neither bytes nor a path.")


def transcribe_batch(pipeline, rows: list[dict], batch_size: int) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="omni_asr_") as tmpdir:
        audio_paths = []
        for i, row in enumerate(rows):
            audio, sample_rate = decode_audio(row["audio"])
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            path = Path(tmpdir) / f"{i:05d}.wav"
            sf.write(path, audio, sample_rate)
            audio_paths.append(str(path))

        transcriptions = pipeline.transcribe(
            audio_paths,
            lang=[LANG_CODE] * len(audio_paths),
            batch_size=min(batch_size, len(audio_paths)),
        )
        return [str(text).strip() for text in transcriptions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=2000)
    parser.add_argument("--model-card", default=DEFAULT_MODEL_CARD)
    parser.add_argument("--dataset-revision", default=NAIJAS2ST_REVISION)
    args = parser.parse_args()

    references = load_pairs(args.pairs)
    print(f"Reference sentence IDs: {len(references):,}")
    print(f"Omnilingual model card: {args.model_card}")

    pipeline = ASRInferencePipeline(model_card=args.model_card)

    dataset = load_dataset(
        DATASET_ID,
        split=args.split,
        streaming=True,
        revision=args.dataset_revision,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))

    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pending_rows: list[dict] = []
    processed = 0
    wer_scores = []
    gold_transcripts = []
    asr_transcripts = []

    def flush_batch(file_handle) -> None:
        nonlocal processed, pending_rows

        if not pending_rows:
            return

        hypotheses = transcribe_batch(
            pipeline=pipeline,
            rows=pending_rows,
            batch_size=args.batch_size,
        )

        for row, asr_text in zip(pending_rows, hypotheses):
            key = alignment_key(str(row["text_id"]), "hausa")
            reference = references[key]
            gold_hausa = str(row["text"]).strip()
            sample_wer = 100.0 * jiwer.wer(gold_hausa, asr_text)

            result = {
                "alignment_id": key,
                "utterance_id": f"{row['user_id']}::{row['text_id']}",
                "speaker_id": row["user_id"],
                "split": args.split,
                "asr_system": args.model_card,
                "hausa_gold": gold_hausa,
                "hausa_asr": asr_text,
                "english_ref": reference["english_ref"],
                "wer": sample_wer,
                "duration": row.get("duration"),
                "snr_db": row.get("snr_db"),
                "speech_rate": row.get("speech_rate"),
            }
            file_handle.write(json.dumps(result, ensure_ascii=False) + "\n")

            processed += 1
            wer_scores.append(sample_wer)
            gold_transcripts.append(gold_hausa)
            asr_transcripts.append(asr_text)

            denominator = str(args.max_samples) if args.max_samples > 0 else "all"
            print(
                f"\rProcessed {processed}/{denominator} | "
                f"last WER={sample_wer:.1f}%",
                end="",
                flush=True,
            )

        pending_rows = []

    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            if str(row["language"]).lower().strip() != "hausa":
                continue

            key = alignment_key(str(row["text_id"]), "hausa")
            if key not in references:
                continue

            pending_rows.append(row)

            reached_limit = (
                args.max_samples > 0
                and processed + len(pending_rows) >= args.max_samples
            )
            if len(pending_rows) >= args.batch_size or reached_limit:
                if reached_limit and args.max_samples > 0:
                    remaining = args.max_samples - processed
                    pending_rows = pending_rows[:remaining]
                flush_batch(f)

            if args.max_samples > 0 and processed >= args.max_samples:
                break

        if pending_rows:
            flush_batch(f)

    print()
    if processed == 0:
        raise RuntimeError("No matching Hausa examples were processed.")

    corpus_wer = 100.0 * jiwer.wer(gold_transcripts, asr_transcripts)
    print("Omnilingual ASR summary")
    print("-----------------------")
    print(f"Examples: {processed}")
    print(f"Corpus WER: {corpus_wer:.2f}%")
    print(f"Mean utterance WER: {mean(wer_scores):.2f}%")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
