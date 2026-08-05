"""Download and format the Hausa FLEURS dataset for Whisper fine-tuning.

Uses google/fleurs (config ha_ng) instead of the originally planned
mozilla-foundation/common_voice_11_0: Mozilla moved Common Voice off the
Hugging Face Hub to their own "Mozilla Data Collective" platform in
October 2025, so the old dataset id is no longer accessible here.

Audio is decoded manually with soundfile rather than through datasets'
built-in Audio decoder, because that decoder now requires torchcodec,
which in turn requires a shared (DLL) FFmpeg build in a version torchcodec
supports (4-8) -- the FFmpeg available here is a static v9 build.
"""

import io

import soundfile as sf
from datasets import Audio, load_dataset
from transformers import WhisperProcessor

MODEL_ID = "openai/whisper-small"
DATASET_ID = "google/fleurs"
LANGUAGE_CONFIG = "ha_ng"
SAMPLING_RATE = 16_000
OUTPUT_DIR = "./data"


def load_raw_dataset():
    train = load_dataset(DATASET_ID, LANGUAGE_CONFIG, split="train")
    test = load_dataset(DATASET_ID, LANGUAGE_CONFIG, split="test")
    train = train.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE, decode=False))
    test = test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE, decode=False))
    return train, test


def prepare_example(example, processor):
    audio_array, sr = sf.read(io.BytesIO(example["audio"]["bytes"]))
    example["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=sr
    ).input_features[0]
    example["labels"] = processor.tokenizer(example["raw_transcription"]).input_ids
    return example


def main():
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="Hausa", task="transcribe")

    train, test = load_raw_dataset()
    train = train.map(prepare_example, fn_kwargs={"processor": processor}, remove_columns=train.column_names)
    test = test.map(prepare_example, fn_kwargs={"processor": processor}, remove_columns=test.column_names)

    train.save_to_disk(f"{OUTPUT_DIR}/train")
    test.save_to_disk(f"{OUTPUT_DIR}/test")


if __name__ == "__main__":
    main()
