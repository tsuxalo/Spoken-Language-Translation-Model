"""Transcribe a raw Hausa .wav file with the fine-tuned Whisper model."""

import sys

import librosa
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_DIR = "nahomazmach/whisper-small-ha"  # Hugging Face Hub repo; use a local path (e.g. "./whisper-small-ha") instead if you trained your own copy
SAMPLING_RATE = 16_000

_device = "cuda" if torch.cuda.is_available() else "cpu"
_processor = None
_model = None


def _load_model(model_dir: str):
    global _processor, _model
    if _model is None:
        _processor = WhisperProcessor.from_pretrained(model_dir)
        _model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(_device)
    return _processor, _model


def transcribe(wav_path: str, model_dir: str = MODEL_DIR) -> str:
    processor, model = _load_model(model_dir)

    audio_array, sr = sf.read(wav_path)
    if sr != SAMPLING_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLING_RATE)

    input_features = processor.feature_extractor(
        audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features.to(_device)

    predicted_ids = model.generate(input_features, language="hausa", task="transcribe")
    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python inference.py <path-to-wav> [model_dir]")
    wav_arg = sys.argv[1]
    model_dir_arg = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR
    print(transcribe(wav_arg, model_dir_arg))
