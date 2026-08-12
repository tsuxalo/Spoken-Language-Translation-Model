"""Transcribe a raw Hausa .wav file and translate it to English.

Cascaded pipeline: our fine-tuned Whisper does Hausa audio -> Hausa text,
then NLLB-200 does Hausa text -> English text. Two pretrained/fine-tuned
models chained together, no new training required for the translation step.
"""

import sys

import librosa
import soundfile as sf
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ASR_MODEL_DIR = "nahomazmach/whisper-small-ha"  # Hugging Face Hub repo; use a local path (e.g. "./whisper-small-ha") instead if you trained your own copy
MT_MODEL_ID = "facebook/nllb-200-distilled-600M"
SAMPLING_RATE = 16_000

_device = "cuda" if torch.cuda.is_available() else "cpu"
_asr_processor = None
_asr_model = None
_mt_tokenizer = None
_mt_model = None


def _load_asr_model(model_dir: str):
    global _asr_processor, _asr_model
    if _asr_model is None:
        _asr_processor = WhisperProcessor.from_pretrained(model_dir)
        _asr_model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(_device)
    return _asr_processor, _asr_model


def _load_mt_model():
    global _mt_tokenizer, _mt_model
    if _mt_model is None:
        _mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_ID)
        _mt_model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL_ID).to(_device)
    return _mt_tokenizer, _mt_model


def transcribe(wav_path: str, model_dir: str = ASR_MODEL_DIR) -> str:
    processor, model = _load_asr_model(model_dir)

    audio_array, sr = sf.read(wav_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # downmix stereo/multi-channel to mono
    if sr != SAMPLING_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLING_RATE)

    input_features = processor.feature_extractor(
        audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features.to(_device)

    predicted_ids = model.generate(input_features, language="hausa", task="transcribe")
    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def translate_to_english(hausa_text: str) -> str:
    tokenizer, model = _load_mt_model()
    tokenizer.src_lang = "hau_Latn"
    inputs = tokenizer(hausa_text, return_tensors="pt").to(_device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    generated_ids = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=256)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_and_translate(wav_path: str, model_dir: str = ASR_MODEL_DIR) -> tuple[str, str]:
    hausa_text = transcribe(wav_path, model_dir)
    return hausa_text, translate_to_english(hausa_text)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Hausa characters (ƙ, ƴ, ...) crash the default Windows console encoding
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python inference.py <path-to-wav> [model_dir]")
    wav_arg = sys.argv[1]
    model_dir_arg = sys.argv[2] if len(sys.argv) > 2 else ASR_MODEL_DIR
    hausa, english = transcribe_and_translate(wav_arg, model_dir_arg)
    print(f"Hausa:   {hausa}")
    print(f"English: {english}")
