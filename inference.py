"""Transcribe a raw Hausa .wav file and translate it to English.

Cascaded pipeline: our fine-tuned Whisper does Hausa audio -> Hausa text,
then NLLB-200 does Hausa text -> English text. Two pretrained/fine-tuned
models chained together, no new training required for the translation step.
"""

import gc
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

from experiments.revisions import (
    NLLB_600M_ID,
    NLLB_600M_REVISION,
    WHISPER_HAUSA_ID,
    WHISPER_HAUSA_REVISION,
)

ASR_MODEL_DIR = WHISPER_HAUSA_ID  # Use a local path instead for a local checkpoint.
ASR_MODEL_REVISION = WHISPER_HAUSA_REVISION
MT_MODEL_ID = NLLB_600M_ID
MT_MODEL_REVISION = NLLB_600M_REVISION
SAMPLING_RATE = 16_000

_device = "cuda" if torch.cuda.is_available() else "cpu"
_asr_processor = None
_asr_model = None
_mt_tokenizer = None
_mt_model = None


def _load_asr_model(model_dir: str):
    global _asr_processor, _asr_model
    if _asr_model is None:
        revision = ASR_MODEL_REVISION if model_dir == ASR_MODEL_DIR else None
        _asr_processor = WhisperProcessor.from_pretrained(model_dir, revision=revision)
        _asr_model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, revision=revision
        ).to(_device).eval()
    return _asr_processor, _asr_model


def _load_mt_model():
    global _mt_tokenizer, _mt_model
    if _mt_model is None:
        _mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_ID, revision=MT_MODEL_REVISION)
        _mt_model = AutoModelForSeq2SeqLM.from_pretrained(
            MT_MODEL_ID, revision=MT_MODEL_REVISION
        ).to(_device).eval()
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

    with torch.inference_mode():
        predicted_ids = model.generate(input_features, language="hausa", task="transcribe")
    return processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def translate_to_english(hausa_text: str) -> str:
    tokenizer, model = _load_mt_model()
    tokenizer.src_lang = "hau_Latn"
    inputs = tokenizer(hausa_text, return_tensors="pt").to(_device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id, max_length=256
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def transcribe_and_translate(wav_path: str, model_dir: str = ASR_MODEL_DIR) -> tuple[str, str]:
    hausa_text = transcribe(wav_path, model_dir)
    return hausa_text, translate_to_english(hausa_text)


def close() -> None:
    """Release cascade models so another deployable system can use the GPU."""
    global _asr_processor, _asr_model, _mt_tokenizer, _mt_model
    _asr_processor = None
    _asr_model = None
    _mt_tokenizer = None
    _mt_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
