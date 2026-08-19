"""Run the three pinned Hausa-to-English systems on one qualitative clip.

The result is row-level evidence, so the default output is under the ignored
``artifacts/comparison-v2/private`` boundary.  The three systems are loaded
and released sequentially to make the memory contract explicit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from comparison_v2 import atomic_write_json
from direct_c1 import C1Runtime, PreparedAudio, read_audio
from experiments.revisions import (
    DIRECT_PILOT_ADAPTER_ID,
    DIRECT_PILOT_ADAPTER_REVISION,
    FLEURS_REVISION,
    NLLB_600M_ID,
    NLLB_600M_REVISION,
    OPENAI_WHISPER_SMALL_ID,
    OPENAI_WHISPER_SMALL_REVISION,
    WHISPER_HAUSA_ID,
    WHISPER_HAUSA_REVISION,
)

ASR_MODEL_ID = WHISPER_HAUSA_ID
ASR_REVISION = WHISPER_HAUSA_REVISION
MT_MODEL_ID = NLLB_600M_ID
MT_REVISION = NLLB_600M_REVISION
DIRECT_BASE_ID = OPENAI_WHISPER_SMALL_ID
DIRECT_BASE_REVISION = OPENAI_WHISPER_SMALL_REVISION
DIRECT_ADAPTER_ID = DIRECT_PILOT_ADAPTER_ID
DIRECT_ADAPTER_REVISION = DIRECT_PILOT_ADAPTER_REVISION

FLEURS_DATASET_REVISION = FLEURS_REVISION
FLEURS_EXAMPLE_ID = 1856
FLEURS_AUDIO_FILE = "5153420622149111720.wav"


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def run_cascade(prepared: PreparedAudio, device: str) -> dict[str, Any]:
    """Run ASR then MT, releasing ASR before loading NLLB."""

    _reset_peak(device)
    started = time.perf_counter()
    asr_processor = WhisperProcessor.from_pretrained(
        ASR_MODEL_ID,
        revision=ASR_REVISION,
        trust_remote_code=False,
    )
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL_ID,
        revision=ASR_REVISION,
        trust_remote_code=False,
        dtype=torch.float32,
    ).to(device).eval()
    _synchronize(device)
    asr_load_seconds = time.perf_counter() - started

    features = asr_processor.feature_extractor(
        prepared.samples,
        sampling_rate=prepared.sampling_rate,
        return_tensors="pt",
    ).input_features.to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        token_ids = asr_model.generate(
            features,
            language="hausa",
            task="transcribe",
        )
    _synchronize(device)
    asr_inference_seconds = time.perf_counter() - started
    hausa_asr = asr_processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    del token_ids, features, asr_model, asr_processor
    _release_cuda()

    started = time.perf_counter()
    mt_tokenizer = AutoTokenizer.from_pretrained(
        MT_MODEL_ID,
        revision=MT_REVISION,
        trust_remote_code=False,
    )
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_MODEL_ID,
        revision=MT_REVISION,
        trust_remote_code=False,
        dtype=torch.float32,
    ).to(device).eval()
    _synchronize(device)
    mt_load_seconds = time.perf_counter() - started
    mt_tokenizer.src_lang = "hau_Latn"
    model_inputs = mt_tokenizer(hausa_asr, return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        token_ids = mt_model.generate(
            **model_inputs,
            forced_bos_token_id=mt_tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=256,
        )
    _synchronize(device)
    mt_inference_seconds = time.perf_counter() - started
    prediction = mt_tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    peak_mb = _peak_memory_mb(device)
    del token_ids, model_inputs, mt_model, mt_tokenizer
    _release_cuda()

    if not prediction:
        raise RuntimeError("cascade returned an empty prediction")
    inference_seconds = asr_inference_seconds + mt_inference_seconds
    return {
        "system": "cascade_real_asr",
        "model_revisions": {"asr": ASR_REVISION, "mt": MT_REVISION},
        "device": device,
        "hausa_asr": hausa_asr,
        "prediction": prediction,
        "load_time_seconds": asr_load_seconds + mt_load_seconds,
        "inference_time_seconds": inference_seconds,
        "real_time_factor": inference_seconds / prepared.duration_seconds,
        "peak_cuda_memory_mb": peak_mb,
    }


def run_direct_pilot(prepared: PreparedAudio, device: str) -> dict[str, Any]:
    """Run the historical Whisper-small PEFT direct pilot."""

    _reset_peak(device)
    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(
        DIRECT_ADAPTER_ID,
        revision=DIRECT_ADAPTER_REVISION,
        trust_remote_code=False,
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        DIRECT_BASE_ID,
        revision=DIRECT_BASE_REVISION,
        trust_remote_code=False,
        dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(
        base_model,
        DIRECT_ADAPTER_ID,
        revision=DIRECT_ADAPTER_REVISION,
    ).to(device).eval()
    _synchronize(device)
    load_seconds = time.perf_counter() - started

    features = processor.feature_extractor(
        prepared.samples,
        sampling_rate=prepared.sampling_rate,
        return_tensors="pt",
    ).input_features.to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        token_ids = model.generate(
            features,
            language="hausa",
            task="translate",
            max_new_tokens=225,
            num_beams=1,
        )
    _synchronize(device)
    inference_seconds = time.perf_counter() - started
    prediction = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    peak_mb = _peak_memory_mb(device)
    del token_ids, features, model, base_model, processor
    _release_cuda()

    if not prediction:
        raise RuntimeError("historical direct pilot returned an empty prediction")
    return {
        "system": "direct_pilot",
        "model_revisions": {
            "base": DIRECT_BASE_REVISION,
            "adapter": DIRECT_ADAPTER_REVISION,
        },
        "device": device,
        "prediction": prediction,
        "load_time_seconds": load_seconds,
        "inference_time_seconds": inference_seconds,
        "real_time_factor": inference_seconds / prepared.duration_seconds,
        "peak_cuda_memory_mb": peak_mb,
    }


def run_c1(prepared: PreparedAudio, device: str) -> dict[str, Any]:
    runtime = C1Runtime(device=device)
    try:
        prediction = runtime.translate_array(
            prepared.samples,
            prepared.sampling_rate,
        )
        if not prediction:
            raise RuntimeError("C1 returned an empty prediction")
        assert runtime.last_telemetry is not None
        return runtime.last_telemetry.to_dict()
    finally:
        runtime.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Public qualitative Hausa audio clip")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--hausa-transcription",
        default=None,
        help="Optional public source transcription; never treated as English reference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/comparison-v2/private/qualitative_fleurs_1856.json"),
        help="Ignored row-level result path",
    )
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    prepared = read_audio(args.audio)
    systems = [
        run_cascade(prepared, device),
        run_direct_pilot(prepared, device),
        run_c1(prepared, device),
    ]
    result = {
        "schema_version": "1.0",
        "label": "Qualitative demonstration only; no gold English translation.",
        "audio": {
            "dataset": "google/fleurs",
            "dataset_revision": FLEURS_DATASET_REVISION,
            "example_id": FLEURS_EXAMPLE_ID,
            "expected_filename": FLEURS_AUDIO_FILE,
            "actual_filename": args.audio.name,
            "sha256": _sha256(args.audio),
            "duration_seconds": prepared.duration_seconds,
            "sampling_rate": prepared.sampling_rate,
            "hausa_transcription": args.hausa_transcription,
            "english_reference": None,
        },
        "systems": systems,
    }
    atomic_write_json(args.output, result)
    for row in systems:
        print(f"{row['system']}: {row['prediction']}")
        print(
            f"  inference={row['inference_time_seconds']:.3f}s "
            f"rtf={row['real_time_factor']:.4f} "
            f"peak_cuda_mib={row['peak_cuda_memory_mb']}"
        )
    print(f"Private result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
