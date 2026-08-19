"""Run a frozen private comparison-v2 evaluation, one system at a time."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from comparison_v2 import (
    DEPLOYABLE_SYSTEMS,
    FULL_DEVELOPMENT_EXAMPLES,
    FULL_DEVELOPMENT_SCOPE,
    PILOT_SCOPE,
    PredictionStore,
    canonical_manifest_hash,
    reject_official_dev,
)
from direct_c1 import C1Runtime, read_audio
from qualitative_demo import (
    ASR_MODEL_ID,
    ASR_REVISION,
    DIRECT_ADAPTER_ID,
    DIRECT_ADAPTER_REVISION,
    DIRECT_BASE_ID,
    DIRECT_BASE_REVISION,
    MT_MODEL_ID,
    MT_REVISION,
)


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _peak(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


ResolvedRows = list[tuple[dict[str, Any], Path]]


def _resolve_rows(
    manifest: dict[str, Any],
    audio_cache: Path,
    *,
    full_development: bool,
) -> ResolvedRows:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise TypeError("private evaluation manifest has no row list")
    expected_scope = FULL_DEVELOPMENT_SCOPE if full_development else PILOT_SCOPE
    actual_scope = str(manifest.get("scope", PILOT_SCOPE))
    if actual_scope != expected_scope:
        raise ValueError(
            f"manifest scope {actual_scope!r} does not match requested {expected_scope!r}"
        )
    if full_development and len(rows) != FULL_DEVELOPMENT_EXAMPLES:
        raise ValueError(
            f"full development manifest must contain {FULL_DEVELOPMENT_EXAMPLES} rows"
        )
    if not full_development and not 10 <= len(rows) <= 20:
        raise ValueError("private pilot manifest must contain 10-20 rows")
    reject_official_dev(rows)
    if {row.get("official_split") for row in rows} != {"train"}:
        raise ValueError("private pilot must originate only from official train")
    if {row.get("project_split") for row in rows} != {"validation"}:
        raise ValueError("private pilot must use only C1 project-validation rows")
    actual_hash = canonical_manifest_hash(rows)
    if manifest.get("membership_sha256") != actual_hash:
        raise ValueError("private pilot membership changed after it was frozen")

    resolved_rows: ResolvedRows = []
    for row in rows:
        locator = row.get("audio_locator") or {}
        if locator.get("split") != "train":
            raise ValueError("audio locator attempted to leave the train split")
        row_index = int(locator["row_index"])
        matches = list(audio_cache.glob(f"{row_index:06d}_*.wav"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one cached train waveform for row {row_index}; found {len(matches)}"
            )
        audio_info = sf.info(matches[0])
        expected_duration = float(row["audio_duration_seconds"])
        if abs(float(audio_info.duration) - expected_duration) > 0.02:
            raise ValueError(f"duration mismatch for private example {row['example_id']}")
        resolved_rows.append((row, matches[0]))
    return resolved_rows


def _progress(system: str, completed: int, total: int) -> None:
    if completed == 1 or completed == total or completed % 25 == 0:
        print(f"{system}: {completed}/{total}", flush=True)


def _append_checkpointed(
    store: PredictionStore,
    row: dict[str, Any],
    completed: int,
    total: int,
) -> None:
    store.append(row, persist=False)
    if completed == total or completed % 10 == 0:
        store.checkpoint()


def _revision_map(system: str) -> dict[str, str]:
    if system == "cascade_real_asr":
        return {ASR_MODEL_ID: ASR_REVISION, MT_MODEL_ID: MT_REVISION}
    if system == "direct_pilot":
        return {
            DIRECT_BASE_ID: DIRECT_BASE_REVISION,
            DIRECT_ADAPTER_ID: DIRECT_ADAPTER_REVISION,
        }
    if system == "direct_c1":
        from direct_c1 import MODEL_ID, MODEL_REVISION

        return {MODEL_ID: MODEL_REVISION}
    raise ValueError(system)


def _prediction_row(
    membership: dict[str, Any],
    *,
    system: str,
    prediction: str,
    load_seconds: float,
    inference_seconds: float,
    peak_mb: float | None,
    error_type: str | None = None,
) -> dict[str, Any]:
    duration = float(membership["audio_duration_seconds"])
    return {
        "example_id": str(membership["example_id"]),
        "alignment_id": str(membership["alignment_id"]),
        "system": system,
        "model_revisions": _revision_map(system),
        "reference": str(membership["reference"]),
        "prediction": prediction,
        "audio_duration_seconds": duration,
        "load_time_seconds": load_seconds,
        "inference_time_seconds": inference_seconds,
        "real_time_factor": inference_seconds / duration,
        "peak_gpu_memory_mb": peak_mb,
        "success": error_type is None and bool(prediction.strip()),
        "error_type": error_type,
    }


def _pending(
    resolved_rows: ResolvedRows,
    store: PredictionStore,
    system: str,
) -> ResolvedRows:
    completed = {
        (str(row["example_id"]), str(row["system"]))
        for row in store.rows
    }
    return [
        item
        for item in resolved_rows
        if (str(item[0]["example_id"]), system) not in completed
    ]


def run_cascade(
    resolved_rows: ResolvedRows,
    store: PredictionStore,
    device: str,
) -> None:
    pending = _pending(resolved_rows, store, "cascade_real_asr")
    if not pending:
        return
    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(
        ASR_MODEL_ID,
        revision=ASR_REVISION,
        trust_remote_code=False,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL_ID,
        revision=ASR_REVISION,
        trust_remote_code=False,
        dtype=torch.float32,
    ).to(device).eval()
    _sync(device)
    asr_load = time.perf_counter() - started
    transcripts: dict[str, tuple[str, float, float | None]] = {}
    for index, (membership, audio_path) in enumerate(pending, start=1):
        prepared = read_audio(audio_path)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        features = processor.feature_extractor(
            prepared.samples,
            sampling_rate=prepared.sampling_rate,
            return_tensors="pt",
        ).input_features.to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            token_ids = model.generate(features, language="hausa", task="transcribe")
        _sync(device)
        inference_seconds = time.perf_counter() - started
        transcript = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
        transcripts[str(membership["example_id"])] = (
            transcript,
            inference_seconds,
            _peak(device),
        )
        del token_ids, features
        _progress("cascade_asr", index, len(pending))
    del model, processor
    _release()

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MT_MODEL_ID,
        revision=MT_REVISION,
        trust_remote_code=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_MODEL_ID,
        revision=MT_REVISION,
        trust_remote_code=False,
        dtype=torch.float32,
    ).to(device).eval()
    _sync(device)
    mt_load = time.perf_counter() - started
    tokenizer.src_lang = "hau_Latn"
    for index, (membership, _) in enumerate(pending, start=1):
        transcript, asr_seconds, asr_peak = transcripts[str(membership["example_id"])]
        try:
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            inputs = tokenizer(transcript, return_tensors="pt").to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                token_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                    max_length=256,
                )
            _sync(device)
            mt_seconds = time.perf_counter() - started
            prediction = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
            row = _prediction_row(
                membership,
                system="cascade_real_asr",
                prediction=prediction,
                load_seconds=asr_load + mt_load,
                inference_seconds=asr_seconds + mt_seconds,
                peak_mb=max(
                    (value for value in (asr_peak, _peak(device)) if value is not None),
                    default=None,
                ),
                error_type=None if prediction else "EmptyPrediction",
            )
            del token_ids, inputs
        except (RuntimeError, ValueError) as exc:
            row = _prediction_row(
                membership,
                system="cascade_real_asr",
                prediction="",
                load_seconds=asr_load + mt_load,
                inference_seconds=asr_seconds,
                peak_mb=asr_peak,
                error_type=type(exc).__name__,
            )
        _append_checkpointed(store, row, index, len(pending))
        _progress("cascade_mt", index, len(pending))
    del model, tokenizer
    _release()


def run_direct_pilot(
    resolved_rows: ResolvedRows,
    store: PredictionStore,
    device: str,
) -> None:
    pending = _pending(resolved_rows, store, "direct_pilot")
    if not pending:
        return
    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(
        DIRECT_ADAPTER_ID,
        revision=DIRECT_ADAPTER_REVISION,
        trust_remote_code=False,
    )
    base = WhisperForConditionalGeneration.from_pretrained(
        DIRECT_BASE_ID,
        revision=DIRECT_BASE_REVISION,
        trust_remote_code=False,
        dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(
        base,
        DIRECT_ADAPTER_ID,
        revision=DIRECT_ADAPTER_REVISION,
    ).to(device).eval()
    _sync(device)
    load_seconds = time.perf_counter() - started
    for index, (membership, audio_path) in enumerate(pending, start=1):
        prepared = read_audio(audio_path)
        started = time.perf_counter()
        peak_mb = None
        try:
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
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
            _sync(device)
            inference_seconds = time.perf_counter() - started
            peak_mb = _peak(device)
            prediction = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
            row = _prediction_row(
                membership,
                system="direct_pilot",
                prediction=prediction,
                load_seconds=load_seconds,
                inference_seconds=inference_seconds,
                peak_mb=peak_mb,
                error_type=None if prediction else "EmptyPrediction",
            )
            del token_ids, features
        except (RuntimeError, ValueError) as exc:
            inference_seconds = time.perf_counter() - started
            row = _prediction_row(
                membership,
                system="direct_pilot",
                prediction="",
                load_seconds=load_seconds,
                inference_seconds=inference_seconds,
                peak_mb=peak_mb,
                error_type=type(exc).__name__,
            )
        _append_checkpointed(store, row, index, len(pending))
        _progress("direct_pilot", index, len(pending))
    del model, base, processor
    _release()


def run_c1(
    resolved_rows: ResolvedRows,
    store: PredictionStore,
    device: str,
) -> None:
    pending = _pending(resolved_rows, store, "direct_c1")
    if not pending:
        return
    runtime = C1Runtime(device=device)
    try:
        runtime._ensure_loaded()
        for index, (membership, audio_path) in enumerate(pending, start=1):
            prepared = read_audio(audio_path)
            started = time.perf_counter()
            try:
                prediction = runtime.translate_array(
                    prepared.samples,
                    prepared.sampling_rate,
                )
                assert runtime.last_telemetry is not None
                telemetry = runtime.last_telemetry
                row = _prediction_row(
                    membership,
                    system="direct_c1",
                    prediction=prediction,
                    load_seconds=runtime.load_time_seconds,
                    inference_seconds=telemetry.inference_time_seconds,
                    peak_mb=telemetry.peak_cuda_memory_mb,
                    error_type=None if prediction else "EmptyPrediction",
                )
            except (RuntimeError, ValueError) as exc:
                row = _prediction_row(
                    membership,
                    system="direct_c1",
                    prediction="",
                    load_seconds=runtime.load_time_seconds,
                    inference_seconds=time.perf_counter() - started,
                    peak_mb=_peak(device),
                    error_type=type(exc).__name__,
                )
            _append_checkpointed(store, row, index, len(pending))
            _progress("direct_c1", index, len(pending))
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_cache_train", type=Path)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--full-development",
        action="store_true",
        help=(
            "Explicitly authorize the complete recovered 1,037-row C1 internal-"
            "validation development scope; official dev remains prohibited"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate scope, membership hash, duration, and audio coverage without inference",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    predictions = args.predictions or Path(
        "artifacts/comparison-v2/private/"
        + (
            "full_development_predictions.json"
            if args.full_development
            else "common_predictions.json"
        )
    )
    if "private" not in {part.casefold() for part in predictions.parts}:
        raise ValueError("row-level predictions must stay under a private directory")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    resolved_rows = _resolve_rows(
        manifest,
        args.audio_cache_train,
        full_development=args.full_development,
    )
    print(
        f"Validated {len(resolved_rows)} {manifest.get('scope', PILOT_SCOPE)} rows; "
        f"membership SHA-256 {manifest['membership_sha256']}",
        flush=True,
    )
    if args.validate_only:
        return 0
    store = PredictionStore(predictions)
    run_cascade(resolved_rows, store, device)
    run_direct_pilot(resolved_rows, store, device)
    run_c1(resolved_rows, store, device)
    store.finalize(
        [str(row["example_id"]) for row, _ in resolved_rows],
        DEPLOYABLE_SYSTEMS,
    )
    print(f"Verified {len(store.rows)} private prediction rows at {predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
