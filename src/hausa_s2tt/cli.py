"""Installed command-line entry points for the research workflows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
NAIJA_REVISION = "898f51582750fe244693794f22e3f4b32c5baf95"
WHISPER_SMALL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
HAUSA_ASR_REVISION = "c4e2b47d88ae8b3ee0a605e09863b93aafca72e3"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def data_main(argv: list[str] | None = None) -> None:
    from .datasets import (
        align_naija_rows,
        iter_dataset_parquet_metadata,
        iter_dataset_viewer_rows,
        load_fleurs_splits,
        speaker_leakage,
        write_pairing_artifacts,
    )

    parser = argparse.ArgumentParser(description="Prepare FLEURS or audit NaijaS2ST")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    fleurs = subparsers.add_parser("fleurs", help="Save official Hausa FLEURS splits")
    fleurs.add_argument("--output-dir", type=Path, default=Path("data/fleurs"))
    fleurs.add_argument("--revision", default=FLEURS_REVISION)
    naija = subparsers.add_parser("naija", help="Audit NaijaS2ST metadata")
    naija.add_argument("--output-dir", type=Path, default=Path("data/naija_pairs"))
    naija.add_argument("--splits", nargs="+", default=["train", "dev"])
    naija.add_argument("--max-rows", type=int, help="Partial smoke audit only")
    naija.add_argument("--workers", type=int, default=2)
    naija.add_argument(
        "--source",
        choices=["parquet", "viewer"],
        default="parquet",
        help="Parquet range reads are reliable for a complete metadata audit",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "fleurs":
        dataset = load_fleurs_splits(revision=args.revision, decode_audio=False)
        counts = {}
        for split in ("train", "validation", "test"):
            dataset[split].save_to_disk(str(args.output_dir / split))
            counts[split] = len(dataset[split])
        result: dict[str, Any] = {
            "dataset": "google/fleurs",
            "config": "ha_ng",
            "revision": args.revision,
            "splits": counts,
            "features_precomputed": False,
            "test_policy": "saved but never loaded by the training entrypoint",
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    else:
        result = {}
        pairs_by_split = {}
        for split in args.splits:
            started = time.perf_counter()
            if args.source == "parquet":
                rows = list(
                    iter_dataset_parquet_metadata(
                        split, workers=args.workers, limit=args.max_rows
                    )
                )
            else:
                rows = list(
                    iter_dataset_viewer_rows(
                        split, workers=args.workers, limit=args.max_rows
                    )
                )
            pairs, audit = align_naija_rows(rows, split=split)
            write_pairing_artifacts(pairs, audit, args.output_dir)
            value = audit.to_dict()
            value["audit_wall_seconds"] = time.perf_counter() - started
            value["partial_audit"] = args.max_rows is not None
            value["metadata_source"] = args.source
            value["audio_decode_validation"] = "not performed"
            result[split] = value
            pairs_by_split[split] = pairs
        overlaps = speaker_leakage(pairs_by_split)
        partial = args.max_rows is not None
        result["cross_split"] = {
            "speaker_overlap": overlaps,
            "overlap_detected": bool(overlaps),
            "speaker_disjoint": None if partial else not overlaps,
            "partial_audit": partial,
        }
        (args.output_dir / "dataset_audit.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def train_main(argv: list[str] | None = None) -> None:
    from .asr import train_asr
    from .config import load_config
    from .direct_s2tt import train_direct_s2tt

    parser = argparse.ArgumentParser(description="Train ASR or direct S2TT")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.kind == "asr":
        result = train_asr(config)
    elif config.kind == "direct_s2tt":
        result = train_direct_s2tt(config)
    else:
        raise SystemExit("Training accepts only asr or direct_s2tt configs")
    print(json.dumps(result, indent=2, default=str))


def inference_main(argv: list[str] | None = None) -> None:
    from .cascade import CascadeTranslator
    from .inference import (
        create_asr_runtime,
        create_direct_runtime,
        create_zero_shot_runtime,
    )
    from .mt import NLLBTranslator

    parser = argparse.ArgumentParser(
        description="Run Hausa ASR, zero-shot/direct S2TT, or cascade inference"
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--task", choices=["asr", "zero_shot", "direct", "cascade"], required=True
    )
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--asr-model-id", default="nahomazmach/whisper-small-ha")
    parser.add_argument("--asr-model-revision")
    parser.add_argument("--mt-model-id", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--mt-model-revision")
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
    )
    parser.add_argument("--chunk-seconds", type=float, default=29.0)
    parser.add_argument("--stride-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    common = {
        "precision": args.precision,
        "chunk_length_seconds": args.chunk_seconds,
        "stride_seconds": args.stride_seconds,
        "batch_size": args.batch_size,
    }
    if args.task == "asr":
        revision = args.model_revision
        if args.model_id is None:
            revision = HAUSA_ASR_REVISION
        runtime = create_asr_runtime(
            args.model_id or "nahomazmach/whisper-small-ha",
            revision=revision,
            **common,
        )
        results = [
            {"file": str(path), "task": "asr", **runtime.process(path).to_dict()}
            for path in args.files
        ]
    elif args.task == "zero_shot":
        revision = args.model_revision
        if args.model_id is None:
            revision = WHISPER_SMALL_REVISION
        runtime = create_zero_shot_runtime(
            args.model_id or "openai/whisper-small",
            revision=revision,
            **common,
        )
        results = [
            {"file": str(path), "task": "zero_shot", **runtime.process(path).to_dict()}
            for path in args.files
        ]
    elif args.task == "direct":
        if not args.model_id:
            raise SystemExit("--model-id is required for direct S2TT inference")
        runtime = create_direct_runtime(
            args.model_id, revision=args.model_revision, **common
        )
        results = [
            {"file": str(path), "task": "direct", **runtime.process(path).to_dict()}
            for path in args.files
        ]
    else:
        asr_revision = args.asr_model_revision
        if args.asr_model_id == "nahomazmach/whisper-small-ha" and not asr_revision:
            asr_revision = HAUSA_ASR_REVISION
        mt_revision = args.mt_model_revision
        if args.mt_model_id == "facebook/nllb-200-distilled-600M" and not mt_revision:
            mt_revision = NLLB_REVISION
        cascade = CascadeTranslator(
            asr=create_asr_runtime(
                args.asr_model_id,
                revision=asr_revision,
                **common,
            ),
            mt=NLLBTranslator(
                args.mt_model_id,
                revision=mt_revision,
                precision=args.precision,
            ),
        )
        results = [
            {"file": str(path), "task": "cascade", **cascade.translate(path).to_dict()}
            for path in args.files
        ]
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def evaluate_asr_main(argv: list[str] | None = None) -> None:
    from .datasets import load_fleurs_splits
    from .evaluation import (
        FinalTestGuard,
        evaluate_asr_rows,
        write_prediction_artifacts,
    )
    from .inference import create_asr_runtime

    parser = argparse.ArgumentParser(description="Evaluate Hausa ASR")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision", default=FLEURS_REVISION)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluations/asr"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit("Final test evaluation requires --confirm-final-test")
    if args.split == "test" and args.max_samples is not None:
        raise SystemExit("Partial final-test metrics are disabled")
    guard = FinalTestGuard(args.output_dir.parent, args.run_name)
    if args.split == "test":
        guard.ensure_unused()
    dataset = load_fleurs_splits(
        revision=args.dataset_revision, splits=(args.split,)
    )[args.split]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    runtime = create_asr_runtime(args.model_id, revision=args.model_revision)
    rows = []
    for item in dataset:
        prediction = runtime.process(item["audio"])
        rows.append(
            {
                "source_id": item["id"],
                "reference": item["raw_transcription"],
                "prediction": prediction.text,
                "audio_duration_seconds": prediction.audio_duration_seconds,
                "inference_seconds": prediction.inference_seconds,
                "real_time_factor": prediction.real_time_factor,
            }
        )
    summary = {
        **evaluate_asr_rows(rows),
        "split": args.split,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
    }
    write_prediction_artifacts(rows, summary, args.output_dir / args.run_name)
    if args.split == "test":
        guard.seal(summary)
    print(json.dumps(summary, indent=2))


def evaluate_s2tt_main(argv: list[str] | None = None) -> None:
    from .cascade import CascadeTranslator
    from .datasets import load_naija_split, pair_naija_dataset
    from .evaluation import (
        FinalTestGuard,
        analysis_flags,
        evaluate_translation_rows,
        write_prediction_artifacts,
    )
    from .inference import (
        create_asr_runtime,
        create_direct_runtime,
        create_zero_shot_runtime,
    )
    from .mt import NLLBTranslator

    parser = argparse.ArgumentParser(description="Evaluate common held-out S2TT audio")
    parser.add_argument(
        "--systems", nargs="+", choices=["zero_shot", "direct", "cascade"], required=True
    )
    parser.add_argument("--direct-model-id")
    parser.add_argument("--direct-model-revision")
    parser.add_argument("--zero-shot-model-id", default="openai/whisper-small")
    parser.add_argument("--zero-shot-model-revision", default=WHISPER_SMALL_REVISION)
    parser.add_argument("--asr-model-id", default="nahomazmach/whisper-small-ha")
    parser.add_argument("--asr-model-revision", default=HAUSA_ASR_REVISION)
    parser.add_argument("--mt-model-id", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--mt-model-revision", default=NLLB_REVISION)
    parser.add_argument("--dataset-revision", default=NAIJA_REVISION)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluations/s2tt"))
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_final_test:
        raise SystemExit("Reserved held-out evaluation requires --confirm-final-test")
    if "direct" in args.systems and not args.direct_model_id:
        raise SystemExit("--direct-model-id is required for direct")
    guard = FinalTestGuard(args.output_dir.parent, args.run_name)
    guard.ensure_unused()
    paired, audit = pair_naija_dataset(
        load_naija_split("dev", revision=args.dataset_revision), split="dev"
    )
    runtimes: dict[str, Any] = {}
    if "zero_shot" in args.systems:
        runtimes["zero_shot"] = create_zero_shot_runtime(
            args.zero_shot_model_id, revision=args.zero_shot_model_revision
        )
    if "direct" in args.systems:
        runtimes["direct"] = create_direct_runtime(
            args.direct_model_id, revision=args.direct_model_revision
        )
    if "cascade" in args.systems:
        runtimes["cascade"] = CascadeTranslator(
            asr=create_asr_runtime(
                args.asr_model_id, revision=args.asr_model_revision
            ),
            mt=NLLBTranslator(args.mt_model_id, revision=args.mt_model_revision),
        )
    system_rows = {name: [] for name in args.systems}
    for item in paired:
        for name, runtime in runtimes.items():
            if name == "cascade":
                prediction = runtime.translate(item["audio"])
                text = prediction.english_text
                seconds = prediction.total_seconds
                rtf = prediction.real_time_factor
                source_text = prediction.hausa_text
            else:
                prediction = runtime.process(item["audio"])
                text = prediction.text
                seconds = prediction.inference_seconds
                rtf = prediction.real_time_factor
                source_text = item.get("source_text")
            system_rows[name].append(
                {
                    "source_id": item["source_text_id"],
                    "target_text_ids": item["target_text_ids"],
                    "alignment_key": item["alignment_key"],
                    "dataset_row_index": item["dataset_row_index"],
                    "speaker_id": item["speaker_id"],
                    "source_transcript": source_text,
                    "reference": item["target_text"],
                    "prediction": text,
                    "duration": item["duration"],
                    "inference_seconds": seconds,
                    "real_time_factor": rtf,
                    "analysis_flags": analysis_flags(
                        str(source_text or ""), item["target_text"], item["duration"]
                    ),
                }
            )
    summaries = {}
    for name, rows in system_rows.items():
        if not rows:
            raise RuntimeError("No aligned evaluation rows were produced")
        summary = {
            **evaluate_translation_rows(rows),
            "system": name,
            "split": "dev (reserved held-out set)",
            "mean_rtf": sum(row["real_time_factor"] for row in rows) / len(rows),
            "aggregate_rtf": sum(row["inference_seconds"] for row in rows)
            / sum(row["duration"] for row in rows),
            "pairing_audit": audit.to_dict(),
            "dataset_revision": args.dataset_revision,
            "model_provenance": {
                "zero_shot": [args.zero_shot_model_id, args.zero_shot_model_revision],
                "direct": [args.direct_model_id, args.direct_model_revision],
                "asr": [args.asr_model_id, args.asr_model_revision],
                "mt": [args.mt_model_id, args.mt_model_revision],
            },
        }
        write_prediction_artifacts(rows, summary, args.output_dir / args.run_name / name)
        summaries[name] = summary
    guard.seal(summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


def estimate_main(argv: list[str] | None = None) -> None:
    from .telemetry import RuntimeMeasurement, estimate_full_run

    parser = argparse.ArgumentParser(description="Project a full run from pilot JSON")
    parser.add_argument("pilot_json", type=Path)
    parser.add_argument("--pilot-examples", type=int, required=True)
    parser.add_argument("--full-examples-per-epoch", type=int, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--gpu-usd-per-hour", type=float)
    args = parser.parse_args(argv)
    payload = json.loads(args.pilot_json.read_text(encoding="utf-8"))
    if isinstance(payload.get("runtime"), dict):
        payload = payload["runtime"]
    pilot = RuntimeMeasurement(
        wall_seconds=float(payload["wall_seconds"]),
        examples=int(payload["examples"]),
        audio_seconds=float(payload["audio_seconds"]),
        gpu_count=int(payload.get("gpu_count", 0)),
        peak_vram_bytes=payload.get("peak_vram_bytes"),
    )
    result = estimate_full_run(
        pilot,
        pilot_examples=args.pilot_examples,
        full_examples_per_epoch=args.full_examples_per_epoch,
        epochs=args.epochs,
        gpu_usd_per_hour=args.gpu_usd_per_hour,
    )
    print(json.dumps(result, indent=2))


def smoke_train_main(argv: list[str] | None = None) -> None:
    """Run three CPU-safe optimizer steps through the real Whisper Trainer path."""
    from datetime import UTC, datetime

    import numpy as np
    from datasets import Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    from .hardware import set_reproducible_seed
    from .telemetry import RuntimeTimer, directory_size_bytes, write_json_artifact
    from .training import SpeechSeq2SeqCollator

    parser = argparse.ArgumentParser(description="Run a three-step Whisper training smoke")
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument(
        "--revision", default="169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--source-id")
    args = parser.parse_args(argv)
    if (args.audio is None) != (args.target is None):
        raise SystemExit("--audio and --target must be provided together")
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    set_reproducible_seed(args.seed)
    run_id = datetime.now(UTC).strftime("training-%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    checkpoint_dir = run_dir / "checkpoint"
    processor = WhisperProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        language="Hausa",
        task="translate",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.revision
    )
    model.freeze_encoder()
    model.generation_config.language = "hausa"
    model.generation_config.task = "translate"
    rows = []
    if args.audio:
        from .audio import load_audio

        samples, sample_rate = load_audio(args.audio)
        audio_seconds = samples.size / sample_rate
        rows = [
            {"audio": {"path": str(args.audio)}, "target_text": args.target}
            for _ in range(4)
        ]
        label_provenance = "user-supplied aligned English reference"
    else:
        for index, target in enumerate(
            ["Good morning.", "Thank you.", "The meeting starts today.", "We are ready."]
        ):
            seconds = 1.0 + index * 0.05
            time_axis = np.arange(round(16_000 * seconds), dtype=np.float32) / 16_000
            audio = 0.01 * np.sin(2 * np.pi * (180 + index * 20) * time_axis)
            rows.append(
                {
                    "audio": {"array": audio, "sampling_rate": 16_000},
                    "target_text": target,
                }
            )
        audio_seconds = 1.05
        label_provenance = "synthetic English strings paired with synthetic tones"
    dataset = Dataset.from_list(rows)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_steps=args.steps,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SpeechSeq2SeqCollator(
            processor=processor,
            target_column="target_text",
        ),
        processing_class=processor,
    )
    with RuntimeTimer() as timer:
        result = trainer.train()
    trainer.save_model(str(checkpoint_dir))
    processor.save_pretrained(str(checkpoint_dir))
    measurement = timer.measurement(
        examples=args.steps, audio_seconds=audio_seconds * args.steps
    )
    summary = {
        "status": "passed",
        "purpose": "API/forward/backward smoke only; not a scientific experiment",
        "steps": trainer.state.global_step,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "task": "translate",
        "target_language": "English",
        "labels": label_provenance,
        "source_id": args.source_id,
        "checkpoint": str(checkpoint_dir),
        "checkpoint_size_bytes": directory_size_bytes(checkpoint_dir),
        "train_metrics": result.metrics,
        "runtime": measurement.to_dict(),
        "transformers_version": __import__("transformers").__version__,
    }
    write_json_artifact(run_dir / "summary.json", summary)
    print(json.dumps({**summary, "artifact": str(run_dir / "summary.json")}, indent=2))
