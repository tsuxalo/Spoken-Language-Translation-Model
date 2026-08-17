"""Preflight checks for the final Hausa speech-translation GPU experiments.

Run before any expensive ASR generation or LoRA training.

Examples:
    python experiments/gpu_preflight.py --stage setup --require-cuda
    python experiments/gpu_preflight.py --stage train --require-cuda
    python experiments/gpu_preflight.py --stage evaluate --require-cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_SCRIPTS = [
    "experiments/prepare_naijas2st_pairs.py",
    "experiments/generate_asr_noise.py",
    "experiments/make_research_splits.py",
    "experiments/train_error_aware_controlled.py",
    "experiments/evaluate_research_suite.py",
    "analysis/analyze_predictions.py",
]

TRAIN_FILES = [
    "experiments/generated/research_train.jsonl",
    "experiments/generated/research_val.jsonl",
]

EVAL_FILES = [
    "experiments/generated/naijas2st_dev_whisper_all.jsonl",
    "outputs/final/clean",
    "outputs/final/noisy",
    "outputs/final/mixed",
]

CORE_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "sacrebleu",
    "jiwer",
    "soundfile",
]


def package_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"IMPORT FAILED: {exc}"
    return str(getattr(module, "__version__", "unknown"))


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "not found"
    text = (result.stdout or result.stderr).strip()
    return text or f"exit={result.returncode}"


def path_status(path_str: str) -> dict:
    path = Path(path_str)
    return {
        "path": path_str,
        "exists": path.exists(),
        "type": "directory" if path.is_dir() else "file" if path.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["setup", "train", "evaluate"],
        default="setup",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if PyTorch cannot see a CUDA GPU.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save the full preflight report.",
    )
    args = parser.parse_args()

    import torch

    report = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "stage": args.stage,
        "packages": {name: package_version(name) for name in CORE_PACKAGES},
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_build": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
        },
        "disk_free_gb": round(shutil.disk_usage(".").free / (1024**3), 2),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status": run_command(["git", "status", "--short"]),
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ]
        ),
        "required_paths": [],
    }

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        report["cuda"]["devices"] = devices

    required_paths = list(REQUIRED_SCRIPTS)
    if args.stage in {"train", "evaluate"}:
        required_paths.extend(TRAIN_FILES)
    if args.stage == "evaluate":
        required_paths.extend(EVAL_FILES)

    report["required_paths"] = [path_status(path) for path in required_paths]

    print("=" * 72)
    print("FINAL CAPSTONE GPU PREFLIGHT")
    print("=" * 72)
    print(f"Stage:             {args.stage}")
    print(f"Python:            {report['python']}")
    print(f"Executable:        {report['executable']}")
    print(f"Git commit:        {report['git_commit']}")
    print(f"Disk free:         {report['disk_free_gb']} GB")
    print(f"CUDA available:    {report['cuda']['available']}")
    print(f"PyTorch CUDA build:{report['cuda']['torch_cuda_build']}")
    print(f"NVIDIA status:     {report['nvidia_smi']}")
    print()

    print("Package versions")
    print("----------------")
    for name, version in report["packages"].items():
        print(f"{name:14} {version}")
    print()

    print("Required paths")
    print("--------------")
    missing_paths = []
    for item in report["required_paths"]:
        marker = "OK" if item["exists"] else "MISSING"
        print(f"[{marker:7}] {item['path']}")
        if not item["exists"]:
            missing_paths.append(item["path"])

    failures = []

    failed_imports = [
        name
        for name, version in report["packages"].items()
        if version.startswith("IMPORT FAILED")
    ]
    if failed_imports:
        failures.append(
            "Package imports failed: " + ", ".join(failed_imports)
        )

    if missing_paths:
        failures.append(
            "Required files/directories are missing: "
            + ", ".join(missing_paths)
        )

    if args.require_cuda and not torch.cuda.is_available():
        failures.append(
            "CUDA is required for this run, but torch.cuda.is_available() is False."
        )

    if report["disk_free_gb"] < 20:
        print()
        print("WARNING: Less than 20 GB of free disk space is available.")

    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Saved report: {path}")

    print()
    if failures:
        print("PREFLIGHT FAILED")
        print("----------------")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PREFLIGHT PASSED")
    print("Safe to proceed to the requested stage.")


if __name__ == "__main__":
    main()
