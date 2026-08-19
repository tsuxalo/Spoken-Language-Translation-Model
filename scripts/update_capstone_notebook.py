"""Rebuild the capstone notebook's comparison sections deterministically."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "capstone_demo.ipynb"


def _source(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def _cell_id(cell_type: str, text: str) -> str:
    payload = f"{cell_type}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id("markdown", text),
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    ast.parse(text)
    return {
        "cell_type": "code",
        "id": _cell_id("code", text),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    badge = markdown(
        r"""
[![Open PR preview in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsuxalo/Spoken-Language-Translation-Model/blob/codex/c1-integration/capstone_demo.ipynb)

> During review, this badge opens the integration branch. The setup cell automatically prefers `main` once the required C1 files are merged there and otherwise uses `codex/c1-integration`.
"""
    )

    eda_markdown = markdown(
        r"""
## 2. Audio Exploratory Data Analysis
"""
    )

    eda_code = code(
        r"""
import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from datasets import Audio as HFAudio
from datasets import load_dataset
from IPython.display import Audio, display

SAMPLING_RATE = 16_000
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"

# Mozilla moved Common Voice off the Hub in 2025, so this EDA uses FLEURS.
ds = load_dataset("google/fleurs", "ha_ng", split="train", revision=FLEURS_REVISION)
ds = ds.cast_column("audio", HFAudio(sampling_rate=SAMPLING_RATE, decode=False))
samples = [ds[index] for index in range(3)]

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for index, sample in enumerate(samples):
    audio_source = (
        io.BytesIO(sample["audio"]["bytes"])
        if sample["audio"]["bytes"]
        else sample["audio"]["path"]
    )
    audio_array, sampling_rate = sf.read(audio_source)
    print(f"Sample {index + 1}: {sample['raw_transcription']}")
    display(Audio(audio_array, rate=sampling_rate))

    librosa.display.waveshow(audio_array, sr=sampling_rate, ax=axes[index, 0])
    axes[index, 0].set_title(f"Sample {index + 1} — Waveform")
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sampling_rate,
        n_mels=80,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(
        mel_spec_db,
        sr=sampling_rate,
        x_axis="time",
        y_axis="mel",
        ax=axes[index, 1],
    )
    axes[index, 1].set_title(f"Sample {index + 1} — Mel-Spectrogram")

plt.tight_layout()
plt.show()
"""
    )

    eda_explanation = markdown(
        r"""
**What this is showing:** each row above is one Hausa clip — an audio player you can listen to, its **waveform** (left: loudness over time), and its **mel-spectrogram** (right: frequency energy over time). Whisper consumes mel features; C1 instead consumes the prepared waveform through its Wav2Vec2 processor.
"""
    )

    data_pipeline_markdown = markdown(
        r"""
## 3. Data Pipeline Verification

The scored C1 path uses the tested waveform contract in `direct_c1.prepare_audio`. The shared FLEURS cell below exercises the same helper for every system so decoded duration, mono conversion, and 16 kHz resampling are consistent.
"""
    )

    setup = code(
        r"""
# Colab setup: clone the repository so local modules and aggregate artifacts
# are available, then install a Colab-safe C1/comparison overlay. PyTorch and
# core compiled scientific packages are intentionally supplied by Colab.
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

IN_COLAB = "google.colab" in sys.modules
REPOSITORY_URL = "https://github.com/tsuxalo/Spoken-Language-Translation-Model.git"
REPOSITORY_RAW_URL = (
    "https://raw.githubusercontent.com/tsuxalo/Spoken-Language-Translation-Model"
)
REPOSITORY_CANDIDATE_REFS = ("main", "codex/c1-integration")
BINARY_DISTRIBUTIONS = ("numpy", "scipy", "pandas", "matplotlib", "pyarrow")


def repository_ref_ready(ref):
    request = urllib.request.Request(
        f"{REPOSITORY_RAW_URL}/{ref}/requirements-colab.txt",
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return True
    except HTTPError as error:
        if error.code == 404:
            return False
        raise


def distribution_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


if IN_COLAB:
    requested_ref = os.environ.get("SLT_REPOSITORY_REF")
    candidate_refs = (requested_ref,) if requested_ref else REPOSITORY_CANDIDATE_REFS
    REPOSITORY_REF = next(
        (ref for ref in candidate_refs if repository_ref_ready(ref)),
        None,
    )
    if REPOSITORY_REF is None:
        raise RuntimeError(
            "Neither main nor codex/c1-integration contains the Colab bootstrap."
        )

    repo_root = Path(
        os.environ.get(
            "SLT_NOTEBOOK_REPO_ROOT",
            "/content/Spoken-Language-Translation-Model",
        )
    )
    if repo_root.exists() and not (repo_root / ".git").is_dir():
        raise RuntimeError(f"Existing Colab path is not a Git checkout: {repo_root}")

    if (repo_root / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(repo_root), "remote", "set-url", "origin", REPOSITORY_URL],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", REPOSITORY_REF],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_root), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    else:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--branch", REPOSITORY_REF,
                "--single-branch", REPOSITORY_URL, str(repo_root),
            ],
            check=True,
        )

    os.chdir(repo_root)

    required_paths = (
        Path("requirements-colab.txt"),
        Path("direct_c1.py"),
        Path("artifacts/comparison-v2/common_manifest_metrics.json"),
        Path("artifacts/gpu-handoff/evaluation_metrics.json"),
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise RuntimeError(
            f"Selected repository ref {REPOSITORY_REF!r} is incomplete: {missing_paths}"
        )

    checked_out_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    print(f"Using repository ref {REPOSITORY_REF}: {checked_out_commit}")

    binary_versions_before = distribution_versions(BINARY_DISTRIBUTIONS)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade-strategy",
            "only-if-needed",
            "-r",
            "requirements-colab.txt",
        ],
        check=True,
    )
    binary_versions_after = distribution_versions(BINARY_DISTRIBUTIONS)
    changed_binary_packages = {
        name: (binary_versions_before[name], binary_versions_after[name])
        for name in BINARY_DISTRIBUTIONS
        if binary_versions_before[name] != binary_versions_after[name]
    }
    if changed_binary_packages:
        print("Compiled packages changed during setup:", changed_binary_packages)
        print("Colab is restarting once. Reconnect, then choose Runtime > Run all.")
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import datasets, librosa, numpy, pandas, pyarrow, scipy, soundfile, transformers; "
                "print('Colab dependency import check passed')"
            ),
        ],
        check=True,
    )
    print("Colab binary stack remained compatible:", binary_versions_after)
else:
    REPOSITORY_REF = os.environ.get("SLT_REPOSITORY_REF", "main")

ARTIFACT_RAW_BASE = (
    f"{REPOSITORY_RAW_URL}/{REPOSITORY_REF}/artifacts/comparison-v2"
)
GPU_ARTIFACT_RAW_BASE = (
    f"{REPOSITORY_RAW_URL}/{REPOSITORY_REF}/artifacts/gpu-handoff"
)


def load_artifact(relative_path):
    local_path = Path("artifacts/comparison-v2") / relative_path
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(f"{ARTIFACT_RAW_BASE}/{relative_path}") as response:
        return json.load(response)


def load_gpu_artifact(relative_path):
    local_path = Path("artifacts/gpu-handoff") / relative_path
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(f"{GPU_ARTIFACT_RAW_BASE}/{relative_path}") as response:
        return json.load(response)
"""
    )

    architecture_markdown = markdown(
        r"""
## 1. Architecture Overview

This project now preserves three deployable Hausa-speech-to-English paths:

1. **Cascade:** audio → fine-tuned Whisper Hausa ASR → written Hausa → base NLLB-200 → English.
2. **Historical direct pilot:** audio → Whisper-small + a Hausa→English LoRA adapter → English.
3. **C1 direct model:** audio → XLS-R/Wav2Vec2 speech encoder → cross-modal connection → mBART-style decoder → English.

C1 is not Whisper, not a PEFT adapter, and does not call an external Hausa ASR or text-MT model. It is direct at the complete inference-graph level.
"""
    )

    architecture_plot = code(
        r"""
import matplotlib.pyplot as plt

flows = [
    ("Cascade", ["Hausa audio", "Whisper ASR", "Hausa text", "NLLB-200", "English"]),
    ("Historical direct pilot", ["Hausa audio", "Whisper + LoRA", "English"]),
    ("C1 direct", ["Hausa audio", "XLS-R encoder", "mBART decoder", "English"]),
]
fig, axes = plt.subplots(3, 1, figsize=(12, 5.8))
for ax, (label, nodes) in zip(axes, flows):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = [0.08 + index * 0.84 / max(len(nodes) - 1, 1) for index in range(len(nodes))]
    for index, (x, node) in enumerate(zip(xs, nodes)):
        ax.text(x, 0.5, node, ha="center", va="center", bbox={"boxstyle": "round", "fc": "#e8f1fb"})
        if index:
            ax.annotate("", xy=(x - 0.055, 0.5), xytext=(xs[index - 1] + 0.055, 0.5), arrowprops={"arrowstyle": "->"})
    ax.text(0.0, 0.9, label, weight="bold", transform=ax.transAxes)
fig.suptitle("Three deployable inference graphs")
plt.tight_layout()
plt.show()
"""
    )

    telemetry_markdown = markdown(
        r"""
## 4. Training Telemetry

The panels below read checked-in aggregate artifacts. Losses are separated by architecture and objective: comparing their absolute magnitudes would not establish which system translates better. The direct pilot has only scalar losses, so it is shown as a table rather than an invented curve. Base NLLB is a pretrained baseline and was not fine-tuned in the primary comparison.
"""
    )

    telemetry_code = code(
        r"""
import matplotlib.pyplot as plt
import pandas as pd

histories = load_artifact("training_histories.json")
by_system = {item["system"]: item for item in histories["series"]}
asr = by_system["cascade_asr"]
c1 = by_system["direct_c1"]
pilot = by_system["direct_pilot"]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].plot([p["step"] for p in asr["training_loss"]], [p["loss"] for p in asr["training_loss"]], label="training loss")
axes[0].plot([p["step"] for p in asr["validation"]], [p["loss"] for p in asr["validation"]], marker="o", label="evaluation loss")
axes[0].set_title("Hausa Whisper ASR")
axes[0].set_xlabel("step")
axes[0].set_ylabel("loss")
axes[0].legend()

axes[1].plot([p["step"] for p in asr["validation"]], [p["wer"] for p in asr["validation"]], marker="o", color="darkorange")
axes[1].set_title("Hausa Whisper ASR evaluation WER")
axes[1].set_xlabel("step")
axes[1].set_ylabel("WER (%)")

axes[2].plot([p["step"] for p in c1["training_loss"]], [p["loss"] for p in c1["training_loss"]], label="25-step mean training loss")
axes[2].plot([p["step"] for p in c1["validation"]], [p["loss"] for p in c1["validation"]], marker="o", label="validation loss")
axes[2].axvline(185, linestyle="--", color="gray", label="selected checkpoint")
axes[2].set_title("C1 XLS-R→mBART")
axes[2].set_xlabel("optimizer step")
axes[2].set_ylabel("loss")
axes[2].legend()
plt.tight_layout()
plt.show()

display(pd.DataFrame([
    {
        "system": "Historical direct pilot", "model": pilot["model"],
        "dataset": pilot["dataset"], "run/step": pilot["run"],
        **pilot["scalar_only"], "provenance": pilot["source_artifact_or_document"],
    },
    {
        "system": "Base NLLB-600M", "model": "facebook/nllb-200-distilled-600M",
        "dataset": "pretraining data outside this project",
        "run/step": "pretrained baseline—not fine-tuned in this experiment",
        "train_loss_average": None, "validation_loss": None,
        "provenance": "model contract; no project training artifact",
    },
]))
print(
    "ASR caption:", asr["model"], asr["dataset"], asr["run"],
    f"steps {asr['training_loss'][0]['step']}–{asr['training_loss'][-1]['step']}",
    asr["source_artifact_or_document"],
)
print(
    "C1 caption:", c1["model"], c1["dataset"], c1["run"],
    f"steps {c1['training_loss'][0]['step']}–{c1['training_loss'][-1]['step']}",
    c1["source_artifact_or_document"],
)
"""
    )

    cascade_markdown = markdown(
        r"""
## 5. Interactive Inference — Cascade

The cascade remains the fine-tuned Hausa Whisper model followed by base `facebook/nllb-200-distilled-600M`. The pinned helper below is called by the shared qualitative demonstration in Section 6B. It unloads ASR before loading MT, and unloads the full cascade before the next deployable system.
"""
    )

    cascade_code = code(
        r"""
import gc
import time

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from direct_c1 import prepare_audio

ASR_MODEL_ID = "nahomazmach/whisper-small-ha"
ASR_REVISION = "c4e2b47d88ae8b3ee0a605e09863b93aafca72e3"
MT_MODEL_ID = "facebook/nllb-200-distilled-600M"
MT_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unload_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_cascade(audio, sampling_rate):
    prepared = prepare_audio(audio, sampling_rate)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL_ID, revision=ASR_REVISION)
    asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_ID, revision=ASR_REVISION).to(DEVICE).eval()
    asr_load = time.perf_counter() - started
    features = asr_processor.feature_extractor(prepared.samples, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    started = time.perf_counter()
    with torch.inference_mode():
        ids = asr_model.generate(features, language="hausa", task="transcribe")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    asr_seconds = time.perf_counter() - started
    hausa = asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    del ids, features, asr_model, asr_processor
    unload_models()

    started = time.perf_counter()
    mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_ID, revision=MT_REVISION)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL_ID, revision=MT_REVISION).to(DEVICE).eval()
    mt_load = time.perf_counter() - started
    mt_tokenizer.src_lang = "hau_Latn"
    mt_inputs = mt_tokenizer(hausa, return_tensors="pt").to(DEVICE)
    started = time.perf_counter()
    with torch.inference_mode():
        ids = mt_model.generate(**mt_inputs, forced_bos_token_id=mt_tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=256)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mt_seconds = time.perf_counter() - started
    english = mt_tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    del ids, mt_inputs, mt_model, mt_tokenizer
    unload_models()
    return {
        "system": "cascade_real_asr", "hausa_asr": hausa, "prediction": english,
        "device": DEVICE, "load_time_seconds": asr_load + mt_load,
        "inference_time_seconds": asr_seconds + mt_seconds,
        "real_time_factor": (asr_seconds + mt_seconds) / prepared.duration_seconds,
        "peak_gpu_memory_mb": peak,
    }
"""
    )

    direct_pilot_markdown = markdown(
        r"""
## 6A — Original 256-example Whisper direct pilot

This historical system remains `openai/whisper-small` plus the published LoRA adapter `nahomazmach/whisper-small-ha-en-direct-pilot`. It uses Whisper's `task="translate"` path and directly generates English. The original pilot trained on 256 NaijaS2ST examples for 50 steps; its historical BLEU/chrF++ came from a separate train-derived pilot validation set.

Hub-based inference is reproducible from `main`, while reproducing the complete historical training run still depends on the diverged `feature/direct-s2tt`-era code and an unpreserved `_pairs.json` membership artifact. Exact pilot training overlap is therefore **UNVERIFIED**.
"""
    )

    direct_pilot_code = code(
        r"""
from peft import PeftModel

DIRECT_BASE_ID = "openai/whisper-small"
DIRECT_BASE_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
DIRECT_ADAPTER_ID = "nahomazmach/whisper-small-ha-en-direct-pilot"
DIRECT_ADAPTER_REVISION = "a91a4a1155c574a24226de53f053e08b6446806d"


def run_direct_pilot(audio, sampling_rate):
    prepared = prepare_audio(audio, sampling_rate)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(DIRECT_ADAPTER_ID, revision=DIRECT_ADAPTER_REVISION)
    base = WhisperForConditionalGeneration.from_pretrained(DIRECT_BASE_ID, revision=DIRECT_BASE_REVISION)
    model = PeftModel.from_pretrained(base, DIRECT_ADAPTER_ID, revision=DIRECT_ADAPTER_REVISION).to(DEVICE).eval()
    load_seconds = time.perf_counter() - started
    features = processor.feature_extractor(prepared.samples, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    started = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(features, language="hausa", task="translate", max_new_tokens=225, num_beams=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    english = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    del ids, features, model, base, processor
    unload_models()
    return {
        "system": "direct_pilot", "prediction": english, "device": DEVICE,
        "load_time_seconds": load_seconds, "inference_time_seconds": inference_seconds,
        "real_time_factor": inference_seconds / prepared.duration_seconds,
        "peak_gpu_memory_mb": peak,
    }
"""
    )

    c1_markdown = markdown(
        r"""
## 6B — C1 full-data XLS-R→mBART direct model

1. The raw Hausa waveform is decoded.
2. Audio is downmixed and resampled to 16 kHz without truncation, chunking, or silent amplitude normalization.
3. `Wav2Vec2Processor` converts the waveform into encoder-ready numeric input and an attention mask.
4. The XLS-R/Wav2Vec2 encoder transforms sound patterns into contextual speech representations.
5. The cross-modal connection passes those representations to the text decoder.
6. The mBART-style decoder generates English tokens autoregressively with C1's frozen beam configuration.
7. The tokenizer decodes token IDs into readable English.

C1 is not Whisper, is not a PEFT adapter, does not call an external Hausa ASR model, and does not call an external text-MT model. Run the standalone CLI with `python direct_c1.py path/to/hausa.wav`.

**Duration scope:** C1's frozen evaluation contract covers clips no longer than 30 seconds. Longer clips are excluded from scored evaluation even if the underlying architecture might technically accept them.
"""
    )

    qualitative_markdown = markdown(
        r"""
### Shared pinned FLEURS example

The next cell runs one fixed public FLEURS Hausa rendition through the cascade, historical direct pilot, and C1. The dataset revision, semantic example ID, and audio filename are all pinned. Systems load and unload sequentially.

**Qualitative demonstration only. FLEURS does not provide a gold English translation for this clip. No BLEU or chrF++ is calculated.**
"""
    )

    qualitative_code = code(
        r"""
import io

import pandas as pd
import soundfile as sf
from datasets import Audio as HFAudio
from datasets import load_dataset
from IPython.display import Audio, display

from comparison_v2 import atomic_write_json
from direct_c1 import C1Runtime

FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_EXAMPLE_ID = 1856
FLEURS_AUDIO_FILE = "5153420622149111720.wav"

test_ds = load_dataset("google/fleurs", "ha_ng", split="test", revision=FLEURS_REVISION)
test_ds = test_ds.cast_column("audio", HFAudio(sampling_rate=16000, decode=False))
matches = test_ds.filter(
    lambda row: int(row["id"]) == FLEURS_EXAMPLE_ID
    and Path(row["audio"]["path"]).name == FLEURS_AUDIO_FILE
)
if len(matches) != 1:
    raise RuntimeError(f"Expected one pinned FLEURS rendition, found {len(matches)}")
sample = matches[0]
audio_source = io.BytesIO(sample["audio"]["bytes"]) if sample["audio"]["bytes"] else sample["audio"]["path"]
audio, sampling_rate = sf.read(audio_source, dtype="float32")
display(Audio(audio, rate=sampling_rate))
print("FLEURS Hausa transcription:", sample["raw_transcription"])

results = []
results.append(run_cascade(audio, sampling_rate))
results.append(run_direct_pilot(audio, sampling_rate))
with C1Runtime(device=DEVICE) as c1_runtime:
    c1_prediction = c1_runtime.translate_array(audio, sampling_rate)
    c1_telemetry = c1_runtime.last_telemetry.to_dict()
    results.append({
        "system": "direct_c1", "prediction": c1_prediction, "device": c1_telemetry["device"],
        "load_time_seconds": c1_telemetry["load_time_seconds"],
        "inference_time_seconds": c1_telemetry["inference_time_seconds"],
        "real_time_factor": c1_telemetry["real_time_factor"],
        "peak_gpu_memory_mb": c1_telemetry["peak_cuda_memory_mb"],
    })

private_result = {
    "label": "Qualitative demonstration only; no gold English translation.",
    "dataset": "google/fleurs", "dataset_revision": FLEURS_REVISION,
    "example_id": FLEURS_EXAMPLE_ID, "audio_file": FLEURS_AUDIO_FILE,
    "hausa_transcription": sample["raw_transcription"], "systems": results,
}
atomic_write_json("artifacts/comparison-v2/private/qualitative_fleurs_1856.json", private_result)
display(pd.DataFrame(results)[[
    "system", "prediction", "inference_time_seconds", "real_time_factor", "peak_gpu_memory_mb", "device"
]])
"""
    )

    warning = markdown(
        r"""
> **Research-only warning:** These systems may produce fluent but semantically incorrect English. Names, numbers, dates, negation, and unusual audio require additional caution.
"""
    )

    results_markdown = markdown(
        r"""
## 7. Translation-quality evidence — kept in scope

The artifact-backed rows below deliberately retain their evaluation-set labels. The sealed cascade result, 128-example direct-pilot validation result, and 1,037-example C1 development result come from different memberships. They are not an apples-to-apples ranking, and C1's validation membership influenced model selection.
"""
    )

    results_code = code(
        r"""
import numpy as np

quality = load_artifact("historical_metrics.json")
quality_df = pd.DataFrame(quality["rows"])
display(quality_df[[
    "system", "system_type", "evaluation_set", "number_of_examples",
    "BLEU", "chrF++", "evidence_scope", "training_or_selection_influence"
]])

panels = [
    ("Historical sealed official dev", ["cascade_real_asr", "gold_hausa_mt_oracle"]),
    ("Historical train-derived pilot val", ["direct_pilot"]),
    ("C1 internal development val", ["direct_c1"]),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (title, names) in zip(axes, panels):
    subset = quality_df[quality_df["system"].isin(names)]
    x = np.arange(len(subset))
    ax.bar(x - 0.18, subset["BLEU"], 0.36, label="BLEU")
    ax.bar(x + 0.18, subset["chrF++"], 0.36, label="chrF++")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["system"], rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.legend()
fig.suptitle("Separate evidence panels — scales do not create a common benchmark")
plt.tight_layout()
plt.show()
"""
    )

    comparison_markdown = markdown(
        r"""
### Full common-membership development evaluation

The exact C1 validation membership was recovered and frozen before inference: 1,037 renditions in 543 `alignment_id` clusters from NaijaS2ST official `train` / C1 project-validation membership. All three systems produced a successful nonempty prediction for every row. The intervals and paired deltas use 1,000 cluster-bootstrap replicates. Historical direct-pilot training overlap could not be fully audited, so this evaluation must not be presented as leakage-free. C1's development membership also influenced its checkpoint/decoding selection, so this is not independent test evidence.

The fresh exported C1 package result below is separate from the historical C1 panel above. The IDs and references match, but the exported float32 package did not bitwise reproduce the historical local-adapter/bfloat16 predictions; see `full_development_provenance.json` for the reconciliation audit.
"""
    )

    comparison_code = code(
        r"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    common_eval = load_artifact("full_development_metrics.json")
except (FileNotFoundError, HTTPError):
    common_eval = None

if common_eval is None:
    print("No full common-membership aggregate is checked in.")
    print("Prepare it with: python comparison_v2.py prepare-full-manifest <local-validation.jsonl>")
else:
    print(common_eval["label"])
    print(common_eval["limitation"])
    metrics = pd.DataFrame(common_eval["metrics"]).T
    display(metrics[[
        "BLEU", "chrF++", "inference_time_seconds_mean",
        "inference_time_seconds_median", "real_time_factor_mean",
        "peak_gpu_memory_mb", "failure_rate", "empty_output_rate",
        "repeated_3gram_rate", "number_mismatch_rate", "negation_omission_rate",
    ]])

    systems = list(common_eval["metrics"])
    intervals = common_eval["bootstrap"]["system_intervals"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("BLEU", "chrF++")):
        values = np.array([common_eval["metrics"][system][metric] for system in systems])
        lower = np.array([intervals[system][metric][0] for system in systems])
        upper = np.array([intervals[system][metric][1] for system in systems])
        ax.bar(systems, values, color="#4c78a8")
        ax.errorbar(
            systems,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            capsize=5,
            color="black",
        )
        ax.set_title(f"{metric} with 95% paired cluster-bootstrap interval")
        ax.tick_params(axis="x", rotation=18)
        ax.set_ylabel("corpus score")
    fig.suptitle(
        f"{common_eval['label']} "
        f"(n={common_eval['membership']['number_of_examples']}; "
        f"clusters={common_eval['membership']['alignment_clusters']})"
    )
    plt.tight_layout()
    plt.show()

    efficiency = metrics[[
        "load_time_seconds", "inference_time_seconds_mean",
        "inference_time_seconds_median", "real_time_factor_mean",
        "peak_gpu_memory_mb", "failure_rate", "empty_output_rate",
    ]]
    display(efficiency)
    print("Observed inference hours:", common_eval["observed_total_inference_hours"])
    print("Paired deltas:", common_eval["bootstrap"]["paired_deltas"])
    print("Human-review queue:", common_eval["qualitative_review_queue"])
"""
    )

    gpu_results_markdown = markdown(
        r"""
## 7B. GPU error-aware MT — separate official-dev panel

This panel answers a different question from the C1 development comparison above. Every system here receives the same **Whisper-produced Hausa text**:

```text
Hausa audio → fixed Whisper Hausa ASR → Hausa text → MT system → English
```

The checked aggregates cover 1,500 official NaijaS2ST `dev` utterances in 500 alignment clusters. Mixed has the highest BLEU/chrF++ point estimates and noisy has the highest SSA-COMET point estimate. The predeclared analysis did not save a direct noisy-versus-mixed paired comparison, so the notebook makes no statistical-superiority claim between them. This was one training seed; cluster-bootstrap intervals measure evaluation-sample uncertainty, not training-seed variability.

Official `dev` has now been observed and is not an untouched set for future tuning. C1 and GPU results remain in separate panels because their input modality, graph, membership, selection history, and scientific question differ.
"""
    )

    gpu_results_code = code(
        r"""
gpu_eval = load_gpu_artifact("evaluation_metrics.json")
gpu_bootstrap = load_gpu_artifact("bootstrap_confidence_intervals.json")
gpu_paired = load_gpu_artifact("paired_deltas.json")
gpu_bins = load_gpu_artifact("wer_bins.json")

gpu_metrics = pd.DataFrame(gpu_eval["systems"]).T
display(gpu_metrics.rename(columns={"bleu": "BLEU", "chrf_pp": "chrF++", "ssa_comet": "SSA-COMET"}))
print("Evaluation scale:", gpu_eval["evaluation_scale"])
print("Gold-Hausa oracle:", gpu_eval["gold_hausa_oracle"])
print("Direct noisy-vs-mixed paired result saved:", gpu_paired["direct_noisy_vs_mixed_saved"])

systems = list(gpu_eval["systems"])
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (metric, label) in zip(
    axes,
    (("bleu", "BLEU"), ("chrf_pp", "chrF++"), ("ssa_comet", "SSA-COMET")),
):
    values = [gpu_eval["systems"][system][metric] for system in systems]
    intervals = {
        row["system"]: (row["ci_2.5"], row["ci_97.5"])
        for row in gpu_bootstrap["intervals"]
        if row["metric"] == metric
    }
    lower = np.array([intervals[system][0] for system in systems])
    upper = np.array([intervals[system][1] for system in systems])
    values_array = np.array(values)
    ax.bar(systems, values_array, color="#6c5ce7")
    ax.errorbar(
        systems,
        values_array,
        yerr=np.vstack([values_array - lower, upper - values_array]),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_title(label)
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("GPU fixed-ASR MT official-dev results — not a C1 leaderboard")
plt.tight_layout()
plt.show()

wer_bins = pd.DataFrame(gpu_bins["bins"])
fig, ax = plt.subplots(figsize=(8, 4.8))
for system in systems:
    subset = wer_bins[wer_bins["system"] == system]
    ax.plot(subset["mean_wer"], subset["mean_sentence_chrf"], marker="o", label=system)
ax.set_xlabel("Mean Whisper Hausa ASR WER within bin (%)")
ax.set_ylabel("Mean sentence chrF++")
ax.set_title("Privacy-safe aggregate WER bins")
ax.legend(ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
"""
    )

    error_markdown = markdown(
        r"""
## 8. A Closer Look: Does More ASR Error Mean Worse Translation?

This historical 25-example error-propagation view is preserved, but its values now come from a checked-in artifact rather than a notebook array. Sentence metrics are noisy at this sample size; the plot is descriptive supporting evidence.
"""
    )

    error_code = code(
        r"""
error_data = load_artifact("error_propagation.json")
points = error_data["rows"]
wer = np.array([point["wer"] for point in points])
bleu = np.array([point["bleu"] for point in points])
chrf = np.array([point["chrf_pp"] for point in points])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, values, ylabel in ((axes[0], bleu, "Sentence BLEU"), (axes[1], chrf, "Sentence chrF++")):
    correlation = np.corrcoef(wer, values)[0, 1]
    ax.scatter(wer, values, alpha=0.75, edgecolor="k", linewidth=0.5)
    slope, intercept = np.polyfit(wer, values, 1)
    xs = np.linspace(wer.min(), wer.max(), 50)
    ax.plot(xs, slope * xs + intercept, color="darkorange", label=f"trend (r={correlation:.2f})")
    ax.set_xlabel("ASR Word Error Rate (%)")
    ax.set_ylabel(ylabel)
    ax.legend()
fig.suptitle(f"Historical cascade error propagation (n={len(points)}; artifact-backed)")
plt.tight_layout()
plt.show()
print("Provenance:", error_data["source_artifact_or_document"])
"""
    )

    notebook["cells"] = [
        badge,
        setup,
        architecture_markdown,
        architecture_plot,
        eda_markdown,
        eda_code,
        eda_explanation,
        data_pipeline_markdown,
        telemetry_markdown,
        telemetry_code,
        cascade_markdown,
        cascade_code,
        direct_pilot_markdown,
        direct_pilot_code,
        c1_markdown,
        qualitative_markdown,
        qualitative_code,
        warning,
        results_markdown,
        results_code,
        comparison_markdown,
        comparison_code,
        gpu_results_markdown,
        gpu_results_code,
        error_markdown,
        error_code,
    ]
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Updated {NOTEBOOK} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
