"""Build the canonical, output-free Notebook 02 deterministically."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/02_direct_s2tt_training.ipynb"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip(), execution_count=None, outputs=[])


cells = [
    markdown(
        """
# Direct Hausa→English speech-to-text translation training

## 1. Purpose

This notebook implements the development-stage direct S2TT experiment: **16 kHz Hausa audio → one encoder-decoder model → English text**. It uses genuine aligned English `target_text` labels, a speaker-disjoint validation split derived from official NaijaS2ST `train`, and leaves official `dev` untouched for Notebook 03.

The notebook is safe when run from top to bottom: all model downloads and training/evaluation stages are disabled by default. A structural smoke verifies mechanics, while matched pilots and full training require separate explicit flags. A smoke success is not evidence of translation quality.

Output labels used below:

- **Measured artifact** — read from a real saved project file.
- **Live output** — appears only when this notebook is executed.
- **Expected output structure** — illustrative schema, never a measurement.
"""
    ),
    markdown(
        """
## 2. Four-notebook project map

Notebook 00 owns alignment and exact project split membership. Notebook 01 will own Hausa ASR and the ASR→NLLB cascade. This Notebook 02 owns direct-model selection and training. Notebook 03—not this notebook—will compare frozen systems on official held-out data.

`00 data → 01 ASR/cascade → 02 direct training → 03 protected final comparison`
"""
    ),
    code(
        """
NOTEBOOK_CONTRACT = {
    "input": "Notebook 00 schema-1.0 train/validation JSONL or seed-42 reconstruction",
    "output": "adapter + processor + validation predictions + telemetry + selection JSON",
    "official_dev_evaluated": False,
}
NOTEBOOK_CONTRACT
"""
    ),
    markdown(
        """
## 3. Direct S2TT definition

ASR maps Hausa audio to Hausa text. Zero-shot Whisper translation maps Hausa audio to English without project-specific English-label training. A cascade explicitly creates Hausa text before MT. **Supervised direct S2TT** optimizes Hausa audio against aligned English targets and has no intermediate Hausa transcript at inference.

Teacher forcing supplies the previous reference English token to the decoder during training. At inference, the decoder is autoregressive: each predicted English token conditions the next one.
"""
    ),
    markdown(
        """
## 4. Environment and Git setup

**Live output.** This cell may clone and install the repository. In Colab it only fast-forwards an existing checkout when the tree is clean; it stops on dirty work rather than discarding it. It never prints tokens or environment variables.
"""
    ),
    code(
        """
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/tsuxalo/Spoken-Language-Translation-Model.git"
REPO_REF = "feature/direct-s2tt"
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    REPO_ROOT = Path("/content/Spoken-Language-Translation-Model")
    if not (REPO_ROOT / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--branch", REPO_REF, "--single-branch", REPO_URL, str(REPO_ROOT)],
            check=True,
        )
    else:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        if status.strip():
            raise RuntimeError("Existing Colab checkout is dirty; preserve or remove it manually.")
        actual_remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if actual_remote.rstrip("/").removesuffix(".git") != REPO_URL.rstrip("/").removesuffix(".git"):
            raise RuntimeError(f"Unexpected origin: {actual_remote}")
        subprocess.run(["git", "fetch", "origin", REPO_REF], cwd=REPO_ROOT, check=True)
        local_ref = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{REPO_REF}"],
            cwd=REPO_ROOT,
            check=False,
        )
        if local_ref.returncode == 0:
            subprocess.run(["git", "switch", REPO_REF], cwd=REPO_ROOT, check=True)
        else:
            subprocess.run(
                ["git", "switch", "--track", "-c", REPO_REF, f"origin/{REPO_REF}"],
                cwd=REPO_ROOT, check=True,
            )
        subprocess.run(
            ["git", "merge", "--ff-only", f"origin/{REPO_REF}"], cwd=REPO_ROOT, check=True
        )
else:
    candidate = Path.cwd().resolve()
    REPO_ROOT = candidate.parent if candidate.name == "notebooks" else candidate
    if not (REPO_ROOT / "pyproject.toml").is_file():
        raise RuntimeError("Run this notebook from the repository root or notebooks directory.")

os.chdir(REPO_ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)], check=True)
for module_name in list(sys.modules):
    if module_name == "hausa_s2tt" or module_name.startswith("hausa_s2tt."):
        del sys.modules[module_name]

import peft
import torch
import transformers

import hausa_s2tt

branch = subprocess.run(
    ["git", "branch", "--show-current"], cwd=REPO_ROOT, check=True,
    capture_output=True, text=True,
).stdout.strip()
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
    capture_output=True, text=True,
).stdout.strip()
print({
    "package_path": hausa_s2tt.__file__, "branch": branch, "commit": commit,
    "python": sys.version.split()[0], "torch": torch.__version__,
    "transformers": transformers.__version__, "peft": peft.__version__,
    "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
})
"""
    ),
    code(
        """
# Every expensive stage is opt-in and separately reviewable.
RUN_STRUCTURAL_SMOKE = False
RUN_BASELINE_EVALUATION = False
RUN_MATCHED_PILOTS = False
FREEZE_ARCHITECTURE_SELECTION = False
HUMAN_REVIEW_COMPLETE = False
RUN_FULL_TRAINING = False
RESUME_FROM_CHECKPOINT = None
GPU_USD_PER_HOUR = None
QUALITATIVE_REVIEW = {
    "base": {"status": "not_reviewed"},
    "from_asr": {"status": "not_reviewed"},
}
"""
    ),
    markdown(
        """
## 5. Hardware and precision

Mixed precision reduces memory and can increase throughput. BF16 is selected only when CUDA reports support; otherwise CUDA uses FP16, while CPU/MPS uses FP32. Gradient checkpointing saves activation memory by recomputing some forward operations during backward, trading extra compute for lower peak memory.
"""
    ),
    code(
        """
from hausa_s2tt.hardware import hardware_snapshot, select_precision

precision = select_precision(torch)
hardware = hardware_snapshot()
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    hardware["gpu_name"] = properties.name
    hardware["gpu_vram_bytes"] = properties.total_memory
print({"precision": precision.to_dict(), "hardware": hardware})
"""
    ),
    markdown(
        """
## 6. Notebook 00 artifact validation

Notebook 00's manifest and JSONL files are authoritative when present. This check validates schema, pinned dataset/revision, generating Git metadata, official-train speaker split policy, seed 42, required fields, exact project membership, and zero speaker overlap. It does **not** load audio.

If the artifacts are absent, gated training reconstructs the split with the same seed and verifies counts against the revision-matched tracked audit. That path is labeled reconstruction, not artifact consumption.
"""
    ),
    code(
        """
from hausa_s2tt.datasets import load_pairing_artifacts, load_revision_matched_audit
from hausa_s2tt.revisions import NAIJA_DATASET_ID, NAIJA_REVISION

PAIRING_DIR = REPO_ROOT / "artifacts/data/naija_s2st"
TRACKED_AUDIT = REPO_ROOT / "reports/naija_s2st_audit_summary.json"
tracked_audit = load_revision_matched_audit(
    TRACKED_AUDIT,
    expected_dataset_id=NAIJA_DATASET_ID,
    expected_dataset_revision=NAIJA_REVISION,
)
if (PAIRING_DIR / "manifest.json").is_file():
    artifact_splits, pairing_manifest = load_pairing_artifacts(
        PAIRING_DIR,
        expected_dataset_id=NAIJA_DATASET_ID,
        expected_dataset_revision=NAIJA_REVISION,
        expected_seed=42,
    )
    print("Measured artifact", {
        "mode": "Notebook 00 exact membership",
        "counts": {name: len(rows) for name, rows in artifact_splits.items()},
        "generating_git_commit": pairing_manifest["git_commit"],
    })
else:
    print("Measured artifact", {
        "mode": "artifacts absent; gated run will reconstruct and verify",
        "accepted_pairs": tracked_audit["train"]["accepted_pairs"],
        "derived_seed_42_split": tracked_audit["derived_seed_42_split"],
    })
"""
    ),
    markdown(
        """
## 7. Held-out-data guard

NaijaS2ST official `dev` is reserved for Notebook 03. Notebook 02 uses only project train/validation derived from official `train`. No individual dev targets, predictions, metrics, prompts, or tuning signals are permitted here. Every saved run must say `official_dev_evaluated: false`.
"""
    ),
    code(
        """
from hausa_s2tt.config import load_config

BASE_PILOT_CONFIG = REPO_ROOT / "configs/direct_s2tt_pilot_base.yaml"
ASR_PILOT_CONFIG = REPO_ROOT / "configs/direct_s2tt_pilot_from_asr.yaml"
base_pilot = load_config(BASE_PILOT_CONFIG)
asr_pilot = load_config(ASR_PILOT_CONFIG)
for candidate in (base_pilot, asr_pilot):
    assert candidate.dataset.train_split == "train"
    assert candidate.dataset.validation_split == "derived_from_train"
    assert candidate.dataset.test_split == "dev"  # reserved name, never loaded by Trainer
    assert candidate.training.seed == 42
print({"official_source_loaded_by_training": "train only", "official_dev_evaluated": False})
"""
    ),
    markdown(
        """
## 8. Architecture candidates

Tier A considered Whisper full tuning, LoRA, freezing, both initializations, and a custom Whisper-encoder/NLLB-decoder bridge. Tier B screened SeamlessM4T-v2, OWSM/OWSM-CTC, AfriHuBERT, MMS, and W2v-BERT 2.0. Tier C screened MCAT, MSRT, speech-encoder/LLM projectors, streaming transducers, and speech-to-speech systems.

The evidence/parameter/license matrix is tracked in `reports/DIRECT_S2TT_ARCHITECTURE_DECISION.md`. Published claims and project estimates are labeled separately there.
"""
    ),
    code(
        """
ARCHITECTURE_REPORT = REPO_ROOT / "reports/DIRECT_S2TT_ARCHITECTURE_DECISION.md"
print({"architecture_report": str(ARCHITECTURE_REPORT), "exists": ARCHITECTURE_REPORT.is_file()})
"""
    ),
    markdown(
        """
## 9. Architecture decision

The provisional implementation is Whisper-small + English translation supervision + rank-16 LoRA + gradient checkpointing + hardware-aware precision. Larger/newer models were not chosen merely for recency: several require 1B–27B parameters, different toolchains, noncommercial licenses, unverified Hausa coverage, or a new modality bridge.

The initialization remains an experimental question. The two pilot files must differ only in model identity, revision, output path, and run name.
"""
    ),
    code(
        """
from hausa_s2tt.direct_s2tt import assert_matched_direct_configs

assert_matched_direct_configs(base_pilot, asr_pilot)
print({
    "candidate_A": [base_pilot.model.id, base_pilot.model.revision],
    "candidate_B": [asr_pilot.model.id, asr_pilot.model.revision],
    "unintended_config_drift": False,
})
"""
    ),
    markdown(
        """
## 10. Selected encoder-decoder design

An encoder-decoder Transformer separates acoustic representation learning from text generation. The feature extractor converts the waveform to log-Mel features; the encoder contextualizes them; decoder cross-attention reads those acoustic states; the autoregressive decoder predicts English tokens.

**Expected output structure.** The following local diagram is explanatory, not a measured result.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 2.8))
labels = ["16 kHz Hausa\\nwaveform", "log-Mel\\nfeatures", "Whisper\\nencoder", "cross-attending\\ndecoder", "English\\ntokens"]
for index, label in enumerate(labels):
    ax.text(index, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round", "fc": "#e8f1fb"})
    if index < len(labels) - 1:
        ax.annotate("", xy=(index + 0.72, 0.5), xytext=(index + 0.28, 0.5), arrowprops={"arrowstyle": "->"})
ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("Direct Hausa→English S2TT inference path")
plt.show()
"""
    ),
    markdown(
        """
## 11. Audio/log-Mel input

A waveform is an ordered sequence of pressure-amplitude samples. The project normalizes audio to mono 16 kHz and rejects malformed, non-finite, empty, or over-30-second training clips. Whisper's feature extractor maps windows of waveform samples into log-Mel spectral energy. A feature extractor is deterministic preprocessing; it is not the learned encoder.
"""
    ),
    code(
        """
print({"sampling_rate_hz": 16_000, "maximum_training_seconds": 30.0, "audio_loaded_now": False})
"""
    ),
    markdown(
        """
## 12. Whisper encoder

The learned speech encoder transforms log-Mel frames into contextual acoustic representations. “Acoustic” does not mean language-free: pretraining and later ASR adaptation can encode language/task biases. The matched initialization pilot tests those biases rather than assuming they help translation.
"""
    ),
    markdown(
        """
## 13. Decoder cross-attention

At each target position, decoder self-attention sees previous English tokens, while cross-attention queries encoder states. This tight speech-to-text coupling is why the selected model does not need an external Hausa transcript or an NLLB stage during inference.
"""
    ),
    markdown(
        """
## 14. English autoregressive decoder

Autoregressive generation predicts one token at a time until an end token or length limit. Translation mode and the Hausa language prompt configure the pretrained task; they do not, by themselves, constitute project fine-tuning. English `target_text` supervision is what makes the trained system direct Hausa→English S2TT.
"""
    ),
    markdown(
        """
## 15. LoRA

LoRA freezes pretrained matrices and learns low-rank updates. Rank controls adapter capacity; alpha scales the update; dropout regularizes it. Here LoRA targets existing `q_proj` and `v_proj` attention modules only after module discovery verifies they exist. The saved adapter still needs its pinned base model.
"""
    ),
    code(
        """
print({
    "strategy": base_pilot.model.efficiency_strategy,
    "rank": base_pilot.model.lora_r,
    "alpha": base_pilot.model.lora_alpha,
    "dropout": base_pilot.model.lora_dropout,
    "targets": base_pilot.model.lora_targets,
})
"""
    ),
    markdown(
        """
## 16. Frozen versus trainable parameters

Frozen parameters participate in forward/backward computation but are not updated. Trainable LoRA parameters receive gradients and optimizer state. Runtime manifests measure exact totals; until a run exists, the chart must remain an honest empty state.
"""
    ),
    code(
        """
import json


def newest_json(root, name):
    matches = sorted(root.glob(f"**/{name}"), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None

smoke_summary_path = newest_json(REPO_ROOT / "artifacts/direct_s2tt/smoke", "summary.json")
fig, ax = plt.subplots(figsize=(7, 4))
if smoke_summary_path:
    payload = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    counts = payload["parameters"]
    frozen = counts["total_parameters"] - counts["trainable_parameters"]
    ax.bar(["Frozen", "Trainable"], [frozen, counts["trainable_parameters"]], color=["#9aa0a6", "#1a73e8"])
    ax.set_ylabel("Parameters"); ax.set_title("Measured frozen versus trainable parameters")
else:
    ax.text(0.5, 0.5, "No measured smoke artifact yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Frozen versus trainable parameters — empty state")
plt.show()
"""
    ),
    markdown(
        """
## 17. Initialization hypotheses

The multilingual base retains pretrained English translation behavior. The Hausa-ASR checkpoint may provide speech representations better adapted to Hausa, but its previous training objective also adapted the model toward Hausa transcription. A matched translation pilot is therefore required to determine whether that initialization helps or hurts Hausa→English translation. Calling this outcome “catastrophic forgetting” before measurement would overstate the evidence.
"""
    ),
    markdown(
        """
## 18. Loss and teacher forcing

Teacher forcing shifts the reference English sequence: the decoder receives earlier reference tokens and cross-entropy penalizes the probability assigned to each next reference token. Training loss measures fit under this conditioning; it is not a translation-quality score. chrF++ and SacreBLEU require generated predictions on unseen validation speakers.
"""
    ),
    code(
        """
print("Expected output structure", {"decoder_inputs": "English labels shifted right", "objective": "token cross-entropy", "quality_metrics": ["chrF++", "SacreBLEU"]})
"""
    ),
    markdown(
        """
## 19. Collator and `-100` masking

A batch groups examples for one forward pass. Padding makes variable-length English labels rectangular; padded label IDs are replaced with `-100`, PyTorch cross-entropy's ignore index, so padding contributes no loss. The collator also removes a duplicated decoder-start prefix when every row has it.
"""
    ),
    code(
        """
from hausa_s2tt.training import SpeechSeq2SeqCollator

print({"collator": SpeechSeq2SeqCollator.__name__, "padding_loss_id": -100, "target_column": "target_text"})
"""
    ),
    markdown(
        """
## 20. Structural smoke test

**Expensive live output (small public checkpoint download; CPU-safe but may take minutes).** This genuine tiny-model gate runs a forward/backward pass, verifies LoRA targets/gradients/frozen weights, performs a few optimizer steps, saves adapter + processor + resumable Trainer state, reloads the local adapter, and executes translation-mode generation.

It establishes mechanics only: translation decoding configuration verified; generation completed; scientific quality claim **none**.
"""
    ),
    code(
        """
SMOKE_ROOT = REPO_ROOT / "artifacts/direct_s2tt/smoke"
if RUN_STRUCTURAL_SMOKE:
    from hausa_s2tt.cli import smoke_train_main
    smoke_train_main(["--output-dir", str(SMOKE_ROOT), "--steps", "3"])
else:
    print("Structural smoke skipped (default). Set RUN_STRUCTURAL_SMOKE=True explicitly.")
"""
    ),
    markdown(
        """
## 21. Pre-training baseline

**Expensive live output (model + validation audio downloads/inference).** Both untouched initializations are evaluated on the same fixed project-validation subset before any LoRA update. Artifacts save IDs, English references, predictions, chrF++, SacreBLEU, validation loss, runtime, peak VRAM when CUDA is present, revisions, and generation settings.
"""
    ),
    code(
        """
from hausa_s2tt.direct_s2tt import evaluate_initialization_baseline

BASELINE_ROOT = REPO_ROOT / "artifacts/experiments/direct_s2tt/baselines"
baseline_results = {}
if RUN_BASELINE_EVALUATION:
    for name, config in (("base", base_pilot), ("from_asr", asr_pilot)):
        baseline_results[name] = evaluate_initialization_baseline(config, BASELINE_ROOT / name)
else:
    print("Pre-training baselines skipped (default).")
"""
    ),
    markdown(
        """
## 22. Matched pilot

An optimizer step applies one accumulated gradient update; with batch size 2 and accumulation 8, the single-device effective batch is 16 examples. An epoch is one pass over the configured training subset. The 50-step pilots are longer than a three-step structural smoke but remain development experiments.

**GPU-only expensive live output.** Both candidates use identical subsets, seed, steps, learning rate, LoRA, effective batch, precision policy, generation, and metrics. The cell refuses CPU and requires `RUN_MATCHED_PILOTS=True`.
"""
    ),
    code(
        """
from hausa_s2tt.direct_s2tt import train_direct_s2tt

matched_pilot_results = {}
if RUN_MATCHED_PILOTS:
    if not torch.cuda.is_available():
        raise RuntimeError("Matched pilots require an explicitly selected CUDA runtime.")
    assert_matched_direct_configs(base_pilot, asr_pilot)
    for name, config in (("base", base_pilot), ("from_asr", asr_pilot)):
        matched_pilot_results[name] = train_direct_s2tt(config)
else:
    print("Matched GPU pilots skipped (default).")
"""
    ),
    markdown(
        """
## 23. Runtime/VRAM telemetry

Startup, first three warm-up optimizer steps, later steady-state steps, validation prediction, and checkpoint writing are stored separately. Throughput, GPU-hours, RTF, and peak VRAM are measurements only when their artifact exists.
"""
    ),
    code(
        """
pilot_summaries = {}
for name, config in (("base", base_pilot), ("from_asr", asr_pilot)):
    path = REPO_ROOT / config.training.output_dir / "training_summary.json"
    if path.is_file():
        pilot_summaries[name] = json.loads(path.read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if pilot_summaries:
    names = list(pilot_summaries)
    axes[0].bar(names, [pilot_summaries[name]["runtime"]["wall_seconds"] for name in names])
    axes[0].set_ylabel("Training wall time (seconds)"); axes[0].set_title("Measured pilot runtime")
    vram = [pilot_summaries[name]["runtime"].get("peak_vram_bytes") for name in names]
    if all(value is not None for value in vram):
        axes[1].bar(names, [value / 2**30 for value in vram])
        axes[1].set_ylabel("Peak allocated VRAM (GiB)"); axes[1].set_title("Measured pilot peak VRAM")
    else:
        axes[1].text(0.5, 0.5, "Peak VRAM not measured", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_title("Pilot peak VRAM — empty state")
else:
    for axis, title in zip(axes, ("Pilot runtime", "Pilot peak VRAM"), strict=True):
        axis.text(0.5, 0.5, "No measured pilot artifacts", ha="center", va="center", transform=axis.transAxes)
        axis.set_xticks([]); axis.set_yticks([]); axis.set_title(f"{title} — empty state")
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        """
## 24. Training curves

Training loss is an optimization diagnostic. Validation loss measures held-out teacher-forced likelihood. Neither curve alone proves good translation; divergence can suggest overfitting and is one input to early stopping.
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(8, 4))
plotted = False
for name, config in (("base", base_pilot), ("from_asr", asr_pilot)):
    history_path = REPO_ROOT / config.training.output_dir / "trainer_history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))["log_history"]
        train_points = [(row["step"], row["loss"]) for row in history if "loss" in row]
        eval_points = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
        if train_points:
            ax.plot(*zip(*train_points), label=f"{name} training loss")
            plotted = True
        if eval_points:
            ax.plot(*zip(*eval_points), marker="o", label=f"{name} validation loss")
            plotted = True
if plotted:
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Cross-entropy loss"); ax.set_title("Measured training/validation loss"); ax.legend()
else:
    ax.text(0.5, 0.5, "No measured training history", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Training curves — empty state")
plt.show()
"""
    ),
    markdown(
        """
## 25. Validation metrics

Primary checkpoint selection uses case-sensitive chrF++ (`word_order=2`); SacreBLEU with `13a` tokenization is complementary, and validation loss is diagnostic. WER/CER are ASR metrics and do not score direct translation. Metric artifacts preserve SacreBLEU signatures and package versions.
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(8, 4))
metric_plotted = False
for name, config in (("base", base_pilot), ("from_asr", asr_pilot)):
    history_path = REPO_ROOT / config.training.output_dir / "trainer_history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))["log_history"]
        for key, label in (("eval_chrf_pp", "chrF++"), ("eval_sacrebleu", "SacreBLEU")):
            points = [(row["step"], row[key]) for row in history if key in row]
            if points:
                ax.plot(*zip(*points), marker="o", label=f"{name} {label}")
                metric_plotted = True
if metric_plotted:
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Score"); ax.set_title("Measured validation translation metrics"); ax.legend()
else:
    ax.text(0.5, 0.5, "No measured validation metric history", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Validation metrics — empty state")
plt.show()
"""
    ),
    markdown(
        """
## 26. Before/after validation examples

Improvement examples must use fixed validation speakers absent from training and preserve ID, Hausa source transcript, English reference, baseline prediction, and post-training prediction. Inspect untranslated Hausa, fluent-but-wrong English, omissions, additions, repetition, names, numbers, dates, and negation.

If the paired artifacts do not exist, the only honest conclusion is: **No measured training improvement is available yet.**
"""
    ),
    code(
        """
def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

comparison_rows = []
baseline_path = BASELINE_ROOT / "base/predictions.jsonl"
post_path = REPO_ROOT / base_pilot.training.output_dir / "validation/predictions.jsonl"
if baseline_path.is_file() and post_path.is_file():
    baseline_by_id = {str(row["source_id"]): row for row in read_jsonl(baseline_path)}
    for row in read_jsonl(post_path):
        before = baseline_by_id.get(str(row["source_id"]))
        if before:
            comparison_rows.append({
                "source_id": row["source_id"], "Hausa": row.get("source_transcript"),
                "English reference": row["reference"], "baseline": before["prediction"],
                "post_training": row["prediction"],
            })
print(comparison_rows[:5] if comparison_rows else "No measured training improvement is available yet.")

fig, ax = plt.subplots(figsize=(10, 2.8))
if comparison_rows:
    changed = sum(row["baseline"] != row["post_training"] for row in comparison_rows)
    ax.bar(["unchanged", "changed"], [len(comparison_rows) - changed, changed])
    ax.set_ylabel("Validation examples"); ax.set_title("Measured before/after prediction changes")
else:
    ax.text(0.5, 0.5, "No paired before/after artifacts", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Before/after validation translations — empty state")
plt.show()
"""
    ),
    markdown(
        """
## 27. Architecture selection

Selection uses only project-validation chrF++ with predeclared tie breakers. Before freezing, a human must review both candidates for source-language leakage, hallucination/repetition, omissions/additions, and names/numbers/dates/negation. The machine-readable artifact refuses any run that does not declare `official_dev_evaluated: false`.

Replace each `QUALITATIVE_REVIEW` value with a mapping containing `status="complete"`, the four named review fields, and Boolean `eligible_for_selection`. A severe qualitative failure may make a candidate ineligible before metric ranking; do not mark a review complete without inspecting the saved validation rows.
"""
    ),
    code(
        """
from hausa_s2tt.direct_s2tt import (
    select_direct_architecture,
    write_notebook03_handoff,
)

SELECTION_PATH = REPO_ROOT / "artifacts/experiments/direct_s2tt/architecture_selection.json"
if FREEZE_ARCHITECTURE_SELECTION:
    if not HUMAN_REVIEW_COMPLETE:
        raise RuntimeError("Complete and document qualitative validation review first.")
    candidates = []
    for name, config in (("base", base_pilot), ("from_asr", asr_pilot)):
        summary_path = REPO_ROOT / config.training.output_dir / "training_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing matched pilot summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["validation_metrics"]
        candidates.append({
            "run_name": config.tracking.run_name, "model_id": config.model.id,
            "model_revision": config.model.revision, "split": "validation",
            "chrf_pp": metrics["validation_chrf_pp"],
            "sacrebleu": metrics["validation_sacrebleu"],
            "validation_loss": metrics["validation_loss"],
            "official_dev_evaluated": False,
            "qualitative_review": QUALITATIVE_REVIEW[name],
        })
    selection = select_direct_architecture(candidates, SELECTION_PATH)
    selected_config = (
        base_pilot
        if selection["selected_model_id"] == base_pilot.model.id
        else asr_pilot
    )
    selected_summary = json.loads(
        (REPO_ROOT / selected_config.training.output_dir / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    handoff = write_notebook03_handoff(
        selected_config,
        selected_summary,
        architecture_selection_path=SELECTION_PATH,
        output_path=REPO_ROOT / "artifacts/experiments/direct_s2tt/notebook03_handoff.json",
        training_status="matched_pilot_selected",
    )
    print("Measured artifact", selection)
    print("Notebook 03 handoff", handoff)
else:
    print("Architecture selection not frozen (default).")
"""
    ),
    markdown(
        """
## 28. Full-run configuration

`direct_s2tt_full.yaml` means the **full accepted dataset with LoRA**, not full-parameter fine-tuning. Full training requires a passed structural smoke, both matched pilots, frozen selection, hardware-matched runtime/storage projection, suitable CUDA device/VRAM, and `RUN_FULL_TRAINING=True`.
"""
    ),
    code(
        """
FULL_BASE_CONFIG = REPO_ROOT / "configs/direct_s2tt_full.yaml"
FULL_ASR_CONFIG = REPO_ROOT / "configs/direct_s2tt_from_asr.yaml"
print({
    "base_full": str(FULL_BASE_CONFIG), "asr_full": str(FULL_ASR_CONFIG),
    "full_training_enabled": RUN_FULL_TRAINING,
})
"""
    ),
    markdown(
        """
## 29. Persistence and resume

A checkpoint contains LoRA adapter weights/config, processor files, Trainer state, optimizer state, and scheduler state in numbered `checkpoint-*` directories. Local inference reads `adapter_config.json`, reloads the pinned base, and attaches the adapter. Resuming must point to a numbered Trainer checkpoint—not merely the final adapter directory.
"""
    ),
    code(
        """
if RESUME_FROM_CHECKPOINT is not None:
    resume_path = Path(RESUME_FROM_CHECKPOINT)
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    missing = [name for name in required if not (resume_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is not resumable; missing {missing}")
    print({"resume_from_checkpoint": str(resume_path), "state_files_verified": required})
else:
    print("Starting checkpoint not selected; RESUME_FROM_CHECKPOINT=None.")
"""
    ),
    markdown(
        """
## 30. Compute/cost projection

A tiny CPU smoke is not a runtime estimator. Projection requires hardware-matched steady-state pilot step samples and separately adds startup, validation, and checkpoint time. The interval uses observed pilot p10–p90 step variation. Cost is calculated only from a user-supplied hourly rate; no Colab price is hard-coded.
"""
    ),
    code(
        """
import math

from hausa_s2tt.telemetry import (
    estimate_training_from_pilot_phases,
    write_json_artifact,
)

projection = None
if SELECTION_PATH.is_file():
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected_config = base_pilot if selection["selected_model_id"] == base_pilot.model.id else asr_pilot
    timing_path = REPO_ROOT / selected_config.training.output_dir / "step_timing.json"
    summary_path = REPO_ROOT / selected_config.training.output_dir / "training_summary.json"
    if timing_path.is_file() and summary_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        full_examples = tracked_audit["derived_seed_42_split"]["training"]["pairs"]
        effective_batch = selected_config.training.train_batch_size * selected_config.training.gradient_accumulation_steps
        steps_per_epoch = math.ceil(full_examples / effective_batch)
        projection = estimate_training_from_pilot_phases(
            timing["steady_step_seconds"], optimizer_steps_per_epoch=steps_per_epoch,
            epochs=10, startup_seconds=sum(summary["phase_seconds"].get(key, 0) for key in ("data_startup", "model_startup")),
            validation_seconds_per_run=summary["phase_seconds"].get("validation_prediction", 0),
            validation_runs=10, checkpoint_seconds_per_save=summary["phase_seconds"].get("checkpoint_writing", 0),
            checkpoint_saves=10, gpu_count=summary["runtime"].get("gpu_count", 0),
            checkpoint_bytes_per_save=summary["checkpoint_size_bytes"],
            examples_per_step=effective_batch, gpu_usd_per_hour=GPU_USD_PER_HOUR,
        )
        write_json_artifact(REPO_ROOT / "artifacts/experiments/direct_s2tt/full_run_projection.json", projection)
print(projection or "No hardware-matched pilot projection is available yet.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if projection:
    estimate_hours = projection["projected_wall_seconds"] / 3600
    low, high = [value / 3600 for value in projection["projected_wall_seconds_interval"]]
    axes[0].bar(["Full run"], [estimate_hours], yerr=[[estimate_hours-low], [high-estimate_hours]], capsize=8)
    axes[0].set_ylabel("Wall time (hours)"); axes[0].set_title("Estimated full-run time")
    axes[0].legend(["Median with observed p10–p90 interval"])
    projected_cost = projection["projected_gpu_cost_usd"]
    if projected_cost is None:
        axes[1].text(0.5, 0.5, "Set GPU_USD_PER_HOUR for cost", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].set_title("Estimated cost — rate absent")
    else:
        axes[1].bar(["Full run"], [projected_cost], color="#1a73e8", label="User-rate estimate")
        axes[1].set_ylabel("Estimated GPU cost (USD)"); axes[1].set_title("Estimated full-run cost")
        axes[1].legend()
else:
    for axis, title in zip(axes, ("Full-run time", "Full-run cost"), strict=True):
        axis.text(0.5, 0.5, "No hardware-matched projection", ha="center", va="center", transform=axis.transAxes)
        axis.set_xticks([]); axis.set_yticks([]); axis.set_title(f"{title} — empty state")
plt.tight_layout(); plt.show()
"""
    ),
    code(
        """
if RUN_FULL_TRAINING:
    smoke_path = newest_json(SMOKE_ROOT, "summary.json")
    projection_path = REPO_ROOT / "artifacts/experiments/direct_s2tt/full_run_projection.json"
    if not smoke_path or not SELECTION_PATH.is_file() or not projection_path.is_file():
        raise RuntimeError("Full-run gates missing: smoke, selection, and projection are required.")
    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires an explicitly selected CUDA runtime.")
    smoke_evidence = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not all(
        [smoke_evidence.get("status") == "passed", smoke_evidence.get("adapter_saved"),
         smoke_evidence.get("processor_saved"), smoke_evidence.get("generation_completed")]
    ):
        raise RuntimeError("The structural smoke artifact does not satisfy all mechanics gates.")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("official_dev_evaluated") is not False or len(selection.get("candidate_runs", [])) != 2:
        raise RuntimeError("Architecture selection is incomplete or used protected data.")
    for config in (base_pilot, asr_pilot):
        if not (REPO_ROOT / config.training.output_dir / "training_summary.json").is_file():
            raise FileNotFoundError("Both matched pilot summaries are required before full training.")
    selected_pilot = base_pilot if selection["selected_model_id"] == base_pilot.model.id else asr_pilot
    selected_summary = json.loads(
        (REPO_ROOT / selected_pilot.training.output_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    measured_peak = selected_summary["runtime"].get("peak_vram_bytes")
    available_vram = torch.cuda.get_device_properties(0).total_memory
    if measured_peak is None or available_vram < measured_peak * 1.10:
        raise RuntimeError("Current GPU lacks a measured 10% VRAM margin over the selected pilot.")
    full_path = FULL_BASE_CONFIG if selection["selected_model_id"] == base_pilot.model.id else FULL_ASR_CONFIG
    full_config = load_config(full_path)
    full_config.training.resume_from_checkpoint = RESUME_FROM_CHECKPOINT
    full_result = train_direct_s2tt(full_config)
    from hausa_s2tt.direct_s2tt import write_notebook03_handoff
    full_handoff = write_notebook03_handoff(
        full_config,
        full_result,
        architecture_selection_path=SELECTION_PATH,
        output_path=REPO_ROOT / "artifacts/experiments/direct_s2tt/notebook03_handoff.json",
        training_status="full_training_completed",
    )
    print("Measured artifact", full_result)
    print("Notebook 03 handoff", full_handoff)
else:
    print("Full project training skipped (default).")
"""
    ),
    markdown(
        """
## 31. Artifact handoff

Notebook 03 should load a frozen selection JSON, selected resolved config, pinned model/dataset revisions, exact project-split provenance/hash, adapter/checkpoint path, processor path, generation config, validation metrics/predictions, Trainer history, runtime/VRAM/parameter telemetry, pilot/full status, and `official_dev_evaluated: false`.

**Expected output structure.** Large model/data/prediction artifacts remain under ignored `artifacts/`; only small reviewed reports belong in Git.
"""
    ),
    code(
        """
NOTEBOOK03_REQUIRED_FIELDS = [
    "selected_model_initialization", "resolved_config", "model_revision", "dataset_revision",
    "project_split_provenance", "adapter_or_checkpoint_path", "processor_path",
    "generation_config", "validation_metrics", "validation_predictions",
    "training_history", "runtime_telemetry",
    "parameter_counts", "pilot_or_full_status", "architecture_selection_artifact",
    "official_dev_evaluated",
]
print("Expected output structure", NOTEBOOK03_REQUIRED_FIELDS)
"""
    ),
    markdown(
        """
## 32. Limitations and ethics

The tracked audit measures metadata, not full audio decode integrity. Dialect/accent, speaker, gender, topic, and recording-condition coverage remain incompletely characterized; gender should not be inferred from names or voices. Direct models can hallucinate fluent English, omit content, leak Hausa, or alter names, numbers, dates, and negation. A surface metric can also disagree with semantic adequacy.

This research prototype must not be the sole translation source for medical, legal, immigration, policing, benefits, education, or other consequential decisions. Publication requires rechecking dataset/base-model/adapter license obligations and documenting human review.
"""
    ),
    markdown(
        """
## 33. Summary

The runnable pipeline now separates five gates: structural smoke, frozen pretraining baselines, matched base-versus-Hausa-ASR pilots, validation-only architecture selection, and explicitly authorized full-data LoRA training. Exact Notebook 00 membership is consumed when available; deterministic reconstruction is provenance-labeled and audit-checked otherwise. Official dev remains sealed for Notebook 03.

Until pilot artifacts exist: **No measured training improvement is available yet.**
"""
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"direct-s2tt-{index:02d}"

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbformat.write(notebook, OUTPUT)
print(OUTPUT)
