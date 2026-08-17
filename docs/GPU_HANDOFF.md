# GPU Experiment Handoff: Final Hausa Speech-Translation Study

## Purpose

This handoff runs the final controlled experiment for the capstone:

**Does adapting Hausa->English MT to the real errors produced by our Hausa ASR system improve end-to-end speech translation?**

The core comparison is NLLB/AfriNLLB vs clean, noisy, and mixed LoRA adaptation, followed by a held-out NaijaS2ST dev evaluation and statistical analysis.

## Important rules

1. Do not run expensive jobs unless `python experiments/gpu_preflight.py --stage setup --require-cuda` passes.
2. Use NaijaS2ST `train` only for training/internal validation.
3. Keep NaijaS2ST `dev` untouched until all three LoRA models are trained.
4. Do not install COMET into the training environment before training. Generate predictions first; run COMET analysis afterward, preferably in a separate environment.
5. Use the same seed and training hyperparameters for clean/noisy/mixed.
6. If batch size must change for memory, use the same effective batch size across all three conditions.
7. Save the terminal output and the files in `results/final/`.

## A. Pull the experiment branch

```powershell
git fetch origin
git switch karun/final-capstone-benchmark
git pull
```

If the branch has another name, use the branch name provided by Karun.

## B. Create the GPU environment

Use a fresh environment where possible:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build appropriate for the machine/cluster first. If the machine already has a working CUDA PyTorch environment, keep it.

Then install the project/research dependencies without COMET:

```powershell
python -m pip install -r requirements-research.txt
```

Do not continue until this passes:

```powershell
python experiments\gpu_preflight.py `
    --stage setup `
    --require-cuda `
    --json-output results\final\preflight_setup.json
```

## C. Build all NaijaS2ST training text pairs

```powershell
python experiments\prepare_naijas2st_pairs.py `
    --split train `
    --max-samples 0 `
    --output experiments\generated\naijas2st_train_pairs_all.jsonl
```

Check:

```powershell
(Get-Content experiments\generated\naijas2st_train_pairs_all.jsonl).Count
Get-Content experiments\generated\naijas2st_train_pairs_all.jsonl -TotalCount 3
```

## D. Generate the full Whisper ASR-error training manifest

```powershell
python experiments\generate_asr_noise.py `
    --pairs experiments\generated\naijas2st_train_pairs_all.jsonl `
    --split train `
    --max-samples 0 `
    --shuffle-buffer 0 `
    --seed 42 `
    --output experiments\generated\naijas2st_train_whisper_all.jsonl
```

After completion:

```powershell
(Get-Content experiments\generated\naijas2st_train_whisper_all.jsonl).Count
```

## E. Create the leakage-resistant internal split

```powershell
python experiments\make_research_splits.py `
    --input experiments\generated\naijas2st_train_whisper_all.jsonl `
    --train-output experiments\generated\research_train.jsonl `
    --val-output experiments\generated\research_val.jsonl `
    --val-fraction 0.10 `
    --seed 42 `
    --stats-output results\final\split_stats.json
```

Required sanity checks in the printed output:

- `alignment_overlap = 0`
- `text_pair_overlap = 0`

Then:

```powershell
python experiments\gpu_preflight.py `
    --stage train `
    --require-cuda `
    --json-output results\final\preflight_train.json
```

**STOP AND SEND KARUN:** the full split output plus `results/final/split_stats.json` before starting the three long LoRA runs.

## F. Train CLEAN

```powershell
python experiments\train_error_aware_controlled.py `
    --train-jsonl experiments\generated\research_train.jsonl `
    --val-jsonl experiments\generated\research_val.jsonl `
    --mode clean `
    --output-dir outputs\final\clean `
    --epochs 3 `
    --batch-size 2 `
    --gradient-accumulation 4 `
    --learning-rate 2e-4 `
    --seed 42
```

## G. Train NOISY

```powershell
python experiments\train_error_aware_controlled.py `
    --train-jsonl experiments\generated\research_train.jsonl `
    --val-jsonl experiments\generated\research_val.jsonl `
    --mode noisy `
    --output-dir outputs\final\noisy `
    --epochs 3 `
    --batch-size 2 `
    --gradient-accumulation 4 `
    --learning-rate 2e-4 `
    --seed 42
```

## H. Train MIXED

```powershell
python experiments\train_error_aware_controlled.py `
    --train-jsonl experiments\generated\research_train.jsonl `
    --val-jsonl experiments\generated\research_val.jsonl `
    --mode mixed `
    --output-dir outputs\final\mixed `
    --epochs 3 `
    --batch-size 2 `
    --gradient-accumulation 4 `
    --learning-rate 2e-4 `
    --seed 42
```

If memory is insufficient, use this setting for **all three** models:

```text
--batch-size 1
--gradient-accumulation 8
--gradient-checkpointing
```

## I. Prepare the untouched final public dev benchmark

Only after clean/noisy/mixed training is complete:

```powershell
python experiments\prepare_naijas2st_pairs.py `
    --split dev `
    --max-samples 0 `
    --output experiments\generated\naijas2st_dev_pairs_all.jsonl
```

```powershell
python experiments\generate_asr_noise.py `
    --pairs experiments\generated\naijas2st_dev_pairs_all.jsonl `
    --split dev `
    --max-samples 0 `
    --shuffle-buffer 0 `
    --seed 42 `
    --output experiments\generated\naijas2st_dev_whisper_all.jsonl
```

Run the evaluation preflight:

```powershell
python experiments\gpu_preflight.py `
    --stage evaluate `
    --require-cuda `
    --json-output results\final\preflight_evaluate.json
```

## J. Evaluate the five main systems

```powershell
python experiments\evaluate_research_suite.py `
    --input experiments\generated\naijas2st_dev_whisper_all.jsonl `
    --source-field hausa_asr `
    --batch-size 4 `
    --adapter "clean=outputs\final\clean" `
    --adapter "noisy=outputs\final\noisy" `
    --adapter "mixed=outputs\final\mixed" `
    --output results\final\whisper_error_aware_suite.csv
```

If evaluation runs out of memory, change `--batch-size 4` to `--batch-size 1`.

## K. Run the clean-transcript oracle

```powershell
python experiments\evaluate_research_suite.py `
    --input experiments\generated\naijas2st_dev_whisper_all.jsonl `
    --source-field hausa_gold `
    --batch-size 4 `
    --baseline nllb `
    --baseline afrinllb `
    --output results\final\gold_mt_oracle.csv
```

## L. Save reproducibility information

```powershell
git rev-parse HEAD | Out-File results\final\git_commit.txt
python -m pip freeze | Out-File results\final\environment_training.txt
nvidia-smi | Out-File results\final\gpu_info.txt
```

```powershell
python -c "from huggingface_hub import dataset_info, model_info; print('NaijaS2ST:', dataset_info('McGill-NLP/NaijaS2ST').sha); print('NLLB600M:', model_info('facebook/nllb-200-distilled-600M').sha); print('NLLB3.3B:', model_info('facebook/nllb-200-3.3B').sha); print('SSA-COMET:', model_info('McGill-NLP/ssa-comet-mtl').sha)" | Out-File results\final\hf_revisions.txt
```

## M. Statistical analysis / COMET

Do this after the prediction CSV exists. Prefer a separate analysis environment so COMET dependency constraints do not alter the training environment.

```powershell
python -m venv .venv-analysis
.\.venv-analysis\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-comet.txt
```

Then:

```powershell
python analysis\analyze_predictions.py `
    --input results\final\whisper_error_aware_suite.csv `
    --output-dir results\final\analysis_whisper `
    --baseline nllb `
    --candidate mixed `
    --bootstrap 1000 `
    --seed 42 `
    --qualitative-n 40 `
    --comet `
    --comet-model McGill-NLP/ssa-comet-mtl `
    --comet-batch-size 8
```

Expected outputs include:

- `system_metrics.csv`
- `bootstrap_confidence_intervals.csv`
- `paired_bootstrap_differences.csv`
- `sentence_metrics.csv`
- `asr_error_correlations.csv`
- `wer_binned_translation.csv`
- `qualitative_examples.csv`
- `wer_vs_chrf.png`
- `analysis_metadata.json`

## What to send back

Please send Karun:

1. `preflight_setup.json`
2. Number of rows in `naijas2st_train_whisper_all.jsonl`
3. Console output from `make_research_splits.py`
4. `split_stats.json`
5. `experiment_metadata.json` from clean/noisy/mixed
6. `whisper_error_aware_suite.csv`
7. `gold_mt_oracle.csv`
8. Entire `results/final/analysis_whisper/` directory
9. `git_commit.txt`, `environment_training.txt`, `gpu_info.txt`, and `hf_revisions.txt`

Do not commit large generated datasets, model checkpoints, or adapter weights unless the team has explicitly decided where those artifacts should live.
