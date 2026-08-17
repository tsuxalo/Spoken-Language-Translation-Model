# Final Capstone Experiment Upgrade

This package adds a controlled error-aware MT study, a strong Omnilingual-ASR
baseline, and cluster-bootstrap statistical analysis.

## Files

- `experiments/make_research_splits.py`
- `experiments/train_error_aware_controlled.py`
- `experiments/evaluate_research_suite.py`
- `experiments/generate_omni_asr.py`
- `analysis/analyze_predictions.py`
- `requirements-research.txt`
- `requirements-omni.txt`

## Core experimental rule

Use NaijaS2ST `train` only to create training/internal validation data.
Keep NaijaS2ST `dev` untouched until final evaluation.

The internal split groups by `alignment_id` because multiple speakers record
the same sentence. This prevents the same Hausa-English text pair from leaking
into both training and validation.

The three LoRA conditions are exposure-matched:

- clean: gold Hausa for every utterance
- noisy: Whisper ASR Hausa for every utterance
- mixed: exactly half gold Hausa and half Whisper ASR Hausa

All three are model-selected against the same noisy validation set.

## Final outputs

The analysis script reports:
- corpus BLEU
- corpus chrF++
- optional SSA-COMET
- 95% cluster-bootstrap confidence intervals
- paired bootstrap differences vs baseline
- ASR substitution/deletion/insertion correlations
- WER-binned degradation curves
- 40 qualitative examples by default

## Paper-aligned strong cascade baseline

The NaijaS2ST paper's cascade uses Omnilingual-ASR 1B LLM followed by
NLLB-200 3.3B. `evaluate_research_suite.py` therefore supports arbitrary
additional Hugging Face MT checkpoints through:

`--model nllb3b=facebook/nllb-200-3.3B`

This lets us compare:
- Whisper-Hausa -> NLLB 600M
- Whisper-Hausa -> error-aware NLLB 600M LoRA
- Omnilingual-ASR 1B -> NLLB 600M
- Omnilingual-ASR 1B -> NLLB 3.3B

The last system mirrors the model family/components used by the paper, but
scores from our public dev evaluation should not be claimed as reproducing the
paper's held-out test-set numbers.
