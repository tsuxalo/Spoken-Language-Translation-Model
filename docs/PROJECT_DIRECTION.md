# Project Direction: Low-Resource Spoken Language Translation

## Working Title

**Direct and Cascaded Bemba-to-English Speech Translation with BIG-C**

## Project Summary

This project studies speech-to-text translation for an under-resourced
African language. Given an audio recording in Bemba, the goal is to
produce an English translation.

We compare two modeling strategies:

1. A cascaded pipeline:
   Bemba speech → Bemba transcription → English translation

2. A direct pipeline:
   Bemba speech → English translation

The project builds on the BIG-C dataset, which contains Bemba speech,
Bemba transcriptions, English translations, and corresponding images.

## Motivation

Modern speech and language systems perform well for high-resource
languages but remain much less reliable for many African languages.
Bemba is widely spoken in Zambia but has substantially fewer public
language resources than English and other high-resource languages.

The original BIG-C work established ASR, machine translation, and direct
speech-translation baselines. Our project follows up using modern
pretrained multilingual models and a reproducible experimental pipeline.

## Current Repository Status

The repository currently implements a Hausa automatic speech recognition
prototype using Whisper-small and FLEURS.

The current implementation performs:

Hausa speech → Hausa transcription

It does not yet perform:

Source-language speech → English translation

The existing Hausa system will be retained as a legacy baseline and
framework smoke test. The primary capstone experiments will use
Bemba-English parallel speech from BIG-C.

## Research Questions

### RQ1: Direct versus cascaded translation

How does direct Bemba-to-English speech translation compare with a
cascaded ASR and text-translation pipeline?

### RQ2: Data efficiency

How does translation performance change when training with 10%, 25%,
50%, and 100% of the available training data?

### RQ3: Error propagation

Which Bemba ASR errors cause the greatest downstream degradation in
English translation?

### RQ4: Multimodal context

Does access to the grounding image improve translation for ambiguous
or context-dependent utterances?

RQ4 is a stretch goal and will only be attempted after the speech-only
baselines are complete.

## Dataset

Primary dataset: BIG-C.

Each standardized example should contain:

- example_id
- audio_path
- image_path
- source_language
- target_language
- source_text
- target_text
- speaker_id
- image_id
- conversation_id
- turn_id
- duration_seconds
- split

Official train, validation, and test splits must be preserved.

All split validation will be performed at the image level to prevent
different utterances grounded in the same image from appearing across
training and evaluation sets.

## Systems

### System A: Bemba ASR

Input: Bemba audio  
Output: Bemba text

Initial model:

- Whisper-small

Evaluation:

- Word Error Rate
- Character Error Rate

### System B: Cascaded speech translation

Input: Bemba audio  
Intermediate output: Bemba transcription  
Final output: English translation

Models:

- Whisper-small for ASR
- NLLB or another multilingual MT model for Bemba-to-English translation

Evaluation:

- SacreBLEU
- ChrF++
- Translation error categories

### System C: Direct speech translation

Input: Bemba audio  
Output: English translation

Candidate models:

- Whisper-small fine-tuned on English targets
- SeamlessM4T if compute permits

Evaluation:

- SacreBLEU
- ChrF++
- Length ratio
- Wrong-language rate
- Manual error analysis

## Experimental Design

All experiments will use:

- Fixed official dataset splits
- Fixed random seeds
- Versioned configuration files
- Saved prediction files
- Reproducible manifest files
- Logged Git commit hashes

Initial data fractions:

- 10%
- 25%
- 50%
- 100%

The same selected examples must be reused across model comparisons.

## Expected Contributions

The project does not claim to introduce a new neural architecture.

Its contributions are:

1. A reproducible BIG-C data and evaluation pipeline
2. A modern Whisper-based Bemba ASR baseline
3. A direct versus cascaded speech-translation comparison
4. Controlled data-scaling experiments
5. An analysis of ASR error propagation
6. A public demonstration and documented codebase

## Evaluation

ASR:

- WER
- CER

Translation:

- SacreBLEU
- ChrF++
- Semantic metric if computationally feasible
- Wrong-language rate
- Output/reference length ratio

Human analysis:

- Correct
- Minor lexical error
- Major meaning error
- Omission
- Hallucination
- Named-entity error
- Number error
- Truncation
- Wrong-language output

## Reproducibility

Every experiment will produce:

- config.yaml
- metrics.json
- predictions.jsonl
- environment.txt
- Git commit hash
- Random seed
- Dataset-manifest checksum

Model checkpoints and raw data will not be committed to Git.

## Risks

### Compute constraints

Mitigation:

- Begin with Whisper-small
- Use mixed precision
- Run 32-example overfitting tests
- Run subset pilots before full training

### Dataset licensing

Mitigation:

- Do not redistribute the raw dataset
- Keep all data outside Git
- Cite the dataset creators
- Review the dataset license before releasing derived model artifacts

### Translation metric limitations

Mitigation:

- Use multiple automatic metrics
- Inspect model outputs manually
- Report qualitative failure examples

## Definition of a Successful Capstone

The minimum successful project includes:

1. A reproducible BIG-C dataset pipeline
2. One working Bemba ASR model
3. One working direct speech-translation model
4. One working cascaded speech-translation system
5. Quantitative comparison on the official test set
6. Manual error analysis
7. A final notebook or interactive demo
8. A polished technical report
