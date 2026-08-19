# C1 integration verification report

## Outcome

The C1 Hausa→English direct S2TT integration landed through merged PRs #5 and #6. The pinned exported package runs as a float32 `SpeechEncoderDecoderModel`; the cascade and historical Whisper+LoRA direct pilot remain runnable; the notebook demonstrates the three graphs sequentially; and all three systems completed a shared 1,037-example development evaluation. This follow-up hardens the Colab bootstrap, records the completed CPU smoke test, and is reviewed separately from those merged PRs.

## Repository isolation

- Remote: `https://github.com/tsuxalo/Spoken-Language-Translation-Model.git`
- Historical integration base: `e00bc441b55b9ca2f31091d56d5ef0893c4536f9` (`Added Work Bank`)
- Colab follow-up base: `0c72eaf9a8b822322f82fd7406da2a9178dc90b2` (includes the final research-poster merge)
- Working branch: `codex/c1-integration`
- Worktree: a separate clean clone; the pre-existing dirty checkout was not modified

## Immutable revisions

| Purpose | ID | Revision |
|---|---|---|
| C1 direct | `lEtoileNoir/Hausa_English_Direct_S2TT` | `cd84a6c2e447b098d772d6ad59b247f16c29075d` |
| Cascade ASR | `nahomazmach/whisper-small-ha` | `c4e2b47d88ae8b3ee0a605e09863b93aafca72e3` |
| Cascade MT | `facebook/nllb-200-distilled-600M` | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` |
| Direct-pilot base | `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` |
| Direct-pilot adapter | `nahomazmach/whisper-small-ha-en-direct-pilot` | `a91a4a1155c574a24226de53f053e08b6446806d` |
| NaijaS2ST | `McGill-NLP/NaijaS2ST` | `898f51582750fe244693794f22e3f4b32c5baf95` |
| FLEURS qualitative clip | `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` |

## Runtime contract

`direct_c1.py` provides lazy `translate_file`, `translate_array`, and `close` entry points. It verifies the package architecture, processor, 16 kHz sampling rate, attention mask, parameter count, and frozen beam configuration. Audio is converted to float32, downmixed, checked for finite/nonempty values, and polyphase-resampled to 16 kHz without ad-hoc normalization, truncation, or chunking. Scored clips over 30 seconds are rejected.

C1 is loaded independently from the historical Whisper/PEFT pilot. `qualitative_demo.py` and `common_pilot_runner.py` unload each system before loading the next. The comparison runner resolves audio lazily, decodes one waveform at a time, refuses non-private prediction paths, checkpoints atomically every ten examples, and resumes from a validated partial Cartesian product.

## Membership and leakage audit

- Frozen evaluation: 1,037 renditions / 543 alignment clusters / 6 speakers
- Audio: 3.604 hours, maximum 29.15 seconds; 1,037/1,037 exact cache matches
- Source: NaijaS2ST official `train`, C1 project `validation`
- Official NaijaS2ST `dev` loaded for the C1 evaluation: **false** (the separate
  fixed-ASR GPU MT benchmark has since observed official `dev`)
- Validation manifest SHA-256: `ad70f94e7d9ca3bc2e287bc25c1d2a64fe06428a0fe5484e76be3c18adafd939`
- Frozen membership SHA-256: `afe66df22fe0a86e3bd88fd9bd9c21ccc12a8a70cf34175072e841284b932bd4`
- C1 train overlap: zero example, alignment, normalized-target, and speaker overlap
- Historical direct-pilot training overlap: **UNVERIFIED** because its exact `_pairs.json` was not preserved

> Direct-pilot training overlap could not be fully audited. This evaluation must not be presented as leakage-free.

## Full three-system development evaluation

All 3,111 predictions completed successfully and were nonempty. Intervals use 1,000 paired cluster-bootstrap replicates over the 543 `alignment_id` clusters. The implementation sums precomputed SacreBLEU 2.6 sufficient statistics; direct-rescoring parity is covered by unit tests and an additional full-matrix spot check.

| System | BLEU (95% CI) | chrF++ (95% CI) | Mean / median inference | Mean RTF | Peak CUDA |
|---|---:|---:|---:|---:|---:|
| Cascade | 12.559 (11.650–13.320) | 35.607 (34.727–36.450) | 0.892 / 0.870 s | 0.0778 | 2,421.5 MiB |
| Historical direct pilot | 0.473 (0.353–0.595) | 14.703 (14.245–15.189) | 0.825 / 0.499 s | 0.0647 | 1,071.9 MiB |
| C1 exported package | 0.377 (0.207–0.543) | 16.574 (16.300–16.839) | 0.527 / 0.521 s | 0.0448 | 3,681.5 MiB |

Observed inference-only totals were 0.257 hours for the cascade, 0.238 hours for the direct pilot, and 0.152 hours for C1. C1's repeated-3-gram rate was 0.39%, compared with 57.67% for the historical direct pilot.

Paired bootstrap deltas:

| Delta | BLEU mean (95% CI) | chrF++ mean (95% CI) |
|---|---:|---:|
| C1 − direct pilot | −0.107 (−0.296–0.088) | 1.869 (1.361–2.321) |
| C1 − cascade | −12.125 (−12.979–−11.288) | −19.021 (−19.826–−18.248) |
| Cascade − direct pilot | 12.019 (11.179–12.861) | 20.889 (19.974–21.769) |

The cascade has the strongest automatic scores on this development membership. This is not independent test evidence: C1's validation membership influenced checkpoint/decoding selection, and direct-pilot overlap is unverified. Automatic scores also do not substitute for the 752 queued human reviews of names, dates, numbers, and negation.

Checked aggregate artifacts:

- `artifacts/comparison-v2/full_development_metrics.json`
- `artifacts/comparison-v2/full_development_provenance.json`
- `artifacts/comparison-v2/common_manifest_metrics.json` (the earlier 10-row canary)

Row-level membership, references, predictions, audio paths, and review queues remain under ignored `artifacts/comparison-v2/private/`.

## Historical C1 reconciliation

The fresh exported-package score is BLEU 0.377 / chrF++ 16.574; the stored historical local-adapter score is 0.499 / 16.668. The audit found:

- identical 1,037 IDs and references;
- identical frozen generation settings;
- matching model-shard ETags between the pinned Hub commit and friendly tag;
- 611 different decoded strings;
- zero of two differing examples restored to the historical string by loading the exported package in bfloat16.

Membership and scoring are therefore not the cause. The historical result came from a local step-185 adapter evaluated in bfloat16, while this integration evaluates the exported merged package in its explicit float32 runtime. The remaining package/runtime numerical difference is not proven to a single operation, so both results retain distinct inference-path labels; the report does not claim bitwise equivalence.

## Verification commands

```powershell
python comparison_v2.py prepare-full-manifest <c1-validation-manifest>
python common_pilot_runner.py artifacts/comparison-v2/private/full_development_manifest.json <c1-train-audio-cache> --device cuda --full-development
python comparison_v2.py score-private-predictions artifacts/comparison-v2/private/full_development_predictions.json
python scripts/update_capstone_notebook.py
python -m pytest -q
python -m ruff check .
python -m compileall -q .
git diff --check
```

Final integration checks:

- unit tests: **50 passed**;
- full-repository Ruff: **passed**;
- compile check: **passed**;
- notebook builder: deterministic; valid nbformat 4.5; every code cell parses;
- sanitized Colab evidence: source parity confirmed; 11/11 code cells executed in
  order on CPU with zero error outputs and no traceback text;
- checked JSON artifacts: parse and cross-artifact membership/metric checks pass;
- private artifacts and dependency overlay: ignored by Git;
- secret-pattern scan: no Hugging Face or OpenAI token pattern found;
- clean Python 3.12 CPU environment: dependencies installed, `pip check` passed,
  C1/comparison imports passed, and the pinned live C1 config resolved as
  `SpeechEncoderDecoderModel`;
- `git diff --check`: passed apart from checkout line-ending notices.

## Remaining limitations

- C1's internal validation influenced model and decoding selection.
- Direct-pilot training membership remains unverified.
- The exported package is not bitwise prediction-equivalent to the historical local-adapter path.
- The FLEURS shared demonstration has no gold English translation and is qualitative only.
- Long-form chunking, quantization/FP16 parity, retraining, a direct-C1
  official-dev evaluation, and formal human scoring were not performed.
- The corrected notebook passed all cells on a Colab CPU runtime after removing the incompatible optional `torchao==0.10.0`; this proves end-to-end compatibility but not CUDA execution. The `MANUAL_COLAB_GPU_GATE` remains open until a complete run reports a CUDA device. This does not alter the independently validated aggregate record of the historical GPU handoff experiment.

## Change-control status

The original C1 consolidation and first Colab bootstrap fix are already merged
through PRs #5 and #6. This later follow-up reuses `codex/c1-integration` for a
separate draft review after incorporating current `main`; it does not merge the
new review, upload model/data artifacts, or change Hugging Face visibility.
Ignored private evaluation rows and local dependency environments remain
outside Git.
