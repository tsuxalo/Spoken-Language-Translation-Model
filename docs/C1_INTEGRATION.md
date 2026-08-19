# C1 direct Hausa→English S2TT integration

## Outcome and immutable scope

C1 is integrated as a standalone `SpeechEncoderDecoderModel` runtime at revision `cd84a6c2e447b098d772d6ad59b247f16c29075d`. It consumes a raw Hausa waveform and directly generates English with an XLS-R/Wav2Vec2 encoder and mBART-style decoder. It never enters the Whisper/PEFT loader used by the historical direct pilot.

The integration work began from `main` commit `e00bc441b55b9ca2f31091d56d5ef0893c4536f9` (`Added Work Bank`) and was consolidated on `codex/c1-integration`. Work was isolated from the pre-existing dirty checkout; that unrelated checkout was not modified.

## Verified model contract

The complete machine-readable contract is [`artifacts/comparison-v2/model_contracts/direct_c1.json`](../artifacts/comparison-v2/model_contracts/direct_c1.json).

| Field | Verified value |
|---|---|
| Model | `lEtoileNoir/Hausa_English_Direct_S2TT` |
| Revision | `cd84a6c2e447b098d772d6ad59b247f16c29075d` |
| Friendly tag | exists, but targets earlier commit `ba385f7…`; contract-file hashes match |
| Architecture | `SpeechEncoderDecoderModel` |
| Encoder / decoder | `wav2vec2` / `mbart` |
| Processor | `Wav2Vec2Processor` (`Wav2Vec2FeatureExtractor` + `MBart50Tokenizer`) |
| Sampling rate / mask | 16 kHz / attention mask returned |
| Remote code | no `auto_map`; `trust_remote_code=False` |
| Frozen generation | beams 5, max new tokens 128, early stopping, no-repeat 3-gram, repetition/length penalties 1.0 |
| Parameters | Hub safetensors metadata and loaded `sum(parameter.numel())`: 792,989,312 |
| Storage | approximately 3.19 GB |
| Model license | Apache-2.0 |
| Training data | NaijaS2ST revision `898f515…`, CC BY 4.0 attribution required |

## Runtime and audio contract

`direct_c1.py` exposes:

```python
translate_file(wav_path) -> str
translate_array(audio, sampling_rate) -> str
close() -> None
```

The runtime loads lazily in float32, enters evaluation/inference mode, passes the processor's attention mask to `generate`, and does not override the pinned generation configuration. `close()` drops references, runs garbage collection, and clears the CUDA cache.

Decoded audio is converted to NumPy float32, downmixed to mono, checked for empty/nonfinite values, and resampled to exactly 16 kHz with polyphase resampling. The helper performs no ad-hoc amplitude normalization, truncation, or chunking; the pinned Wav2Vec2 feature extractor's verified internal `do_normalize=true` behavior remains part of the model contract. Duration is measured before resampling.

> C1’s frozen evaluation contract covers clips no longer than 30 seconds. Longer clips are excluded from scored evaluation even if the underlying architecture might technically accept them.

Run:

```powershell
python direct_c1.py path\to\hausa.wav
```

## Dependencies

Install a CPU or CUDA PyTorch build appropriate for the machine first; do not install a CPU-only wheel into a shared GPU requirements file. Then:

```powershell
pip install -r requirements-c1.txt
```

`requirements-c1.txt` pins the tested Transformers/PEFT/datasets/audio/metric stack. COMET remains isolated in `requirements-comet.txt`.

## Provenance and membership discovery

The aggregate discovery table is [`artifacts/comparison-v2/provenance_discovery.json`](../artifacts/comparison-v2/provenance_discovery.json). The exact C1 validation manifest was recovered locally: 1,037 rows, 543 alignment groups, all from NaijaS2ST official `train`, six held-out project-validation speakers, maximum duration 29.15 seconds, and matching file/membership hashes. The split audit records no speaker, pair, or normalized-English-target overlap with project training.

Row-level manifests, English references, predictions, speaker IDs, and audio paths remain under ignored `artifacts/comparison-v2/private/`. They are not checked in. The NaijaS2ST card at the pinned revision states CC BY 4.0 and requires attribution.

The historical direct pilot's exact `_pairs.json` training membership was not found. Therefore:

> Direct-pilot training overlap could not be fully audited. This pilot must not be presented as leakage-free.

## Evaluation evidence

Historical metrics are in [`artifacts/comparison-v2/historical_metrics.json`](../artifacts/comparison-v2/historical_metrics.json). Rows retain evaluation membership and evidence scope. The sealed cascade official-dev score, historical direct-pilot score, and C1 internal-development score are not directly comparable.

The notebook runs one pinned FLEURS `ha_ng` test rendition (`id=1856`, audio `5153420622149111720.wav`, dataset revision `70bb2e84…`) through all three systems. This is qualitative only: FLEURS supplies no gold English translation, so BLEU/chrF++ are not calculated.

`comparison_v2.py` first prepared a 10-row canary and then froze the complete 1,037-row project-validation membership. `common_pilot_runner.py` ran all three systems sequentially on the same 543 `alignment_id` clusters, checkpointing private predictions atomically every ten rows so an interrupted run could resume. The harness rejects official `dev`, validates the frozen membership hash and system×example Cartesian product, computes corpus BLEU/chrF++ and degeneration heuristics, and uses 1,000 paired cluster-bootstrap replicates keyed by `alignment_id`. The exact SacreBLEU bootstrap is implemented with precomputed per-cluster sufficient statistics; tests verify parity with direct corpus rescoring.

The checked aggregates are [`artifacts/comparison-v2/full_development_metrics.json`](../artifacts/comparison-v2/full_development_metrics.json) and [`artifacts/comparison-v2/full_development_provenance.json`](../artifacts/comparison-v2/full_development_provenance.json). Row-level membership, references, predictions, and the 752-row human-review queue remain ignored and private.

| System | BLEU (95% cluster CI) | chrF++ (95% cluster CI) | Mean inference | Mean RTF | Peak CUDA |
|---|---:|---:|---:|---:|---:|
| Cascade | 12.56 (11.65–13.32) | 35.61 (34.73–36.45) | 0.892 s | 0.0778 | 2,421 MiB |
| Historical direct pilot | 0.47 (0.35–0.60) | 14.70 (14.25–15.19) | 0.825 s | 0.0647 | 1,072 MiB |
| C1 direct | 0.38 (0.21–0.54) | 16.57 (16.30–16.84) | 0.527 s | 0.0448 | 3,682 MiB |

All 3,111 system/example predictions succeeded and were nonempty. Observed inference totals were 0.257 hours for the cascade, 0.238 hours for the historical direct pilot, and 0.152 hours for C1, excluding model loading and scoring. The cascade had the strongest automatic translation scores on this development membership. C1 exceeded the direct pilot on chrF++ by a bootstrap mean of 1.87 points (95% CI 1.36–2.32), while their BLEU delta was inconclusive. These statements are development observations, not leakage-free or independent test claims.

The fresh float32 C1 package scored BLEU 0.377 / chrF++ 16.574, slightly below the stored historical local-adapter result of 0.499 / 16.668. All 1,037 references and IDs match, but 611 decoded strings differ. A bfloat16 spot check of the published package did not recreate the stored local-adapter outputs. The reports therefore keep the fresh exported-package result and historical result separate instead of silently treating them as bitwise-equivalent.

## Training history availability

| Component | Evidence |
|---|---|
| Hausa Whisper ASR | historical step/evaluation log values recovered from the notebook; training dataset revision is **UNVERIFIED** |
| Historical direct pilot | scalar train/validation loss only; no curve is invented |
| C1 | real step loss history and four validation losses recovered; fixed-window means are documented |
| Base NLLB-600M | pretrained baseline—not fine-tuned in this primary comparison |
| Cascade overall | no single combined training loss exists |

All plotted values come from [`artifacts/comparison-v2/training_histories.json`](../artifacts/comparison-v2/training_histories.json). Loss magnitudes across objectives are not translation-quality evidence.

## Scientific limitations

- C1's 1,037-row development membership influenced model/decoding selection; it is not independent evidence.
- Direct-pilot training overlap is unverified.
- Historical system metrics use different memberships and cannot support a ranking.
- The shared FLEURS clip has no gold English reference.
- Numbers, dates, names, negation, dialect variation, noise, and unusual audio require human review.
- Long-form chunking, quantization, FP16 parity, retraining, and official-dev reruns are outside this integration.
- The published merged C1 package is not bitwise prediction-equivalent to the historical local-adapter evaluation path.
- Free-Colab smoke status is **UNVERIFIED** until somebody runs the exact notebook in Colab.
