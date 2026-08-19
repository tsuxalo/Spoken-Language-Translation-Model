# Spoken-Language-Translation-Model
Translating Hausa Speech to English — and Asking Whether Machine Learning Can Bypass the Need for a Written Language

Collaborative Project with Nahom Azmach, Salim Gloyd, and Karun Mokha

## What this project does

The goal is to take a recording of someone speaking Hausa and get back written **English** text of what they said. The repository now preserves three deployable systems:

- **A cascade** — two separate models chained together: one converts Hausa speech to Hausa text (Automatic Speech Recognition), the other translates that Hausa text to English. This is the project's main, fully-built pipeline — see **Part 1**.
- **The historical Whisper direct pilot** — Whisper-small plus a LoRA adapter that generates English without a Hausa-text intermediate step — see **Part 2**.
- **C1 direct S2TT** — a full-data XLS-R/Wav2Vec2 speech encoder connected to an mBART-style English decoder. It is a self-contained `SpeechEncoderDecoderModel`, not Whisper or PEFT — see [`docs/C1_INTEGRATION.md`](docs/C1_INTEGRATION.md).

All three reuse pretrained speech or language representations rather than training from scratch. Their inference graphs remain distinct, which is why their loaders and evidence are kept separate.

The Whisper-based cascade and historical pilot use `whisper-small`; C1 instead
uses an XLS-R/Wav2Vec2 encoder and mBART decoder. The repository is organized
for single-consumer-GPU inference by loading and unloading systems sequentially.

Below: general setup, the deployable approaches, and evidence-scoped comparison
panels that preserve the boundaries between development and official-dev results.

## Where things stand right now

- Environment, GPU/CUDA support, and all dependencies: set up and confirmed working.
- **The cascade** (Hausa ASR fine-tune + NLLB-200 translation): fully built, trained, evaluated, and published. WER 44.7%, cascade translation BLEU ~8–10 on real ASR output. See **Part 1**.
- **Historical direct pilot**: published Hub inference is preserved. BLEU 0.24 on its own 128-example train-derived pilot validation set. See **Part 2**.
- **C1 direct S2TT**: integrated at immutable revision `cd84a6c2e447b098d772d6ad59b247f16c29075d`, with a CLI, tested audio contract, Colab demo, and artifact-backed documentation.
- **Shared development evaluation:** all three systems completed the same 1,037-example / 543-cluster C1 project-validation membership with 3,111 successful nonempty predictions. The checked result is development evidence, not an independent or leakage-free test.
- **Evidence status:** historical scores from different datasets remain separate from the new common-membership result. The cascade has the strongest automatic scores on the shared development membership; C1 improves chrF++ over the historical direct pilot but not BLEU conclusively.
- **GPU error-aware MT experiment:** completed on the 1,500-example / 500-cluster official NaijaS2ST `dev` set. Mixed leads the BLEU/chrF++ point estimates and noisy leads SSA-COMET; no noisy-versus-mixed statistical-superiority claim is supported. See [`docs/GPU_EXPERIMENT_RESULTS.md`](docs/GPU_EXPERIMENT_RESULTS.md).
- Notebook: Sections 6A/6B preserve the historical pilot and add C1, then run one pinned FLEURS clip through all three deployable systems sequentially.

## Phase 1 — Getting the machine ready

Before any training can happen, the machine needs Python, the ML libraries (PyTorch, Hugging Face's `transformers` and `datasets`, etc.), and `ffmpeg` (a tool used under the hood for handling audio).

The one detail worth understanding here: PyTorch (the library that actually does the math for training neural networks) can run either on the CPU or on the GPU. GPUs are built to do many small calculations at once, which happens to be exactly what training a neural network needs, so training on a GPU is often 10-50x faster than on a CPU. Our machine has an NVIDIA RTX 5050 GPU, so we made sure PyTorch was installed with CUDA support (CUDA is NVIDIA's software layer that lets programs like PyTorch talk to the GPU). Without this step, all the later training would silently run on the CPU and take drastically longer.

## Phase 2 — The notebook

Alongside the plain Python scripts, we built `capstone_demo.ipynb`, a Jupyter notebook that walks through both approaches interactively, with charts, playable audio, and runnable models instead of just terminal output. There are two reasons for having both a notebook and standalone scripts:

- **The scripts (`data_prep.py`, `train.py`, `inference.py`) are the cascade's "real" pipeline** — meant to be run as-is, in order, to actually produce a trained model.
- **The notebook is for exploring and explaining** what's happening at each stage, and it's built to run on Google Colab, not just locally. Not everyone on the team has a GPU on their own machine, and Colab provides free (if limited) GPU access in a browser. The very first cell in the notebook is a Colab compatibility check: if it detects it's running on Colab, it installs all the needed libraries automatically, since Colab doesn't come with them preinstalled the way our local environment does.

The notebook covers architecture, audio exploration, artifact-backed training telemetry, the cascade, Section 6A's historical direct pilot, Section 6B's C1 model, a shared qualitative clip, and evidence-scoped results/error-propagation views.

## Part 1: The Cascade Approach

### Getting and preparing the data (`data_prep.py`)

A model can't learn from raw audio and raw text directly — both need to be converted into a numeric form it can actually process.

**The dataset:** we're using Hausa speech clips from Google's FLEURS dataset (specifically its `ha_ng` — Hausa, Nigeria — portion), each clip paired with its correct written transcription. The original project plan called for a different dataset (Mozilla's Common Voice), but Mozilla moved that dataset off the platform we were pulling it from in October 2025, so we switched to FLEURS, which contains a comparable set of Hausa recordings and is freely accessible.

**Turning audio into numbers (mel-spectrograms):** raw audio is just a long list of amplitude values over time, which isn't a great format for a model to learn patterns from. Instead, we convert each clip into a **mel-spectrogram** — essentially a picture of the audio, showing which frequencies (pitches) are present at each moment in time, on a scale that roughly matches how human hearing perceives pitch. This is the same format Whisper was originally trained on, so it's the format it expects.

**Turning text into numbers (tokenization):** similarly, the written Hausa transcription for each clip is broken into small chunks (tokens) and converted into a sequence of numbers, using the same vocabulary Whisper already knows. This is what the model is being asked to predict, one token at a time, from the audio.

After this conversion, everything is saved to a `./data` folder (a training set and a separate held-out test set) so training doesn't have to redo this conversion work every time.

### Training setup (`train.py`)

This is where the actual learning happens: showing the model many examples of (audio in → correct text out) and adjusting its internal parameters so its predictions get closer to being right.

A few choices worth explaining:

- **Batch size and gradient accumulation:** ideally, a model looks at many examples at once per learning step, since that gives a more stable sense of "which direction to adjust in." But more examples at once also means more GPU memory. Our GPU has a limited 8GB of memory, so we process only 2 clips at a time, but accumulate the learning signal over 4 of those small batches before actually updating the model — approximating a batch of 8 without needing the memory for 8 at once.
- **Mixed precision (fp16):** normally numbers inside the model are stored with high precision, using more memory and compute than usually necessary. We tell the training to use lower-precision numbers where it's safe to do so, which roughly halves memory use and speeds things up, with a negligible effect on the end result.
- **Measuring progress (Word Error Rate):** after each pass through the data, the model is asked to transcribe some clips it wasn't trained on, and we compare its output to the correct answer using **Word Error Rate (WER)** — essentially, what percentage of words were inserted, deleted, or substituted incorrectly. Lower is better. A WER of 0% would mean perfect transcription.

Before committing to a full run, we first verified this setup with a short trial (a couple of steps on a handful of examples) to confirm nothing was broken, then ran full training: 3 complete passes over all ~3,259 training clips on the local GPU, taking about 4.3 hours. Training loss dropped steadily the whole way — from 13.7 at the start down to 0.86 by the end — and WER on the held-out test set improved with each pass: 50.5% after epoch 1, 45.0% after epoch 2, and **44.7%** after epoch 3. That means the fine-tuned model gets a bit under half the words right on speech it never saw during training, a solid result for 3 epochs on a comparatively small, low-resource-language dataset.

### Trying the model out, and translating to English (`inference.py`)

Once a model is trained, `inference.py` is the piece that makes it actually useful: give it a path to a `.wav` audio file, and it returns the transcribed Hausa text. It loads the fine-tuned model, converts the given audio into the same mel-spectrogram format used during training (so the model sees data in the format it expects), and asks the model to generate a text prediction.

We first validated the mechanics (loading, audio conversion, generation, decoding back to text) using the base, not-yet-fine-tuned Whisper model — its output was recognizably in the right neighborhood phonetically but not accurate, as expected for a model that hadn't seen Hausa yet. Now that training is complete, `inference.py` and the notebook load the actual fine-tuned model.

The fine-tuned model is published on the **Hugging Face Hub** at [nahomazmach/whisper-small-ha](https://huggingface.co/nahomazmach/whisper-small-ha), rather than only existing as a ~1GB folder on one laptop. That means anyone — a groupmate, the Colab notebook, a grader — can load it directly with `WhisperForConditionalGeneration.from_pretrained("nahomazmach/whisper-small-ha")` and get real transcriptions immediately, without needing to run any of the training themselves.

**Getting to English:** the project's original goal was Hausa audio → **English** text, not just Hausa text. Rather than training one giant model to do both jobs at once, `inference.py` **chains a second, separate pretrained model** onto the output of the first:

```
[ Hausa audio ] → (our fine-tuned Whisper) → [ Hausa text ] → (NLLB-200) → [ English text ]
```

Meta's **NLLB-200** ([`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)) is a pretrained text-to-text translation model that explicitly supports Hausa→English, so this required no additional training — just loading a second model and adding one more `generate()` call. This "cascaded" ASR + MT setup is the standard architecture most production speech-translation systems actually use, rather than one end-to-end audio-to-foreign-text model. `inference.py` exposes `transcribe_and_translate()`, returning both the Hausa transcription and its English translation.

The notebook's Section 5 does a related but more visual version of this: it downloads both models, runs 5 examples from the held-out test set through the full cascade, and displays a table with the correct Hausa transcription, our model's Hausa prediction, the WER score, and the English translation for each one — a quick, human-readable way to see quality on real examples.

### Cascade Results

| Metric | Epoch 1 | Epoch 2 | Epoch 3 (final) |
|---|---|---|---|
| Eval WER | 50.5% | 45.0% | **44.7%** |
| Eval loss | 0.750 | 0.702 | 0.712 |

![Cascade training loss and WER over training steps](images/cascade_training_curve.png)

**What this picture is showing, in plain terms:** the left chart is the model's **training loss** — a single number that measures "how wrong is the model right now," calculated after every batch of examples it sees. It starts high (the model knows nothing about Hausa yet) and drops fast, then levels off — that flattening-out is normal and expected; it means the model has learned most of what it's going to learn from this amount of data, and further big improvements would need either more data or more training time. The right chart is **WER**, checked three times (once per epoch, i.e. once per full pass through the training data) on audio the model never trained on — it's the real test of whether the model actually got better, not just memorized the training examples. Both charts agreeing (loss down, WER down) is a good sign: the model is really learning to transcribe Hausa, not just overfitting.

Trained model: [nahomazmach/whisper-small-ha](https://huggingface.co/nahomazmach/whisper-small-ha) on the Hugging Face Hub.

**A concrete before/after**, on the same held-out test clip used again in Part 2 below, for direct comparison:

- **Ground truth:** *An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbor.*
- **Base Whisper (untrained on Hausa):** *Aung kwa tanta ferietun ginaginang deke yung sarulin samania Hong Kong dekin shiki mei wal kiya wan da akabayenata gabung ruwan Victoria Habo.*
- **Fine-tuned model (Hausa text):** *An kwatanta, feryatin gina-ginan da ke yin tsarin samaniya, Hong Kong, da ginshiki mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbour.*
- **Cascade's English translation (fine-tuned Hausa → NLLB-200):** *"The crystal structure of the skyline, Hong Kong, and the sparkling column that is depicted by the Victoria Harbour waterfront are illustrated."* — a bit awkwardly phrased, but the meaning (Hong Kong's skyline, glittering buildings, Victoria Harbour) comes through clearly, even after passing through our imperfect Hausa ASR output first.

The base model is barely phonetically related to the correct answer; the fine-tuned model gets nearly every word right. That gap is what the 3 epochs of training actually bought us.

## Part 2: The Direct Approach (Pilot)

The cascade above works and gets the meaning across, but its phrasing can be awkward, especially when it's translating our ASR model's own transcription errors rather than clean Hausa text — errors compound across the two chained models. That raises an obvious question: what if we skip the written-Hausa step entirely, and train a single model to go straight from Hausa audio to English text?

```
[ Hausa audio ]  →  (Whisper encoder/decoder, task="translate", LoRA-adapted)  →  [ English text ]
```

**What we built:** We had already set up infrastructure for exactly this on a separate branch (`feature/direct-s2tt`) ... a LoRA fine-tune of `openai/whisper-small` in its `task=translate` mode (Whisper natively supports outputting English directly, since it was originally trained on translation pairs alongside transcription), trained on real Hausa-audio-to-English-text pairs from [McGill-NLP/NaijaS2ST](https://huggingface.co/datasets/McGill-NLP/NaijaS2ST) (a dataset with genuine English translations, unlike FLEURS which only has Hausa text). Rather than duplicating that work, we ran it: the branch's training code would have downloaded NaijaS2ST's entire ~69GB train split before using any of it, so we first used its own metadata-only audit tooling to find which 2 of its 115 data shards contained the most usable pairs, downloaded only those (a few GB instead of 69GB), and trained a real pilot. Full write-up, the exact scripts run, and reproduction steps are in [`direct_pilot/RESULTS.md`](direct_pilot/RESULTS.md).

**Training:** 256 training examples, 50 steps (3.125 epochs), LoRA — meaning only a small add-on set of weights (1.77 million parameters, 0.7% of the model's 243 million total) gets trained, while the base model stays frozen. Training loss averaged 30.98; validation loss came out to 4.04.

**Measuring translation quality (BLEU and chrF++):** WER (used for the cascade above) doesn't apply here, since we're not checking whether individual transcribed words match a script — we're comparing two English sentences for meaning. Instead we use two standard machine-translation metrics: **BLEU** compares chunks of consecutive words between the model's output and the correct answer (higher = closer phrasing), and **chrF++** does something similar at the character level, which tends to be a bit more forgiving of minor wording differences. Both range from 0 (no overlap) to 100 (a perfect match) — real translations, even good ones, often score in the 30s–40s, since there's more than one correct way to phrase a sentence.

**The same test clip as Part 1, this time through the direct model:**

- **Ground truth:** *An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbor.*
- **Direct pilot's English output:** *"The government has been working on the development of the Hong Kong-based economic system in the region, which is the only way to help the country win the war."*

Fluent-sounding English — grammatically fine — but essentially unrelated to the actual content. That's what a validation BLEU of 0.24 looks like in practice: the model has learned to produce plausible English sentences, but hasn't learned to actually translate Hausa yet.

The pilot model is published on the Hugging Face Hub at [nahomazmach/whisper-small-ha-en-direct-pilot](https://huggingface.co/nahomazmach/whisper-small-ha-en-direct-pilot) — see the notebook's Section 6 for a runnable demo using this exact clip.

### C1 direct S2TT

Run C1 locally with:

```powershell
python direct_c1.py path\to\hausa.wav
```

The CLI verifies the frozen processor/model/generation contract, measures duration from decoded audio, rejects scored clips over 30 seconds, runs float32 inference, and reports timing/real-time factor/peak CUDA memory. See [`docs/C1_INTEGRATION.md`](docs/C1_INTEGRATION.md) for the immutable model contract, dependencies, provenance audit, and limitations.

## Part 3: Evidence-scoped comparison

The repository now has two deliberately separate result panels. They answer
different questions and must not be merged into one leaderboard.

### Panel A — C1 direct S2TT development evidence

```text
Hausa audio → direct SpeechEncoderDecoderModel → English
```

The frozen common membership contains 1,037 examples in 543 alignment
clusters, drawn from NaijaS2ST official `train` / project validation. It is
selection-influenced development evidence, not an independent test. On that
membership, the cascade scored BLEU 12.56 / chrF++ 35.61, the historical direct
pilot 0.47 / 14.70, and the exported C1 package 0.38 / 16.57. The exported C1
package does not bitwise reproduce the historical local-adapter result; both
inference paths remain explicitly separate.

### Panel B — GPU error-aware MT official-dev evidence

```text
Hausa audio → fixed Whisper Hausa ASR → Hausa text → MT system → English
```

The MT systems were evaluated on the same fixed-ASR Hausa text for 1,500
official-dev utterances in 500 clusters. The canonical point estimates are:

| MT system | BLEU | chrF++ | SSA-COMET |
|---|---:|---:|---:|
| NLLB | 13.33 | 37.51 | 0.4446 |
| AfriNLLB | 14.21 | 38.27 | 0.4494 |
| Clean LoRA | 14.21 | 38.31 | 0.4469 |
| Noisy LoRA | 15.26 | 39.12 | **0.4663** |
| Mixed LoRA | **15.36** | **39.36** | 0.4630 |

Mixed leads the BLEU/chrF++ point estimates and noisy leads the SSA-COMET point
estimate. Both noisy and mixed improve over base NLLB in the predeclared paired
cluster bootstrap, but no direct noisy-versus-mixed paired comparison was
saved. This was one training seed, and the intervals describe evaluation-cluster
uncertainty—not training-seed variability. Official NaijaS2ST `dev` has now
been observed and is not an untouched holdout for future tuning.

Full results, recovery provenance, immutable revisions, raw-artifact hashes,
and privacy boundaries are in
[`docs/GPU_EXPERIMENT_RESULTS.md`](docs/GPU_EXPERIMENT_RESULTS.md). The public
machine-readable package is under [`artifacts/gpu-handoff/`](artifacts/gpu-handoff/).

Regenerate and validate it from an authorized private evidence package with:

```powershell
python scripts/build_gpu_handoff_artifacts.py --private-root <private-artifact-root>
python scripts/validate_gpu_handoff_artifacts.py
python scripts/plot_gpu_handoff.py
```

### C1 comparison detail

The historical cascade, historical direct pilot, and C1 development metrics were produced on different memberships. They are retained in [`artifacts/comparison-v2/historical_metrics.json`](artifacts/comparison-v2/historical_metrics.json) and displayed in separate notebook panels; they must not be ranked as a common benchmark.

The notebook's one shared FLEURS example is **qualitative only** because FLEURS supplies a Hausa transcription but no gold English translation. The earlier 10-example canary remains in [`artifacts/comparison-v2/common_manifest_metrics.json`](artifacts/comparison-v2/common_manifest_metrics.json). The complete shared evaluation is in [`artifacts/comparison-v2/full_development_metrics.json`](artifacts/comparison-v2/full_development_metrics.json), with membership and reproducibility details in [`artifacts/comparison-v2/full_development_provenance.json`](artifacts/comparison-v2/full_development_provenance.json).

On the 1,037-example common membership, the cascade scored BLEU 12.56 / chrF++ 35.61, the historical direct pilot 0.47 / 14.70, and the exported C1 package 0.38 / 16.57. C1's chrF++ delta over the direct pilot was +1.87 (95% paired cluster-bootstrap CI +1.36 to +2.32); the BLEU delta included zero. C1's validation membership influenced selection and direct-pilot training overlap remains unverified, so these values must not be presented as independent or leakage-free test evidence.

The existing 25-example ASR-error analysis is preserved with values moved to [`artifacts/comparison-v2/error_propagation.json`](artifacts/comparison-v2/error_propagation.json). Its moderate negative correlations are descriptive supporting evidence, not a causal or definitive benchmark result.

