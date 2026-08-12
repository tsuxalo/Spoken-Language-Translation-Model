# Spoken-Language-Translation-Model
A Framework for Translating Hausa Audio Recordings into English Text

Collaborative Project with N Azamach & K Mokha

## What this project does

The goal is to take a recording of someone speaking Hausa and get back written text of what they said. This is called Automatic Speech Recognition (ASR). Rather than building an ASR model from nothing (which needs enormous amounts of data and compute), we're taking a model that Open AI already trained on many languages — called **Whisper** — and teaching it specifically how to transcribe Hausa. This is called **fine-tuning**: instead of starting from zero, we start from a model that already understands "what speech sounds like" in general, and nudge it toward doing well on Hausa specifically.

We're using the smallest version of Whisper (`whisper-small`) so that it can realistically be trained on a single consumer GPU rather than needing a data center.

Below is what each phase of the pipeline does and why, followed by where things currently stand.

## Phase 1 — Getting the machine ready

Before any training can happen, the machine needs Python, the ML libraries (PyTorch, Hugging Face's `transformers` and `datasets`, etc.), and `ffmpeg` (a tool used under the hood for handling audio).

The one detail worth understanding here: PyTorch (the library that actually does the math for training neural networks) can run either on the CPU or on the GPU. GPUs are built to do many small calculations at once, which happens to be exactly what training a neural network needs, so training on a GPU is often 10-50x faster than on a CPU. Our machine has an NVIDIA RTX 5050 GPU, so we made sure PyTorch was installed with CUDA support (CUDA is NVIDIA's software layer that lets programs like PyTorch talk to the GPU). Without this step, all the later training would silently run on the CPU and take drastically longer.

## Phase 2 — The notebook

Alongside the plain Python scripts, we built `capstone_demo.ipynb`, a Jupyter notebook that walks through the same pipeline interactively, with charts and playable audio instead of just terminal output. There are two reasons for having both a notebook and standalone scripts:

- **The scripts (`data_prep.py`, `train.py`, `inference.py`) are the "real" pipeline** — meant to be run as-is, in order, to actually produce a trained model.
- **The notebook is for exploring and explaining** what's happening at each stage, and it's built to run on Google Colab, not just locally. Not everyone on the team has a GPU on their own machine, and Colab provides free (if limited) GPU access in a browser. The very first cell in the notebook is a Colab compatibility check: if it detects it's running on Colab, it installs all the needed libraries automatically, since Colab doesn't come with them preinstalled the way our local environment does.

The notebook is organized into five sections that mirror the project's phases: an architecture overview, audio exploration, data pipeline verification, training telemetry (charts of how training is going), and interactive inference (trying the model out on real examples).

## Phase 3 — Getting and preparing the data (`data_prep.py`)

A model can't learn from raw audio and raw text directly — both need to be converted into a numeric form it can actually process.

**The dataset:** we're using Hausa speech clips from Google's FLEURS dataset (specifically its `ha_ng` — Hausa, Nigeria — portion), each clip paired with its correct written transcription. The original project plan called for a different dataset (Mozilla's Common Voice), but Mozilla moved that dataset off the platform we were pulling it from in October 2025, so we switched to FLEURS, which contains a comparable set of Hausa recordings and is freely accessible.

**Turning audio into numbers (mel-spectrograms):** raw audio is just a long list of amplitude values over time, which isn't a great format for a model to learn patterns from. Instead, we convert each clip into a **mel-spectrogram** — essentially a picture of the audio, showing which frequencies (pitches) are present at each moment in time, on a scale that roughly matches how human hearing perceives pitch. This is the same format Whisper was originally trained on, so it's the format it expects.

**Turning text into numbers (tokenization):** similarly, the written Hausa transcription for each clip is broken into small chunks (tokens) and converted into a sequence of numbers, using the same vocabulary Whisper already knows. This is what the model is being asked to predict, one token at a time, from the audio.

After this conversion, everything is saved to a `./data` folder (a training set and a separate held-out test set) so training doesn't have to redo this conversion work every time.

## Phase 4 — Training setup (`train.py`)

This is where the actual learning happens: showing the model many examples of (audio in → correct text out) and adjusting its internal parameters so its predictions get closer to being right.

A few choices worth explaining:

- **Batch size and gradient accumulation:** ideally, a model looks at many examples at once per learning step, since that gives a more stable sense of "which direction to adjust in." But more examples at once also means more GPU memory. Our GPU has a limited 8GB of memory, so we process only 2 clips at a time, but accumulate the learning signal over 4 of those small batches before actually updating the model — approximating a batch of 8 without needing the memory for 8 at once.
- **Mixed precision (fp16):** normally numbers inside the model are stored with high precision, using more memory and compute than usually necessary. We tell the training to use lower-precision numbers where it's safe to do so, which roughly halves memory use and speeds things up, with a negligible effect on the end result.
- **Measuring progress (Word Error Rate):** after each pass through the data, the model is asked to transcribe some clips it wasn't trained on, and we compare its output to the correct answer using **Word Error Rate (WER)** — essentially, what percentage of words were inserted, deleted, or substituted incorrectly. Lower is better. A WER of 0% would mean perfect transcription.

Before committing to a full run, we first verified this setup with a short trial (a couple of steps on a handful of examples) to confirm nothing was broken, then ran full training: 3 complete passes over all ~3,259 training clips on the local GPU, taking about 4.3 hours. Training loss dropped steadily the whole way — from 13.7 at the start down to 0.86 by the end — and WER on the held-out test set improved with each pass: 50.5% after epoch 1, 45.0% after epoch 2, and **44.7%** after epoch 3. That means the fine-tuned model gets a bit under half the words right on speech it never saw during training, a solid result for 3 epochs on a comparatively small, low-resource-language dataset.

## Phase 5 — Trying the model out, and translating to English (`inference.py`)

Once a model is trained, `inference.py` is the piece that makes it actually useful: give it a path to a `.wav` audio file, and it returns the transcribed Hausa text. It loads the fine-tuned model, converts the given audio into the same mel-spectrogram format used during training (so the model sees data in the format it expects), and asks the model to generate a text prediction.

We first validated the mechanics (loading, audio conversion, generation, decoding back to text) using the base, not-yet-fine-tuned Whisper model — its output was recognizably in the right neighborhood phonetically but not accurate, as expected for a model that hadn't seen Hausa yet. Now that training is complete, `inference.py` and the notebook load the actual fine-tuned model.

The fine-tuned model is published on the **Hugging Face Hub** at [nahomazmach/whisper-small-ha](https://huggingface.co/nahomazmach/whisper-small-ha), rather than only existing as a ~1GB folder on one laptop. That means anyone — a groupmate, the Colab notebook, a grader — can load it directly with `WhisperForConditionalGeneration.from_pretrained("nahomazmach/whisper-small-ha")` and get real transcriptions immediately, without needing to run any of the training themselves.

**Getting to English:** the project's original goal was Hausa audio → **English** text, not just Hausa text. Rather than training one giant model to do both jobs at once, `inference.py` now **chains a second, separate pretrained model** onto the output of the first:

```
[ Hausa audio ] → (our fine-tuned Whisper) → [ Hausa text ] → (NLLB-200) → [ English text ]
```

Meta's **NLLB-200** ([`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)) is a pretrained text-to-text translation model that explicitly supports Hausa→English, so this required no additional training — just loading a second model and adding one more `generate()` call. This "cascaded" ASR + MT setup is the standard architecture most production speech-translation systems actually use, rather than one end-to-end audio-to-foreign-text model. `inference.py` now exposes `transcribe_and_translate()`, returning both the Hausa transcription and its English translation.

On the same test clip used throughout this README, the cascade produces: *"The crystal structure of the skyline, Hong Kong, and the sparkling column that is depicted by the Victoria Harbour waterfront are illustrated."* — a bit awkwardly phrased, but the meaning (Hong Kong's skyline, glittering buildings, Victoria Harbour) comes through clearly, even after passing through our imperfect Hausa ASR output first.

The notebook's final section does a related but more visual version of this: it downloads both models, runs 5 examples from the held-out test set through the full cascade, and displays a table with the correct Hausa transcription, our model's Hausa prediction, the WER score, and now the English translation for each one — a quick, human-readable way to see quality on real examples.

## Results

| Metric | Epoch 1 | Epoch 2 | Epoch 3 (final) |
|---|---|---|---|
| Eval WER | 50.5% | 45.0% | **44.7%** |
| Eval loss | 0.750 | 0.702 | 0.712 |

Trained model: [nahomazmach/whisper-small-ha](https://huggingface.co/nahomazmach/whisper-small-ha) on the Hugging Face Hub.

**A concrete before/after**, on the same held-out test clip:

- **Ground truth:** *An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbor.*
- **Base Whisper (untrained on Hausa):** *Aung kwa tanta ferietun ginaginang deke yung sarulin samania Hong Kong dekin shiki mei wal kiya wan da akabayenata gabung ruwan Victoria Habo.*
- **Fine-tuned model:** *An kwatanta, feryatin gina-ginan da ke yin tsarin samaniya, Hong Kong, da ginshiki mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbour.*

The base model is barely phonetically related to the correct answer; the fine-tuned model gets nearly every word right. That gap is what the 3 epochs of training actually bought us.

## Where things stand right now

- Environment, GPU/CUDA support, and all dependencies: set up and confirmed working.
- Data pipeline: fully built and run — the processed training and test data exist locally.
- Training pipeline: fully built, run to completion (3 epochs), and confirmed to actually improve the model — see Results above.
- Inference pipeline: fully built and confirmed working against the real fine-tuned model, not just the base model.
- Model: fine-tuned and published to the Hugging Face Hub, so it can be used without anyone retraining it locally.
- **English translation: done.** The pipeline now goes all the way from Hausa audio to English text via a cascaded ASR + NLLB-200 setup — see Phase 5 above.
- Notebook: mirrors all of the above and is Colab-ready — Section 4 (training telemetry) and Section 5 (inference + translation demo) both work out of the box using the published model and NLLB-200, with no local setup required.

## Coming next

The cascade (ASR → NLLB) is working and gets the meaning across, but its phrasing can be awkward, especially when it's translating our ASR model's own transcription errors rather than clean Hausa text — errors compound across the two chained models. Possible next steps:

- Try a **direct** speech-to-text-translation model (Hausa audio → English text in one model, no intermediate Hausa text step) and compare quality against the cascade. This needs real Hausa-audio-paired-with-English-text training data, which the cascade approach doesn't require but a direct model does.
- Evaluate translation quality properly (e.g., BLEU/chrF against reference English text) rather than eyeballing individual examples.
- Revisit the training methodology: our current WER is measured against the same FLEURS test split that was evaluated during training every epoch, rather than a completely held-out final partition — worth tightening up before treating 44.7% as a fully clean number.

## Notes
- ~~build the cascade model and a notebook. NA has already began the ASR portion. Extending it with NLLB?~~ Done — see Phase 5 and "Coming next" above.
- build the direct s2tt model and a notebook. Speech Encoder-Text decoder Transformer
- Our final submission should be a ipnyb file that include the outputs from both notebooks, comparing the architectures
- focusing on hausa due to the amount of available data, with an intent to make this applicable to all spoken languages later. Note that hausa is a tonal language
- note the difference between ASR vs speech encoder
