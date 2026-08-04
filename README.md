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
- **Mixed precision (fp16):** normally, numbers inside the model are stored with high precision, using more memory and compute than usually necessary. We tell the training to use lower-precision numbers where it's safe to do so, which roughly halves memory use and speeds things up, with a negligible effect on the end result.
- **Measuring progress (Word Error Rate):** after each pass through the data, the model is asked to transcribe some clips it wasn't trained on, and we compare its output to the correct answer using **Word Error Rate (WER)** — essentially, what percentage of words were inserted, deleted, or substituted incorrectly. Lower is better. A WER of 0% would mean perfect transcription.

We've verified this entire training setup actually runs correctly on our GPU (loading data, computing loss, updating the model, computing WER) using a very short trial run of a couple of steps on a handful of examples — enough to confirm nothing is broken, without committing to the time a full training run takes. Full training (multiple complete passes over all ~3,259 training clips) has not been run yet.

## Phase 5 — Trying the model out (`inference.py`)

Once a model is trained, `inference.py` is the piece that makes it actually useful: give it a path to a `.wav` audio file, and it returns the transcribed Hausa text. It loads the fine-tuned model, converts the given audio into the same mel-spectrogram format used during training (so the model sees data in the format it expects), and asks the model to generate a text prediction.

We validated this works correctly using the base, not-yet-fine-tuned Whisper model on a real Hausa clip. The output was recognizably in the right neighborhood phonetically but not accurate — which is expected, since that model hasn't been taught Hausa specifically yet. This test confirmed the mechanics (loading, audio conversion, generation, decoding back to text) all work; transcription quality itself will depend on completing Phase 4's full training run.

The notebook's final section does a related but more visual version of this: it runs 5 examples from the held-out test set through the fine-tuned model and displays a table comparing the correct transcription, the model's prediction, and the WER score for each one — a quick, human-readable way to sanity-check quality.

## Where things stand right now

- Environment, GPU/CUDA support, and all dependencies: set up and confirmed working.
- Data pipeline: fully built and run — the processed training and test data already exist locally.
- Training pipeline: fully built and confirmed to run correctly, but not yet run to completion. The model has not actually been fine-tuned on Hausa yet.
- Inference pipeline: fully built and confirmed to work mechanically, using the untrained base model as a stand-in.
- Notebook: mirrors all of the above, Colab-ready, with the sections for training and inference results built to display a clear "you need to train first" message if run before a fine-tuned model exists.

The next step is running full training, after which the training-loss/WER charts and the 5-example results table in the notebook will actually have something to show.
