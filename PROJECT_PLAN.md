# Hausa Speech-to-Text End-to-End Transformer: Execution Plan

## 1. Project Objective
Build and fine-tune a lightweight, end-to-end oral language transformer for Hausa Automatic Speech Recognition (ASR). The goal is to produce a working pipeline that takes raw Hausa audio and outputs transcribed text.

## 2. Architecture & Data Sources
We will avoid training from scratch and instead use a pre-trained multilingual backbone fine-tuned on an open-source dataset. 

*   **Dataset:** We will use the Mozilla Common Voice Corpus 11.0. 
    *   The Common Voice dataset includes Hausa speech data alongside 100 other languages. 
    *   The Hugging Face dataset identifier is `mozilla-foundation/common_voice_11_0`, and we will specifically load the `ha` (Hausa) configuration.
*   **Model Architecture:** We will use a pre-trained Whisper model.
    *   Models like `openai/whisper-small` have been successfully fine-tuned on the Common Voice Hausa dataset for Automatic Speech Recognition.
    *   *Alternative Architecture Note:* `facebook/mms-1b-all` is a 1-billion parameter model based on Wav2Vec2 that supports transcribing over 1,000 languages. However, `whisper-small` is lighter for local prototyping.

## 3. Phase 1: Environment & Dependency Setup
Agent, please open an Ubuntu/bash terminal and execute the following commands to configure the workspace and ensure local audio processing libraries are installed:

1.  `sudo apt-get update && sudo apt-get install -y ffmpeg` (Required for local audio resampling).
2.  `python3 -m venv venv`
3.  `source venv/bin/activate`
4.  Create a `requirements.txt` file with the following:
    ```text
    transformers>=4.30
    datasets
    torch
    torchaudio
    librosa
    evaluate
    jiwer
    accelerate
    ```
5.  `pip install -r requirements.txt`

## 4. Phase 2: Project Scaffolding
Agent, create the following directory structure in the root folder:
*   `data_prep.py` (Script to download and format the Hausa dataset).
*   `train.py` (Script containing the Hugging Face Seq2SeqTrainer logic).
*   `inference.py` (Script to test the model with a sample `.wav` file).
*   `.gitignore` (Standard Python gitignore, plus ignoring data/ and models/ folders).

## 5. Phase 3: Data Ingestion (Agent Instructions for `data_prep.py`)
Agent, write the Python code for `data_prep.py` to do the following:
1.  Import `load_dataset` and `Audio` from the `datasets` library.
2.  Load the Hausa train and test splits from `mozilla-foundation/common_voice_11_0`.
3.  Cast the audio column to a sampling rate of 16,000 kHz, as this is required for Whisper and MMS model inputs.
4.  Initialize the `WhisperProcessor` from `openai/whisper-small`.
5.  Map the dataset to extract input features (Mel-spectrograms) from the audio arrays and token labels from the text transcriptions.
6.  Save the processed dataset to a local `./data` directory.

## 6. Phase 4: Model Training (Agent Instructions for `train.py`)
Agent, write the Python code for `train.py` to do the following:
1.  Load the processed dataset from the `./data` directory.
2.  Initialize the `WhisperForConditionalGeneration` model from `openai/whisper-small`.
3.  Set up a Data Collator specifically for speech-to-text to handle dynamic padding of both audio features and text tokens.
4.  Define a compute metrics function using the `jiwer` library to calculate Word Error Rate (WER).
5.  Configure `Seq2SeqTrainingArguments` optimized for a local GPU (e.g., fp16=True, batch size of 8 or 16, gradient accumulation steps).
6.  Initialize the `Seq2SeqTrainer` and execute `trainer.train()`.

## 7. Phase 5: Version Control
Agent, once the files are generated, execute the following commands in the terminal:
1.  `git init`
2.  `git add .`
3.  `git commit -m "Initial commit: End-to-end Hausa ASR pipeline scaffolding"`