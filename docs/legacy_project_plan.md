# Hausa Speech-to-Text Pipeline: Master Execution Plan

Agent, we are building an end-to-end Hausa ASR transformer using an "on the go" workflow. You will write the modular Python scripts while simultaneously building a Jupyter Notebook (`capstone_demo.ipynb`) to test and visualize each phase.

## Phase 1: Environment Correction
Execute the following terminal commands to ensure PyTorch is using CUDA for the local RTX 5050 GPU:
1. `pip uninstall torch torchaudio -y`
2. `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

## Phase 2: Create the Colab-Ready Capstone Notebook
Generate a Jupyter Notebook named `capstone_demo.ipynb` in the root folder. 

**Notebook Requirements:**
*   **Markdown Header:** The very first cell must be Markdown and contain this exact badge code (I will update USERNAME later):
    `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/hausa-asr-transformer/blob/main/capstone_demo.ipynb)`
*   **Cell 1 (Colab Setup):** The first code cell must be:
    ```python
    # Run this cell if executing in Google Colab to install dependencies
    import sys
    if 'google.colab' in sys.modules:
        !pip install transformers datasets torch torchaudio librosa evaluate jiwer accelerate
    ```
*   **Structure:** Add placeholder Markdown sections for: "1. Architecture Overview", "2. Audio Exploratory Data Analysis", "3. Data Pipeline Verification", "4. Training Telemetry", and "5. Interactive Inference."

## Phase 3: Data Ingestion (`data_prep.py` & Notebook Sec 2)
1.  **Write `data_prep.py`:** Use `datasets` to load `mozilla-foundation/common_voice_11_0` (Hausa split: `ha`). Cast audio to 16kHz, initialize `openai/whisper-small` processor, and map the dataset to extract log-mel spectrograms. Save processed data to `./data`.
2.  **Update Notebook Section 2:** Write code in the notebook to load 3 raw Hausa audio clips. Use `IPython.display.Audio` to make them playable, and use `librosa` and `matplotlib` to plot their Waveforms and Mel-Spectrograms side-by-side.

## Phase 4: Training Setup (`train.py` & Notebook Sec 4)
1.  **Write `train.py`:** Initialize `WhisperForConditionalGeneration` (`openai/whisper-small`). Set `per_device_train_batch_size=2` and `gradient_accumulation_steps=4` (optimized for an 8GB VRAM RTX 5050). Configure the `Seq2SeqTrainer` using `jiwer` to calculate Word Error Rate (WER).
2.  **Update Notebook Section 4:** Write a cell in the notebook that parses the training logs (once generated) and uses `seaborn` or `matplotlib` to plot Training Loss and WER over time.

## Phase 5: Inference (`inference.py` & Notebook Sec 5)
1.  **Write `inference.py`:** Write a function that takes a raw Hausa `.wav` file, runs it through the fine-tuned model, and returns the transcribed text.
2.  **Update Notebook Section 5:** Write a cell to run 5 test samples through the model. Output a Pandas DataFrame displaying: `[Sample ID | Ground Truth Hausa | Model Output | WER Score]`.