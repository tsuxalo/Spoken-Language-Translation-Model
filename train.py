"""Fine-tune whisper-small on the processed Hausa (FLEURS ha_ng) ASR dataset."""

from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from experiments.revisions import (
    OPENAI_WHISPER_SMALL_ID,
    OPENAI_WHISPER_SMALL_REVISION,
)

MODEL_ID = OPENAI_WHISPER_SMALL_ID
DATA_DIR = "./data"
OUTPUT_DIR = "./whisper-small-ha"


@dataclass
class SpeechSeq2SeqCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def main():
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        revision=OPENAI_WHISPER_SMALL_REVISION,
        language="Hausa",
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=OPENAI_WHISPER_SMALL_REVISION,
    )
    model.generation_config.language = "hausa"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_dataset = load_from_disk(f"{DATA_DIR}/train")
    eval_dataset = load_from_disk(f"{DATA_DIR}/test")

    data_collator = SpeechSeq2SeqCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        warmup_steps=50,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
