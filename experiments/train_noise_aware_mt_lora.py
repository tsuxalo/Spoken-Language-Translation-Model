"""LoRA fine-tune NLLB on clean + real ASR-corrupted Hausa.

Research hypothesis:

A generic MT model is normally trained on clean written Hausa, but at
inference time our cascade feeds it imperfect ASR Hausa.

We reduce this train/test mismatch by adapting NLLB on:

    clean Hausa -> English

and

    ASR-generated Hausa -> the SAME English

LoRA keeps the experiment computationally realistic by training a small
fraction of NLLB's parameters rather than updating the entire 600M model.
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    NllbTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

SOURCE_LANG = "hau_Latn"
TARGET_LANG = "eng_Latn"


def read_jsonl(path: str) -> list[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def build_examples(
    rows: list[dict],
    mode: str,
) -> list[dict]:
    """Construct clean/noisy Hausa -> English training examples.

    For clean examples we deduplicate by alignment ID because multiple
    speakers can read the same underlying sentence.

    For noisy examples we KEEP the multiple ASR versions because speaker and
    acoustic variation can produce different recognition errors. Those
    different errors are exactly the useful supervision we want.
    """

    examples = []
    clean_seen = set()

    for row in rows:
        target = row["english_ref"]

        if mode in {"clean", "mixed"}:
            alignment_id = row["alignment_id"]

            if alignment_id not in clean_seen:
                examples.append(
                    {
                        "source_text": row["hausa_gold"],
                        "target_text": target,
                        "source_type": "clean",
                    }
                )

                clean_seen.add(alignment_id)

        if mode in {"noisy", "mixed"}:
            noisy_text = row["hausa_asr"].strip()

            if noisy_text:
                examples.append(
                    {
                        "source_text": noisy_text,
                        "target_text": target,
                        "source_type": "asr_noise",
                    }
                )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-jsonl",
        required=True,
    )

    parser.add_argument(
        "--dev-jsonl",
        required=True,
    )

    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["clean", "noisy", "mixed"],
        help="Which source distribution to train on.",
    )

    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/nllb-hausa-error-aware-lora",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    args = parser.parse_args()

    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)

    train_examples = build_examples(
        train_rows,
        args.mode,
    )

    dev_examples = build_examples(
        dev_rows,
        args.mode,
    )

    print(f"Training examples:   {len(train_examples):,}")
    print(f"Validation examples: {len(dev_examples):,}")
    print(f"Training mode:       {args.mode}")

    train_dataset = Dataset.from_list(
        train_examples
    )

    dev_dataset = Dataset.from_list(
        dev_examples
    )

    tokenizer = NllbTokenizerFast.from_pretrained(
        DEFAULT_MODEL,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG,
    )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
    )

    # Adapt only attention query/value projections.
    # This updates a very small subset of the 600M model's parameters.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # Disable caching during training because the model needs gradients.
    model.config.use_cache = False

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["source_text"],
            max_length=256,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=256,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_dev = dev_dataset.map(
        preprocess,
        batched=True,
        remove_columns=dev_dataset.column_names,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        per_device_eval_batch_size=args.batch_size,

        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,

        fp16=torch.cuda.is_available(),

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        logging_steps=25,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # For a PEFT model this stores the compact LoRA adapter rather than
    # duplicating the entire 600M-parameter NLLB checkpoint.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print()
    print(f"LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()