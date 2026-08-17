"""Controlled clean/noisy/mixed LoRA adaptation for Hausa -> English NLLB.

This script is intentionally stricter than the earlier pilot:
- every condition receives the same number of training rows;
- mixed is an exact 50/50 clean/noisy mixture (up to one row);
- all conditions are selected using the same noisy validation distribution;
- seed and experiment metadata are written to disk.

Input JSONL rows are expected to contain:
alignment_id, utterance_id, speaker_id, hausa_gold, hausa_asr, english_ref.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    NllbTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "hau_Latn"
TARGET_LANG = "eng_Latn"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def eligible_rows(rows: list[dict]) -> list[dict]:
    required = ("hausa_gold", "hausa_asr", "english_ref")
    kept = []
    for row in rows:
        if all(str(row.get(field, "")).strip() for field in required):
            kept.append(row)
    return kept


def build_examples(rows: list[dict], mode: str, seed: int) -> list[dict]:
    """Build exposure-matched examples.

    Clean/noisy/mixed all produce exactly one MT example per input utterance.
    This removes the confound where mixed/noisy conditions see more optimizer
    updates simply because they contain more rows.
    """
    rows = eligible_rows(rows)
    rng = random.Random(seed)

    mixed_noisy_indices: set[int] = set()
    if mode == "mixed":
        n_noisy = len(rows) // 2
        mixed_noisy_indices = set(rng.sample(range(len(rows)), n_noisy))

    examples = []
    for i, row in enumerate(rows):
        clean = str(row["hausa_gold"]).strip()
        noisy = str(row["hausa_asr"]).strip()
        target = str(row["english_ref"]).strip()

        if mode == "clean":
            source = clean
            source_type = "clean"
        elif mode == "noisy":
            source = noisy
            source_type = "asr_noise"
        elif mode == "mixed":
            if i in mixed_noisy_indices:
                source = noisy
                source_type = "asr_noise"
            else:
                source = clean
                source_type = "clean"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        examples.append(
            {
                "source_text": source,
                "target_text": target,
                "source_type": source_type,
            }
        )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--mode", required=True, choices=["clean", "noisy", "mixed"])
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use if GPU memory is tight.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)

    train_examples = build_examples(train_rows, args.mode, args.seed)

    # Every condition is selected on the same deployment-like distribution.
    val_examples = build_examples(val_rows, "noisy", args.seed)

    if not train_examples or not val_examples:
        raise RuntimeError("Training or validation examples are empty.")

    source_counts = {}
    for example in train_examples:
        source_counts[example["source_type"]] = (
            source_counts.get(example["source_type"], 0) + 1
        )

    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")
    print(f"Training source counts: {source_counts}")
    print("Validation source distribution: 100% ASR noise")

    tokenizer = NllbTokenizerFast.from_pretrained(
        DEFAULT_MODEL,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG,
    )

    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / total_params:.4f}%)"
    )

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["source_text"],
            max_length=args.max_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=args.max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "base_model": args.base_model,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_source_counts": source_counts,
        "validation_distribution": "noisy",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_parameters": trainable_params,
        "total_parameters_with_adapter": total_params,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Saved adapter and metadata to: {output_dir}")


if __name__ == "__main__":
    main()
