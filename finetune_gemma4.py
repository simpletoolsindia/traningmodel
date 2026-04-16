#!/usr/bin/env python3
"""
Fine-tune Gemma 4 E4B — YOUR Custom Format
==========================================
Train using YOUR custom XML prompt + JSON tool call response format.

Dataset: training_data.jsonl (from generate_gemma_dataset.py)
Input:   <init>...<User:>task</User:>
Output:  {"message_type": "tool_call", "tool_call": {...}}

Usage:
    python finetune_gemma4.py --dataset training_data.jsonl --epochs 3
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 — Your Custom Format")
    p.add_argument("--model", default="unsloth/gemma-4-E4B-it")
    p.add_argument("--dataset", default="training_data.jsonl")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--output-dir", default="./gemma4-custom-agent")
    p.add_argument("--max-samples", type=int, default=None,
                    help="Limit dataset size for quick testing")
    return p.parse_args()


def load_jsonl_dataset(path, max_samples=None):
    """Load YOUR custom format JSONL dataset."""
    data = []
    print(f"Loading YOUR custom format dataset: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            row = json.loads(line)
            if "text" in row:
                data.append(row)

    print(f"  Loaded {len(data):,} examples")
    if data:
        avg = sum(len(d["text"]) for d in data) / len(data)
        print(f"  Avg length: ~{avg:.0f} chars (~{int(avg * 0.25)} tokens)")

    return data


def formatting_func(example, tokenizer):
    """Format YOUR custom text field for training."""
    return example["text"]


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Gemma 4 E4B — YOUR Custom Format Fine-tuning")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Input:      <init>...<User:>...</User:>")
    print(f"Output:     {{'message_type': 'tool_call', 'tool_call': {{...}}}}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch:      {args.batch_size} x {args.gradient_accumulation}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples:,}")
    print("=" * 70 + "\n")

    # GPU check
    import torch
    if not torch.cuda.is_available():
        print("ERROR: GPU required")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    if not Path(args.dataset).exists():
        print(f"ERROR: {args.dataset} not found")
        print("Run: python generate_gemma_dataset.py --rows 50000 --output training_data.jsonl")
        sys.exit(1)

    dataset = load_jsonl_dataset(args.dataset, args.max_samples)
    if not len(dataset):
        print("ERROR: Empty dataset")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Load model
    from unsloth import FastLanguageModel
    print("\nLoading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Apply LoRA
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Training args
    from unsloth import UnslothTrainingArguments
    from trl import SFTTrainer

    bf16 = torch.cuda.is_bf16_supported()

    train_args = UnslothTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=100,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        fp16=not bf16,
        bf16=bf16,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        packing=True,
    )

    # Formatting function — handles YOUR custom text field
    def formatter(example):
        return example["text"]

    # Trainer — use formatting_func instead of dataset_text_field
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatter,
        max_seq_length=args.max_seq_length,
        args=train_args,
    )

    steps = len(dataset) // (args.batch_size * args.gradient_accumulation)
    total_steps = steps * args.epochs
    print(f"\nSteps: {steps}/epoch x {args.epochs} epochs = {total_steps:,}")
    print(f"Precision: {'BF16' if bf16 else 'FP16'}")
    print("\nStarting training...")
    print("-" * 70)

    start = datetime.now()
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_model(f"{args.output_dir}/interrupt")
        sys.exit(1)

    duration = datetime.now() - start
    print(f"\nTraining complete: {duration}")
    print(f"Steps: {trainer.state.global_step:,}")

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(f"{args.output_dir}/training_summary.json", "w") as f:
        json.dump({
            "model": args.model,
            "examples": len(dataset),
            "epochs": args.epochs,
            "lora_rank": args.lora_rank,
            "duration": str(duration),
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Model saved: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
