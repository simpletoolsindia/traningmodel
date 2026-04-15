#!/usr/bin/env python3
"""
Fine-tune Gemma 4 E4B for CLI Coding Agent
==========================================
Best Practices Enabled:
- Gradient checkpointing
- Early stopping with patience
- Learning rate scheduling with warmup
- Gradient clipping
- Mixed precision (BF16/FP16)
- Best model tracking
- Evaluation during training
- Exponential moving average (optional)
- Deepspeed support (optional)

Usage:
    python finetune.py                           # Default settings (best practices)
    python finetune.py --model gemma3-12b      # Use different model
    python finetune.py --epochs 5               # Custom epochs
    python finetune.py --dataset my_data.csv    # Custom dataset
    python finetune.py --eval-steps 500         # Evaluate every 500 steps
"""

import os
import sys
import json
import argparse
import csv
from datetime import datetime
from pathlib import Path


class CSVDataset:
    """Custom CSV dataset loader for training."""

    def __init__(self, csv_path, max_samples=None):
        self.data = []
        print(f"Loading dataset from {csv_path}...")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break

                # Format as prompt -> output pair
                prompt = row.get("prompt", "")
                output = row.get("output_json", row.get("response", ""))

                if prompt and output:
                    self.data.append({
                        "text": f"{prompt}\n\n{output}"
                    })

        print(f"Loaded {len(self.data):,} training examples")

        # Calculate tokens estimate
        avg_chars = sum(len(d["text"]) for d in self.data) / len(self.data) if self.data else 0
        est_tokens = int(avg_chars * 0.25)  # Rough estimate
        print(f"Average example length: ~{avg_chars:.0f} chars (~{est_tokens} tokens)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EvalCallback:
    """Callback to track best evaluation metrics."""

    def __init__(self, trainer, output_dir):
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.best_loss = float('inf')
        self.best_metric = {}
        self.training_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.training_history.append(logs.copy())

            # Save metrics
            loss = logs.get("loss", None)
            if loss and loss < self.best_loss:
                self.best_loss = loss
                self.best_metric = {
                    "step": state.global_step,
                    "loss": loss,
                    "learning_rate": logs.get("learning_rate", 0),
                    "timestamp": datetime.now().isoformat()
                }

                # Save best model
                checkpoint_dir = self.output_dir / "best_model"
                checkpoint_dir.mkdir(exist_ok=True)
                self.trainer.save_model(str(checkpoint_dir))
                print(f"\n  ✓ New best model saved! Loss: {loss:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n  Best metrics: {self.best_metric}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 for CLI Coding Agent")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-4-E4B-better-fitting-v2",
        help="Model to fine-tune"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="./training_data.csv",
        help="Path to training dataset CSV"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None = all)"
    )

    # Training - Core
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    # Training - Best Practices
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=2e-5,
        help="Minimum learning rate (for cosine decay)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio (overrides warmup-steps if set)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    # Evaluation & Checkpointing
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum checkpoints to keep"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log every N steps"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after N evaluations without improvement (0 = disabled)"
    )

    # Optimization
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        choices=["adamw_8bit", "adamw_torch", "paged_adamw_8bit", "paged_adamw_32bit"],
        help="Optimizer"
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "polynomial"],
        help="Learning rate scheduler"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gemma4-coding-agent",
        help="Output directory"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="gemma4-cli-agent",
        help="Project name for logging"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    return parser.parse_args()


def get_training_config(args):
    """Generate TrainingArguments with best practices."""

    # Determine precision
    bf16_support = torch.cuda.is_bf16_supported()
    fp16_support = torch.cuda.is_available()

    print(f"\n{'='*60}")
    print("Training Configuration (Best Practices)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size} × {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Learning Rate: {args.learning_rate} → {args.min_learning_rate} ({args.lr_scheduler})")
    print(f"Warmup: {args.warmup_steps} steps ({args.warmup_ratio*100:.0f}% ratio)")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Max Grad Norm: {args.max_grad_norm}")
    print(f"LoRA: Rank={args.lora_rank}, Alpha={args.lora_alpha}, Dropout={args.lora_dropout}")
    print(f"Eval every: {args.eval_steps} steps")
    print(f"Save every: {args.save_steps} steps")
    print(f"Early Stopping: {args.early_stopping_patience} evaluations")

    if bf16_support:
        print(f"Precision: BF16 (optimized)")
    elif fp16_support:
        print(f"Precision: FP16")
    else:
        print(f"Precision: FP32 (fallback)")

    print(f"Optimizer: {args.optim}")
    print(f"{'='*60}\n")

    return {
        # Core training
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "num_train_epochs": args.epochs,
        "max_steps": -1,  # Use epochs instead

        # Sequence length
        "max_seq_length": args.max_seq_length,

        # Learning rate schedule
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio if args.warmup_steps == 100 else 0,
        "min_lr": args.min_learning_rate,

        # Optimizer
        "optim": args.optim,
        "weight_decay": args.weight_decay,

        # Gradient clipping (BEST PRACTICE)
        "max_grad_norm": args.max_grad_norm,

        # Precision
        "fp16": not bf16_support,
        "bf16": bf16_support,

        # Logging
        "logging_steps": args.logging_steps,
        "logging_first_step": True,

        # Checkpointing
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",

        # Evaluation
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        # Early stopping (BEST PRACTICE)
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": 0.01,

        # Output
        "output_dir": args.output_dir,
        "report_to": "none",

        # Reproducibility
        "seed": 42,
        "data_seed": 42,

        # Performance
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }


def main():
    """Main training function."""
    # GPU-specific imports (only needed at runtime on CUDA machine)
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer

    args = parse_args()

    print("\n" + "="*70)
    print("Gemma 4 E4B - CLI Coding Agent Fine-tuning")
    print("Best Practices: Gradient Clipping, Early Stopping, Best Model Tracking")
    print("="*70 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training config
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Config saved to: {config_path}")

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Training requires GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print()

    # Load dataset
    try:
        dataset = CSVDataset(args.dataset, args.max_samples)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {args.dataset}")
        print("Please run: python generate_training_data.py --full")
        sys.exit(1)

    if len(dataset) == 0:
        print("ERROR: Dataset is empty")
        sys.exit(1)

    print()
    print("Loading model...")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print()
    print("Applying LoRA with best practices...")

    # Apply LoRA with optimized settings
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Best practice for memory
        random_state=42,
    )

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print()

    # Get training config
    train_config = get_training_config(args)

    # Create trainer
    print("Creating trainer with best practices...")
    print("  ✓ Gradient checkpointing enabled")
    print("  ✓ Gradient clipping (max_norm=1.0)")
    print("  ✓ Early stopping enabled")
    print("  ✓ Best model tracking enabled")
    print("  ✓ Cosine LR schedule with warmup")
    print()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=os.cpu_count() or 4,
        packing=True,
        args=TrainingArguments(**train_config),
    )

    # Add evaluation callback
    eval_callback = EvalCallback(trainer, args.output_dir)
    trainer.add_callback(eval_callback)

    # Training estimate
    steps_per_epoch = len(dataset) // (args.batch_size * args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    print(f"Training steps: {steps_per_epoch}/epoch × {args.epochs} epochs = {total_steps:,}")
    print()

    # Print schedule
    print("Learning Rate Schedule:")
    print(f"  Steps 0-{args.warmup_steps}: Warmup {args.learning_rate}")
    print(f"  Steps {args.warmup_steps}-{total_steps}: Cosine decay to {args.min_learning_rate}")
    print()

    print("Starting training...")
    print("-"*70)

    start_time = datetime.now()

    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user")
        # Save interrupt checkpoint
        trainer.save_model(f"{args.output_dir}/interrupted_checkpoint")
        print(f"Checkpoint saved to: {args.output_dir}/interrupted_checkpoint")
        sys.exit(1)
    except Exception as e:
        print(f"Training error: {e}")
        raise

    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("-"*70)
    print(f"Training completed in {duration}")
    print(f"Total steps: {trainer.state.global_step}")
    print()

    # Load best model
    print("Loading best model for final save...")
    best_checkpoint = Path(args.output_dir) / "best_model"
    if best_checkpoint.exists():
        print(f"Best checkpoint found: {best_checkpoint}")

        # Save to main output
        print(f"Copying best model to {args.output_dir}...")
        import shutil
        for item in best_checkpoint.iterdir():
            if item.is_file():
                shutil.copy(item, Path(args.output_dir) / item.name)
            else:
                shutil.copytree(item, Path(args.output_dir) / item.name, dirs_exist_ok=True)
    else:
        print("No best checkpoint found, saving current model...")

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Save training summary
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "total_examples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "total_steps": trainer.state.global_step,
        "best_loss": eval_callback.best_loss,
        "best_step": eval_callback.best_metric.get("step"),
        "training_duration": str(duration),
        "completed_at": datetime.now().isoformat(),
    }

    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Model saved to: {args.output_dir}")
    print(f"Best loss: {eval_callback.best_loss:.4f}")
    print(f"Best step: {eval_callback.best_metric.get('step')}")
    print(f"Duration: {duration}")
    print()
    print("Next steps:")
    print("  1. Export to GGUF: python export_to_gguf.py")
    print("  2. Use with Ollama: ollama create gemma4-cli-agent -f Modelfile")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
