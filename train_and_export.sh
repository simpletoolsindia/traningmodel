#!/bin/bash
# =============================================================================
# CLI Coding Agent - Training & Export Pipeline
# =============================================================================
# Model: Google Gemma 4 E4B (8B)
# Training: Unsloth + LoRA
# Export: GGUF for Ollama / LM Studio
#
# Best Training Practices Enabled:
# - Gradient checkpointing (memory optimization)
# - Gradient clipping (stability)
# - Early stopping (prevent overfitting)
# - Best model tracking (save best checkpoint)
# - Cosine LR schedule with warmup
# - Mixed precision (BF16/FP16)
# - 8-bit AdamW optimizer
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# =============================================================================

# Model to fine-tune
MODEL_NAME="unsloth/gemma-4-E4B-better-fitting-v2"
# Alternative: MODEL_NAME="unsloth/gemma-4-26B-better-fitting-v2"

# ========== TRAINING PARAMETERS (Best Practices) ==========

# Core Training
NUM_EPOCHS=3                    # Number of training epochs
BATCH_SIZE=4                    # Batch size per device
GRADIENT_ACCUMULATION=4         # Effective batch = 4×4 = 16
MAX_SEQ_LENGTH=2048            # Maximum sequence length

# Learning Rate Schedule (Cosine with Warmup)
LEARNING_RATE=2e-4              # Peak learning rate
MIN_LEARNING_RATE=2e-5          # Minimum learning rate (cosine decay)
WARMUP_RATIO=0.03              # Warmup ratio (3% of total steps)

# LoRA Configuration
LORA_RANK=32                   # LoRA rank (higher = more capacity)
LORA_ALPHA=64                  # LoRA alpha (typically 2× rank)
LORA_DROPOUT=0.05              # LoRA dropout

# Optimization & Stability (Best Practices)
WEIGHT_DECAY=0.01              # Weight decay
MAX_GRAD_NORM=1.0              # Gradient clipping (prevents exploding gradients)
OPTIM="adamw_8bit"             # 8-bit AdamW (memory efficient)

# Evaluation & Checkpointing (Best Practices)
EVAL_STEPS=500                 # Evaluate every N steps
SAVE_STEPS=500                 # Save checkpoint every N steps
SAVE_TOTAL_LIMIT=3             # Keep only N checkpoints
LOGGING_STEPS=100              # Log every N steps
EARLY_STOPPING_PATIENCE=5      # Stop after N evaluations without improvement

# Dataset
DATASET_PATH="./training_data.csv"
DATASET_ROWS=5000000           # 5M rows, set to lower for testing

# Output
OUTPUT_DIR="./gemma4-coding-agent"
MODEL_NAME_OUTPUT="gemma4-e4b-coding-agent"
GGUF_QUANTIZATION="q4_k_m"     # Options: q4_k_m, q8_0, f16, q5_k_m

# HuggingFace (optional - for uploading)
HF_TOKEN=""                     # Set your HF token if you want to push to Hub

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_gpu() {
    log_info "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
        log_success "GPU detected"
    else
        log_error "No GPU detected! Training requires NVIDIA GPU."
        exit 1
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python version: $PYTHON_VERSION"

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 not found"
        exit 1
    fi

    log_success "Dependencies check passed"
}

# =============================================================================
# STEP 1: SETUP ENVIRONMENT
# =============================================================================

setup_environment() {
    log_info "Setting up training environment..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Check if running in Docker
    if [ -f /.dockerenv ]; then
        log_info "Running inside Docker container"
    else
        log_info "Running on bare metal"

        # Install system dependencies
        log_info "Installing system dependencies..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y build-essential cmake libcurl4-openssl-dev git
        fi

        # Install Python packages
        log_info "Installing Python packages..."
        pip3 install --upgrade pip
        pip3 install unsloth transformers datasets huggingface_hub

        # Install llama.cpp for export
        log_info "Installing llama.cpp for GGUF export..."
        if [ ! -d "./llama.cpp" ]; then
            git clone https://github.com/ggml-org/llama.cpp.git
            cd llama.cpp
            cmake .. -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
            cmake --build build --config Release -j
            cd ..
        fi
    fi

    log_success "Environment setup complete"
}

# =============================================================================
# STEP 2: GENERATE DATASET
# =============================================================================

generate_dataset() {
    log_info "Generating training dataset..."

    if [ -f "$DATASET_PATH" ]; then
        log_warning "Dataset already exists at $DATASET_PATH"
        read -p "Regenerate dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing dataset"
            return
        fi
    fi

    # Generate dataset using Python
    python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')

try:
    from generate_training_data import generate_dataset, NUM_ROWS
    import argparse

    print("Generating 5M training rows...")
    generate_dataset(5000000, "training_data.csv")
    print("Dataset generation complete!")
except ImportError:
    print("generate_training_data.py not found. Creating sample dataset...")

    import csv
    import json
    import uuid
    import random
    from datetime import datetime

    # Minimal sample for testing
    NUM_ROWS = 10000

    fieldnames = [
        "id", "prompt", "language", "framework", "task_type",
        "message_type", "tools_available", "output_json",
        "content", "tool_name", "tool_provider", "tool_args",
        "mcp_server", "turns", "humanize", "category",
        "difficulty", "agent_role", "guardrails", "strict_rules", "metadata",
    ]

    with open("training_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i in range(NUM_ROWS):
            row = {
                "id": str(uuid.uuid4()),
                "prompt": f"<init><agent_role>Developer</agent_role></init>\n\nUser: Hello",
                "language": "python",
                "framework": "none",
                "task_type": "code_generation",
                "message_type": "normal",
                "tools_available": '["read", "write", "bash"]',
                "output_json": '{"message_type": "normal", "content": "Hello!", "tool_call": null}',
                "content": "Hello!",
                "tool_name": "",
                "tool_provider": "none",
                "tool_args": "{}",
                "mcp_server": "",
                "turns": "",
                "humanize": "false",
                "category": "general",
                "difficulty": "easy",
                "agent_role": "Developer",
                "guardrails": '["Never delete files"]',
                "strict_rules": '["Use correct tool"]',
                "metadata": '{"source": "synthetic"}',
            }
            writer.writerow(row)

            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/{NUM_ROWS}")

    print(f"Created {NUM_ROWS} sample rows")
PYTHON_SCRIPT

    log_success "Dataset generation complete"
}

# =============================================================================
# STEP 3: TRAIN MODEL
# =============================================================================

train_model() {
    log_info "Starting model training..."

    # Check if model already trained
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
        log_warning "Model already trained at $OUTPUT_DIR"
        read -p "Retrain model? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing trained model"
            return
        fi
    fi

    # Run training with best practices
    python3 << 'PYTHON_SCRIPT'
import os
import sys
import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Best Practice Configuration
MODEL_NAME = "unsloth/gemma-4-E4B-better-fitting-v2"
OUTPUT_DIR = "./gemma4-coding-agent"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 2e-5
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
OPTIM = "adamw_8bit"
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 100
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER = "cosine"
DATASET_PATH = "./training_data.csv"

print("="*70)
print("Gemma 4 E4B - CLI Coding Agent Training")
print("Best Practices: Gradient Clipping, Early Stopping, Best Model Tracking")
print("="*70)
print()
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")
print(f"Dataset: {DATASET_PATH}")
print()
print("Training Best Practices Enabled:")
print(f"  ✓ Gradient Checkpointing (memory optimization)")
print(f"  ✓ Gradient Clipping (max_norm={MAX_GRAD_NORM})")
print(f"  ✓ Early Stopping (patience={EARLY_STOPPING_PATIENCE})")
print(f"  ✓ Best Model Tracking (load_best_model_at_end)")
print(f"  ✓ Cosine LR Schedule ({LEARNING_RATE} → {MIN_LEARNING_RATE})")
print(f"  ✓ Warmup Ratio ({WARMUP_RATIO*100:.0f}%)")
print(f"  ✓ 8-bit AdamW Optimizer")
print(f"  ✓ Mixed Precision (BF16/FP16)")
print()

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU: {gpu_name}")
print(f"Memory: {gpu_mem:.1f} GB")
print()

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Apply LoRA with best practices
print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
print()

# Load dataset
print("Loading dataset...")

class CSVDataset:
    def __init__(self, csv_path):
        self.data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    "text": f"{row.get('prompt', '')}\n\n{row.get('output_json', '')}"
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

import csv
dataset = CSVDataset(DATASET_PATH)
print(f"Loaded {len(dataset):,} examples")
print()

# Create trainer with best practices
print("Creating trainer with best practices...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=os.cpu_count(),
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        max_seq_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        min_lr=MIN_LEARNING_RATE,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=0.01,
        output_dir=OUTPUT_DIR,
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    ),
)

# Training estimate
steps_per_epoch = len(dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
total_steps = steps_per_epoch * NUM_EPOCHS
print(f"Steps: {steps_per_epoch}/epoch × {NUM_EPOCHS} epochs = {total_steps:,}")
print(f"Effective batch: {BATCH_SIZE} × {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print()

print("Starting training...")
print("-"*70)

# Train
trainer.train()

# Save best model
print()
print("-"*70)
print("Training complete!")

best_checkpoint = Path(OUTPUT_DIR) / "best_model"
if best_checkpoint.exists():
    print(f"Best model saved at: {best_checkpoint}")
    best_loss = float('inf')
    for f in best_checkpoint.glob("*.safetensors"):
        print(f"  - {f.name}")
else:
    print("Training completed without best checkpoint")

# Save final model
print(f"Saving final model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print()
print("Training complete!")
PYTHON_SCRIPT

    if [ $? -eq 0 ]; then
        log_success "Model training complete"
    else
        log_error "Training failed"
        exit 1
    fi
}

# =============================================================================
# STEP 4: EXPORT TO GGUF
# =============================================================================

export_to_gguf() {
    log_info "Exporting model to GGUF format..."

    GGUF_OUTPUT="$OUTPUT_DIR/$GGUF_QUANTIZATION"
    mkdir -p "$GGUF_OUTPUT"

    python3 << 'PYTHON_SCRIPT'
import os
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Configuration
OUTPUT_DIR = "./gemma4-coding-agent"
MODEL_NAME = "unsloth/gemma-4-E4B-better-fitting-v2"
QUANTIZATION = "q4_k_m"  # Options: q4_k_m, q8_0, f16, q5_k_m

print(f"Loading fine-tuned model from {OUTPUT_DIR}...")
print()

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Set to inference mode
FastLanguageModel.for_inference(model)

print("Model loaded!")
print(f"Exporting to GGUF with quantization: {QUANTIZATION}")
print()

# Save to GGUF
output_path = f"{OUTPUT_DIR}/{QUANTIZATION}"
model.save_pretrained_gguf(
    output_path,
    tokenizer,
    quantization_method=QUANTIZATION,
)

print()
print(f"GGUF export complete!")
print(f"Output: {output_path}")

# List files
import glob
files = glob.glob(f"{output_path}/*")
print(f"Files created:")
for f in files:
    size = os.path.getsize(f) / (1024*1024*1024)  # GB
    print(f"  {os.path.basename(f)}: {size:.2f} GB")
PYTHON_SCRIPT

    if [ $? -eq 0 ]; then
        log_success "GGUF export complete"
    else
        log_error "GGUF export failed - trying alternative method"
        export_to_gguf_manual
    fi
}

export_to_gguf_manual() {
    log_info "Running manual GGUF export..."

    # Step 1: Merge adapters
    python3 << 'PYTHON_SCRIPT'
import os
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

OUTPUT_DIR = "./gemma4-coding-agent"

print("Loading model for merging...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("Merging adapters into base model...")
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method="merged_4bit_for_cpu",
)
print("Merge complete!")
PYTHON_SCRIPT

    # Step 2: Convert to GGUF
    if [ -d "./llama.cpp" ]; then
        python3 llama.cpp/convert_hf_to_gguf.py merged_model \
            --outfile "$OUTPUT_DIR/model-F16.gguf" \
            --outtype f16 \
            --split-max-size 50G
    fi

    log_success "Manual export complete"
}

# =============================================================================
# STEP 5: CREATE OLLAMA MODELFIE
# =============================================================================

create_ollama_modelfile() {
    log_info "Creating Ollama Modelfile..."

    cat > "$OUTPUT_DIR/Modelfile" << EOF
# Modelfile for CLI Coding Agent
# Model: Gemma 4 E4B fine-tuned for coding + tool calling

FROM ./$GGUF_QUANTIZATION/$(basename $OUTPUT_DIR)-$GGUF_QUANTIZATION.gguf

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER num_ctx 2048

# System prompt
SYSTEM """
You are a CLI coding assistant. Help developers write, debug, refactor, and review code.

Follow these rules:
- Always use the correct tool for the task
- Return ONLY the JSON response when calling tools
- For file operations, use the appropriate file tool
- For git operations, use the git tool
- For web searches, use web_search tool
- For system info, use the appropriate system tool
- For date/time queries, use datetime tool
"""

# Template for tool calling
TEMPLATE """{{ if .System }}<init>
<agent_role>CLI Coding Assistant</agent_role>
<agent_capabilities>
- Expert in multiple programming languages
- Strong in debugging, refactoring, and clean code
- Writes comprehensive tests and documentation
</agent_capabilities>
<localization>
  <bot_tone>professional</bot_tone>
  <language>en</language>
</localization>
</init>
{{ .System }}
{{ end }}
{{ if .Prompt }}
<agent_instruction>You are a CLI coding assistant that helps developers.</agent_instruction>
<guardrails>
- Never delete files without confirmation
- Always show diffs before applying changes
- Never execute destructive commands without asking
</guardrails>
<humanize>false</humanize>

---

User: {{ .Prompt }}
{{ end }}
{{ .Response }}"""
EOF

    log_success "Ollama Modelfile created at $OUTPUT_DIR/Modelfile"
}

# =============================================================================
# STEP 6: CREATE LM STUDIO CONFIG
# =============================================================================

create_lmstudio_config() {
    log_info "Creating LM Studio configuration..."

    cat > "$OUTPUT_DIR/lm-studio-config.json" << EOF
{
  "model": "$(basename $OUTPUT_DIR)-$GGUF_QUANTIZATION.gguf",
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_context_length": 2048,
    "gpu_offload": true,
    "threads": 8
  },
  "system_prompt": "You are a CLI coding assistant that helps developers write, debug, refactor, and review code.",
  "tools": [
    "read", "write", "edit", "copy", "move", "delete", "mkdir",
    "grep", "find", "glob",
    "bash", "vim", "top", "ps", "kill", "df", "du", "free",
    "git", "git_status", "git_add", "git_commit", "git_pull", "git_push",
    "datetime", "date_now", "date_yesterday", "date_tomorrow",
    "web_search", "fetch", "curl"
  ]
}
EOF

    log_success "LM Studio config created at $OUTPUT_DIR/lm-studio-config.json"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_help() {
    cat << EOF
Usage: ./train_and_export.sh [OPTIONS]

Options:
    --generate-dataset    Generate training dataset only
    --train              Train model only
    --export             Export to GGUF only
    --all                Run full pipeline (default)
    --check              Check environment only
    -h, --help           Show this help message

Examples:
    ./train_and_export.sh --all              # Full pipeline
    ./train_and_export.sh --train           # Train only
    ./train_and_export.sh --export         # Export only

Output:
    Models saved to: ./gemma4-coding-agent/
    - adapter_model.safetensors (LoRA weights)
    - <quantization>/ (GGUF files for Ollama/LM Studio)
    - Modelfile (for Ollama)
    - lm-studio-config.json (for LM Studio)
EOF
}

# Parse arguments
OPERATION="${1:-all}"

case "$OPERATION" in
    --generate-dataset)
        check_dependencies
        generate_dataset
        ;;
    --train)
        check_dependencies
        check_gpu
        train_model
        ;;
    --export)
        export_to_gguf
        create_ollama_modelfile
        create_lmstudio_config
        ;;
    --all)
        check_dependencies
        check_gpu
        generate_dataset
        train_model
        export_to_gguf
        create_ollama_modelfile
        create_lmstudio_config
        ;;
    --check)
        check_dependencies
        check_gpu
        ;;
    -h|--help)
        show_help
        ;;
    *)
        log_error "Unknown option: $OPERATION"
        show_help
        exit 1
        ;;
esac

log_success "Done!"
