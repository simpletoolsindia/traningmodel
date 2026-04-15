#!/usr/bin/env python3
"""
Export Fine-tuned Model to GGUF Format
=======================================

Converts the fine-tuned LoRA model to GGUF for use with:
- Ollama
- LM Studio
- llama.cpp
- Jan AI
- Open WebUI

Usage:
    python export_to_gguf.py                           # Default Q4_K_M
    python export_to_gguf.py --quant q8_0            # Q8 quantization
    python export_to_gguf.py --model ./my-model       # Custom model path
"""

import os
import sys
import argparse
import glob
import shutil
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export model to GGUF format")

    parser.add_argument(
        "--model",
        type=str,
        default="./gemma4-coding-agent",
        help="Path to fine-tuned model directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: model dir)"
    )

    parser.add_argument(
        "--quant",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16", "q5_k_m", "q2_k", "q3_k_m"],
        help="Quantization method"
    )

    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push to HuggingFace Hub (format: username/repo-name)"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for pushing"
    )

    return parser.parse_args()


def export_with_unsloth(args):
    """Export using Unsloth's built-in method."""
    print("=" * 70)
    print("Exporting with Unsloth (Recommended)")
    print("=" * 70)
    print()

    try:
        import torch
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: Unsloth not installed")
        print("Run: pip install unsloth")
        return False

    print(f"Loading model from: {args.model}")
    print(f"Quantization: {args.quant}")
    print()

    # Load model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

    # Set inference mode
    FastLanguageModel.for_inference(model)

    # Output path
    output_dir = args.output_dir or args.model
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting to: {output_dir}")
    print("This may take several minutes...")
    print()

    # Export to GGUF
    try:
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method=args.quant,
        )
    except Exception as e:
        print(f"Export error: {e}")
        print()
        print("Trying alternative method...")
        return export_manual(args)

    # List output files
    print()
    print("Export complete!")
    print()
    print("Files created:")

    for f in glob.glob(f"{output_dir}/*"):
        if f.endswith('.gguf'):
            size_gb = os.path.getsize(f) / (1024**3)
            print(f"  {os.path.basename(f)}: {size_gb:.2f} GB")

    return True


def export_manual(args):
    """Manual export using llama.cpp."""
    print("=" * 70)
    print("Manual Export with llama.cpp")
    print("=" * 70)
    print()

    # Check for llama.cpp
    llama_dir = "./llama.cpp"

    if not os.path.exists(llama_dir):
        print("Cloning llama.cpp...")
        os.system("git clone https://github.com/ggml-org/llama.cpp.git")
        print()

    print("Building llama.cpp (this may take a while)...")
    os.system(f"""
        cd {llama_dir} && \
        cmake .. -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON && \
        cmake --build build --config Release -j
    """)

    print()
    print("Merging model...")

    try:
        import torch
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Merge and save
        output_dir = args.output_dir or args.model
        os.makedirs(output_dir, exist_ok=True)

        print("Merging adapters into base model...")
        model.save_pretrained_merged(
            "merged_model",
            tokenizer,
            save_method="merged_4bit_for_cpu",
        )

        print()
        print("Converting to GGUF...")

        # Convert to GGUF
        os.system(f"""
            python {llama_dir}/convert_hf_to_gguf.py merged_model \
                --outfile {output_dir}/model-F16.gguf \
                --outtype f16
        """)

        return True

    except Exception as e:
        print(f"Manual export failed: {e}")
        return False


def create_modelfile(output_dir, quant):
    """Create Ollama Modelfile."""
    print()
    print("Creating Ollama Modelfile...")

    dq = '"'  # Double quote helper
    ts = "{{"  # Escaped brace for f-strings
    te = "}}"

    modelfile = f"""# Modelfile for CLI Coding Agent
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Model: Gemma 4 E4B fine-tuned
# Quantization: {quant}

FROM ./{quant}/model.gguf

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1

# System prompt for coding agent behavior
SYSTEM {dq}You are an expert CLI coding assistant.

Follow these rules:
- Use the correct tool for each task
- Return ONLY valid JSON when calling tools
- File operations: read, write, edit, copy, move, delete, mkdir
- Git operations: git_status, git_add, git_commit, git_pull, git_push
- System: bash, top, ps, df, free
- Web: web_search, fetch
- DateTime: datetime, date_now, date_yesterday, date_tomorrow

Available tools format:
{dq}{{{ts}
  {dq}message_type{dq}: {dq}tool_call{dq},
  {dq}tool_call{dq}: {{
    {dq}name{dq}: {dq}tool_name{dq},
    {dq}arguments{dq}: {{}}
  }}
{dq}{te}{dq}

TEMPLATE {dq}<init>
<agent_role>CLI Coding Assistant</agent_role>
<agent_capabilities>
- Expert in multiple programming languages
- Strong debugging and problem-solving skills
- Writes clean, maintainable code
</agent_capabilities>
<localization>
  <bot_tone>professional</bot_tone>
  <language>en</language>
</localization>
</init>

---

User: {ts}.Prompt{te}

Assistant: {dq}
"""

    filepath = os.path.join(output_dir, "Modelfile")
    with open(filepath, 'w') as f:
        f.write(modelfile)

    print(f"  Created: {filepath}")

    return filepath


def create_lmstudio_config(output_dir, quant):
    """Create LM Studio configuration."""
    print()
    print("Creating LM Studio configuration...")

    import json

    config = {
        "model_name": "Gemma 4 E4B CLI Coding Agent",
        "quantization": quant,
        "generated": datetime.now().isoformat(),
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_context_length": 2048,
            "gpu_offload": True,
            "threads": 8
        },
        "system_prompt": "You are an expert CLI coding assistant. Use tools via JSON format.",
        "tools": {
            "file_operations": [
                "read", "write", "edit", "copy", "move", "delete", "mkdir",
                "stat", "exists", "list"
            ],
            "search": ["grep", "find", "glob"],
            "system": [
                "bash", "vim", "top", "ps", "kill",
                "df", "du", "free", "uptime", "uname", "whoami"
            ],
            "git": [
                "git", "git_status", "git_add", "git_commit",
                "git_pull", "git_push", "git_branch", "git_stash",
                "git_log", "git_diff", "git_checkout", "git_merge"
            ],
            "datetime": [
                "datetime", "date_now", "date_yesterday",
                "date_tomorrow", "date_add", "date_diff"
            ],
            "web": ["web_search", "fetch", "curl", "ping"]
        }
    }

    filepath = os.path.join(output_dir, "lm-studio-config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Created: {filepath}")

    return filepath


def create_readme(output_dir, quant):
    """Create README with usage instructions."""
    print()
    print("Creating README...")

    readme = f"""# Gemma 4 E4B - CLI Coding Agent

Fine-tuned model for CLI coding assistance with tool calling capabilities.

## Model Details

- **Base Model**: Google Gemma 4 E4B (8B)
- **Fine-tuning**: LoRA with Unsloth
- **Quantization**: {quant.upper()}
- **Purpose**: CLI coding agent with tool calling

## Tools Supported

### File Operations
- `read`, `write`, `edit`, `copy`, `move`, `delete`, `mkdir`
- `stat`, `exists`, `list`

### Search
- `grep`, `find`, `glob`

### System Commands
- `bash`, `vim`, `top`, `ps`, `kill`
- `df`, `du`, `free`, `uptime`, `uname`, `whoami`

### Git Operations
- `git`, `git_status`, `git_add`, `git_commit`
- `git_pull`, `git_push`, `git_branch`, `git_stash`
- `git_log`, `git_diff`, `git_checkout`, `git_merge`

### DateTime
- `datetime`, `date_now`, `date_yesterday`
- `date_tomorrow`, `date_add`, `date_diff`

### Web
- `web_search`, `fetch`, `curl`, `ping`

## Usage

### Ollama

```bash
# Import model
ollama create gemma4-cli-agent -f Modelfile

# Run
ollama run gemma4-cli-agent

# Example prompt
/User: List all Python files in src directory
```

### LM Studio

1. Open LM Studio
2. Click "Import Model"
3. Select the GGUF file
4. Load the model
5. Use the chat interface

### llama.cpp

```bash
# Run with CLI
./llama.cpp/build/bin/llama-cli \\
    -m {quant}/model.gguf \\
    -c 2048 \\
    --temp 0.7 \\
    -p "You are a CLI coding assistant..."

# Run server
./llama.cpp/build/bin/llama-server \\
    -m {quant}/model.gguf \\
    -c 2048 \\
    --port 8080
```

### Python (transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./gemma4-coding-agent",
    quantization_config=load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("./gemma4-coding-agent")
```

## Response Format

The model returns JSON in this format:

```json
{{
  "message_type": "tool_call",
  "tool_call": {{
    "name": "bash",
    "arguments": {{
      "command": "ls -la",
      "working_dir": "/project",
      "timeout": 30
    }}
  }}
}}
```

For normal responses:

```json
{{
  "message_type": "normal",
  "content": "Here's the Python code...",
  "tool_call": null,
  "mcp_call": null
}}
```

## Training Details

- **Dataset**: {5_000_000:,} examples
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **LoRA Rank**: 32
- **Sequence Length**: 2048

## Files

- `*.gguf` - Quantized model files
- `Modelfile` - Ollama configuration
- `lm-studio-config.json` - LM Studio configuration
- `tokenizer*` - Tokenizer files
- `adapter_model.safetensors` - LoRA adapter weights

## License

Gemma 4 terms apply. Fine-tuned model inherits base model license.
"""

    filepath = os.path.join(output_dir, "README.md")
    with open(filepath, 'w') as f:
        f.write(readme)

    print(f"  Created: {filepath}")

    return filepath


def main():
    """Main export function."""
    args = parse_args()

    print("=" * 70)
    print("GGUF Export for CLI Coding Agent")
    print("=" * 70)
    print()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print("Please run training first: python finetune.py")
        sys.exit(1)

    output_dir = args.output_dir or args.model

    # Export
    success = export_with_unsloth(args)

    if not success:
        sys.exit(1)

    # Create configs
    create_modelfile(output_dir, args.quant)
    create_lmstudio_config(output_dir, args.quant)
    create_readme(output_dir, args.quant)

    # Push to Hub if requested
    if args.push_to_hub:
        print()
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")

        try:
            from unsloth import FastLanguageModel
            from transformers import AutoTokenizer

            model, tokenizer = FastLanguageModel.from_pretrained(args.model)

            model.push_to_hub_gguf(
                args.push_to_hub,
                tokenizer,
                quantization_method=args.quant,
                token=args.hf_token,
            )

            print(f"Pushed to: https://huggingface.co/{args.push_to_hub}")

        except Exception as e:
            print(f"Push failed: {e}")
            print("Model saved locally only.")

    print()
    print("=" * 70)
    print("Export complete!")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Use with Ollama: ollama create gemma4-cli-agent -f Modelfile")
    print("  2. Use with LM Studio: Import the GGUF file")
    print("  3. Run with llama.cpp: ./llama-server -m model.gguf")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
