# Gemma 4 CLI Coding Agent — Training Pipeline

> **Base Model**: Google Gemma 4 E4B (8B)
> **Training Platform**: RunPod (Unsloth + LoRA)
> **Format**: JSONL with `messages` array (HuggingFace/Unsloth standard)

---

## What Was Fixed

The original `generate_training_data.py` had **critical format errors** that caused training to fail with Unsloth:

| Problem | Original (WRONG) | Fixed (CORRECT) |
|---------|-----------------|-----------------|
| Format | CSV with `prompt` + `output_json` concatenated as plain text | JSONL with `messages` array |
| Template | Custom XML (`<init>`, `<agent_role>`, etc.) | Gemma 4 native (`<\|turn>user` / `<\|turn>model`) |
| Role names | `"assistant"` in output + custom `"tool_call"` key | Standard `"assistant"` role (template maps to `"model"`) |
| Tool calls | `{"role": "assistant", "tool_call": {...}}` (malformed) | `<tool_call>\n{...}\n</tool_call>` in content text |
| Dataset type | CSV (slow, hard to parse) | JSONL (fast, streaming) |
| Input to trainer | `{text: "prompt\n\noutput"}` | Pre-tokenized `text` column with chat template |

---

## Complete Pipeline

### Step 1: Generate Dataset (Local Machine — no GPU needed)

```bash
# Small sample (fast, for testing)
python generate_gemma_dataset.py --rows 50000 --output my_data.jsonl

# Full dataset (5M rows)
python generate_gemma_dataset.py --full
```

Output: `gemma_dataset.jsonl` — JSONL with `messages` array format

```json
{
  "messages": [
    {"role": "system", "content": "You are a Senior Python Developer..."},
    {"role": "user", "content": "Write a FastAPI endpoint"},
    {"role": "assistant", "content": "Here's the FastAPI endpoint..."}
  ],
  "metadata": {"language": "python", "difficulty": "medium", ...}
}
```

### Step 2: Validate (Optional but recommended)

```bash
python validate_dataset.py my_data.jsonl --max 10000
```

### Step 3: Convert & Tokenize (RunPod GPU — required for Unsloth)

Upload `my_data.jsonl` to RunPod, then:

```bash
python convert_for_gemma.py \
    --input my_data.jsonl \
    --output gemma_tokenized.jsonl \
    --model unsloth/gemma-4-E4B-it \
    --max-seq-length 2048
```

This applies the Gemma 4 chat template (`<|turn>user` / `<|turn>model`) and outputs pre-tokenized JSONL.

### Step 4: Train

```bash
python finetune_gemma4.py \
    --dataset gemma_tokenized.jsonl \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --output ./gemma4-coding-agent
```

### Step 5: Export to GGUF

```bash
python export_to_gguf.py
```

---

## File Reference

| File | Purpose |
|------|---------|
| `generate_gemma_dataset.py` | Generate JSONL dataset (run locally) |
| `convert_for_gemma.py` | Tokenize with Gemma 4 chat template (run on GPU) |
| `finetune_gemma4.py` | Train with Unsloth (run on GPU) |
| `validate_dataset.py` | Validate JSONL format |
| `export_to_gguf.py` | Export LoRA adapter to GGUF |

---

## Dataset Format (Gemma 4 Standard)

### Message Types

**Normal (text response):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a coding assistant..."},
    {"role": "user", "content": "Explain SOLID principles"},
    {"role": "assistant", "content": "SOLID principles are..."}
  ]
}
```

**Tool Call:**
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Run pip install pytest"},
    {"role": "assistant", "content": "<tool_call>\n{\"id\": \"call_xxx\", \"name\": \"bash\", \"arguments\": {\"command\": \"pip install pytest\"}}\n</tool_call>"}
  ]
}
```

**Multi-turn (with tool response):**
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Check git status"},
    {"role": "assistant", "content": "<tool_call>\n{...}\n</tool_call>"},
    {"role": "tool", "content": "On branch main, nothing to commit"},
    {"role": "assistant", "content": "Git is clean. Ready to commit."}
  ]
}
```

### Tokenized Output

After `convert_for_gemma.py`, each row has a `text` column with the full chat template:

```
<|turn>user
You are a Senior Python Developer. Expert in Python, Django...

Run pip install pytest<turn|>
<|turn>model
<tool_call>
{"id": "call_xxx", "name": "bash", "arguments": {"command": "pip install pytest"}}
</tool_call><turn|>
```

This is what Unsloth's SFTTrainer consumes with `dataset_text_field="text"`.

---

## Gemma 4 Chat Template

Gemma 4 uses `<|turn>role` / `<turn|>` delimiters:

- `<|turn>user` — User message
- `<|turn>model` — Assistant message (internal, template converts `"assistant"` role to `"model"`)
- `<|turn>system` — System message (inline with first user)
- `<|end_of_turn>` — End of turn marker (optional)

The template is applied by `convert_for_gemma.py` using:
```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma4")
```

---

## RunPod Settings

| Setting | Value |
|---------|-------|
| GPU | RTX 4090 24GB or A100 40GB |
| Instance | `u-8-80-s-uncached` (24GB) or `a100-80` (80GB) |
| Max seq length | 2048 |
| Batch size | 4 (with grad accum 4) |
| LoRA rank | 32 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Packing | True |

Install on RunPod:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers trl bitsandbytes
```
