#!/usr/bin/env python3
"""
Convert JSONL dataset to Gemma 4 Unsloth training format.
Run this on your GPU machine (RunPod) AFTER generating the JSONL dataset locally.

Usage:
    python convert_for_gemma.py --input gemma_dataset.jsonl --output gemma_tokenized.jsonl

Options:
    --model       Model name (default: unsloth/gemma-4-E4B-it)
    --max-seq     Max sequence length (default: 2048)
    --max-samples Max samples (default: all)
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to Gemma 4 tokenized format")
    parser.add_argument("--input", required=True, help="Input JSONL (generate_gemma_dataset.py output)")
    parser.add_argument("--output", required=True, help="Output tokenized JSONL for Unsloth")
    parser.add_argument("--model", default="unsloth/gemma-4-E4B-it",
                        help="Model for tokenizer (needed for chat template)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Gemma 4 Dataset Converter for Unsloth")
    print("=" * 60)
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    print(f"Model:    {args.model}")
    print(f"Max seq:  {args.max_seq_length}")
    print()

    # Check for required packages
    try:
        import torch
        from transformers import AutoTokenizer
        from unsloth.chat_templates import get_chat_template
    except ImportError as e:
        print("ERROR: Missing required packages!")
        print("Please install: pip install torch transformers unsloth")
        print(f"Import error: {e}")
        return 1

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma4")

    # Verify template
    test_msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    test_out = tokenizer.apply_chat_template(test_msgs, tokenize=False)
    assert "<|turn>user" in test_out and "<|turn>model" in test_out, \
        f"Template check failed! Got: {test_out[:100]}"
    print(f"  Chat template verified (gemma4)")
    print(f"  Special tokens: <|turn>, <turn|>, <|end_of_turn|>")

    # Read input
    print("\nReading input JSONL...")
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break
            rows.append(json.loads(line))
    print(f"  Loaded {len(rows):,} rows")

    # Tokenize
    print("\nTokenizing...")
    tokenized_rows = []
    skipped = 0
    too_long = 0

    for i, row in enumerate(rows):
        try:
            messages = row["messages"]

            # Apply Gemma 4 chat template
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )

            # Check length
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > args.max_seq_length:
                too_long += 1
                if too_long <= 3:
                    print(f"  Warning: Row {i} exceeds {args.max_seq_length} tokens ({len(tokens)}), skipping")
                continue

            tokenized_rows.append({"text": text})

            if (i + 1) % 50000 == 0:
                print(f"  Processed {i+1:,} / {len(rows):,} rows")

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skipping row {i}: {e}")

    print(f"\n  Tokenized: {len(tokenized_rows):,}")
    print(f"  Skipped (too long): {too_long}")
    print(f"  Skipped (error): {skipped}")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for item in tokenized_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    avg_chars = sum(len(t["text"]) for t in tokenized_rows) / len(tokenized_rows)
    est_tokens = int(avg_chars * 0.25)
    print(f"\n  Average text length: ~{avg_chars:.0f} chars (~{est_tokens} tokens)")
    print(f"  File size estimate: ~{len(tokenized_rows) * avg_chars / (1024*1024):.0f} MB")

    print()
    print("=" * 60)
    print("Conversion complete!")
    print(f"Next step: Train with finetune_gemma4.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())