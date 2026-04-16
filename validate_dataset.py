#!/usr/bin/env python3
"""
Validate Gemma 4 dataset format (no torch required).
Run this locally before uploading to RunPod.

Usage:
    python validate_dataset.py training_data.jsonl
    python validate_dataset.py training_data.jsonl --max 1000
"""

import json
import sys
import argparse


def validate_row_text(row_text, idx):
    """Validate a single row (text field format - our custom format)."""
    errors = []

    # Must have exactly two parts: prompt and output_json
    parts = row_text.split('\n\n')
    if len(parts) < 2:
        errors.append(f"Row {idx}: Not enough parts (expected prompt\\n\\noutput_json)")
        return errors

    # First part is prompt, last part is output_json
    prompt = '\n\n'.join(parts[:-1])
    output_str = parts[-1]

    if not prompt.strip():
        errors.append(f"Row {idx}: Empty prompt")

    # Parse output as JSON
    try:
        output = json.loads(output_str)
    except json.JSONDecodeError as e:
        errors.append(f"Row {idx}: Output JSON parse error - {e}")
        return errors

    # Check message_type
    msg_type = output.get("message_type")
    if msg_type not in {"normal", "tool_call", "mcp_call", "multi_turn"}:
        errors.append(f"Row {idx}: Unknown message_type '{msg_type}'")

    # Validate message_type-specific fields
    if msg_type == "tool_call":
        tc = output.get("tool_call")
        if not tc:
            errors.append(f"Row {idx}: tool_call message missing tool_call object")
        elif not tc.get("id"):
            errors.append(f"Row {idx}: tool_call missing 'id'")
        elif not tc.get("name"):
            errors.append(f"Row {idx}: tool_call missing 'name'")
        elif tc.get("arguments") is None:
            errors.append(f"Row {idx}: tool_call missing 'arguments'")

    elif msg_type == "mcp_call":
        mcp = output.get("mcp_call")
        if not mcp:
            errors.append(f"Row {idx}: mcp_call message missing mcp_call object")
        elif not mcp.get("id"):
            errors.append(f"Row {idx}: mcp_call missing 'id'")
        elif not mcp.get("name"):
            errors.append(f"Row {idx}: mcp_call missing 'name'")
        elif mcp.get("arguments") is None:
            errors.append(f"Row {idx}: mcp_call missing 'arguments'")

    elif msg_type == "multi_turn":
        turns = output.get("turns")
        if not turns:
            errors.append(f"Row {idx}: multi_turn message missing turns array")
        elif len(turns) == 0:
            errors.append(f"Row {idx}: multi_turn has empty turns array")
        else:
            for ti, turn in enumerate(turns):
                if "turn_id" not in turn:
                    errors.append(f"Row {idx}, turn {ti}: missing turn_id")
                if "type" not in turn:
                    errors.append(f"Row {idx}, turn {ti}: missing type")
                if turn.get("type") == "tool_call":
                    inner_tc = turn.get("tool_call")
                    if not inner_tc:
                        errors.append(f"Row {idx}, turn {ti}: tool_call type missing tool_call object")
                    elif not inner_tc.get("id"):
                        errors.append(f"Row {idx}, turn {ti}: tool_call missing 'id'")
                    elif not inner_tc.get("name"):
                        errors.append(f"Row {idx}, turn {ti}: tool_call missing 'name'")
                    elif inner_tc.get("arguments") is None:
                        errors.append(f"Row {idx}, turn {ti}: tool_call missing 'arguments'")

    elif msg_type == "normal":
        if not output.get("content"):
            errors.append(f"Row {idx}: normal message missing content")

    # Check metadata
    metadata = output.get("metadata", {})
    required_meta = ["language", "framework", "difficulty", "category"]
    for key in required_meta:
        if key not in metadata:
            errors.append(f"Row {idx}: metadata missing '{key}'")

    # Verify metadata values are reasonable
    valid_languages = {"python", "javascript", "typescript", "java", "kotlin",
                       "go", "rust", "shell", "other", "en"}
    lang = metadata.get("language", "")
    if lang and lang not in valid_languages:
        errors.append(f"Row {idx}: unknown language '{lang}'")

    valid_difficulties = {"easy", "medium", "hard"}
    diff = metadata.get("difficulty", "")
    if diff and diff not in valid_difficulties:
        errors.append(f"Row {idx}: unknown difficulty '{diff}'")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSONL file to validate")
    parser.add_argument("--max", type=int, default=None, help="Max rows to validate")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Validating: {args.file}")
    print(f"Format: Custom XML prompt + JSON output (text field)")
    print(f"Max rows: {args.max or 'all'}")
    print()

    total = 0
    errors = 0
    error_list = []
    msg_type_counts = {}

    with open(args.file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.max and idx >= args.max:
                break

            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors += 1
                error_list.append(f"Row {idx}: JSON parse error - {e}")
                continue

            # New format: {"text": "prompt\n\noutput_json"}
            if "text" not in row:
                errors += 1
                error_list.append(f"Row {idx}: Missing 'text' field")
                continue

            text = row["text"]

            row_errors = validate_row_text(text, idx)
            if row_errors:
                errors += 1
                error_list.extend(row_errors)
            else:
                # Track message type distribution (by parsing output)
                try:
                    output = json.loads(text.split('\n\n')[-1])
                    mt = output.get("message_type", "unknown")
                    msg_type_counts[mt] = msg_type_counts.get(mt, 0) + 1
                except Exception:
                    pass

            if args.verbose and row_errors:
                for err in row_errors[:3]:
                    print(f"  ERROR: {err}")

            if (idx + 1) % 100000 == 0:
                print(f"  Validated: {idx+1:,} rows | Errors: {errors}")

    print()
    print("=" * 60)
    print(f"VALIDATION RESULTS")
    print(f"  Total rows checked: {total:,}")
    print(f"  Rows with errors:   {errors:,}")
    print(f"  Error rate:         {errors/total*100:.1f}%")
    if msg_type_counts:
        print(f"  Message types:      {msg_type_counts}")
    print("=" * 60)

    if errors > 0:
        print(f"\nFirst 10 errors:")
        for e in error_list[:10]:
            print(f"  - {e}")
        return 1

    print("\n  All rows pass! Dataset is valid for Gemma 4 training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
