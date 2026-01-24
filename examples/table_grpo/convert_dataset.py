#!/usr/bin/env python3
"""Convert OCR dataset to GRPO format.

Moves assistant messages to 'solution' field for GRPO training.

Input format:
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "images": [...],
    ...
}

Output format:
{
    "messages": [
        {"role": "user", "content": "..."}
    ],
    "solution": "...",
    "images": [...],
    ...
}
"""

import json
import argparse
from pathlib import Path


def convert_dataset(input_path: str, output_path: str) -> None:
    """Convert dataset from chat format to GRPO format."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    converted = []
    for item in data:
        messages = item.get('messages', [])

        # Find user and assistant messages
        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']

        if not user_messages or not assistant_messages:
            print(f"Skipping item: missing user or assistant message")
            continue

        # Create new item with only user message and solution
        new_item = {
            'messages': user_messages,
            'solution': assistant_messages[0]['content'],
        }

        # Copy other fields (images, document_path, etc.)
        for key, value in item.items():
            if key != 'messages':
                new_item[key] = value

        converted.append(new_item)

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted)} items")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert OCR dataset to GRPO format')
    parser.add_argument('--input', '-i', type=str, default='dataset_42706.json',
                        help='Input dataset file (default: dataset_42706.json)')
    parser.add_argument('--output', '-o', type=str, default='dataset_42706_grpo.json',
                        help='Output dataset file (default: dataset_42706_grpo.json)')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output

    convert_dataset(str(input_path), str(output_path))


if __name__ == '__main__':
    main()
