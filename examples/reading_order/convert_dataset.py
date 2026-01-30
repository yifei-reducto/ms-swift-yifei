#!/usr/bin/env python3
"""Convert reading order SFT dataset to GRPO format.

Moves assistant messages to 'solution' field for GRPO training with
the KendallTau reward function.

Input format (SFT):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "<image>..."},
        {"role": "assistant", "content": "赵钱孙李周吴郑王..."}
    ],
    "images": "..."
}

Output format (GRPO):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "<image>..."}
    ],
    "solution": "赵钱孙李周吴郑王...",
    "images": "..."
}
"""

import json
import argparse
from pathlib import Path

string_in_start = "Return only a "
string_out = "Return only a list of labels in reading order (e.g., \"孙赵钱...\"). Do not output anything else."

def replace_string_in_content(content: str) -> str:
    """Replace instruction in content by splitting on string_in_start and appending string_out."""
    parts = content.split(string_in_start, 1)
    if len(parts) > 1:
        return parts[0] + string_out
    return content

def convert_dataset(input_path: str, output_path: str) -> None:
    """Convert dataset from SFT chat format to GRPO format."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted = []
    skipped = 0

    for item in data:
        messages = item.get('messages', [])

        # Separate messages by role
        non_assistant_messages = []
        assistant_content = None

        for msg in messages:
            role = msg.get('role', '')
            if role == 'assistant':
                assistant_content = msg.get('content', '')
            elif role == 'user':
                # Replace strings in user content
                new_msg = msg.copy()
                new_msg['content'] = replace_string_in_content(msg.get('content', ''))
                non_assistant_messages.append(new_msg)
            else:
                non_assistant_messages.append(msg)

        # Skip if no assistant message (no ground truth)
        if assistant_content is None:
            skipped += 1
            continue

        # Skip if no user message
        has_user = any(m.get('role') == 'user' for m in non_assistant_messages)
        if not has_user:
            skipped += 1
            continue

        # Create new item with solution field
        new_item = {
            'messages': non_assistant_messages,
            'solution': assistant_content,
        }

        # Copy other fields (images, document_path, etc.)
        for key, value in item.items():
            if key != 'messages':
                new_item[key] = value

        converted.append(new_item)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted)} items")
    if skipped > 0:
        print(f"Skipped {skipped} items (missing user or assistant message)")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert reading order SFT dataset to GRPO format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='train_boxes_to_labels_baijiaxing.json',
        help='Input dataset file (default: train_boxes_to_labels_baijiaxing.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output dataset file (default: <input>_grpo.json)'
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input

    # Generate output path if not specified
    if args.output is None:
        input_stem = Path(args.input).stem
        output_name = f"{input_stem}_grpo.json"
    else:
        output_name = args.output
    output_path = script_dir / output_name

    convert_dataset(str(input_path), str(output_path))


if __name__ == '__main__':
    main()
