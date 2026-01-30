#!/usr/bin/env python3
"""Script to analyze assistant output statistics from the dataset."""

import json
import argparse
from collections import Counter


def get_assistant_content(messages):
    """Extract assistant content from a message list."""
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def analyze_dataset(file_path):
    """Analyze assistant output statistics from the dataset."""
    with open(file_path, "r") as f:
        data = json.load(f)

    assistant_outputs = []
    for item in data:
        content = get_assistant_content(item.get("messages", []))
        assistant_outputs.append(content)

    # Calculate statistics
    lengths = [len(content) for content in assistant_outputs]
    num_labels = [len(content.split(",")) for content in assistant_outputs]

    print(f"Dataset: {file_path}")
    print(f"Total samples: {len(assistant_outputs)}")
    print()

    # Character length statistics
    print("=== Character Length Statistics ===")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Avg length: {sum(lengths) / len(lengths):.2f}")
    print()

    # Number of labels statistics
    print("=== Number of Labels Statistics ===")
    print(f"Max labels: {max(num_labels)}")
    print(f"Min labels: {min(num_labels)}")
    print(f"Avg labels: {sum(num_labels) / len(num_labels):.2f}")
    print()

    # Distribution of number of labels
    label_counts = Counter(num_labels)
    print("=== Label Count Distribution ===")
    for count in sorted(label_counts.keys()):
        print(f"  {count} labels: {label_counts[count]} samples")
    print()

    # Show examples with max length
    max_len = max(lengths)
    print("=== Sample(s) with Max Length ===")
    for i, (content, length) in enumerate(zip(assistant_outputs, lengths)):
        if length == max_len:
            print(f"Index {i}: {content}")
            print(f"  Length: {length}, Labels: {len(content.split(','))}")
            break  # Show only one example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze assistant output statistics")
    parser.add_argument(
        "file_path",
        nargs="?",
        default="train_boxes_to_labels.json",
        help="Path to the dataset JSON file",
    )
    args = parser.parse_args()

    analyze_dataset(args.file_path)
