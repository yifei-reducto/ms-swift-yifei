#!/usr/bin/env python3
"""Sample a subset from GRPO dataset with stratified sampling.

Sampling strategy:
1. Split dataset by solution length into shortest 50% and longest 50%
2. Shortest 50%:
   - Keep ALL datapoints containing <checked> or <unchecked>
   - Sample 10% from remaining datapoints
3. Longest 50%:
   - Keep ALL datapoints containing <checked> or <unchecked>
   - Sample 20% from remaining datapoints
"""

import json
import argparse
import random
from pathlib import Path


def has_checkbox(item: dict) -> bool:
    """Check if item's solution contains checkbox tags."""
    solution = item.get('solution', '')
    return '<checked>' in solution or '<unchecked>' in solution


def sample_dataset(input_path: str, output_path: str, seed: int = 42) -> None:
    """Sample dataset with stratified sampling based on length and checkbox content."""
    random.seed(seed)

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {input_path}")

    # Sort by solution length
    data_sorted = sorted(data, key=lambda x: len(x.get('solution', '')))

    # Split at median
    median_idx = len(data_sorted) // 2
    short_half = data_sorted[:median_idx]
    long_half = data_sorted[median_idx:]

    print(f"\nSplit at median index {median_idx}:")
    print(f"  Short half: {len(short_half)} items")
    print(f"  Long half: {len(long_half)} items")

    # Process short half
    short_checkbox = [item for item in short_half if has_checkbox(item)]
    short_non_checkbox = [item for item in short_half if not has_checkbox(item)]
    short_sampled = random.sample(short_non_checkbox, int(len(short_non_checkbox) * 0.1))

    print(f"\nShort half breakdown:")
    print(f"  With checkbox: {len(short_checkbox)} (keeping all)")
    print(f"  Without checkbox: {len(short_non_checkbox)} (sampling 10% = {len(short_sampled)})")

    # Process long half
    long_checkbox = [item for item in long_half if has_checkbox(item)]
    long_non_checkbox = [item for item in long_half if not has_checkbox(item)]
    long_sampled = random.sample(long_non_checkbox, int(len(long_non_checkbox) * 0.2))

    print(f"\nLong half breakdown:")
    print(f"  With checkbox: {len(long_checkbox)} (keeping all)")
    print(f"  Without checkbox: {len(long_non_checkbox)} (sampling 20% = {len(long_sampled)})")

    # Combine all kept items
    final_data = short_checkbox + short_sampled + long_checkbox + long_sampled

    # Shuffle
    random.shuffle(final_data)

    print(f"\nFinal dataset: {len(final_data)} items")
    print(f"  Total checkbox items: {len(short_checkbox) + len(long_checkbox)}")
    print(f"  Total sampled non-checkbox: {len(short_sampled) + len(long_sampled)}")

    # Calculate compression ratio
    compression = (1 - len(final_data) / len(data)) * 100
    print(f"  Compression: {compression:.1f}% reduction")

    # Save output
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"\nOutput saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Sample GRPO dataset with stratified sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', type=str, default='dataset_42706_grpo.json',
                        help='Input dataset file (default: dataset_42706_grpo.json)')
    parser.add_argument('--output', '-o', type=str, default='dataset_42706_sampled.json',
                        help='Output dataset file (default: dataset_42706_sampled.json)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output

    sample_dataset(str(input_path), str(output_path), args.seed)


if __name__ == '__main__':
    main()
