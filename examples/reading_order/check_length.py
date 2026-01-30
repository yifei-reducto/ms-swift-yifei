import argparse
import json

import numpy as np
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_token_length(text: str) -> int:
    return len(tokenizer.encode(text))


def get_prompt_token_length(item: dict) -> int:
    """Apply chat template to messages and count tokens."""
    messages = item["messages"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return get_token_length(prompt)


def print_statistics(lengths: list[int], dataset_path: str) -> None:
    """Calculate and print prompt length statistics."""
    lengths_array = np.array(lengths)
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(lengths)}")
    print(f"Mean:    {np.mean(lengths_array):.1f}")
    print(f"P50:     {np.percentile(lengths_array, 50):.1f}")
    print(f"P90:     {np.percentile(lengths_array, 90):.1f}")
    print(f"P95:     {np.percentile(lengths_array, 95):.1f}")
    print(f"P99:     {np.percentile(lengths_array, 99):.1f}")
    print(f"Max:     {max(lengths)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt token lengths in GRPO datasets")
    parser.add_argument("dataset", help="Path to the GRPO dataset JSON file")
    args = parser.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)

    lengths = [get_prompt_token_length(item) for item in data]
    print_statistics(lengths, args.dataset)


if __name__ == "__main__":
    main()