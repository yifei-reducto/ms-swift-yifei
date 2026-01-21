#!/usr/bin/env python3
"""
Prepare local dataset for table GRPO training.

This script downloads the fintabnet-html dataset from HuggingFace,
saves images locally, and creates a JSONL file in the format required
by ms-swift for GRPO training.

Usage:
    python prepare_dataset.py

The script will create:
    - images/: folder containing table images
    - train_table_grpo.jsonl: JSONL dataset file
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    # Configuration
    num_samples = 2000
    output_dir = Path(__file__).parent
    images_dir = output_dir / "images"
    jsonl_path = output_dir / "train_table_grpo.jsonl"

    # Create images directory
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {num_samples} samples from apoidea/fintabnet-html (en subset)...")
    dataset = load_dataset(
        "apoidea/fintabnet-html",
        "en",
        split=f"train[:{num_samples}]",
        trust_remote_code=True
    )

    print(f"Processing {len(dataset)} samples...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            # Save image
            img_filename = f"{i:05d}.png"
            img_path = images_dir / img_filename

            # The image is already a PIL Image from the dataset
            image = sample["image"]
            image.save(img_path, format="PNG")

            # Create JSONL entry in ms-swift GRPO format
            # - messages: prompt with <image> placeholder
            # - images: list of relative image paths
            # - html_table: ground truth for reward function
            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>Parse this table image into HTML code. Output only the raw HTML table using <table>, <thead>, <tbody>, <tr>, <th>, <td> tags with colspan and rowspan attributes where needed. Do not include any CSS styles or explanations."
                    }
                ],
                "images": [f"images/{img_filename}"],
                "html_table": sample["html_table"]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDataset preparation complete!")
    print(f"  Images saved to: {images_dir}")
    print(f"  JSONL saved to: {jsonl_path}")
    print(f"  Total samples: {len(dataset)}")


if __name__ == "__main__":
    main()
