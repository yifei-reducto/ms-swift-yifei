import os
import json
import numpy as np
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_token_length(text: str) -> int:
    return len(tokenizer.encode(text))

dataset_dir = "/home/ubuntu/ms-swift-yifei/examples/table_grpo/dataset_14495_sampled_0124.json"

# Load dataset
with open(dataset_dir, "r") as f:
    data = json.load(f)

# Get token lengths for all solution fields
solution_lengths = [get_token_length(item["solution"]) for item in data]

# Calculate percentiles
p50 = np.percentile(solution_lengths, 50)
p90 = np.percentile(solution_lengths, 90)
p95 = np.percentile(solution_lengths, 95)
p99 = np.percentile(solution_lengths, 99)

print(f"Total samples: {len(solution_lengths)}")
print(f"P50: {p50:.0f} tokens")
print(f"P90: {p90:.0f} tokens")
print(f"P95: {p95:.0f} tokens")
print(f"P99: {p99:.0f} tokens")
print(f"Max: {max(solution_lengths)} tokens")
print(f"Min: {min(solution_lengths)} tokens")
