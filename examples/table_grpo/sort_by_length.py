import json

dataset_path = "/home/ubuntu/ms-swift-yifei/examples/table_grpo/train_table_grpo.jsonl"
output_path = "/home/ubuntu/ms-swift-yifei/examples/table_grpo/train_table_grpo_sorted.jsonl"

# Load all entries from the dataset
entries = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line.strip())
        entries.append(entry)

# Sort by length of html_table field
entries.sort(key=lambda x: len(x.get('html_table', '')))

# Write sorted entries to new file
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Sorted {len(entries)} entries by html_table length")
print(f"Saved to: {output_path}")