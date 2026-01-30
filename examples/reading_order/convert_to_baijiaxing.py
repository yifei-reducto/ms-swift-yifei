#!/usr/bin/env python3
"""Convert letter labels (A, B, C, ..., AA, AB, ...) to 百家姓 (Chinese surnames)."""

import json
import re
import argparse
import random

# 百家姓 - 200 common Chinese surnames
BAIJIAXING = [
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王", "冯", "陈", "褚", "卫", "蒋", "沈", "韩", "杨",
    "朱", "秦", "尤", "许", "何", "吕", "施", "张", "孔", "曹", "严", "华", "金", "魏", "陶", "姜",
    "戚", "谢", "邹", "喻", "柏", "水", "窦", "章", "云", "苏", "潘", "葛", "奚", "范", "彭", "郎",
    "鲁", "韦", "昌", "马", "苗", "凤", "花", "方", "俞", "任", "袁", "柳", "酆", "鲍", "史", "唐",
    "费", "廉", "岑", "薛", "雷", "贺", "倪", "汤", "滕", "殷", "罗", "毕", "郝", "邬", "安", "常",
    "乐", "于", "时", "傅", "皮", "卞", "齐", "康", "伍", "余", "元", "卜", "顾", "孟", "平", "黄",
    "和", "穆", "萧", "尹", "姚", "邵", "湛", "汪", "祁", "毛", "禹", "狄", "米", "贝", "明", "臧",
    "计", "伏", "成", "戴", "谈", "宋", "茅", "庞", "熊", "纪", "舒", "屈", "项", "祝", "董", "梁",
    "杜", "阮", "蓝", "闵", "席", "季", "麻", "强", "贾", "路", "娄", "危", "江", "童", "颜", "郭",
    "梅", "盛", "林", "刁", "钟", "徐", "邱", "骆", "高", "夏", "蔡", "田", "樊", "胡", "凌", "霍",
    "虞", "万", "支", "柯", "昝", "管", "卢", "莫", "经", "房", "裘", "缪", "干", "解", "应", "宗",
    "丁", "宣", "贲", "邓", "郁", "单", "杭", "洪", "包", "诸", "左", "石", "崔", "吉", "钮", "龚",
    "程", "嵇", "邢", "滑", "裴", "陆", "荣", "翁"
]

def get_letter_label(index):
    """Convert index to letter label (0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ...)."""
    result = ""
    while True:
        result = chr(ord('A') + index % 26) + result
        index = index // 26 - 1
        if index < 0:
            break
    return result


def build_label_mapping(num_labels, offset=0):
    """Build mapping from letter labels to 百家姓.

    Args:
        num_labels: Number of labels to map.
        offset: Starting index in BAIJIAXING (for diversity in training data).
    """
    mapping = {}
    for i in range(num_labels):
        letter = get_letter_label(i)
        mapping[letter] = BAIJIAXING[(i + offset) % len(BAIJIAXING)]
    return mapping


def extract_labels_from_content(content):
    """Extract all letter labels from the user content."""
    # Match patterns like "A: [123, 456, 789, 012]" in the labeled boxes section
    pattern = r'^([A-Z]+): \[\d+, \d+, \d+, \d+\]'
    labels = []
    for line in content.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            labels.append(match.group(1))
    return labels


def replace_labels_in_user_content(content, mapping):
    """Replace letter labels with 百家姓 in user content."""
    new_content = content

    # Sort by length descending to replace longer labels first (AA before A)
    sorted_labels = sorted(mapping.keys(), key=len, reverse=True)

    for letter in sorted_labels:
        surname = mapping[letter]
        # Replace in labeled boxes section: "A: [" -> "赵: ["
        new_content = re.sub(
            rf'\b{letter}: \[',
            f'{surname}: [',
            new_content
        )
        # Replace in the example at the end: "C, A, B, ..."
        # This handles the example in parentheses
        new_content = re.sub(
            rf'"\s*{letter}\s*,',
            f'"{surname},',
            new_content
        )
        new_content = re.sub(
            rf',\s*{letter}\s*,',
            f', {surname},',
            new_content
        )
        new_content = re.sub(
            rf',\s*{letter}\s*\.\.\."',
            f', {surname}, ..."',
            new_content
        )

    return new_content


def replace_labels_in_assistant_content(content, mapping):
    """Replace letter labels with 百家姓 in assistant output."""
    # The assistant content is like "A, B, C, D"
    # Output should be concatenated without spaces or commas: "赵钱孙李"
    labels = [label.strip() for label in content.split(',')]
    new_labels = [mapping.get(label, label) for label in labels]
    return ''.join(new_labels)


def convert_sample(sample):
    """Convert a single sample's labels from letters to 百家姓."""
    messages = sample.get("messages", [])

    # Find user and assistant messages
    user_content = None
    assistant_content = None
    user_idx = None
    assistant_idx = None

    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            user_idx = i
        elif msg.get("role") == "assistant":
            assistant_content = msg.get("content", "")
            assistant_idx = i

    if user_content is None or assistant_content is None:
        return sample

    # Extract labels and build mapping
    labels = extract_labels_from_content(user_content)
    if not labels:
        return sample

    # Random offset for diversity in training data
    offset = random.randint(0, len(BAIJIAXING) - 1)
    mapping = build_label_mapping(len(labels), offset=offset)

    # Replace labels in both user and assistant content
    new_user_content = replace_labels_in_user_content(user_content, mapping)
    new_assistant_content = replace_labels_in_assistant_content(assistant_content, mapping)

    # Create new sample with updated content
    new_sample = sample.copy()
    new_sample["messages"] = messages.copy()
    new_sample["messages"][user_idx] = {**messages[user_idx], "content": new_user_content}
    new_sample["messages"][assistant_idx] = {**messages[assistant_idx], "content": new_assistant_content}

    return new_sample


def convert_dataset(input_path, output_path):
    """Convert entire dataset from letter labels to 百家姓."""
    with open(input_path, "r") as f:
        data = json.load(f)

    converted_data = [convert_sample(sample) for sample in data]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted_data)} samples")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert letter labels to 百家姓")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="val_boxes_to_labels.json",
        help="Path to input dataset JSON file",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="val_boxes_to_labels_baijiaxing.json",
        help="Path to output dataset JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    convert_dataset(args.input_path, args.output_path)
