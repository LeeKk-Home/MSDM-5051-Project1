"""Create original synthetic fixtures for checking the benchmark interface."""
import json
from pathlib import Path


def make_records(split):
    records = []
    indexes = range(12) if split == "train" else range(20, 25)
    for i in indexes:
        values = {
            "english": f"Record {i}: algorithms transform text into tokens. A small tokenizer learns repeated patterns and preserves every space.\n",
            "chinese": f"第{i}段：算法把文本转成词元。我们比较运行速度、词表大小和序列长度，并保留所有空格。你好，香港！\n",
            "code": f"def square_{i}(value):\n    # preserve indentation\n    return value * value + {i}\n",
            "unicode": f"sample {i}\t中文 English café e\u0301 🙂🚀\n  two spaces\n",
            "repetitive": (f"item{i} " + "abababa " * (i % 3 + 2)) * 3,
        }
        for category, text in values.items():
            records.append({"id": f"{split}-{category}-{i}", "category": category, "text": text})
    if split == "eval":
        records.extend([
            {"id": "eval-empty", "category": "unicode", "text": ""},
            {"id": "eval-unseen", "category": "unicode", "text": "龘🧬𐐀\u0000\r\n"},
            {"id": "eval-long", "category": "repetitive", "text": "abababa " * 128 + " unique ending"},
        ])
    return records


def main():
    directory = Path(__file__).resolve().parent
    for split in ("train", "eval"):
        path = directory / f"smoke-{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in make_records(split)), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
