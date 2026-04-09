import json
import os
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict


def format_true(example):
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ],
        "flag": True,
    }


def format_false(example):
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ],
        "flag": False,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Zigeng/DMax-LLaDA-2.0-Mini-Math-Trajectories",
        help="Path or name for load_dataset()"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./my_data",
        help="Output directory for processed jsonl files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw = load_dataset(args.dataset_path)

    ds_true = raw.map(format_true, remove_columns=["question", "answer"])
    ds_false = raw.map(format_false, remove_columns=["question", "answer"])

    merged = DatasetDict({
        split: concatenate_datasets([ds_true[split], ds_false[split]])
        for split in ds_true.keys()
    })

    merged = merged.shuffle(seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    for split, ds in merged.items():
        out_path = os.path.join(args.out_dir, f"postprocess_{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in ds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {split}: {out_path}, n={len(ds)}")


if __name__ == "__main__":
    main()