#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = SCRIPT_DIR / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_gsm8k.verl_dynamic_mc import make_stage1_records_from_gsm8k_example


def _validation_count(*, total: int, validation_size: int, validation_ratio: float | None) -> int:
    if validation_ratio is not None:
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("--validation-ratio must be between 0 and 1.")
        return max(1, min(total, round(total * validation_ratio)))
    return max(0, min(total, validation_size))


def _select_partition(
    source: Dataset,
    *,
    split: str,
    partition: str,
    validation_size: int,
    validation_ratio: float | None,
    split_seed: int,
) -> tuple[Dataset, int]:
    if partition == "all":
        return source, 0
    if partition == "test":
        if split != "test":
            raise ValueError("--partition test requires --split test.")
        return source, 0
    if split != "train":
        raise ValueError("--partition train/val is only valid with --split train.")

    total = len(source)
    val_count = _validation_count(total=total, validation_size=validation_size, validation_ratio=validation_ratio)
    if val_count <= 0:
        raise ValueError("Validation holdout must be non-empty when using --partition train or --partition val.")

    shuffled = list(range(total))
    random.Random(split_seed).shuffle(shuffled)
    val_indices = set(shuffled[:val_count])
    if partition == "val":
        selected = [idx for idx in range(total) if idx in val_indices]
    elif partition == "train":
        selected = [idx for idx in range(total) if idx not in val_indices]
    else:
        raise ValueError("--partition must be one of all, train, val, test.")
    return source.select(selected), val_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create VERL-native dynamic MC Stage 1 seed parquet.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--partition",
        default=None,
        choices=["all", "train", "val", "test"],
        help=(
            "Logical partition to export. Use train/val for a deterministic holdout from GSM8K train; "
            "use test only with --split test. Defaults to all for backward compatibility."
        ),
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=512,
        help="Number of GSM8K train questions held out for validation when --partition is train or val.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=None,
        help="Optional validation holdout ratio. If set, overrides --validation-size.",
    )
    parser.add_argument("--split-seed", type=int, default=7, help="Seed for deterministic train/val splitting.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional GSM8K question cap.")
    parser.add_argument(
        "--stage1-prompt-count",
        type=int,
        default=None,
        help="Neutral candidate-generation prompts per question. Defaults to 4.",
    )
    parser.add_argument(
        "--incorrect-target-count",
        type=int,
        default=None,
        help="Deprecated alias: neutral mode uses this as stage1_prompt_count - 1.",
    )
    parser.add_argument("--prompt-mode", choices=["neutral", "role"], default="neutral")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    args = parser.parse_args()
    partition = args.partition or ("test" if args.split == "test" else "all")

    if args.stage1_prompt_count is not None:
        seed_count = max(1, args.stage1_prompt_count)
    elif args.incorrect_target_count is not None:
        seed_count = max(1, args.incorrect_target_count + 1)
    else:
        seed_count = 4
    role_incorrect_count = max(0, seed_count - 1)

    source = load_dataset("openai/gsm8k", "main")[args.split]
    source_total = len(source)
    source = source.add_column("_source_index", list(range(source_total)))
    source, validation_count = _select_partition(
        source,
        split=args.split,
        partition=partition,
        validation_size=args.validation_size,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
    )
    if args.num_samples is not None:
        source = source.select(range(min(args.num_samples, len(source))))

    records = []
    for row_index, example in enumerate(source):
        source_index = int(example.get("_source_index", row_index))
        records.extend(
            make_stage1_records_from_gsm8k_example(
                split=args.split,
                index=source_index,
                example=example,
                incorrect_target_count=role_incorrect_count,
                prompt_mode=args.prompt_mode,
            )
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    print(
        f"[create_dynamic_mc_seed] split={args.split} partition={partition} "
        f"source_questions={source_total} validation_holdout={validation_count} "
        f"questions={len(source)} records={len(records)} prompt_mode={args.prompt_mode} "
        f"stage1_prompt_count={seed_count} split_seed={args.split_seed} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
