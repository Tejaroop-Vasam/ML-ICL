#!/usr/bin/env python3
"""Script to download, verify, and cache all ICL datasets."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_dataset, list_datasets

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_verify")


def verify_dataset(name: str, cache_dir: str, seed: int, force_reload: bool):
    logger.info("=" * 60)
    logger.info(f"PROCESSING DATASET: {name.upper()}")
    logger.info("=" * 60)

    loader = get_dataset(name, cache_dir=cache_dir, seed=seed)
    splits = loader.load_data(force_reload=force_reload)

    print(f"\n--- {name.upper()} Dataset Summary ---")
    print(f"Classes ({loader.num_classes}): {loader.classes}")
    print(f"Label Mapping: {loader.id2label}")
    print(f"Task Instruction: {loader.task_instruction}")

    print("\nSplit Sizes:")
    for split_name, examples in splits.items():
        avg_words = sum(e.num_words for e in examples) / max(len(examples), 1)
        print(f"  • {split_name:12s}: {len(examples):7d} examples (avg {avg_words:.1f} words/example)")

    # Test stratified sampling
    test_subset = loader.get_stratified_subset(split="test", n_samples=min(10, len(splits["test"])))
    print(f"\nStratified Test Sample (n={len(test_subset)}):")
    for ex in test_subset[:3]:
        preview = ex.text[:80] + "..." if len(ex.text) > 80 else ex.text
        print(f"  [id={ex.id}] [{ex.label_name}] {preview}")

    # Test prompt generation across formats
    demo_pool = loader.get_candidate_demonstration_pool(pool_size=2)
    test_ex = test_subset[0]

    print("\n--- Prompt Format Previews (2-shot) ---")
    for fmt in ["plain", "structured", "rationale"]:
        print(f"\n>>> Format '{fmt}':")
        prompt = loader.format_prompt(
            test_example=test_ex,
            demonstrations=demo_pool,
            format_type=fmt,
        )
        # Indent for display
        for line in prompt.split("\n"):
            print(f"  | {line}")

    logger.info(f"Dataset {name} verified successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Download and verify ICL benchmark datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + list_datasets(),
        help="Dataset to download and verify (default: 'all')",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/processed",
        help="Local cache directory for JSONL splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force re-download and re-cache raw datasets",
    )

    args = parser.parse_args()

    targets = list_datasets() if args.dataset == "all" else [args.dataset]
    for d in targets:
        verify_dataset(d, args.cache_dir, args.seed, args.force_reload)

    logger.info("All requested datasets successfully downloaded, processed, and verified.")


if __name__ == "__main__":
    main()
