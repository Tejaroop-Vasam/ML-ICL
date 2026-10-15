#!/usr/bin/env python3
"""Sequence length and token profiler for ICL datasets."""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_dataset, list_datasets


def compute_length_stats(lengths: List[int]) -> Dict[str, float]:
    arr = np.array(lengths)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }


def profile_datasets(cache_dir: str = "data/processed", split: str = "test"):
    print("=" * 80)
    print(f"SEQUENCE LENGTH PROFILING (Split: '{split.upper()}')")
    print("=" * 80)

    stats_table = []
    for name in list_datasets():
        loader = get_dataset(name, cache_dir=cache_dir)
        try:
            examples = loader.get_split(split)
        except KeyError:
            examples = loader.get_split("train")

        word_lengths = [ex.num_words for ex in examples]
        char_lengths = [len(ex.text) for ex in examples]

        w_stats = compute_length_stats(word_lengths)
        c_stats = compute_length_stats(char_lengths)

        stats_table.append({
            "dataset": name,
            "count": w_stats["count"],
            "words_mean": w_stats["mean"],
            "words_median": w_stats["p50"],
            "words_p90": w_stats["p90"],
            "words_max": w_stats["max"],
            "chars_mean": c_stats["mean"],
            "chars_median": c_stats["p50"],
        })

    print(f"\n{'Dataset':<10} | {'Count':<7} | {'Mean Words':<12} | {'Median Words':<14} | {'p90 Words':<11} | {'Max Words':<10}")
    print("-" * 75)
    for row in stats_table:
        print(
            f"{row['dataset']:<10} | "
            f"{row['count']:<7d} | "
            f"{row['words_mean']:<12.1f} | "
            f"{row['words_median']:<14.1f} | "
            f"{row['words_p90']:<11.1f} | "
            f"{row['words_max']:<10d}"
        )

    # Highlight SST-2 vs IMDb comparison
    sst2_row = next((r for r in stats_table if r["dataset"] == "sst2"), None)
    imdb_row = next((r for r in stats_table if r["dataset"] == "imdb"), None)
    if sst2_row and imdb_row:
        ratio = imdb_row["words_mean"] / max(sst2_row["words_mean"], 1)
        print("\n" + "=" * 80)
        print(f"📊 Length Contrast: IMDb reviews are ~{ratio:.1f}x longer on average than SST-2 sentences.")
        print(f"   • SST-2: Mean {sst2_row['words_mean']:.1f} words (Median {sst2_row['words_median']:.1f})")
        print(f"   • IMDb:  Mean {imdb_row['words_mean']:.1f} words (Median {imdb_row['words_median']:.1f})")
        print("   This validates using IMDb to evaluate length-induced degradation while keeping the sentiment task fixed.")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Profile dataset sequence lengths.")
    parser.add_argument("--cache_dir", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    profile_datasets(cache_dir=args.cache_dir, split=args.split)


if __name__ == "__main__":
    main()
