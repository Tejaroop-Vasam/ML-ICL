#!/usr/bin/env python3
"""Batch convert processed JSONL datasets into compressed NumPy .npz archives."""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_dataset, list_datasets, NumpyDataset


def convert_all(cache_dir: str = "data/processed"):
    print("=" * 60)
    print("CONVERTING DATASETS TO NUMPY .NPZ FORMAT")
    print("=" * 60)

    for name in list_datasets():
        loader = get_dataset(name, cache_dir=cache_dir)
        splits = loader.load_data()

        print(f"\nProcessing {name.upper()}:")
        for split_name, examples in splits.items():
            t0 = time.time()
            np_ds = NumpyDataset.from_examples(
                examples=examples,
                label_names=loader.classes,
                dataset_name=name,
                split=split_name,
            )
            out_path = Path(cache_dir) / name / f"{split_name}.npz"
            np_ds.save_npz(out_path)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            dur = (time.time() - t0) * 1000
            print(f"  • {split_name:12s}: {len(np_ds):6d} items -> {out_path.name} ({size_mb:.2f} MB, {dur:.1f}ms)")

    print("\nAll datasets successfully exported to NumPy .npz archives.")


if __name__ == "__main__":
    convert_all()
