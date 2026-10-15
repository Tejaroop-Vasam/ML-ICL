"""Benchmark loading & sampling performance: NumPy (.npz) vs JSONL."""

import time
import json
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import NumpyDataset


def benchmark():
    npz_path = Path("data/processed/imdb/test.npz")
    jsonl_path = Path("data/processed/imdb/test.jsonl")

    if not npz_path.exists() or not jsonl_path.exists():
        print("Missing dataset files.")
        return

    print("=" * 70)
    print("BENCHMARK: JSONL vs. NumPy (.npz) on IMDb Test Split (25,000 long reviews)")
    print("=" * 70)

    # 1. Loading speed
    t0 = time.time()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        json_rows = [json.loads(line) for line in f]
    t_json = (time.time() - t0) * 1000

    t0 = time.time()
    np_ds = NumpyDataset.load_npz(npz_path)
    t_npz = (time.time() - t0) * 1000

    print(f"\n1. LOAD TIME (25,000 items):")
    print(f"   • JSONL Parsing:     {t_json:7.1f} ms")
    print(f"   • NumPy (.npz) Load: {t_npz:7.1f} ms  ({t_json / max(t_npz, 0.01):.1f}x faster!)")

    # 2. File size
    json_mb = jsonl_path.stat().st_size / (1024 * 1024)
    npz_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"\n2. DISK FOOTPRINT:")
    print(f"   • JSONL:             {json_mb:7.2f} MB")
    print(f"   • Compressed .npz:   {npz_mb:7.2f} MB  ({(1 - npz_mb / json_mb) * 100:.1f}% space saved)")

    # 3. Stratified Sampling speed
    t0 = time.time()
    sample = np_ds.stratified_sample(n_samples=250, seed=42)
    t_sample = (time.time() - t0) * 1000
    print(f"\n3. VECTORIZED SAMPLING (n=250):")
    print(f"   • NumPy Stratified Sample: {t_sample:.2f} ms")

    print("=" * 70)


if __name__ == "__main__":
    benchmark()
