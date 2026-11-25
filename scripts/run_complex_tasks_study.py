#!/usr/bin/env python3
"""Run Spider, BIRD and CLAPNQ with the reviewer runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.review_experiments.__main__ import main
from src.review_experiments.config import SQL_DATASETS


if __name__ == "__main__":
    main(["run", "--datasets", *SQL_DATASETS, "clapnq", *sys.argv[1:]])
