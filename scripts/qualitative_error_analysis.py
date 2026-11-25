#!/usr/bin/env python3
"""Export qualitative annotation packets from completed experiment trials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.review_experiments.__main__ import main


if __name__ == "__main__":
    run_dirs = sys.argv[1:] or [
        str(checkpoint.parent)
        for checkpoint in sorted(Path("ml_icl_review_runs").glob("*/checkpoint.sqlite"))
    ]
    main(["export", *run_dirs])
