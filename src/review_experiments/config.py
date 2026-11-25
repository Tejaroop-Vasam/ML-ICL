"""Defaults and validation for the reviewer experiment suite."""

from dataclasses import dataclass, field
from pathlib import Path

CLASS_DATASETS = ("imdb", "yelp_polarity", "trec", "yahoo_answers")
SQL_DATASETS = ("spider", "bird")
DEFAULT_SAMPLES = {
    "imdb": 40, "yelp_polarity": 12, "trec": 12, "yahoo_answers": 20,
    "spider": 12, "bird": 12, "clapnq": 12,
}
DEFAULT_MODEL = "Qwen/Qwen3-8B"
WEIGHT_MODES = ("nf4", "fp16", "bf16")
NATIVE_WINDOW = 32_768
CLASS_MAX_NEW = 16
SQL_MAX_NEW = 512
QA_MAX_NEW = 384


@dataclass(frozen=True)
class ReviewConfig:
    samples: dict[str, int | None] = field(default_factory=lambda: DEFAULT_SAMPLES.copy())
    stress_n: int | None = 4
    stress_prompt_targets: tuple[int, ...] = (2048, 8192, 16384, 30000)
    min_qa_reference_words: int = 50
    imdb_slice_tokens: int = 128
    seed: int = 42
    pool_size: int = 200
    reference_k: int = 4
    k_sweep: tuple[int, ...] = (2, 4, 8)
    random_seeds: tuple[int, ...] = (42,)
    sql_demo_k: int = 2
    qa_candidates: int = 64
    qa_demo_k: int = 2
    profile_only: bool = False
    retry_oom: bool = False
    max_prompt_tokens: int = NATIVE_WINDOW
    output_root: Path = Path("ml_icl_review_runs")
    cache_root: Path = Path("ml_icl_benchmark_data")
    spider_path: Path | None = None
    bird_dev_path: Path | None = None
    bird_train_path: Path | None = None

    def __post_init__(self):
        if set(self.samples) != set(DEFAULT_SAMPLES):
            raise ValueError("Sample counts must be specified for each supported dataset.")
        if all(n == 0 for n in self.samples.values()):
            raise ValueError("Select at least one dataset with a nonzero sample count.")
        if any(n is not None and n < 0 for n in (*self.samples.values(), self.stress_n)):
            raise ValueError("Sample counts must be nonnegative or 'all'.")
        if not 0 < self.max_prompt_tokens <= NATIVE_WINDOW:
            raise ValueError("The prompt cap must be between 1 and 32,768 tokens.")
        shot_counts = (self.reference_k, *self.k_sweep)
        if min(shot_counts) < 1 or max(shot_counts) > self.pool_size:
            raise ValueError("Choose positive shot counts that fit in the candidate pool.")
        if min(self.sql_demo_k, self.qa_demo_k, self.imdb_slice_tokens) < 0:
            raise ValueError("Demonstration counts and review slice length must be nonnegative.")
        if self.qa_candidates < 2 or self.min_qa_reference_words < 1:
            raise ValueError("Use at least two QA candidates and a positive reference length.")
        if any(target <= 0 or target + QA_MAX_NEW > NATIVE_WINDOW for target in self.stress_prompt_targets):
            raise ValueError("Stress targets must leave room for QA generation inside the native window.")
