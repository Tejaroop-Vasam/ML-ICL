"""Base DataLoader for In-Context Learning datasets."""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np

from src.data.example import ICLExample

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for all ICL dataset loaders."""

    def __init__(
        self,
        cache_dir: Optional[str] = "data/processed",
        seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/processed")
        self.seed = seed
        self._splits: Dict[str, List[ICLExample]] = {}

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset (e.g., 'sst2', 'ag_news', 'trec', 'imdb')."""
        pass

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """List of class names ordered by integer label."""
        pass

    @property
    @abstractmethod
    def task_instruction(self) -> str:
        """Task instruction to prepend to prompts."""
        pass

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def id2label(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.classes)}

    @property
    def label2id(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.classes)}

    @property
    def default_input_prefix(self) -> str:
        return "Input: "

    @property
    def default_label_prefix(self) -> str:
        return "Label: "

    @abstractmethod
    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        """Download or read raw data and convert to List[ICLExample] per split."""
        pass

    def get_cache_path(self, split: str) -> Path:
        return self.cache_dir / self.dataset_name / f"{split}.jsonl"

    def load_data(self, force_reload: bool = False) -> Dict[str, List[ICLExample]]:
        """Load dataset splits, using local JSONL cache if present.

        Args:
            force_reload: If True, re-fetches raw dataset and updates cache.
        """
        cache_folder = self.cache_dir / self.dataset_name
        cache_folder.mkdir(parents=True, exist_ok=True)

        splits_to_load = ["train", "validation", "test"]
        all_cached = all(self.get_cache_path(s).exists() for s in ["train", "test"])

        if all_cached and not force_reload:
            logger.info(f"Loading cached {self.dataset_name} from {cache_folder}")
            self._splits = {}
            for s in splits_to_load:
                path = self.get_cache_path(s)
                if path.exists():
                    self._splits[s] = self._read_jsonl(path)
            return self._splits

        logger.info(f"Fetching raw data for {self.dataset_name}...")
        self._splits = self._fetch_raw_splits()

        # Save to cache
        for split_name, examples in self._splits.items():
            cache_file = self.get_cache_path(split_name)
            self._write_jsonl(examples, cache_file)
            logger.info(f"Cached {len(examples)} examples for {self.dataset_name}:{split_name} -> {cache_file}")

        return self._splits

    def get_split(self, split: str = "test") -> List[ICLExample]:
        """Retrieve a specific split ('train', 'validation', or 'test')."""
        if not self._splits:
            self.load_data()
        if split not in self._splits:
            raise KeyError(f"Split '{split}' not found in {list(self._splits.keys())}")
        return self._splits[split]

    def get_stratified_subset(
        self,
        split: str = "test",
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[ICLExample]:
        """Get a balanced/stratified subset of examples for manageable evaluation.

        Args:
            split: Which split to sample from ('test' or 'validation' or 'train').
            n_samples: Total number of samples. If None or >= total size, returns all.
            seed: Random seed for deterministic reproducibility.
        """
        examples = self.get_split(split)
        total_len = len(examples)
        if n_samples is None or n_samples >= total_len:
            return examples

        rnd = np.random.RandomState(seed if seed is not None else self.seed)

        # Group indices by class label
        by_class: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        for idx, ex in enumerate(examples):
            if ex.label in by_class:
                by_class[ex.label].append(idx)

        # Distribute n_samples evenly across classes
        per_class = n_samples // self.num_classes
        remainder = n_samples % self.num_classes

        selected_indices: List[int] = []
        for class_id, idxs in sorted(by_class.items()):
            k = per_class + (1 if class_id < remainder else 0)
            k = min(k, len(idxs))
            chosen = rnd.choice(idxs, size=k, replace=False)
            selected_indices.extend(chosen)

        rnd.shuffle(selected_indices)
        return [examples[i] for i in selected_indices]

    def get_candidate_demonstration_pool(
        self,
        pool_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[ICLExample]:
        """Retrieve candidate demonstrations from the training set.

        Args:
            pool_size: If provided, returns a stratified pool of candidate demonstrations.
            seed: Random seed.
        """
        return self.get_stratified_subset(
            split="train",
            n_samples=pool_size,
            seed=seed if seed is not None else self.seed,
        )

    def format_prompt(
        self,
        test_example: ICLExample,
        demonstrations: Optional[List[ICLExample]] = None,
        format_type: str = "plain",
        include_instruction: bool = True,
        custom_instruction: Optional[str] = None,
    ) -> str:
        """Construct full prompt containing task instruction, few-shot demonstrations, and test query.

        Args:
            test_example: The query example to classify.
            demonstrations: Optional list of few-shot demonstration examples.
            format_type: 'plain' (Format A), 'structured' (Format B), or 'rationale' (Format C).
            include_instruction: Whether to prepend the task description.
            custom_instruction: Custom task description string.
        """
        sections: List[str] = []

        # 1. Instruction
        if include_instruction:
            instr = custom_instruction or self.task_instruction
            if instr:
                sections.append(instr.strip())

        # 2. Demonstrations
        if demonstrations:
            for demo in demonstrations:
                demo_str = demo.format_example(
                    format_type=format_type,
                    include_label=True,
                    input_prefix=self.default_input_prefix,
                    label_prefix=self.default_label_prefix,
                )
                sections.append(demo_str)

        # 3. Test Example
        test_str = test_example.format_example(
            format_type=format_type,
            include_label=False,
            input_prefix=self.default_input_prefix,
            label_prefix=self.default_label_prefix,
        )
        sections.append(test_str)

        delimiter = "\n\n"
        return delimiter.join(sections)

    @staticmethod
    def _read_jsonl(path: Path) -> List[ICLExample]:
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(ICLExample.from_dict(json.loads(line)))
        return examples

    @staticmethod
    def _write_jsonl(examples: List[ICLExample], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
