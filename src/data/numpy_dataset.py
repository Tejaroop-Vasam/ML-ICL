"""Fast, lightweight NumPy-backed dataset representation for In-Context Learning."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np

from src.data.example import ICLExample


class NumpyDataset:
    """Vectorized, NumPy-backed representation of an ICL dataset split.

    Stores texts, integer labels, and IDs as contiguous NumPy arrays for
    instantaneous binary loading, boolean masking, and vectorized stratified sampling.
    """

    def __init__(
        self,
        texts: Union[np.ndarray, List[str]],
        labels: Union[np.ndarray, List[int]],
        ids: Optional[Union[np.ndarray, List[str]]] = None,
        label_names: Optional[List[str]] = None,
        dataset_name: str = "dataset",
        split: str = "test",
    ):
        self.texts = np.asarray(texts, dtype=object)
        self.labels = np.asarray(labels, dtype=np.int32)
        if ids is None:
            self.ids = np.array([f"{dataset_name}_{split}_{i}" for i in range(len(self.texts))], dtype=object)
        else:
            self.ids = np.asarray(ids, dtype=object)

        self.label_names = label_names or [str(c) for c in sorted(np.unique(self.labels))]
        self.dataset_name = dataset_name
        self.split = split

        if len(self.texts) != len(self.labels):
            raise ValueError(f"texts ({len(self.texts)}) and labels ({len(self.labels)}) length mismatch.")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: Union[int, slice, np.ndarray]) -> Union[Tuple[str, int, str], "NumpyDataset"]:
        if isinstance(idx, (int, np.integer)):
            return str(self.texts[idx]), int(self.labels[idx]), str(self.ids[idx])
        return NumpyDataset(
            texts=self.texts[idx],
            labels=self.labels[idx],
            ids=self.ids[idx],
            label_names=self.label_names,
            dataset_name=self.dataset_name,
            split=self.split,
        )

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def class_counts(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.label_names[c]: int(cnt) for c, cnt in zip(unique, counts)}

    def stratified_sample(self, n_samples: int, seed: int = 42) -> "NumpyDataset":
        """Vectorized stratified sampling using NumPy.

        Selects balanced samples across all classes in 2 lines of vector operations.
        """
        if n_samples >= len(self):
            return self

        rng = np.random.default_rng(seed)
        unique_classes = np.unique(self.labels)
        per_class = n_samples // len(unique_classes)
        remainder = n_samples % len(unique_classes)

        chosen_indices = []
        for i, c in enumerate(unique_classes):
            cls_pool = np.where(self.labels == c)[0]
            k = per_class + (1 if i < remainder else 0)
            k = min(k, len(cls_pool))
            chosen = rng.choice(cls_pool, size=k, replace=False)
            chosen_indices.append(chosen)

        sampled_idx = np.concatenate(chosen_indices)
        rng.shuffle(sampled_idx)
        return self[sampled_idx]

    def filter_by_class(self, class_id: int) -> "NumpyDataset":
        """Fast boolean mask filtering by class."""
        mask = self.labels == class_id
        return self[mask]

    def filter_by_word_length(self, min_words: int = 0, max_words: Optional[int] = None) -> "NumpyDataset":
        """Filter dataset by text word count using vectorized NumPy array operations."""
        lengths = np.array([len(str(t).split()) for t in self.texts])
        mask = lengths >= min_words
        if max_words is not None:
            mask = mask & (lengths <= max_words)
        return self[mask]

    def save_npz(self, filepath: Union[str, Path]):
        """Save dataset to a compressed .npz archive."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            texts=self.texts,
            labels=self.labels,
            ids=self.ids,
            label_names=np.array(self.label_names, dtype=object),
            dataset_name=np.array(self.dataset_name),
            split=np.array(self.split),
        )

    @classmethod
    def load_npz(cls, filepath: Union[str, Path]) -> "NumpyDataset":
        """Instantaneously load dataset from a .npz archive."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"NPZ archive not found: {path}")

        data = np.load(path, allow_pickle=True)
        dataset_name = str(data["dataset_name"]) if "dataset_name" in data else "dataset"
        split = str(data["split"]) if "split" in data else "test"
        label_names = list(data["label_names"]) if "label_names" in data else None

        return cls(
            texts=data["texts"],
            labels=data["labels"],
            ids=data["ids"] if "ids" in data else None,
            label_names=label_names,
            dataset_name=dataset_name,
            split=split,
        )

    def to_examples(self) -> List[ICLExample]:
        """Convert NumPy arrays back into List[ICLExample] for prompt generation."""
        examples = []
        for i in range(len(self.texts)):
            lbl = int(self.labels[i])
            examples.append(
                ICLExample(
                    id=str(self.ids[i]),
                    dataset=self.dataset_name,
                    text=str(self.texts[i]),
                    label=lbl,
                    label_name=self.label_names[lbl] if lbl < len(self.label_names) else str(lbl),
                    split=self.split,
                )
            )
        return examples

    @classmethod
    def from_examples(
        cls,
        examples: List[ICLExample],
        label_names: Optional[List[str]] = None,
        dataset_name: str = "dataset",
        split: str = "test",
    ) -> "NumpyDataset":
        """Construct NumpyDataset from a list of ICLExample objects."""
        texts = [e.text for e in examples]
        labels = [e.label for e in examples]
        ids = [e.id for e in examples]
        return cls(
            texts=texts,
            labels=labels,
            ids=ids,
            label_names=label_names,
            dataset_name=dataset_name,
            split=split,
        )
