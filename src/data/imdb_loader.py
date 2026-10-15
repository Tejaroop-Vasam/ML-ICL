"""IMDb Large Movie Review DataLoader for long-input sentiment classification."""

import re
import logging
from typing import List, Dict
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample

logger = logging.getLogger(__name__)


class IMDbDataLoader(BaseDataLoader):
    """DataLoader for IMDb binary sentiment classification (long inputs / full reviews)."""

    @property
    def dataset_name(self) -> str:
        return "imdb"

    @property
    def classes(self) -> List[str]:
        return ["negative", "positive"]

    @property
    def task_instruction(self) -> str:
        return "Classify the sentiment of each movie review as either 'positive' or 'negative'."

    @property
    def default_input_prefix(self) -> str:
        return "Review: "

    @property
    def default_label_prefix(self) -> str:
        return "Sentiment: "

    @staticmethod
    def _clean_review_text(raw_text: str) -> str:
        """Clean HTML breaks and extraneous whitespace from IMDb reviews."""
        text = re.sub(r"<br\s*/?>", "\n", raw_text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        """Fetch IMDb dataset from Hugging Face Hub ('stanfordnlp/imdb')."""
        logger.info("Loading IMDb from Hugging Face ('stanfordnlp/imdb')...")
        hf_dataset = load_dataset("stanfordnlp/imdb")

        splits: Dict[str, List[ICLExample]] = {}

        # Test split (25,000 reviews)
        test_examples = []
        for idx, item in enumerate(hf_dataset["test"]):
            label = int(item["label"])
            if label not in (0, 1):
                continue
            cleaned = self._clean_review_text(item["text"])
            test_examples.append(
                ICLExample(
                    id=f"imdb_test_{idx}",
                    dataset=self.dataset_name,
                    text=cleaned,
                    label=label,
                    label_name=self.classes[label],
                    split="test",
                )
            )
        splits["test"] = test_examples

        # Train split (25,000 reviews) -> hold out 2,500 for validation
        train_raw = list(hf_dataset["train"])
        train_indices, val_indices = train_test_split(
            range(len(train_raw)),
            test_size=min(2500, int(len(train_raw) * 0.1)),
            random_state=self.seed,
            stratify=[x["label"] for x in train_raw],
        )

        train_examples = []
        for idx in train_indices:
            item = train_raw[idx]
            label = int(item["label"])
            if label not in (0, 1):
                continue
            cleaned = self._clean_review_text(item["text"])
            train_examples.append(
                ICLExample(
                    id=f"imdb_train_{idx}",
                    dataset=self.dataset_name,
                    text=cleaned,
                    label=label,
                    label_name=self.classes[label],
                    split="train",
                )
            )
        splits["train"] = train_examples

        val_examples = []
        for idx in val_indices:
            item = train_raw[idx]
            label = int(item["label"])
            if label not in (0, 1):
                continue
            cleaned = self._clean_review_text(item["text"])
            val_examples.append(
                ICLExample(
                    id=f"imdb_val_{idx}",
                    dataset=self.dataset_name,
                    text=cleaned,
                    label=label,
                    label_name=self.classes[label],
                    split="validation",
                )
            )
        splits["validation"] = val_examples

        logger.info(
            f"Loaded IMDb: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )
        return splits
