"""AG News DataLoader for 4-class topic classification."""

import logging
from typing import List, Dict
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample

logger = logging.getLogger(__name__)


class AGNewsDataLoader(BaseDataLoader):
    """DataLoader for AG News 4-class topic classification (short-to-medium inputs)."""

    @property
    def dataset_name(self) -> str:
        return "ag_news"

    @property
    def classes(self) -> List[str]:
        # 0: World, 1: Sports, 2: Business, 3: Sci/Tech
        return ["World", "Sports", "Business", "Sci/Tech"]

    @property
    def task_instruction(self) -> str:
        return (
            "Classify the following news article into one of the four categories: "
            "World, Sports, Business, or Sci/Tech."
        )

    @property
    def default_input_prefix(self) -> str:
        return "Article: "

    @property
    def default_label_prefix(self) -> str:
        return "Topic: "

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        """Fetch AG News dataset from Hugging Face Hub ('fancyzhx/ag_news' or 'SetFit/ag_news')."""
        logger.info("Loading AG News from Hugging Face ('fancyzhx/ag_news')...")
        try:
            hf_dataset = load_dataset("fancyzhx/ag_news")
        except Exception:
            logger.info("Falling back to 'SetFit/ag_news'...")
            hf_dataset = load_dataset("SetFit/ag_news")

        splits: Dict[str, List[ICLExample]] = {}

        # Process test split
        test_examples = []
        for idx, item in enumerate(hf_dataset["test"]):
            text = item.get("text", "").strip()
            label = int(item["label"])
            test_examples.append(
                ICLExample(
                    id=f"ag_news_test_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=self.classes[label],
                    split="test",
                )
            )
        splits["test"] = test_examples

        # Process train split: create held-out validation split (10% of train, max 5000)
        train_raw = list(hf_dataset["train"])
        train_indices, val_indices = train_test_split(
            range(len(train_raw)),
            test_size=min(5000, int(len(train_raw) * 0.1)),
            random_state=self.seed,
            stratify=[x["label"] for x in train_raw],
        )

        train_examples = []
        for idx in train_indices:
            item = train_raw[idx]
            text = item.get("text", "").strip()
            label = int(item["label"])
            train_examples.append(
                ICLExample(
                    id=f"ag_news_train_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=self.classes[label],
                    split="train",
                )
            )
        splits["train"] = train_examples

        val_examples = []
        for idx in val_indices:
            item = train_raw[idx]
            text = item.get("text", "").strip()
            label = int(item["label"])
            val_examples.append(
                ICLExample(
                    id=f"ag_news_val_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=self.classes[label],
                    split="validation",
                )
            )
        splits["validation"] = val_examples

        logger.info(
            f"Loaded AG News: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )
        return splits
