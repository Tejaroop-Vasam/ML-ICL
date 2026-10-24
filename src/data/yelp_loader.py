"""Yelp Polarity DataLoader for binary sentiment classification.

Dataset: fancyzhx/yelp_polarity (~560k train / 38k test)
Labels: 0 = negative, 1 = positive
"""

import logging
from typing import List, Dict
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample

logger = logging.getLogger(__name__)


class YelpPolarityDataLoader(BaseDataLoader):
    """DataLoader for Yelp Polarity binary sentiment classification.

    Much larger than SST-2 (~560k training reviews). Reviews are longer
    (multi-sentence) which tests the model's ability to handle longer inputs.
    """

    @property
    def dataset_name(self) -> str:
        return "yelp_polarity"

    @property
    def classes(self) -> List[str]:
        # 1 = negative, 2 = positive in raw Yelp; remapped to 0/1
        return ["negative", "positive"]

    @property
    def task_instruction(self) -> str:
        return (
            "Classify the sentiment of the following Yelp review as either "
            "'positive' or 'negative'."
        )

    @property
    def default_input_prefix(self) -> str:
        return "Review: "

    @property
    def default_label_prefix(self) -> str:
        return "Sentiment: "

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        """Fetch Yelp Polarity from Hugging Face Hub ('fancyzhx/yelp_polarity')."""
        logger.info("Loading Yelp Polarity from Hugging Face ('fancyzhx/yelp_polarity')...")
        hf_dataset = load_dataset("fancyzhx/yelp_polarity")

        splits: Dict[str, List[ICLExample]] = {}

        # Process test split (capped at 5,000 for fast loading/caching; ample for n_eval<=1000)
        MAX_TEST = 5_000
        test_ds = hf_dataset["test"]
        if len(test_ds) > MAX_TEST:
            test_ds = test_ds.shuffle(seed=self.seed).select(range(MAX_TEST))

        test_examples = []
        for idx, item in enumerate(test_ds):
            text = item.get("text", "").strip()
            label = int(item["label"])  # 0 = negative, 1 = positive
            test_examples.append(
                ICLExample(
                    id=f"yelp_test_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=self.classes[label],
                    split="test",
                )
            )
        splits["test"] = test_examples

        # Process train split: select 20k directly using Hugging Face dataset indexing
        # without converting 560,000 examples to a Python list (avoids OOM and RAM thrashing)
        MAX_TRAIN = 20_000
        train_ds = hf_dataset["train"]
        if len(train_ds) > MAX_TRAIN:
            train_sub = train_ds.shuffle(seed=self.seed).select(range(MAX_TRAIN))
        else:
            train_sub = train_ds

        val_size = min(2000, int(len(train_sub) * 0.1))
        train_size = len(train_sub) - val_size

        train_examples = []
        for idx in range(train_size):
            item = train_sub[idx]
            text = item.get("text", "").strip()
            label = int(item["label"])
            train_examples.append(
                ICLExample(
                    id=f"yelp_train_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=self.classes[label],
                    split="train",
                )
            )
        splits["train"] = train_examples

        val_examples = []
        for idx in range(train_size, len(train_sub)):
            item = train_sub[idx]
            text = item.get("text", "").strip()
            label = int(item["label"])
            val_examples.append(
                ICLExample(
                    id=f"yelp_val_{idx - train_size}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=self.classes[label],
                    split="validation",
                )
            )
        splits["validation"] = val_examples

        logger.info(
            f"Loaded Yelp Polarity: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )
        return splits
