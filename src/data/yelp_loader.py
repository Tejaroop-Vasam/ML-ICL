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

        # Process test split (38,000 examples — we will subsample at eval time)
        test_examples = []
        for idx, item in enumerate(hf_dataset["test"]):
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

        # Process train split: subsample 20k for reasonable memory use,
        # then carve out 10% as validation
        train_raw = list(hf_dataset["train"])

        # Subsample to 20k to keep processing fast (stratified)
        MAX_TRAIN = 20_000
        if len(train_raw) > MAX_TRAIN:
            subsample_indices, _ = train_test_split(
                range(len(train_raw)),
                train_size=MAX_TRAIN,
                random_state=self.seed,
                stratify=[x["label"] for x in train_raw],
            )
            train_raw = [train_raw[i] for i in sorted(subsample_indices)]

        train_indices, val_indices = train_test_split(
            range(len(train_raw)),
            test_size=min(2000, int(len(train_raw) * 0.1)),
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
        for idx in val_indices:
            item = train_raw[idx]
            text = item.get("text", "").strip()
            label = int(item["label"])
            val_examples.append(
                ICLExample(
                    id=f"yelp_val_{idx}",
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
