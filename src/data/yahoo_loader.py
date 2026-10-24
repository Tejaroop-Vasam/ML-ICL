"""Yahoo Answers Topics DataLoader for 10-class topic classification.

Dataset: community-datasets/yahoo_answers_topics (~1.4M train / 60k test)
Labels: 0-9 mapping to 10 topic categories.
"""

import logging
from typing import List, Dict
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample

logger = logging.getLogger(__name__)

# 10 topic labels in Yahoo Answers Topics
YAHOO_CLASSES = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government",
]


class YahooAnswersDataLoader(BaseDataLoader):
    """DataLoader for Yahoo Answers Topics 10-class topic classification.

    Replaces AG News. Yahoo Answers has 10 classes vs AG News's 4,
    making it a harder multi-class ICL challenge. Inputs are Q&A style
    (question + best_answer concatenated) which tests longer-context handling.
    """

    @property
    def dataset_name(self) -> str:
        return "yahoo_answers"

    @property
    def classes(self) -> List[str]:
        return YAHOO_CLASSES

    @property
    def task_instruction(self) -> str:
        return (
            "Classify the following Yahoo Answers question into one of the 10 topic categories: "
            "Society & Culture, Science & Mathematics, Health, Education & Reference, "
            "Computers & Internet, Sports, Business & Finance, Entertainment & Music, "
            "Family & Relationships, or Politics & Government."
        )

    @property
    def default_input_prefix(self) -> str:
        return "Question: "

    @property
    def default_label_prefix(self) -> str:
        return "Topic: "

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        """Fetch Yahoo Answers Topics from Hugging Face Hub."""
        logger.info("Loading Yahoo Answers Topics from Hugging Face ('community-datasets/yahoo_answers_topics')...")
        hf_dataset = load_dataset("community-datasets/yahoo_answers_topics")

        splits: Dict[str, List[ICLExample]] = {}

        # Build text from question_title + question_content (+ best_answer if available)
        def build_text(item: dict) -> str:
            parts = []
            title = (item.get("question_title") or "").strip()
            content = (item.get("question_content") or "").strip()
            answer = (item.get("best_answer") or "").strip()
            if title:
                parts.append(title)
            if content and content != title:
                parts.append(content)
            if answer:
                parts.append(f"Answer: {answer}")
            return " ".join(parts)

        # Process test split (capped at 5,000 for fast loading/caching; ample for n_eval<=1000)
        MAX_TEST = 5_000
        test_ds = hf_dataset["test"]
        if len(test_ds) > MAX_TEST:
            test_ds = test_ds.shuffle(seed=self.seed).select(range(MAX_TEST))

        test_examples = []
        for idx, item in enumerate(test_ds):
            text = build_text(item).strip()
            label = int(item["topic"])
            if label < 0 or label >= len(YAHOO_CLASSES):
                continue
            test_examples.append(
                ICLExample(
                    id=f"yahoo_test_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=YAHOO_CLASSES[label],
                    split="test",
                )
            )
        splits["test"] = test_examples

        # Process train split: select 20k directly using Hugging Face dataset indexing
        # without converting 1,400,000 examples to a Python list (avoids OOM and RAM thrashing)
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
            text = build_text(item).strip()
            label = int(item["topic"])
            if label < 0 or label >= len(YAHOO_CLASSES):
                continue
            train_examples.append(
                ICLExample(
                    id=f"yahoo_train_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=YAHOO_CLASSES[label],
                    split="train",
                )
            )
        splits["train"] = train_examples

        val_examples = []
        for idx in range(train_size, len(train_sub)):
            item = train_sub[idx]
            text = build_text(item).strip()
            label = int(item["topic"])
            if label < 0 or label >= len(YAHOO_CLASSES):
                continue
            val_examples.append(
                ICLExample(
                    id=f"yahoo_val_{idx - train_size}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=YAHOO_CLASSES[label],
                    split="validation",
                )
            )
        splits["validation"] = val_examples

        logger.info(
            f"Loaded Yahoo Answers: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )
        return splits
