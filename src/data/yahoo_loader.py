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

        # Process test split (60,000 examples — subsampled at eval time)
        test_examples = []
        for idx, item in enumerate(hf_dataset["test"]):
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

        # Train split: subsample to 20k (stratified) then 10% validation
        train_raw = list(hf_dataset["train"])

        MAX_TRAIN = 20_000
        if len(train_raw) > MAX_TRAIN:
            subsample_indices, _ = train_test_split(
                range(len(train_raw)),
                train_size=MAX_TRAIN,
                random_state=self.seed,
                stratify=[x["topic"] for x in train_raw],
            )
            train_raw = [train_raw[i] for i in sorted(subsample_indices)]

        train_indices, val_indices = train_test_split(
            range(len(train_raw)),
            test_size=min(2000, int(len(train_raw) * 0.1)),
            random_state=self.seed,
            stratify=[x["topic"] for x in train_raw],
        )

        train_examples = []
        for idx in train_indices:
            item = train_raw[idx]
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
        for idx in val_indices:
            item = train_raw[idx]
            text = build_text(item).strip()
            label = int(item["topic"])
            if label < 0 or label >= len(YAHOO_CLASSES):
                continue
            val_examples.append(
                ICLExample(
                    id=f"yahoo_val_{idx}",
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
