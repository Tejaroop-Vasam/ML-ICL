"""TREC (Question Classification) DataLoader for 6 coarse question types."""

import logging
from typing import List, Dict
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample

logger = logging.getLogger(__name__)


class TRECDataLoader(BaseDataLoader):
    """DataLoader for TREC coarse question classification (6 categories)."""

    COARSE_LABELS = [
        "Description",   # 0: DESC
        "Entity",        # 1: ENTY
        "Abbreviation",  # 2: ABBR
        "Human",         # 3: HUM
        "Numeric",       # 4: NUM
        "Location",      # 5: LOC
    ]

    COARSE_CODES = ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]

    @property
    def dataset_name(self) -> str:
        return "trec"

    @property
    def classes(self) -> List[str]:
        return self.COARSE_LABELS

    @property
    def task_instruction(self) -> str:
        return (
            "Classify each question into one of the following 6 question types: "
            "Description, Entity, Abbreviation, Human, Numeric, or Location."
        )

    @property
    def default_input_prefix(self) -> str:
        return "Question: "

    @property
    def default_label_prefix(self) -> str:
        return "Type: "

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        """Fetch TREC dataset from Hugging Face Hub ('SetFit/TREC-QC')."""
        logger.info("Loading TREC from Hugging Face ('SetFit/TREC-QC')...")
        hf_dataset = load_dataset("SetFit/TREC-QC")

        splits: Dict[str, List[ICLExample]] = {}

        # Test split (500 questions)
        test_examples = []
        for idx, item in enumerate(hf_dataset["test"]):
            text = item.get("text", "").strip()
            coarse_id = int(item["label_coarse"])
            label_name = self.classes[coarse_id]
            test_examples.append(
                ICLExample(
                    id=f"trec_test_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=coarse_id,
                    label_name=label_name,
                    split="test",
                    metadata={
                        "code": self.COARSE_CODES[coarse_id],
                        "fine_label": item.get("label_text", ""),
                    },
                )
            )
        splits["test"] = test_examples

        # Train split (5452 questions) -> carve out 500 as validation
        train_raw = list(hf_dataset["train"])
        train_indices, val_indices = train_test_split(
            range(len(train_raw)),
            test_size=min(500, int(len(train_raw) * 0.1)),
            random_state=self.seed,
            stratify=[x["label_coarse"] for x in train_raw],
        )

        train_examples = []
        for idx in train_indices:
            item = train_raw[idx]
            text = item.get("text", "").strip()
            coarse_id = int(item["label_coarse"])
            label_name = self.classes[coarse_id]
            train_examples.append(
                ICLExample(
                    id=f"trec_train_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=coarse_id,
                    label_name=label_name,
                    split="train",
                    metadata={
                        "code": self.COARSE_CODES[coarse_id],
                        "fine_label": item.get("label_text", ""),
                    },
                )
            )
        splits["train"] = train_examples

        val_examples = []
        for idx in val_indices:
            item = train_raw[idx]
            text = item.get("text", "").strip()
            coarse_id = int(item["label_coarse"])
            label_name = self.classes[coarse_id]
            val_examples.append(
                ICLExample(
                    id=f"trec_val_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=coarse_id,
                    label_name=label_name,
                    split="validation",
                    metadata={
                        "code": self.COARSE_CODES[coarse_id],
                        "fine_label": item.get("label_text", ""),
                    },
                )
            )
        splits["validation"] = val_examples

        logger.info(
            f"Loaded TREC: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )
        return splits
