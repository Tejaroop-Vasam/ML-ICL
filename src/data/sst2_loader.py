"""SST-2 (Stanford Sentiment Treebank) DataLoader for sentiment classification."""

import logging
from typing import List, Dict
from datasets import load_dataset

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample

logger = logging.getLogger(__name__)


class SST2DataLoader(BaseDataLoader):
    """DataLoader for SST-2 binary sentiment classification (short inputs)."""

    @property
    def dataset_name(self) -> str:
        return "sst2"

    @property
    def classes(self) -> List[str]:
        return ["negative", "positive"]

    @property
    def task_instruction(self) -> str:
        return "Classify the sentiment of each review as either 'positive' or 'negative'."

    @property
    def default_input_prefix(self) -> str:
        return "Review: "

    @property
    def default_label_prefix(self) -> str:
        return "Sentiment: "

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        """Fetch SST-2 dataset from Hugging Face Hub (SetFit/sst2 with labeled splits)."""
        logger.info("Loading SST-2 from Hugging Face ('SetFit/sst2')...")
        hf_dataset = load_dataset("SetFit/sst2")

        splits: Dict[str, List[ICLExample]] = {}
        for split_name in ["train", "validation", "test"]:
            if split_name not in hf_dataset:
                continue

            examples = []
            for idx, item in enumerate(hf_dataset[split_name]):
                text = item.get("text", "").strip()
                label = int(item["label"])
                # Ensure binary label
                if label not in (0, 1):
                    continue

                label_name = self.classes[label]
                example = ICLExample(
                    id=f"sst2_{split_name}_{idx}",
                    dataset=self.dataset_name,
                    text=text,
                    label=label,
                    label_name=label_name,
                    split=split_name,
                )
                examples.append(example)

            splits[split_name] = examples
            logger.info(f"Loaded {len(examples)} examples for SST-2 split '{split_name}'")

        return splits
