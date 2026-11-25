"""Base LLM interface for In-Context Learning evaluation."""

from abc import ABC, abstractmethod
from typing import List

from src.label_parsing import extract_legacy_label


class BaseLLM(ABC):
    """Abstract base class for LLMs used in ICL inference."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0, **kwargs) -> str:
        """Generate response text for the given prompt."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the exact number of tokens in the prompt using the model's tokenizer."""
        pass

    def extract_label(self, generated_text: str, candidate_classes: List[str]) -> str:
        """Robustly extract the predicted label from model output.

        Matches candidate class names (case-insensitive) appearing in the generated text.
        """
        label, _ = extract_legacy_label(generated_text, candidate_classes)
        return label
