"""Base LLM interface for In-Context Learning evaluation."""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


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
        clean_gen = generated_text.strip()
        first_line = clean_gen.split("\n")[0].strip()

        # 1. Exact match at start of first line
        for cls in candidate_classes:
            pattern = rf"^\b{re.escape(cls)}\b"
            if re.search(pattern, first_line, re.IGNORECASE):
                return cls

        # 2. Search anywhere in first line
        for cls in candidate_classes:
            pattern = rf"\b{re.escape(cls)}\b"
            if re.search(pattern, first_line, re.IGNORECASE):
                return cls

        # 3. Search anywhere in full generated output
        for cls in candidate_classes:
            pattern = rf"\b{re.escape(cls)}\b"
            if re.search(pattern, clean_gen, re.IGNORECASE):
                return cls

        # Default fallback: return first candidate class if nothing matches
        return candidate_classes[0]
