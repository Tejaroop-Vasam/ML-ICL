"""Deterministic Mock LLM for local testing and dry-run validation."""

from typing import List
from src.models.base_model import BaseLLM


class MockLLM(BaseLLM):
    """Simulates an LLM for rapid pipeline testing without downloading weights."""

    def __init__(self, model_name: str = "mock-llama", seed: int = 42):
        super().__init__(model_name)
        self.seed = seed

    def count_tokens(self, text: str) -> int:
        """Approximates token count (~1.3 tokens per word)."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def generate(self, prompt: str, max_new_tokens: int = 16, **kwargs) -> str:
        """Deterministic pseudo-prediction based on keywords in prompt."""
        lower = prompt.lower()
        if "sentiment" in lower:
            # Check for negative clues in the prompt query
            if any(w in lower for w in ["terrible", "awful", "bad", "worst", "waste", "boring"]):
                return "negative"
            return "positive"
        elif "topic" in lower or "category" in lower:
            if any(w in lower for w in ["football", "baseball", "coach", "game", "team"]):
                return "Sports"
            if any(w in lower for w in ["oil", "stocks", "market", "economy", "bank"]):
                return "Business"
            if any(w in lower for w in ["software", "google", "apple", "tech", "computer"]):
                return "Sci/Tech"
            return "World"
        elif "question" in lower:
            if "how many" in lower or "how much" in lower:
                return "Numeric"
            if "who" in lower:
                return "Human"
            if "where" in lower:
                return "Location"
            return "Description"
        return "positive"
