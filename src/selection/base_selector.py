"""Base interface for demonstration selection strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional
from src.data.example import ICLExample


class BaseSelector(ABC):
    """Abstract base class for all ICL demonstration selection strategies."""

    def __init__(self, candidate_pool: List[ICLExample], seed: int = 42):
        """
        Args:
            candidate_pool: The training examples pool from which demonstrations can be drawn.
            seed: Random seed for reproducibility.
        """
        self.candidate_pool = candidate_pool
        self.seed = seed

    @abstractmethod
    def select_demonstrations(
        self,
        test_example: ICLExample,
        k: int,
        **kwargs,
    ) -> List[ICLExample]:
        """Select k demonstrations from the candidate pool for the given test example.

        Args:
            test_example: The query sample being evaluated.
            k: Number of demonstrations to select.
            **kwargs: Additional parameters (e.g. token_budget, ordering).

        Returns:
            List of k selected demonstration examples.
        """
        pass
