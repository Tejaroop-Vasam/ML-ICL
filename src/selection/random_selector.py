"""Random demonstration selection baseline."""

from typing import List, Optional
import numpy as np

from src.data.example import ICLExample
from src.selection.base_selector import BaseSelector


class RandomSelector(BaseSelector):
    """Selects k demonstrations uniformly at random from the candidate pool."""

    def __init__(
        self,
        candidate_pool: List[ICLExample],
        seed: int = 42,
        stratified: bool = True,
    ):
        super().__init__(candidate_pool, seed)
        self.stratified = stratified

    def select_demonstrations(
        self,
        test_example: ICLExample,
        k: int,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[ICLExample]:
        """Sample k demonstrations at random.

        Args:
            test_example: Query example (not used in random selection, but excluded if present in pool).
            k: Number of demonstrations (if k=0, returns empty list).
            seed: Override random seed for evaluating variance across multiple sampling runs.
        """
        if k <= 0:
            return []

        # Filter out candidate identical to test example if in pool
        candidates = [c for c in self.candidate_pool if c.id != test_example.id]
        if not candidates:
            return []

        rng = np.random.RandomState(seed if seed is not None else self.seed)

        if not self.stratified:
            indices = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
            return [candidates[i] for i in indices]

        # Stratified sampling: balance representation across classes
        by_class = {}
        for c in candidates:
            by_class.setdefault(c.label, []).append(c)

        classes = sorted(by_class.keys())
        selected = []
        cls_idx = 0
        while len(selected) < k and any(by_class.values()):
            c_label = classes[cls_idx % len(classes)]
            cls_pool = by_class[c_label]
            if cls_pool:
                chosen_idx = rng.randint(0, len(cls_pool))
                selected.append(cls_pool.pop(chosen_idx))
            cls_idx += 1

        rng.shuffle(selected)
        return selected[:k]
