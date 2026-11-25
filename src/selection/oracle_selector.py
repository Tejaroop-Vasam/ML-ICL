"""Oracle / Gold Demonstration Selector for disentangling retriever vs LLM bottlenecks.

Provides ground-truth aligned, class-balanced, or gold-relevant demonstrations
from the candidate pool to serve as an empirical upper bound, isolating retrieval
errors in BGE-M3 ranking from LLM reasoning limits.
"""

import logging
from typing import List, Optional, Dict
import numpy as np

from src.data.example import ICLExample
from src.retrieval import BGE_M3_ID
from src.selection.base_selector import BaseSelector
from src.selection.ordering import order_demonstrations

logger = logging.getLogger(__name__)


class OracleSelector(BaseSelector):
    """Oracle demonstration selector providing gold-standard exemplar sets.

    Modes:
      - 'balanced_relevant' (default): Selects top semantically relevant exemplars
        while enforcing strict class balance across the candidate label space.
      - 'label_congruent': Selects exemplars sharing the exact target ground-truth label.
      - 'stratified_gold': Selects verified canonical class exemplars.
    """

    def __init__(
        self,
        candidate_pool: List[ICLExample],
        mode: str = "balanced_relevant",
        model_name: str = BGE_M3_ID,
        seed: int = 42,
    ):
        super().__init__(candidate_pool, seed)
        self.mode = mode
        self.model_name = model_name
        self._similarity_selector = None

    def _get_similarity_selector(self):
        if self._similarity_selector is None:
            from src.selection.similarity_selector import SimilaritySelector
            self._similarity_selector = SimilaritySelector(
                candidate_pool=self.candidate_pool,
                model_name=self.model_name,
                seed=self.seed,
            )
        return self._similarity_selector

    def select_demonstrations(
        self,
        test_example: ICLExample,
        k: int,
        ordering: str = "similar_last",
        **kwargs,
    ) -> List[ICLExample]:
        """Select oracle demonstrations for test_example.

        Args:
            test_example: The evaluation query with known ground-truth label.
            k: Total number of demonstrations to select.
            ordering: Ordering strategy.
        """
        if k <= 0 or not self.candidate_pool:
            return []

        # Available candidates excluding test query
        candidates = [c for c in self.candidate_pool if c.id != test_example.id]
        if not candidates:
            return []

        if self.mode == "label_congruent":
            # Filter strictly to same label
            matched = [c for c in candidates if c.label == test_example.label]
            pool_to_use = matched if len(matched) >= k else candidates
            sim_sel = self._get_similarity_selector()
            # Rank by similarity within the label pool
            scores = sim_sel.compute_similarity_scores(test_example, pool_to_use)
            ranked_indices = np.argsort(scores)[::-1][:k]
            selected = [pool_to_use[i] for i in ranked_indices]
            sel_scores = [scores[i] for i in ranked_indices]
            return order_demonstrations(selected, ordering=ordering, similarity_scores=sel_scores)

        elif self.mode == "balanced_relevant":
            # Partition candidates by class label
            class_groups: Dict[int, List[ICLExample]] = {}
            for c in candidates:
                class_groups.setdefault(c.label, []).append(c)

            unique_labels = sorted(class_groups.keys())
            num_classes = len(unique_labels)
            base_per_class = k // num_classes
            rem = k % num_classes

            selected: List[ICLExample] = []
            selected_scores: List[float] = []

            # Ensure the query's ground truth class gets priority for remainder
            sorted_labels = sorted(
                unique_labels,
                key=lambda l: (0 if l == test_example.label else 1, l)
            )

            sim_sel = self._get_similarity_selector()
            for idx, label in enumerate(sorted_labels):
                quota = base_per_class + (1 if idx < rem else 0)
                if quota <= 0:
                    continue
                cands_in_class = class_groups.get(label, [])
                if not cands_in_class:
                    continue

                scores = sim_sel.compute_similarity_scores(test_example, cands_in_class)
                top_idx = np.argsort(scores)[::-1][:quota]
                for ti in top_idx:
                    selected.append(cands_in_class[ti])
                    selected_scores.append(float(scores[ti]))

            return order_demonstrations(selected, ordering=ordering, similarity_scores=selected_scores)

        else:
            # Fallback to standard similarity
            sim_sel = self._get_similarity_selector()
            return sim_sel.select_demonstrations(test_example, k=k, ordering=ordering)
