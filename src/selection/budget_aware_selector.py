"""Budget-Aware and Diversity-Aware Demonstration Selection Algorithm.

Formulation:
    S(d_i, x | D) = alpha * sim(d_i, x) + beta * div(d_i, D) - gamma * cost(d_i)
"""

import logging
from typing import List, Optional, Tuple, Dict
import numpy as np

from src.data.example import ICLExample
from src.selection.similarity_selector import SimilaritySelector
from src.selection.ordering import order_demonstrations

logger = logging.getLogger(__name__)


class BudgetAwareSelector(SimilaritySelector):
    """Novel demonstration selection method balancing semantic similarity,

    diversity against already-chosen examples, and token length cost under a strict budget.
    """

    def __init__(
        self,
        candidate_pool: List[ICLExample],
        alpha: float = 1.0,    # Weight for semantic relevance to test query
        beta: float = 0.5,     # Weight for diversity relative to chosen set
        gamma: float = 0.2,    # Penalty weight for token length cost
        token_budget: Optional[int] = None, # Total token limit for demonstrations
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = "data/cache/embeddings",
        seed: int = 42,
    ):
        super().__init__(
            candidate_pool=candidate_pool,
            model_name=model_name,
            cache_dir=cache_dir,
            seed=seed,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.token_budget = token_budget

    def select_demonstrations(
        self,
        test_example: ICLExample,
        k: int,
        token_budget: Optional[int] = None,
        ordering: str = "similar_last",
        top_m_candidates: int = 50,
        **kwargs,
    ) -> List[ICLExample]:
        """Greedily select demonstrations maximizing relevance and diversity while penalizing cost.

        Args:
            test_example: The query to classify.
            k: Maximum number of demonstrations to select.
            token_budget: Explicit token budget limit. Overrides self.token_budget if set.
            ordering: Ordering strategy for chosen demonstrations.
            top_m_candidates: Candidate pre-filter pool size for efficient greedy search.
        """
        if k <= 0 or not self.candidate_pool:
            return []

        effective_budget = token_budget if token_budget is not None else self.token_budget

        model = self._get_model()
        query_emb = model.encode(
            [test_example.text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        candidate_embs = self.get_candidate_embeddings()
        rel_scores = np.dot(candidate_embs, query_emb)

        # Filter query if in pool
        for idx, c in enumerate(self.candidate_pool):
            if c.id == test_example.id:
                rel_scores[idx] = -1e9

        # Filter to top M candidate indices to bound search complexity
        m = min(top_m_candidates, len(self.candidate_pool))
        cand_indices = list(np.argsort(rel_scores)[::-1][:m])

        # Precompute normalized token costs among top M
        lengths = [
            self.candidate_pool[i].num_tokens or self.candidate_pool[i].num_words
            for i in cand_indices
        ]
        max_len = max(max(lengths), 1)
        norm_costs = {idx: lengths[p] / max_len for p, idx in enumerate(cand_indices)}

        selected_indices: List[int] = []
        selected_embs: List[np.ndarray] = []
        current_token_count = 0

        for step in range(k):
            best_score = -float("inf")
            best_idx = None
            best_len = 0

            for idx in cand_indices:
                if idx in selected_indices:
                    continue

                c_len = self.candidate_pool[idx].num_tokens or self.candidate_pool[idx].num_words
                if effective_budget is not None and (current_token_count + c_len > effective_budget):
                    continue

                relevance = rel_scores[idx]

                # Diversity: distance from closest already-selected demonstration
                if not selected_embs:
                    diversity = 1.0
                else:
                    sim_to_selected = [
                        float(np.dot(candidate_embs[idx], sel_emb))
                        for sel_emb in selected_embs
                    ]
                    # Higher distance = higher diversity
                    diversity = 1.0 - max(sim_to_selected)

                cost = norm_costs[idx]

                score = (self.alpha * relevance) + (self.beta * diversity) - (self.gamma * cost)

                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_len = c_len

            if best_idx is None:
                # No remaining candidate fits budget or passes criteria
                break

            selected_indices.append(best_idx)
            selected_embs.append(candidate_embs[best_idx])
            current_token_count += best_len

        selected_demos = [self.candidate_pool[i] for i in selected_indices]
        selected_sims = [float(rel_scores[i]) for i in selected_indices]

        # Apply ordering policy
        return order_demonstrations(
            demonstrations=selected_demos,
            ordering=ordering,
            similarity_scores=selected_sims,
            seed=self.seed,
        )
