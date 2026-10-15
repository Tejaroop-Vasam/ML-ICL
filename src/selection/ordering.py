"""Demonstration ordering strategies."""

from typing import List, Optional
import numpy as np
from src.data.example import ICLExample


def order_demonstrations(
    demonstrations: List[ICLExample],
    ordering: str = "similar_last",
    similarity_scores: Optional[List[float]] = None,
    seed: int = 42,
) -> List[ICLExample]:
    """Order a given set of demonstrations according to specified strategy.

    Args:
        demonstrations: The demonstrations to order.
        ordering: One of:
            - 'similar_last': Most similar demonstration placed closest to query (at the end).
            - 'similar_first': Most similar demonstration placed first (furthest from query).
            - 'random': Shuffled with random seed.
            - 'original': Preserves candidate selection order.
            - 'fixed_reference': Deterministically sorted by example id across all queries.
        similarity_scores: Semantic similarity score for each demonstration in demonstrations.
        seed: Random seed for 'random' ordering.

    Returns:
        Ordered list of demonstrations.
    """
    if len(demonstrations) <= 1:
        return list(demonstrations)

    if ordering == "original":
        return list(demonstrations)

    elif ordering == "random":
        rng = np.random.RandomState(seed)
        shuffled = list(demonstrations)
        rng.shuffle(shuffled)
        return shuffled

    elif ordering == "fixed_reference":
        # Global fixed reference ordering based on deterministic example id
        return sorted(demonstrations, key=lambda x: str(x.id))

    elif ordering in ("similar_first", "similar_last"):
        if similarity_scores is None or len(similarity_scores) != len(demonstrations):
            # Fall back to original if no similarity scores provided
            return list(demonstrations)

        # Pair examples with scores
        paired = list(zip(demonstrations, similarity_scores))

        if ordering == "similar_first":
            # Descending: highest similarity at index 0 (start of prompt)
            sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=True)
        else:
            # similar_last: Ascending, so highest similarity is at the end (adjacent to query)
            sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=False)

        return [item[0] for item in sorted_pairs]

    else:
        raise ValueError(
            f"Unknown ordering '{ordering}'. Expected 'similar_last', 'similar_first', 'random', or 'fixed_reference'."
        )
