"""Unit tests for demonstration selection strategies and ordering."""

import unittest
from src.data.example import ICLExample
from src.selection import (
    RandomSelector,
    SimilaritySelector,
    BudgetAwareSelector,
    order_demonstrations,
)


class TestSelectionAndOrdering(unittest.TestCase):

    def setUp(self):
        self.pool = [
            ICLExample(
                id="d1", dataset="toy", text="great film loved it", label=1, label_name="positive", split="train"
            ),
            ICLExample(
                id="d2", dataset="toy", text="terrible movie awful waste", label=0, label_name="negative", split="train"
            ),
            ICLExample(
                id="d3", dataset="toy", text="wonderful acting and plot", label=1, label_name="positive", split="train"
            ),
            ICLExample(
                id="d4", dataset="toy", text="boring and slow disaster", label=0, label_name="negative", split="train"
            ),
        ]
        self.query = ICLExample(
            id="q1", dataset="toy", text="loved this great movie", label=1, label_name="positive", split="test"
        )

    def test_random_selector(self):
        sel = RandomSelector(self.pool, seed=42)
        demos = sel.select_demonstrations(self.query, k=2)
        self.assertEqual(len(demos), 2)
        self.assertTrue(all(isinstance(d, ICLExample) for d in demos))

    def test_ordering(self):
        demos = self.pool[:3]
        scores = [0.2, 0.9, 0.5]

        # similar_first: highest score at index 0
        first_order = order_demonstrations(demos, ordering="similar_first", similarity_scores=scores)
        self.assertEqual(first_order[0].id, "d2")

        # similar_last: highest score at the end
        last_order = order_demonstrations(demos, ordering="similar_last", similarity_scores=scores)
        self.assertEqual(last_order[-1].id, "d2")

    def test_budget_aware_selector(self):
        sel = BudgetAwareSelector(self.pool, alpha=1.0, beta=0.5, gamma=0.1, seed=42)
        demos = sel.select_demonstrations(self.query, k=2, token_budget=50)
        self.assertLessEqual(len(demos), 2)
        self.assertTrue(all(isinstance(d, ICLExample) for d in demos))


if __name__ == "__main__":
    unittest.main()
