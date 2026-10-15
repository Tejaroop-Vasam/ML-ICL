"""Unit tests for ICL data loaders."""

import unittest
from pathlib import Path
from src.data import get_dataset, list_datasets, ICLExample


class TestDataLoaders(unittest.TestCase):

    def test_list_datasets(self):
        datasets = list_datasets()
        self.assertIn("sst2", datasets)
        self.assertIn("ag_news", datasets)
        self.assertIn("trec", datasets)
        self.assertIn("imdb", datasets)

    def test_sst2_loader(self):
        loader = get_dataset("sst2")
        splits = loader.load_data()
        self.assertIn("train", splits)
        self.assertIn("test", splits)
        self.assertEqual(loader.num_classes, 2)
        self.assertEqual(loader.classes, ["negative", "positive"])

        # Test stratified subset
        sample = loader.get_stratified_subset("test", n_samples=10, seed=42)
        self.assertEqual(len(sample), 10)
        self.assertTrue(all(isinstance(ex, ICLExample) for ex in sample))

        # Test prompt formatting
        demos = loader.get_candidate_demonstration_pool(pool_size=2)
        prompt = loader.format_prompt(sample[0], demonstrations=demos, format_type="plain")
        self.assertIn("Review:", prompt)
        self.assertIn("Sentiment:", prompt)

    def test_agnews_loader(self):
        loader = get_dataset("ag_news")
        splits = loader.load_data()
        self.assertEqual(loader.num_classes, 4)
        sample = loader.get_stratified_subset("test", n_samples=8, seed=42)
        self.assertEqual(len(sample), 8)
        # Check class balance in stratified sample
        labels = [ex.label for ex in sample]
        self.assertEqual(sorted(list(set(labels))), [0, 1, 2, 3])

    def test_trec_loader(self):
        loader = get_dataset("trec")
        splits = loader.load_data()
        self.assertEqual(loader.num_classes, 6)
        sample = loader.get_stratified_subset("test", n_samples=12, seed=42)
        self.assertEqual(len(sample), 12)
        self.assertTrue(sample[0].text.strip().endswith("?") or len(sample[0].text) > 0)

    def test_imdb_loader(self):
        loader = get_dataset("imdb")
        splits = loader.load_data()
        self.assertEqual(loader.num_classes, 2)
        sample = loader.get_stratified_subset("test", n_samples=6, seed=42)
        self.assertEqual(len(sample), 6)
        # Verify text was cleaned (no raw <br />)
        for ex in sample:
            self.assertNotIn("<br />", ex.text)
            self.assertGreater(ex.num_words, 0)


if __name__ == "__main__":
    unittest.main()
