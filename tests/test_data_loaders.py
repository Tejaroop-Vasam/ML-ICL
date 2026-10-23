"""Unit tests for ICL data loaders."""

import unittest
from pathlib import Path
from src.data import get_dataset, list_datasets, ICLExample


class TestDataLoaders(unittest.TestCase):

    def test_list_datasets(self):
        datasets = list_datasets()
        self.assertIn("yelp_polarity", datasets)
        self.assertIn("yahoo_answers", datasets)
        self.assertIn("trec", datasets)
        self.assertIn("imdb", datasets)

    def test_yelp_loader(self):
        loader = get_dataset("yelp_polarity")
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

    def test_yahoo_loader(self):
        loader = get_dataset("yahoo_answers")
        splits = loader.load_data()
        self.assertEqual(loader.num_classes, 10)
        sample = loader.get_stratified_subset("test", n_samples=10, seed=42)
        self.assertEqual(len(sample), 10)
        # Check that labels are valid topic IDs
        labels = [ex.label for ex in sample]
        self.assertTrue(all(0 <= lbl < 10 for lbl in labels))

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

    def test_numpy_dataset(self):
        from src.data import NumpyDataset
        loader = get_dataset("yelp_polarity")
        np_ds = loader.get_numpy_dataset("test")
        self.assertIsInstance(np_ds, NumpyDataset)
        self.assertGreater(len(np_ds), 0)

        # Test indexing & slicing
        text, label, item_id = np_ds[0]
        self.assertIsInstance(text, str)
        self.assertIn(label, [0, 1])

        # Test stratified sampling
        sample = np_ds.stratified_sample(n_samples=10, seed=42)
        self.assertEqual(len(sample), 10)
        self.assertEqual(list(sample.class_counts.values()), [5, 5])

        # Test length filtering
        long_subset = np_ds.filter_by_word_length(min_words=20)
        self.assertTrue(all(len(t.split()) >= 20 for t in long_subset.texts))


if __name__ == "__main__":
    unittest.main()
