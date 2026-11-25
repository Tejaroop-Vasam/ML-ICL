import unittest

from src.review_experiments.classification import ClassificationItem, classification_prompt


class ClassificationPromptTest(unittest.TestCase):
    def test_zero_shot_preserves_dataset_prefixes(self):
        for dataset, suffix in (
            ("imdb", "Review: Text\nSentiment:"),
            ("yelp_polarity", "Review: Text\nSentiment:"),
            ("trec", "Question: Text\nType:"),
            ("yahoo_answers", "Question: Text\nTopic:"),
        ):
            with self.subTest(dataset=dataset):
                prompt = classification_prompt(dataset, "  Text  ", instruction="  Classify.  ")
                self.assertEqual(prompt, f"Classify.\n\n{suffix}")

    def test_demo_order_and_plain_and_structured_layouts(self):
        demos = [
            ClassificationItem("a", "  Great.  ", "positive"),
            ClassificationItem("b", "Poor.", "negative"),
        ]
        plain = classification_prompt("imdb", "Mixed.", demos, "Classify.")
        structured = classification_prompt("imdb", "Mixed.", demos, "Classify.", structured=True)
        self.assertEqual(
            plain, "Classify.\n\nReview: Great.\nSentiment: positive\n\n"
                   "Review: Poor.\nSentiment: negative\n\nReview: Mixed.\nSentiment:",
        )
        self.assertEqual(
            structured, "Classify.\n\n[Input] Great.\n[Label] positive\n\n"
                        "[Input] Poor.\n[Label] negative\n\n[Input] Mixed.\n[Label]",
        )

    def test_empty_content_free_prompt_preserves_delimiters(self):
        prompt = classification_prompt("imdb", "")
        self.assertEqual(
            prompt, "Classify the sentiment of each movie review as either 'positive' or 'negative'."
                    "\n\nReview: \nSentiment:",
        )
