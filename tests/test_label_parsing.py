import unittest

from src.label_parsing import TREC_LABELS, rescore_classification_trial, score_classification_output


class ClassificationScoringTest(unittest.TestCase):
    def test_extracts_one_distinct_label(self):
        result = score_classification_output("Type: **numeric**. Numeric answer.", TREC_LABELS, "Numeric")
        self.assertEqual(result["prediction"], "Numeric")
        self.assertTrue(result["is_correct"])

    def test_invalid_output_does_not_default_to_a_class(self):
        for text, status in (("", "empty"), ("The city is Ottawa.", "no_label"),
                             ("Entity or Location?", "ambiguous")):
            with self.subTest(text=text):
                result = score_classification_output(text, TREC_LABELS, TREC_LABELS[0])
                self.assertIsNone(result["prediction"])
                self.assertFalse(result["is_correct"])
                self.assertEqual(result["parse_status"], status)

    def test_rescores_previous_export_from_raw_output(self):
        old = {"task": "classification", "status": "ok", "dataset": "trec", "reference": "Numeric",
               "raw_output": "Type: Numeric", "prediction": None, "strict_prediction": None,
               "is_correct": False, "unique_mention_correct": False}
        result = rescore_classification_trial(old)
        self.assertTrue(result["is_correct"])
        self.assertNotIn("strict_prediction", result)
        self.assertEqual(rescore_classification_trial(result), result)
        self.assertFalse(old["is_correct"])

    def test_preserves_likelihood_predictions_and_other_tasks(self):
        likelihood = {"task": "classification", "status": "ok", "raw_output": None,
                      "prediction": "negative", "is_correct": False,
                      "label_log_likelihood": {"positive": -2.0, "negative": -1.0},
                      "calibrated_correct": True}
        rescore = rescore_classification_trial(likelihood)
        self.assertEqual(rescore, {**likelihood, "score_method": "label_log_likelihood"})
        sql = {"task": "sql", "status": "ok", "raw_output": "SELECT 1", "is_correct": True}
        self.assertEqual(rescore_classification_trial(sql), sql)
