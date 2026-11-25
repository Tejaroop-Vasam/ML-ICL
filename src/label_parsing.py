"""Classification label vocabularies and response scoring."""

import re
from functools import lru_cache


SENTIMENT_LABELS = ("negative", "positive")
TREC_LABELS = ("Description", "Entity", "Abbreviation", "Human", "Numeric", "Location")
YAHOO_TOPICS = (
    "Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
    "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
    "Family & Relationships", "Politics & Government",
)
CLASSIFICATION_LABELS = {
    "imdb": SENTIMENT_LABELS, "yelp_polarity": SENTIMENT_LABELS,
    "trec": TREC_LABELS, "yahoo_answers": YAHOO_TOPICS,
}
CLASSIFICATION_SCORING = "unique_label_mention"


@lru_cache(maxsize=None)
def label_patterns(labels):
    return tuple((label, re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)) for label in labels)


def score_classification_output(output, labels, reference):
    """Accept one distinct candidate label anywhere in a response, without a fallback."""
    text = output.strip()
    mentions = [label for label, pattern in label_patterns(tuple(labels)) if pattern.search(text)]
    prediction = mentions[0] if len(mentions) == 1 else None
    if prediction is not None:
        status = "unique_mention"
    elif mentions:
        status = "ambiguous"
    else:
        status = "no_label" if text else "empty"
    return {
        "prediction": prediction, "is_correct": prediction is not None and prediction == reference,
        "mentions": mentions, "parse_status": status, "score_method": CLASSIFICATION_SCORING,
    }


