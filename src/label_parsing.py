"""Label matching shared by the original evaluator and the review diagnostics."""

import re
from collections.abc import Sequence


def extract_legacy_label(output: str, labels: Sequence[str]) -> tuple[str, str]:
    """Return the original label prediction and the match route it used."""
    text = output.strip()
    first_line = text.split("\n", 1)[0].strip()
    patterns = {label: rf"\b{re.escape(label)}\b" for label in labels}
    regions = (
        (first_line, "^", "leading"),
        (first_line, "", "first_line"),
        (text, "", "later_line"),
    )
    for region, prefix, route in regions:
        for label in labels:
            if re.search(prefix + patterns[label], region, re.IGNORECASE):
                return label, route
    return labels[0], "defaulted_to_first_label"
