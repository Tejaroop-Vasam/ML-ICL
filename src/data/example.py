"""Data structures and representations for In-Context Learning (ICL) examples."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class ICLExample:
    """Standardized representation for a single ICL example across any dataset."""
    id: str
    dataset: str
    text: str
    label: int
    label_name: str
    split: str
    num_words: int = 0
    num_tokens: Optional[int] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.num_words and self.text:
            self.num_words = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ICLExample":
        return cls(**data)

    def format_example(
        self,
        format_type: str = "plain",
        include_label: bool = True,
        input_prefix: str = "Input: ",
        label_prefix: str = "Label: ",
    ) -> str:
        """Format example into a prompt string.

        Args:
            format_type: 'plain' (Format A), 'structured' (Format B), or 'rationale' (Format C)
            include_label: If True, appends the ground truth label (used for demonstrations).
                          If False, leaves label blank with prefix (used for test query).
            input_prefix: Custom prefix for input text.
            label_prefix: Custom prefix for label.
        """
        clean_text = self.text.strip()
        label_str = self.label_name if include_label else ""

        if format_type == "plain":
            # Format A: Plain input and label
            res = f"{input_prefix}{clean_text}\n{label_prefix}{label_str}"
            return res.strip() if not include_label else res

        elif format_type == "structured":
            # Format B: Explicit task and structured fields
            lines = [
                f"[Input] {clean_text}",
                f"[Label] {label_str}" if include_label else "[Label] ",
            ]
            return "\n".join(lines).strip() if not include_label else "\n".join(lines)

        elif format_type == "rationale":
            # Format C: Input, rationale, and label
            rat = self.rationale or self.metadata.get("rationale", "")
            lines = [f"[Input] {clean_text}"]
            if rat:
                lines.append(f"[Reason] {rat}")
            lines.append(f"[Label] {label_str}" if include_label else "[Label] ")
            return "\n".join(lines).strip() if not include_label else "\n".join(lines)

        else:
            raise ValueError(f"Unknown format_type '{format_type}'. Expected 'plain', 'structured', or 'rationale'.")
