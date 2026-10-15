"""Standardized ICL experiment evaluation engine."""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import numpy as np

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample
from src.selection.base_selector import BaseSelector
from src.models.base_model import BaseLLM
from src.evaluation.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Encapsulates the complete result of an evaluation run."""
    dataset: str
    method: str
    k: int
    num_samples: int
    accuracy: float
    macro_f1: float
    mean_prompt_tokens: float
    median_prompt_tokens: float
    max_prompt_tokens: int
    duration_seconds: float
    metrics: Dict[str, Any]
    predictions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ICLEvaluator:
    """Evaluates zero-shot or few-shot ICL performance on a dataset."""

    def __init__(self, data_loader: BaseDataLoader, llm: BaseLLM):
        self.loader = data_loader
        self.llm = llm

    def run_evaluation(
        self,
        selector: Optional[BaseSelector] = None,
        k: int = 0,
        n_eval: Optional[int] = 100,
        format_type: str = "plain",
        ordering: str = "similar_last",
        token_budget: Optional[int] = None,
        seed: int = 42,
        split: str = "test",
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Execute evaluation over the specified test split.

        Args:
            selector: Demonstration selector (None for zero-shot).
            k: Number of demonstrations. If k=0, runs zero-shot.
            n_eval: Number of evaluation test samples to evaluate.
            format_type: 'plain' (Format A), 'structured' (Format B), 'rationale' (Format C).
            ordering: Demonstration ordering policy.
            token_budget: Optional maximum token budget for demonstrations.
            seed: Random seed for sampling evaluation queries and stochastic selection.
            split: 'test' or 'validation'.
            show_progress: Whether to log progress.
        """
        method_name = "zero_shot" if (selector is None or k == 0) else type(selector).__name__
        test_samples = self.loader.get_stratified_subset(split=split, n_samples=n_eval, seed=seed)

        start_time = time.time()
        y_true: List[int] = []
        y_pred: List[int] = []
        token_counts: List[int] = []
        record_list: List[Dict[str, Any]] = []

        total = len(test_samples)
        logger.info(
            f"Starting {self.loader.dataset_name} eval: method={method_name}, k={k}, "
            f"n_eval={total}, format={format_type}, ordering={ordering}"
        )

        for i, sample in enumerate(test_samples):
            # Select demonstrations
            if selector is not None and k > 0:
                demos = selector.select_demonstrations(
                    test_example=sample,
                    k=k,
                    ordering=ordering,
                    token_budget=token_budget,
                    seed=seed + i,
                )
            else:
                demos = []

            # Format prompt
            prompt = self.loader.format_prompt(
                test_example=sample,
                demonstrations=demos,
                format_type=format_type,
            )

            # Count prompt tokens
            num_tokens = self.llm.count_tokens(prompt)
            token_counts.append(num_tokens)

            # Generate and extract prediction
            output_text = self.llm.generate(prompt)
            pred_class = self.llm.extract_label(output_text, self.loader.classes)
            pred_id = self.loader.label2id.get(pred_class, 0)

            y_true.append(sample.label)
            y_pred.append(pred_id)

            record_list.append({
                "sample_id": sample.id,
                "gold_label": sample.label_name,
                "gold_id": sample.label,
                "pred_label": pred_class,
                "pred_id": pred_id,
                "is_correct": bool(sample.label == pred_id),
                "num_demos": len(demos),
                "prompt_tokens": num_tokens,
                "raw_output": output_text,
            })

            if show_progress and (i + 1) % max(1, total // 5) == 0:
                cur_acc = np.mean([r["is_correct"] for r in record_list])
                logger.info(f"Progress [{i+1}/{total}]: running acc = {cur_acc * 100:.1f}%")

        duration = time.time() - start_time
        metrics = compute_classification_metrics(y_true, y_pred, self.loader.classes)

        token_arr = np.array(token_counts)
        mean_tokens = float(np.mean(token_arr))
        median_tokens = float(np.median(token_arr))
        max_tokens = int(np.max(token_arr))

        result = EvaluationResult(
            dataset=self.loader.dataset_name,
            method=method_name,
            k=k,
            num_samples=total,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            mean_prompt_tokens=mean_tokens,
            median_prompt_tokens=median_tokens,
            max_prompt_tokens=max_tokens,
            duration_seconds=duration,
            metrics=metrics,
            predictions=record_list,
        )

        logger.info(
            f"Completed {self.loader.dataset_name} ({method_name}, k={k}): "
            f"Accuracy = {result.accuracy * 100:.2f}%, Macro-F1 = {result.macro_f1:.4f}, "
            f"Avg Tokens = {mean_tokens:.1f} ({duration:.2f}s)"
        )

        return result
