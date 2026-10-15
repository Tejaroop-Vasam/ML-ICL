#!/usr/bin/env python3
"""Unified experiment runner for ICL research benchmarks."""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_dataset, list_datasets
from src.selection import (
    RandomSelector,
    SimilaritySelector,
    BudgetAwareSelector,
)
from src.models import get_model
from src.evaluation import ICLEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("icl_experiments")


def save_results(results: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved experiment results to {output_path}")


# -------------------------------------------------------------
# Experiment 1: Establish Baselines (Zero-Shot vs Random vs Sim)
# -------------------------------------------------------------
def run_exp1_baselines(args):
    """Compare Zero-Shot, Random k-shot, and Similarity-based k-shot."""
    logger.info(">>> RUNNING EXPERIMENT 1: BASELINES (Zero-shot vs Random vs Similarity)")
    loader = get_dataset(args.dataset)
    llm = get_model(args.model)
    evaluator = ICLEvaluator(loader, llm)

    candidate_pool = loader.get_candidate_demonstration_pool(pool_size=args.pool_size, seed=42)
    random_selector = RandomSelector(candidate_pool=candidate_pool, seed=42)
    similarity_selector = SimilaritySelector(candidate_pool=candidate_pool, seed=42)

    configs = [
        ("Zero-Shot", None, 0),
        ("Random Few-Shot", random_selector, args.k),
        ("Similarity Few-Shot", similarity_selector, args.k),
    ]

    all_results = {}
    for name, selector, k_val in configs:
        seed_accuracies = []
        seed_f1s = []
        token_counts = []
        res_list = []

        for seed in args.seeds:
            res = evaluator.run_evaluation(
                selector=selector,
                k=k_val,
                n_eval=args.n_eval,
                seed=seed,
                format_type=args.format,
                ordering=args.ordering,
            )
            seed_accuracies.append(res.accuracy)
            seed_f1s.append(res.macro_f1)
            token_counts.append(res.mean_prompt_tokens)
            res_list.append(res.to_dict())

        all_results[name] = {
            "accuracy_mean": float(np.mean(seed_accuracies)),
            "accuracy_std": float(np.std(seed_accuracies)),
            "macro_f1_mean": float(np.mean(seed_f1s)),
            "macro_f1_std": float(np.std(seed_f1s)),
            "mean_prompt_tokens": float(np.mean(token_counts)),
            "runs": res_list,
        }

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 1 RESULTS: {args.dataset.upper()} (k={args.k}, n={args.n_eval}, seeds={len(args.seeds)})")
    print("=" * 80)
    print(f"{'Method':<25} | {'Accuracy (%)':<18} | {'Macro-F1':<16} | {'Avg Tokens':<10}")
    print("-" * 75)
    for method, stats in all_results.items():
        acc_str = f"{stats['accuracy_mean']*100:.2f} ± {stats['accuracy_std']*100:.2f}"
        f1_str = f"{stats['macro_f1_mean']:.4f} ± {stats['macro_f1_std']:.4f}"
        print(f"{method:<25} | {acc_str:<18} | {f1_str:<16} | {stats['mean_prompt_tokens']:<10.1f}")
    print("=" * 80 + "\n")

    out_file = Path(args.output_dir) / f"exp1_{args.dataset}_{args.model.replace('/', '_')}.json"
    save_results({"experiment": "exp1_baselines", "dataset": args.dataset, "results": all_results}, out_file)


# -------------------------------------------------------------
# Experiment 2: Demonstration Scaling (k-shot sweep)
# -------------------------------------------------------------
def run_exp2_k_sweep(args):
    """Evaluate performance as k increases: k in {0, 2, 4, 8, 16}."""
    logger.info(">>> RUNNING EXPERIMENT 2: DEMONSTRATION COUNT SWEEP")
    loader = get_dataset(args.dataset)
    llm = get_model(args.model)
    evaluator = ICLEvaluator(loader, llm)

    candidate_pool = loader.get_candidate_demonstration_pool(pool_size=args.pool_size, seed=42)
    random_selector = RandomSelector(candidate_pool=candidate_pool, seed=42)
    similarity_selector = SimilaritySelector(candidate_pool=candidate_pool, seed=42)

    k_values = [0, 2, 4, 8, 16] if not args.custom_k else [int(x) for x in args.custom_k.split(",")]
    methods = [("Random", random_selector), ("Similarity", similarity_selector)]

    results = {name: {} for name, _ in methods}
    for name, selector in methods:
        for k in k_values:
            curr_sel = None if k == 0 else selector
            res = evaluator.run_evaluation(
                selector=curr_sel,
                k=k,
                n_eval=args.n_eval,
                seed=args.seeds[0],
                format_type=args.format,
            )
            results[name][k] = {
                "accuracy": res.accuracy,
                "macro_f1": res.macro_f1,
                "mean_prompt_tokens": res.mean_prompt_tokens,
            }

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2 RESULTS: {args.dataset.upper()} - K SWEEP")
    print("=" * 80)
    print(f"{'k':<5} | {'Random Acc (%)':<16} | {'Sim Acc (%)':<16} | {'Random Tokens':<14} | {'Sim Tokens':<12}")
    print("-" * 75)
    for k in k_values:
        r_acc = results["Random"][k]["accuracy"] * 100
        s_acc = results["Similarity"][k]["accuracy"] * 100
        r_tok = results["Random"][k]["mean_prompt_tokens"]
        s_tok = results["Similarity"][k]["mean_prompt_tokens"]
        print(f"{k:<5d} | {r_acc:<16.2f} | {s_acc:<16.2f} | {r_tok:<14.1f} | {s_tok:<12.1f}")
    print("=" * 80 + "\n")

    out_file = Path(args.output_dir) / f"exp2_{args.dataset}_{args.model.replace('/', '_')}.json"
    save_results({"experiment": "exp2_k_sweep", "dataset": args.dataset, "results": results}, out_file)


# -------------------------------------------------------------
# Experiment 4: Demonstration Ordering Study
# -------------------------------------------------------------
def run_exp4_ordering(args):
    """Study the effect of demonstration order holding examples identical."""
    logger.info(">>> RUNNING EXPERIMENT 4: DEMONSTRATION ORDERING")
    loader = get_dataset(args.dataset)
    llm = get_model(args.model)
    evaluator = ICLEvaluator(loader, llm)

    candidate_pool = loader.get_candidate_demonstration_pool(pool_size=args.pool_size, seed=42)
    similarity_selector = SimilaritySelector(candidate_pool=candidate_pool, seed=42)

    orderings = ["similar_last", "similar_first", "random", "fixed_reference"]
    order_results = {}

    for ord_policy in orderings:
        res = evaluator.run_evaluation(
            selector=similarity_selector,
            k=args.k,
            n_eval=args.n_eval,
            ordering=ord_policy,
            seed=42,
            format_type=args.format,
        )
        order_results[ord_policy] = {
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "mean_tokens": res.mean_prompt_tokens,
        }

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 4 RESULTS: {args.dataset.upper()} - ORDERING STUDY (k={args.k})")
    print("=" * 80)
    print(f"{'Ordering Policy':<20} | {'Accuracy (%)':<16} | {'Macro-F1':<16} | {'Avg Tokens':<10}")
    print("-" * 75)
    for pol, stats in order_results.items():
        print(f"{pol:<20} | {stats['accuracy']*100:<16.2f} | {stats['macro_f1']:<16.4f} | {stats['mean_tokens']:<10.1f}")
    print("=" * 80 + "\n")

    out_file = Path(args.output_dir) / f"exp4_{args.dataset}_{args.model.replace('/', '_')}.json"
    save_results({"experiment": "exp4_ordering", "dataset": args.dataset, "results": order_results}, out_file)


# -------------------------------------------------------------
# Experiment 5: Demonstration Formatting Study
# -------------------------------------------------------------
def run_exp5_formatting(args):
    """Study formatting styles: Format A (plain) vs Format B (structured) vs Format C (rationale)."""
    logger.info(">>> RUNNING EXPERIMENT 5: FORMATTING STUDY")
    loader = get_dataset(args.dataset)
    llm = get_model(args.model)
    evaluator = ICLEvaluator(loader, llm)

    candidate_pool = loader.get_candidate_demonstration_pool(pool_size=args.pool_size, seed=42)
    similarity_selector = SimilaritySelector(candidate_pool=candidate_pool, seed=42)

    formats = ["plain", "structured", "rationale"]
    fmt_results = {}

    for fmt in formats:
        res = evaluator.run_evaluation(
            selector=similarity_selector,
            k=args.k,
            n_eval=args.n_eval,
            format_type=fmt,
            seed=42,
        )
        fmt_results[fmt] = {
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "mean_tokens": res.mean_prompt_tokens,
        }

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 5 RESULTS: {args.dataset.upper()} - FORMATTING STUDY (k={args.k})")
    print("=" * 80)
    print(f"{'Format Style':<20} | {'Accuracy (%)':<16} | {'Macro-F1':<16} | {'Avg Tokens':<10}")
    print("-" * 75)
    for fmt, stats in fmt_results.items():
        print(f"{fmt:<20} | {stats['accuracy']*100:<16.2f} | {stats['macro_f1']:<16.4f} | {stats['mean_tokens']:<10.1f}")
    print("=" * 80 + "\n")

    out_file = Path(args.output_dir) / f"exp5_{args.dataset}_{args.model.replace('/', '_')}.json"
    save_results({"experiment": "exp5_formatting", "dataset": args.dataset, "results": fmt_results}, out_file)


# -------------------------------------------------------------
# Novel Method: Budget-Aware Selection
# -------------------------------------------------------------
def run_proposed_method(args):
    """Evaluate Budget-Aware Selection against Random and Similarity under budget limits."""
    logger.info(">>> RUNNING PROPOSED BUDGET-AWARE SELECTION EVALUATION")
    loader = get_dataset(args.dataset)
    llm = get_model(args.model)
    evaluator = ICLEvaluator(loader, llm)

    candidate_pool = loader.get_candidate_demonstration_pool(pool_size=args.pool_size, seed=42)
    random_selector = RandomSelector(candidate_pool=candidate_pool, seed=42)
    sim_selector = SimilaritySelector(candidate_pool=candidate_pool, seed=42)
    budget_selector = BudgetAwareSelector(
        candidate_pool=candidate_pool,
        alpha=1.0,
        beta=0.5,
        gamma=0.2,
        token_budget=args.token_budget,
        seed=42,
    )

    strategies = [
        ("Random Selection", random_selector),
        ("Similarity (k-NN)", sim_selector),
        ("Proposed (Budget-Aware)", budget_selector),
    ]

    comp_results = {}
    for name, selector in strategies:
        res = evaluator.run_evaluation(
            selector=selector,
            k=args.k,
            n_eval=args.n_eval,
            token_budget=args.token_budget,
            seed=42,
            format_type=args.format,
        )
        comp_results[name] = {
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "mean_tokens": res.mean_prompt_tokens,
            "max_tokens": res.max_prompt_tokens,
        }

    print("\n" + "=" * 80)
    print(f"PROPOSED METHOD BENCHMARK: {args.dataset.upper()} (Budget={args.token_budget or 'None'}, k={args.k})")
    print("=" * 80)
    print(f"{'Method':<25} | {'Accuracy (%)':<16} | {'Macro-F1':<14} | {'Avg Tokens':<12} | {'Max Tokens':<10}")
    print("-" * 85)
    for name, stats in comp_results.items():
        print(f"{name:<25} | {stats['accuracy']*100:<16.2f} | {stats['macro_f1']:<14.4f} | {stats['mean_tokens']:<12.1f} | {stats['max_tokens']:<10d}")
    print("=" * 85 + "\n")

    out_file = Path(args.output_dir) / f"proposed_{args.dataset}_{args.model.replace('/', '_')}.json"
    save_results({"experiment": "proposed_method", "dataset": args.dataset, "results": comp_results}, out_file)


def main():
    parser = argparse.ArgumentParser(description="Run ICL experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp1_baselines",
        choices=["exp1_baselines", "exp2_k_sweep", "exp4_ordering", "exp5_formatting", "proposed_method", "all"],
        help="Which experiment group to execute",
    )
    parser.add_argument("--dataset", type=str, default="sst2", choices=list_datasets())
    parser.add_argument("--model", type=str, default="mock", help="Model name or 'mock'")
    parser.add_argument("--k", type=int, default=4, help="Number of demonstrations")
    parser.add_argument("--n_eval", type=int, default=100, help="Number of evaluation test samples")
    parser.add_argument("--pool_size", type=int, default=500, help="Candidate pool size from train split")
    parser.add_argument("--token_budget", type=int, default=None, help="Demonstration token budget constraint")
    parser.add_argument("--format", type=str, default="plain", choices=["plain", "structured", "rationale"])
    parser.add_argument("--ordering", type=str, default="similar_last", choices=["similar_last", "similar_first", "random", "fixed_reference"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds for multi-run variance")
    parser.add_argument("--custom_k", type=str, default=None, help="Comma-separated list of k values for k_sweep")
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    dispatch = {
        "exp1_baselines": run_exp1_baselines,
        "exp2_k_sweep": run_exp2_k_sweep,
        "exp4_ordering": run_exp4_ordering,
        "exp5_formatting": run_exp5_formatting,
        "proposed_method": run_proposed_method,
    }

    if args.experiment == "all":
        for exp_name, func in dispatch.items():
            func(args)
    else:
        dispatch[args.experiment](args)


if __name__ == "__main__":
    main()
