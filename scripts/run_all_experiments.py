#!/usr/bin/env python3
"""
Run ALL models × ALL prompting techniques and auto-generate comparison plots.

Usage:
    # Run all experiments on all locally available models (small scale for testing):
    .venv/bin/python scripts/run_all_experiments.py

    # Run only on specific models:
    .venv/bin/python scripts/run_all_experiments.py --models 0.6b 1.7b

    # Run on all datasets:
    .venv/bin/python scripts/run_all_experiments.py --datasets yelp_polarity yahoo_answers trec imdb

    # Larger evaluation:
    .venv/bin/python scripts/run_all_experiments.py --n_eval 100 --k 4
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_dataset
from src.models import get_model
from src.evaluation import ICLEvaluator
from src.selection import RandomSelector, SimilaritySelector, BudgetAwareSelector

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")


# ─── Configuration ────────────────────────────────────────────────────────────

# Models that are downloaded locally (skip those that exceed your RAM)
AVAILABLE_MODELS = ["0.6b"]

# Prompting dimensions
DATASETS      = ["yelp_polarity", "yahoo_answers", "trec"]
FORMATS       = ["plain", "structured", "rationale"]
ORDERINGS     = ["similar_last", "similar_first", "random"]
K_VALUES      = [0, 2, 4, 8]
SEEDS         = [42, 101, 2024]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved → {path}")


# ─── Experiment Runners ────────────────────────────────────────────────────────

def run_baselines(model_name, dataset, n_eval, k, seeds, out_dir):
    """Zero-Shot vs Random vs Similarity across seeds."""
    logger.info(f"\n{'='*60}")
    logger.info(f"[BASELINES] model={model_name} dataset={dataset} k={k}")
    logger.info(f"{'='*60}")

    loader = get_dataset(dataset)
    llm    = get_model(model_name)
    ev     = ICLEvaluator(loader, llm)
    pool   = loader.get_candidate_demonstration_pool(pool_size=200, seed=42)

    configs = [
        ("Zero-Shot",       None,                           0),
        ("Random",          RandomSelector(pool, seed=42),  k),
        ("Similarity-kNN",  SimilaritySelector(pool, seed=42), k),
    ]

    results = {}
    for name, selector, k_val in configs:
        accs, f1s = [], []
        for seed in seeds:
            r = ev.run_evaluation(selector=selector, k=k_val, n_eval=n_eval, seed=seed, show_progress=False)
            accs.append(r.accuracy)
            f1s.append(r.macro_f1)
        results[name] = {
            "accuracy_mean": float(sum(accs) / len(accs)),
            "accuracy_std":  float((sum((x - sum(accs)/len(accs))**2 for x in accs) / len(accs))**0.5),
            "macro_f1_mean": float(sum(f1s) / len(f1s)),
            "macro_f1_std":  float((sum((x - sum(f1s)/len(f1s))**2 for x in f1s) / len(f1s))**0.5),
        }

    save_json({"experiment": "exp1_baselines", "dataset": dataset, "model": model_name, "results": results},
              out_dir / f"baselines_{dataset}_{model_name}.json")
    return results


def run_format_study(model_name, dataset, n_eval, k, out_dir):
    """Compare plain vs structured vs rationale prompt formats."""
    logger.info(f"\n[FORMATS] model={model_name} dataset={dataset} k={k}")

    loader = get_dataset(dataset)
    llm    = get_model(model_name)
    ev     = ICLEvaluator(loader, llm)
    pool   = loader.get_candidate_demonstration_pool(pool_size=200, seed=42)
    sel    = SimilaritySelector(pool, seed=42)

    results = {}
    for fmt in FORMATS:
        r = ev.run_evaluation(selector=sel, k=k, n_eval=n_eval, format_type=fmt, seed=42, show_progress=False)
        results[fmt] = {"accuracy": r.accuracy, "macro_f1": r.macro_f1, "mean_tokens": r.mean_prompt_tokens}

    save_json({"experiment": "exp5_formatting", "dataset": dataset, "model": model_name, "results": results},
              out_dir / f"formats_{dataset}_{model_name}.json")
    return results


def run_ordering_study(model_name, dataset, n_eval, k, out_dir):
    """Compare similar_last vs similar_first vs random ordering."""
    logger.info(f"\n[ORDERING] model={model_name} dataset={dataset} k={k}")

    loader = get_dataset(dataset)
    llm    = get_model(model_name)
    ev     = ICLEvaluator(loader, llm)
    pool   = loader.get_candidate_demonstration_pool(pool_size=200, seed=42)
    sel    = SimilaritySelector(pool, seed=42)

    results = {}
    for ordering in ORDERINGS:
        r = ev.run_evaluation(selector=sel, k=k, n_eval=n_eval, ordering=ordering, seed=42, show_progress=False)
        results[ordering] = {"accuracy": r.accuracy, "macro_f1": r.macro_f1, "mean_tokens": r.mean_prompt_tokens}

    save_json({"experiment": "exp4_ordering", "dataset": dataset, "model": model_name, "results": results},
              out_dir / f"ordering_{dataset}_{model_name}.json")
    return results


def run_k_sweep(model_name, dataset, n_eval, out_dir):
    """Sweep k ∈ {0, 2, 4, 8} to study demonstration scaling."""
    logger.info(f"\n[K-SWEEP] model={model_name} dataset={dataset}")

    loader = get_dataset(dataset)
    llm    = get_model(model_name)
    ev     = ICLEvaluator(loader, llm)
    pool   = loader.get_candidate_demonstration_pool(pool_size=200, seed=42)

    results = {"Random": {}, "Similarity-kNN": {}}
    for k in K_VALUES:
        for method_name, sel_cls in [("Random", RandomSelector), ("Similarity-kNN", SimilaritySelector)]:
            sel = None if k == 0 else sel_cls(pool, seed=42)
            r = ev.run_evaluation(selector=sel, k=k, n_eval=n_eval, seed=42, show_progress=False)
            results[method_name][str(k)] = {"accuracy": r.accuracy, "macro_f1": r.macro_f1,
                                             "mean_prompt_tokens": r.mean_prompt_tokens}

    save_json({"experiment": "exp2_k_sweep", "dataset": dataset, "model": model_name, "results": results},
              out_dir / f"k_sweep_{dataset}_{model_name}.json")
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def generate_plots(results_dir: Path, figures_dir: Path):
    """Generate all plots from saved JSON result files."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})
    figures_dir.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        exp_type = data.get("experiment", "")
        results  = data.get("results", {})
        dataset  = data.get("dataset", "").upper()
        model    = data.get("model", "")
        stem     = json_file.stem

        # ── Baselines bar chart ──────────────────────────────────────────────
        if exp_type == "exp1_baselines":
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            fig.suptitle(f"Baselines: {dataset}  [{model}]", fontweight="bold")
            methods = list(results.keys())
            colors  = ["#4C72B0", "#55A868", "#C44E52"]

            for ax, metric, label in zip(axes, ["accuracy_mean", "macro_f1_mean"], ["Accuracy (%)", "Macro-F1"]):
                means = [results[m][metric] * (100 if "accuracy" in metric else 1) for m in methods]
                stds  = [results[m][metric.replace("mean", "std")] * (100 if "accuracy" in metric else 1) for m in methods]
                bars  = ax.bar(methods, means, yerr=stds, capsize=5, color=colors[:len(methods)], alpha=0.85, edgecolor="black")
                for bar, v, s in zip(bars, means, stds):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (s or 0) + 1,
                            f"{v:.1f}" + ("%" if "acc" in metric else ""), ha="center", fontsize=9, fontweight="bold")
                ax.set_ylabel(label, fontweight="bold")
                ax.set_ylim(0, max(means) * 1.35)
            plt.tight_layout()
            out = figures_dir / f"{stem}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plot → {out}")

        # ── Format comparison bar chart ──────────────────────────────────────
        elif exp_type == "exp5_formatting":
            fig, ax = plt.subplots(figsize=(7, 4.5))
            fmts   = list(results.keys())
            accs   = [results[f]["accuracy"] * 100 for f in fmts]
            bars   = ax.bar(fmts, accs, color=["#4C72B0", "#55A868", "#C44E52"][:len(fmts)], alpha=0.85, edgecolor="black")
            for bar, v in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{v:.1f}%", ha="center", fontweight="bold")
            ax.set_ylabel("Accuracy (%)", fontweight="bold")
            ax.set_title(f"Prompt Format Study: {dataset}  [{model}]", fontweight="bold")
            ax.set_ylim(0, max(accs) * 1.3)
            plt.tight_layout()
            out = figures_dir / f"{stem}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plot → {out}")

        # ── Ordering comparison bar chart ────────────────────────────────────
        elif exp_type == "exp4_ordering":
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ords  = list(results.keys())
            accs  = [results[o]["accuracy"] * 100 for o in ords]
            bars  = ax.bar(ords, accs, color=sns.color_palette("muted", len(ords)), alpha=0.85, edgecolor="black")
            for bar, v in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{v:.1f}%", ha="center", fontweight="bold")
            ax.set_ylabel("Accuracy (%)", fontweight="bold")
            ax.set_title(f"Ordering Study: {dataset}  [{model}]", fontweight="bold")
            ax.set_ylim(0, max(accs) * 1.3)
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            out = figures_dir / f"{stem}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plot → {out}")

        # ── k-sweep line chart ───────────────────────────────────────────────
        elif exp_type == "exp2_k_sweep":
            fig, ax = plt.subplots(figsize=(8, 4.5))
            markers = {"Random": "o", "Similarity-kNN": "s"}
            colors  = {"Random": "#4C72B0", "Similarity-kNN": "#C44E52"}
            for method, k_dict in results.items():
                k_vals = sorted([int(k) for k in k_dict])
                accs   = [k_dict[str(k)]["accuracy"] * 100 for k in k_vals]
                ax.plot(k_vals, accs, marker=markers.get(method, "o"), label=method,
                        linewidth=2.2, markersize=7, color=colors.get(method))
            ax.set_xlabel("Number of Demonstrations (k)", fontweight="bold")
            ax.set_ylabel("Accuracy (%)", fontweight="bold")
            ax.set_title(f"k-Shot Scaling: {dataset}  [{model}]", fontweight="bold")
            ax.set_xticks(K_VALUES)
            ax.legend(frameon=True)
            plt.tight_layout()
            out = figures_dir / f"{stem}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plot → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all models × all prompting techniques and plot results.")
    parser.add_argument("--models",   nargs="+", default=AVAILABLE_MODELS,
                        help="Model aliases to evaluate (default: 0.6b 1.7b)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Datasets to run on (default: yelp_polarity yahoo_answers trec)")
    parser.add_argument("--n_eval",   type=int, default=30,
                        help="Test samples per evaluation (default: 30 for speed)")
    parser.add_argument("--k",        type=int, default=4,
                        help="Number of few-shot demonstrations (default: 4)")
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42],
                        help="Random seeds for baseline variance (default: 42)")
    parser.add_argument("--out_dir",  type=str, default="results",
                        help="Directory to save JSON results and figures")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip experiments, only generate plots from existing JSONs")
    args = parser.parse_args()

    out_dir     = Path(args.out_dir)
    figures_dir = out_dir / "figures"

    if not args.plot_only:
        for model_name in args.models:
            logger.info(f"\n{'#'*70}")
            logger.info(f"# MODEL: {model_name}")
            logger.info(f"{'#'*70}")
            for dataset in args.datasets:
                run_baselines(model_name, dataset, args.n_eval, args.k, args.seeds, out_dir)
                run_format_study(model_name, dataset, args.n_eval, args.k, out_dir)
                run_ordering_study(model_name, dataset, args.n_eval, args.k, out_dir)
                run_k_sweep(model_name, dataset, args.n_eval, out_dir)

    logger.info("\nGenerating plots...")
    generate_plots(out_dir, figures_dir)
    logger.info(f"\nDone! All figures saved in: {figures_dir}/")


if __name__ == "__main__":
    main()
