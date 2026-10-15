"""Publication-ready plotting utilities for ICL experiment results."""

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set consistent, clean academic aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def _load_data(data_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to accept either a JSON file path or a Python dictionary."""
    if isinstance(data_source, (str, Path)):
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif isinstance(data_source, dict):
        return data_source
    else:
        raise TypeError(f"Expected file path or dict, got {type(data_source)}")


def plot_baselines(
    data_source: Union[str, Path, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    metric: str = "accuracy",  # 'accuracy' or 'macro_f1'
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot bar chart comparing Zero-Shot, Random Few-Shot, and Similarity Few-Shot with seed variance.

    Args:
        data_source: JSON path or dictionary from exp1_baselines.
        save_path: Optional path to save the generated figure.
        metric: 'accuracy' or 'macro_f1'.
        title: Custom plot title.
    """
    data = _load_data(data_source)
    results = data.get("results", data)
    dataset_name = data.get("dataset", "Benchmark").upper()

    methods = list(results.keys())
    metric_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    means = [results[m].get(metric_key, 0.0) * (100.0 if metric == "accuracy" else 1.0) for m in methods]
    stds = [results[m].get(std_key, 0.0) * (100.0 if metric == "accuracy" else 1.0) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bar_colors = colors[:len(methods)]

    bars = ax.bar(methods, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.88, edgecolor="black", linewidth=1.2)

    # Annotate values on top of bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        y_pos = bar.get_height() + (std_val if std_val else 1.0) + 1.0
        label_fmt = f"{mean_val:.1f}%" if metric == "accuracy" else f"{mean_val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label_fmt, ha="center", va="bottom", fontweight="bold")

    y_label = "Accuracy (%)" if metric == "accuracy" else "Macro-F1 Score"
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title or f"Baseline Performance Comparison: {dataset_name}", fontweight="bold", pad=12)
    ax.set_ylim(0, max(means + [80]) + 15)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved baseline plot to {save_path}")

    return fig


def plot_k_scaling(
    data_source: Union[str, Path, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    metric: str = "accuracy",  # 'accuracy' or 'macro_f1'
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot performance vs. number of demonstrations (k in {0, 2, 4, 8, 16}).

    Args:
        data_source: JSON path or dictionary from exp2_k_sweep.
        save_path: Optional path to save the generated figure.
    """
    data = _load_data(data_source)
    results = data.get("results", data)
    dataset_name = data.get("dataset", "Benchmark").upper()

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"Random": "o", "Similarity": "s", "Budget-Aware": "^"}
    colors = {"Random": "#4C72B0", "Similarity": "#C44E52", "Budget-Aware": "#55A868"}

    for method_name, k_dict in results.items():
        k_vals = sorted([int(k) for k in k_dict.keys()])
        vals = [k_dict[str(k)][metric] * (100.0 if metric == "accuracy" else 1.0) for k in k_vals]

        marker = markers.get(method_name, "o")
        color = colors.get(method_name, None)
        ax.plot(
            k_vals,
            vals,
            marker=marker,
            label=f"{method_name} Selection",
            linewidth=2.2,
            markersize=7,
            color=color,
        )

    ax.set_xlabel("Number of Demonstrations (k)", fontweight="bold")
    y_label = "Accuracy (%)" if metric == "accuracy" else "Macro-F1 Score"
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title or f"Demonstration Scaling Curve: {dataset_name}", fontweight="bold", pad=12)
    ax.set_xticks(k_vals)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved k-scaling plot to {save_path}")

    return fig


def plot_tokens_vs_accuracy(
    data_source: Union[str, Path, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot Total Prompt Tokens vs. Classification Accuracy.

    (Directly addresses Reviewer 2: evaluates demonstration token cost efficiency).
    """
    data = _load_data(data_source)
    results = data.get("results", data)
    dataset_name = data.get("dataset", "Benchmark").upper()

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, stats in results.items():
        # Handle either nested dicts or summary stats
        if "accuracy" in stats and "mean_tokens" in stats:
            tok = stats["mean_tokens"]
            acc = stats["accuracy"] * 100.0
            ax.scatter(tok, acc, s=140, label=method_name, edgecolors="black", linewidths=1.2)
        elif "runs" in stats:
            toks = [r["mean_prompt_tokens"] for r in stats["runs"]]
            accs = [r["accuracy"] * 100.0 for r in stats["runs"]]
            ax.scatter(toks, accs, s=120, label=method_name, alpha=0.85, edgecolors="black")

    ax.set_xlabel("Average Prompt Tokens", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(title or f"Prompt Token Budget vs. Accuracy: {dataset_name}", fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved tokens vs accuracy plot to {save_path}")

    return fig


def plot_proposed_vs_baselines(
    data_source: Union[str, Path, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar plot comparing Accuracy and Token Cost for Proposed vs Baselines side-by-side."""
    data = _load_data(data_source)
    results = data.get("results", data)
    dataset_name = data.get("dataset", "Benchmark").upper()

    methods = list(results.keys())
    accs = [results[m]["accuracy"] * 100.0 for m in methods]
    toks = [results[m].get("mean_tokens", results[m].get("mean_prompt_tokens", 0.0)) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Subplot 1: Accuracy
    colors = ["#4C72B0", "#C44E52", "#2CA02C"]
    b1 = ax1.bar(methods, accs, color=colors[:len(methods)], alpha=0.85, edgecolor="black", linewidth=1.1)
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title("Classification Accuracy", fontweight="bold")
    ax1.set_ylim(0, max(accs + [70]) + 15)
    for b, val in zip(b1, accs):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{val:.1f}%", ha="center", fontweight="bold")

    # Subplot 2: Token Cost
    b2 = ax2.bar(methods, toks, color=colors[:len(methods)], alpha=0.85, edgecolor="black", linewidth=1.1)
    ax2.set_ylabel("Average Prompt Tokens", fontweight="bold")
    ax2.set_title("Prompt Token Footprint (Lower is Better)", fontweight="bold")
    for b, val in zip(b2, toks):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 25, f"{val:.0f}", ha="center", fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha="right")

    fig.suptitle(title or f"Proposed Budget-Aware Method vs Baselines on {dataset_name}", fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved proposed vs baselines plot to {save_path}")

    return fig


def plot_all_results(results_dir: str = "results", output_dir: str = "results/figures"):
    """Scan results/ folder and automatically generate figures for all completed experiments."""
    res_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_files = list(res_path.glob("*.json"))
    if not json_files:
        print(f"No result JSON files found in {results_dir}")
        return

    print(f"Generating figures from {len(json_files)} result files...")
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            exp_type = data.get("experiment", "")

            if exp_type == "exp1_baselines":
                plot_baselines(data, save_path=out_path / f"{f.stem}_accuracy.png")
            elif exp_type == "exp2_k_sweep":
                plot_k_scaling(data, save_path=out_path / f"{f.stem}_scaling.png")
            elif exp_type == "proposed_method":
                plot_proposed_vs_baselines(data, save_path=out_path / f"{f.stem}_comparison.png")
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    print(f"All available figures generated in '{output_dir}/'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot ICL experiment results.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--file", type=str, default=None, help="Plot single specific JSON result file")
    parser.add_argument("--type", type=str, default="baselines", choices=["baselines", "k_scaling", "tokens", "proposed"])
    args = parser.parse_args()

    if args.file:
        if args.type == "baselines":
            plot_baselines(args.file, save_path=Path(args.output_dir) / "baseline.png")
        elif args.type == "k_scaling":
            plot_k_scaling(args.file, save_path=Path(args.output_dir) / "scaling.png")
        elif args.type == "proposed":
            plot_proposed_vs_baselines(args.file, save_path=Path(args.output_dir) / "proposed.png")
    else:
        plot_all_results(args.results_dir, args.output_dir)
