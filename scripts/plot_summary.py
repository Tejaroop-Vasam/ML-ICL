#!/usr/bin/env python3
"""Generate cross-model comparison plots combining results from 0.6b, 1.7b, 4b, and 8b."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 200})

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["0.6b", "1.7b", "4b", "8b"]
DATASETS = ["yelp_polarity", "yahoo_answers", "trec", "imdb"]


def generate_cross_model_scaling_plots():
    """Plot Accuracy vs Model Scale (0.6B -> 1.7B -> 4B -> 8B) for each dataset."""
    for dataset in DATASETS:
        # Collect baseline results across models
        data_by_model = {}
        for m in MODEL_ORDER:
            f = RESULTS_DIR / f"baselines_{dataset}_{m}.json"
            if f.exists():
                with open(f) as fp:
                    data_by_model[m] = json.load(fp).get("results", {})

        if not data_by_model:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ["Zero-Shot", "Random", "Similarity-kNN"]
        markers = {"Zero-Shot": "^", "Random": "o", "Similarity-kNN": "s"}
        colors = {"Zero-Shot": "#4C72B0", "Random": "#55A868", "Similarity-kNN": "#C44E52"}

        models_present = [m for m in MODEL_ORDER if m in data_by_model]

        for method in methods:
            accs = []
            valid_models = []
            for m in models_present:
                if method in data_by_model[m]:
                    accs.append(data_by_model[m][method]["accuracy_mean"] * 100)
                    valid_models.append(m)

            if accs:
                ax.plot(
                    valid_models,
                    accs,
                    marker=markers[method],
                    color=colors[method],
                    linewidth=2.4,
                    markersize=8,
                    label=method,
                )
                for x_val, y_val in zip(valid_models, accs):
                    ax.text(x_val, y_val + 1.2, f"{y_val:.1f}%", ha="center", fontsize=9, fontweight="bold")

        ax.set_title(f"Model Scaling Performance: {dataset.upper()}", fontweight="bold", pad=12)
        ax.set_xlabel("Model Size (Parameters)", fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.legend(frameon=True, facecolor="white")
        plt.tight_layout()

        out_path = FIGURES_DIR / f"cross_model_scaling_{dataset}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def generate_overall_summary_heatmap():
    """Generate a heatmap of Similarity-kNN Accuracy across all models and datasets."""
    grid = np.full((len(DATASETS), len(MODEL_ORDER)), np.nan)

    for i, ds in enumerate(DATASETS):
        for j, m in enumerate(MODEL_ORDER):
            f = RESULTS_DIR / f"baselines_{ds}_{m}.json"
            if f.exists():
                with open(f) as fp:
                    res = json.load(fp).get("results", {})
                    if "Similarity-kNN" in res:
                        grid[i, j] = res["Similarity-kNN"]["accuracy_mean"] * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=MODEL_ORDER,
        yticklabels=[d.upper() for d in DATASETS],
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
        linewidths=1,
        linecolor="white",
    )
    ax.set_title("Similarity-kNN Few-Shot Accuracy across Models & Datasets", fontweight="bold", pad=12)
    ax.set_xlabel("Model Size", fontweight="bold")
    ax.set_ylabel("Dataset", fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES_DIR / "summary_heatmap_all_models.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_cross_model_scaling_plots()
    generate_overall_summary_heatmap()
