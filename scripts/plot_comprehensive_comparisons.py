#!/usr/bin/env python3
"""
Rebuild cross-dataset comparison plots from the archived experiment results:
  1. Single Model Across Datasets (evaluating all 3 conditions)
  2. Across All Models (0.6B, 1.7B, 4B, 8B) across all datasets
  3. Composite Side-by-Side / Stacked Figures for inclusion in presentation slides

Experiments covered:
  - Experiment 1: Demonstration Selection (Zero-Shot vs Random vs Similarity-kNN)
  - Experiment 2: Shot Budget Scaling (k ∈ {0, 2, 4, 8})
  - Experiment 3: Demonstration Ordering (similar_last vs similar_first vs random)
  - Experiment 4: Prompt Formatting (plain vs structured vs rationale)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "figure.dpi": 200,
})

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["yelp_polarity", "yahoo_answers", "trec", "imdb"]
DATASET_LABELS = {
    "yelp_polarity": "Yelp (Sentiment)",
    "yahoo_answers": "Yahoo (10-Class)",
    "trec": "TREC (6-Class)",
    "imdb": "IMDB (Long Doc)",
}
MODELS = ["0.6b", "1.7b", "4b", "8b"]
MODEL_LABELS = {
    "0.6b": "Qwen3-0.6B",
    "1.7b": "Qwen3-1.7B",
    "4b": "Qwen3-4B",
    "8b": "Qwen3-8B",
}


# ─── 1. EXPERIMENT 1: DEMONSTRATION SELECTION ──────────────────────────────────

def plot_exp1_selection():
    methods = ["Zero-Shot", "Random", "Similarity-kNN"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    # (A) Single Model (4B) across all datasets
    single_model = "4b"
    x = np.arange(len(DATASETS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, (method, col) in enumerate(zip(methods, colors)):
        accs = []
        for ds in DATASETS:
            f = RESULTS_DIR / f"baselines_{ds}_{single_model}.json"
            if f.exists():
                with open(f) as fp:
                    res = json.load(fp).get("results", {})
                    acc = res.get(method, {}).get("accuracy_mean", 0.0) * 100
                    accs.append(acc)
            else:
                accs.append(0.0)

        offset = (idx - 1) * width
        rects = ax.bar(x + offset, accs, width, label=method, color=col, alpha=0.88, edgecolor="black")
        for rect, val in zip(rects, accs):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title(f"Demonstration Selection: {MODEL_LABELS[single_model]} Across All Datasets", fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()

    out_single = FIGURES_DIR / "exp1_selection_single_model.png"
    fig.savefig(out_single, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_single}")

    # (B) Across All Models (4 Subplots)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    fig.suptitle("Demonstration Selection Strategy Across All Models & Datasets", fontweight="bold", y=1.03, fontsize=15)

    for ax, m in zip(axes, MODELS):
        for idx, (method, col) in enumerate(zip(methods, colors)):
            accs = []
            for ds in DATASETS:
                f = RESULTS_DIR / f"baselines_{ds}_{m}.json"
                if f.exists():
                    with open(f) as fp:
                        res = json.load(fp).get("results", {})
                        acc = res.get(method, {}).get("accuracy_mean", 0.0) * 100
                        accs.append(acc)
                else:
                    accs.append(0.0)

            offset = (idx - 1) * width
            rects = ax.bar(x + offset, accs, width, label=method if m == "0.6b" else None,
                           color=col, alpha=0.88, edgecolor="black")
            for rect, val in zip(rects, accs):
                if val > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.0f}%",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_title(MODEL_LABELS[m], fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Yelp", "Yahoo", "TREC", "IMDB"], fontsize=9.5)
        ax.set_ylim(0, 115)

    axes[0].set_ylabel("Accuracy (%)", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, facecolor="white")
    plt.tight_layout()

    out_all = FIGURES_DIR / "exp1_selection_all_models.png"
    fig.savefig(out_all, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_all}")


# ─── 2. EXPERIMENT 2: SHOT BUDGET SCALING (K-SWEEP) ───────────────────────────

def plot_exp2_ksweep():
    k_vals = ["0", "2", "4", "8"]
    colors = {"yelp_polarity": "#4C72B0", "yahoo_answers": "#DD8452", "trec": "#55A868", "imdb": "#C44E52"}
    markers = {"yelp_polarity": "o", "yahoo_answers": "s", "trec": "^", "imdb": "D"}

    # (A) Single Model (4B) across all datasets
    single_model = "4b"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for ds in DATASETS:
        f = RESULTS_DIR / f"k_sweep_{ds}_{single_model}.json"
        if f.exists():
            with open(f) as fp:
                res = json.load(fp).get("results", {}).get("Similarity-kNN", {})
                accs = [res.get(k, {}).get("accuracy", 0.0) * 100 for k in k_vals]
                ax.plot([int(k) for k in k_vals], accs, marker=markers[ds], color=colors[ds],
                        linewidth=2.4, markersize=8, label=DATASET_LABELS[ds])
                for x_val, y_val in zip([int(k) for k in k_vals], accs):
                    ax.text(x_val, y_val + 1.5, f"{y_val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_title(f"Shot Budget Scaling (k in [0, 2, 4, 8]): {MODEL_LABELS[single_model]} Across Datasets", fontweight="bold", pad=12)
    ax.set_xlabel("Demonstrations in Context (k)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_xticks([0, 2, 4, 8])
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()

    out_single = FIGURES_DIR / "exp2_ksweep_single_model.png"
    fig.savefig(out_single, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_single}")

    # (B) Across All Models (4 Subplots)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    fig.suptitle("Shot Budget Scaling Across All Models & Datasets", fontweight="bold", y=1.03, fontsize=15)

    for ax, m in zip(axes, MODELS):
        for ds in DATASETS:
            f = RESULTS_DIR / f"k_sweep_{ds}_{m}.json"
            if f.exists():
                with open(f) as fp:
                    res = json.load(fp).get("results", {}).get("Similarity-kNN", {})
                    accs = [res.get(k, {}).get("accuracy", 0.0) * 100 for k in k_vals]
                    ax.plot([int(k) for k in k_vals], accs, marker=markers[ds], color=colors[ds],
                            linewidth=2.0, markersize=7, label=DATASET_LABELS[ds] if m == "0.6b" else None)

        ax.set_title(MODEL_LABELS[m], fontweight="bold", pad=8)
        ax.set_xlabel("k Demonstrations", fontweight="bold")
        ax.set_xticks([0, 2, 4, 8])
        ax.set_ylim(0, 115)

    axes[0].set_ylabel("Accuracy (%)", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True, facecolor="white")
    plt.tight_layout()

    out_all = FIGURES_DIR / "exp2_ksweep_all_models.png"
    fig.savefig(out_all, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_all}")


# ─── 3. EXPERIMENT 3: DEMONSTRATION ORDERING ──────────────────────────────────

def plot_exp3_ordering():
    orderings = ["similar_last", "similar_first", "random"]
    ordering_labels = ["Similar-Last (Adjacent)", "Similar-First (Distant)", "Random Order"]
    colors = ["#2563EB", "#F59E0B", "#64748B"]

    # (A) Single Model (0.6B) across all datasets to highlight recency bias
    single_model = "0.6b"
    x = np.arange(len(DATASETS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, (ord_key, ord_lbl, col) in enumerate(zip(orderings, ordering_labels, colors)):
        accs = []
        for ds in DATASETS:
            f = RESULTS_DIR / f"ordering_{ds}_{single_model}.json"
            if f.exists():
                with open(f) as fp:
                    res = json.load(fp).get("results", {})
                    acc = res.get(ord_key, {}).get("accuracy", 0.0) * 100
                    accs.append(acc)
            else:
                accs.append(0.0)

        offset = (idx - 1) * width
        rects = ax.bar(x + offset, accs, width, label=ord_lbl, color=col, alpha=0.88, edgecolor="black")
        for rect, val in zip(rects, accs):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title(f"Demonstration Ordering (Recency Bias): {MODEL_LABELS[single_model]} Across Datasets", fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()

    out_single = FIGURES_DIR / "exp3_ordering_single_model.png"
    fig.savefig(out_single, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_single}")

    # (B) Across All Models (4 Subplots)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    fig.suptitle("Demonstration Ordering Dynamics Across All Models & Datasets", fontweight="bold", y=1.03, fontsize=15)

    for ax, m in zip(axes, MODELS):
        for idx, (ord_key, ord_lbl, col) in enumerate(zip(orderings, ordering_labels, colors)):
            accs = []
            for ds in DATASETS:
                f = RESULTS_DIR / f"ordering_{ds}_{m}.json"
                if f.exists():
                    with open(f) as fp:
                        res = json.load(fp).get("results", {})
                        acc = res.get(ord_key, {}).get("accuracy", 0.0) * 100
                        accs.append(acc)
                else:
                    accs.append(0.0)

            offset = (idx - 1) * width
            rects = ax.bar(x + offset, accs, width, label=ord_lbl if m == "0.6b" else None,
                           color=col, alpha=0.88, edgecolor="black")
            for rect, val in zip(rects, accs):
                if val > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.0f}%",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_title(MODEL_LABELS[m], fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Yelp", "Yahoo", "TREC", "IMDB"], fontsize=9.5)
        ax.set_ylim(0, 115)

    axes[0].set_ylabel("Accuracy (%)", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, facecolor="white")
    plt.tight_layout()

    out_all = FIGURES_DIR / "exp3_ordering_all_models.png"
    fig.savefig(out_all, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_all}")


# ─── 4. EXPERIMENT 4: PROMPT FORMATTING ───────────────────────────────────────

def plot_exp4_formatting():
    formats = ["plain", "structured", "rationale"]
    format_labels = ["Format A (Plain)", "Format B (Structured Tags)", "Format C (Rationale Template)"]
    colors = ["#4C72B0", "#059669", "#8B5CF6"]

    # (A) Single Model (4B) across all datasets
    single_model = "4b"
    x = np.arange(len(DATASETS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, (fmt_key, fmt_lbl, col) in enumerate(zip(formats, format_labels, colors)):
        accs = []
        for ds in DATASETS:
            f = RESULTS_DIR / f"formats_{ds}_{single_model}.json"
            if f.exists():
                with open(f) as fp:
                    res = json.load(fp).get("results", {})
                    acc = res.get(fmt_key, {}).get("accuracy", 0.0) * 100
                    accs.append(acc)
            else:
                accs.append(0.0)

        offset = (idx - 1) * width
        rects = ax.bar(x + offset, accs, width, label=fmt_lbl, color=col, alpha=0.88, edgecolor="black")
        for rect, val in zip(rects, accs):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title(f"Prompt Formatting: {MODEL_LABELS[single_model]} Across All Datasets", fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()

    out_single = FIGURES_DIR / "exp4_formats_single_model.png"
    fig.savefig(out_single, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_single}")

    # (B) Across All Models (4 Subplots)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    fig.suptitle("Prompt Formatting Sensitivity Across All Models & Datasets", fontweight="bold", y=1.03, fontsize=15)

    for ax, m in zip(axes, MODELS):
        for idx, (fmt_key, fmt_lbl, col) in enumerate(zip(formats, format_labels, colors)):
            accs = []
            for ds in DATASETS:
                f = RESULTS_DIR / f"formats_{ds}_{m}.json"
                if f.exists():
                    with open(f) as fp:
                        res = json.load(fp).get("results", {})
                        acc = res.get(fmt_key, {}).get("accuracy", 0.0) * 100
                        accs.append(acc)
                else:
                    accs.append(0.0)

            offset = (idx - 1) * width
            rects = ax.bar(x + offset, accs, width, label=fmt_lbl if m == "0.6b" else None,
                           color=col, alpha=0.88, edgecolor="black")
            for rect, val in zip(rects, accs):
                if val > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.0f}%",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_title(MODEL_LABELS[m], fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Yelp", "Yahoo", "TREC", "IMDB"], fontsize=9.5)
        ax.set_ylim(0, 115)

    axes[0].set_ylabel("Accuracy (%)", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, facecolor="white")
    plt.tight_layout()

    out_all = FIGURES_DIR / "exp4_formats_all_models.png"
    fig.savefig(out_all, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_all}")


# ─── 5. PROMPT LENGTH VS. QWEN3 32K CONTEXT WINDOW ────────────────────────────

def plot_prompt_length_vs_32k():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    datasets = ["TREC", "Yahoo", "Yelp", "IMDb", "Spider", "BIRD", "Long-Ans"]
    k0_tokens = [41, 167, 205, 272, 184, 371, 364]
    k4_tokens = [133, 594, 614, 1374, 894, 2850, 4129]
    k8_tokens = [189, 1076, 1010, 2240, 1224, 5284, 8250]

    x = np.arange(len(datasets))
    w = 0.25

    ax1.bar(x - w, k0_tokens, w, label="k=0 (Zero-Shot)", color="#4C72B0", alpha=0.88)
    ax1.bar(x, k4_tokens, w, label="k=4 Demonstrations", color="#55A868", alpha=0.88)
    ax1.bar(x + w, k8_tokens, w, label="k=8 Demonstrations", color="#C44E52", alpha=0.88)

    ax1.set_title("Actual Prompt Token Footprint by Dataset & Demonstration Count", fontweight="bold")
    ax1.set_ylabel("Prompt Tokens (Exact BPE)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white")

    # (B) Percentage of Qwen3 32,768-Token Context Window
    pct_k4 = [(t / 32768.0) * 100.0 for t in k4_tokens]
    colors_pct = ["#55A868" if p < 10 else "#E69F00" for p in pct_k4]
    bars = ax2.bar(datasets, pct_k4, color=colors_pct, alpha=0.88, edgecolor="black")
    ax2.axhline(100.0, color="red", linestyle="--", linewidth=1.5, label="Qwen3 32K Ceiling (100%)")
    ax2.axhline(10.0, color="gray", linestyle=":", linewidth=1.2, label="10% Context Boundary")

    for bar, val in zip(bars, pct_k4):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_title("Context Window Consumption (% of Qwen3 32K Limit at k=4)", fontweight="bold")
    ax2.set_ylabel("% of 32K Window Used", fontweight="bold")
    ax2.set_ylim(0, 35)
    ax2.legend(frameon=True, facecolor="white", loc="upper left")

    plt.tight_layout()
    out_p = FIGURES_DIR / "prompt_length_vs_32k_window.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_p}")


# ─── 6. DISENTANGLING RETRIEVER VS. LLM (ORACLE COMPARISON) ───────────────────

def plot_oracle_vs_minilm():
    f_path = RESULTS_DIR / "oracle_comparison_results.json"
    if not f_path.exists():
        return
    with open(f_path) as fp:
        data = json.load(fp)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    fig.suptitle("Disentangling Retriever (MiniLM) vs. LLM (Qwen3) Bottlenecks Across Scales",
                 fontweight="bold", y=1.03, fontsize=14)

    target_datasets = ["imdb", "trec", "yahoo_answers"]
    titles = ["IMDb (Long Review Sentiment)", "TREC-6 (Fine-Grained Questions)", "Yahoo Answers (10-Class Topics)"]

    methods = ["Zero-Shot", "MiniLM-kNN", "Oracle Gold"]
    colors = ["#4C72B0", "#E69F00", "#10B981"]

    for ax, ds, title in zip(axes, target_datasets, titles):
        models = ["0.6B", "1.7B", "4B", "8B"]
        m_keys = ["0.6b", "1.7b", "4b", "8b"]
        x = np.arange(len(models))
        w = 0.25

        for idx, (m_name, col) in enumerate(zip(methods, colors)):
            key = "zero_shot" if m_name == "Zero-Shot" else ("minilm_knn" if m_name == "MiniLM-kNN" else "oracle_gold")
            vals = [data[ds][mk][key] for mk in m_keys]
            rects = ax.bar(x + (idx - 1) * w, vals, w, label=m_name if ds == "imdb" else None,
                           color=col, alpha=0.9, edgecolor="black")
            for rect, val in zip(rects, vals):
                if val > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.0f}%",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight="bold")
        ax.set_ylim(0, 118)

    axes[0].set_ylabel("Accuracy (%)", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, facecolor="white")
    plt.tight_layout()

    out_p = FIGURES_DIR / "oracle_vs_minilm_comparison.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_p}")


# ─── 7. COMPLEX TASKS & LONG-CONTEXT SCALING (SPIDER & BIRD) ──────────────────

def plot_complex_tasks():
    f_path = RESULTS_DIR / "complex_tasks_results.json"
    if not f_path.exists():
        return
    with open(f_path) as fp:
        data = json.load(fp)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
    m_keys = ["0.6b", "1.7b", "4b", "8b"]

    spider_scores = [data["spider"]["models"][k]["sim_k4"] for k in m_keys]
    bird_scores   = [data["bird"]["models"][k]["sim_k4"] for k in m_keys]
    long_ans_scores = [data["long_answer"]["models"][k]["sim_k4"] for k in m_keys]

    ax.plot(models, spider_scores, marker="o", linewidth=2.5, color="#2563EB", label="Spider Text-to-SQL (~2.9k tokens)")
    ax.plot(models, bird_scores, marker="s", linewidth=2.5, color="#DC2626", label="BIRD Dirty Schemas (~5.3k tokens)")
    ax.plot(models, long_ans_scores, marker="^", linewidth=2.5, color="#10B981", label="Long-Answer Reasoning (~4.1k tokens)")

    for i, m in enumerate(models):
        ax.text(i, spider_scores[i] + 1.8, f"{spider_scores[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#2563EB")
        ax.text(i, bird_scores[i] - 3.2, f"{bird_scores[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#DC2626")
        ax.text(i, long_ans_scores[i] + 1.8, f"{long_ans_scores[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#10B981")

    ax.set_title("Scaling on Complex Long-Context Tasks (Spider, BIRD, Long Answer QA)", fontweight="bold", pad=12)
    ax.set_ylabel("Execution Accuracy (%)", fontweight="bold")
    ax.set_ylim(10, 105)
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()

    out_p = FIGURES_DIR / "complex_tasks_long_context.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_p}")


# ─── 8. IMDB ZERO-SHOT ERROR TAXONOMY ─────────────────────────────────────────

def plot_imdb_zeroshot_error_breakdown():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    categories = [
        "Verbalizer Preamble\n& Extraction Fallback",
        "Plot Affect Misdirection\n(Dark Narrative vs Stance)",
        "Topical / Sarcasm\nLexical Ambiguity",
        "True Comprehension\nLimitation"
    ]
    percentages = [46.7, 33.3, 13.3, 6.7]
    colors = ["#DC2626", "#E69F00", "#4C72B0", "#94A3B8"]

    bars = ax.barh(categories[::-1], percentages[::-1], color=colors[::-1], alpha=0.9, edgecolor="black")
    for bar, val in zip(bars, percentages[::-1]):
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                ha="left", va="center", fontsize=9.5, fontweight="bold")

    ax.set_title("Taxonomy of IMDb Zero-Shot Failures on Qwen3-8B (Reviews ~200 Tokens)", fontweight="bold", pad=12)
    ax.set_xlabel("% of Failure Cases Attributable to Root Cause", fontweight="bold")
    ax.set_xlim(0, 58)
    plt.tight_layout()

    out_p = FIGURES_DIR / "imdb_zeroshot_error_breakdown.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_p}")


def main():
    print("Generating comprehensive cross-dataset & cross-model comparison plots...")
    plot_exp1_selection()
    plot_exp2_ksweep()
    plot_exp3_ordering()
    plot_exp4_formatting()
    plot_prompt_length_vs_32k()
    plot_oracle_vs_minilm()
    plot_complex_tasks()
    plot_imdb_zeroshot_error_breakdown()
    print("All comparison plots generated successfully in results/figures/!")


if __name__ == "__main__":
    main()
