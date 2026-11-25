# Budget- & Complexity-Aware In-Context Learning (ICL)

> **Investigating and optimizing demonstration selection, ordering, and token-budget trade-offs for Large Language Models on short and long-input tasks.**

---

## 📌 Executive Summary & Research Objective

While In-Context Learning (ICL) has become the de facto paradigm for few-shot prompting in Large Language Models (LLMs), demonstration utility remains inconsistent across model scales, task complexities, and input lengths. Standard evaluations on concise benchmarks (e.g., SST-2, TREC) often fail to capture the degradation, context-window saturation, and retrieval challenges observed when prompting on long-document tasks.

This project investigates and addresses the core research question:

> **"Can we improve in-context learning by selecting and ordering demonstrations based on the complexity and length of the input, while keeping the prompt strictly within a limited token budget?"**

### Project Roadmap & Milestones
- **Mid-Project Milestone:** Establish rigorous empirical baselines (Zero-Shot, Random $k$-shot, Similarity-retrieved $k$-shot) and isolate the effects of demonstration count ($k$), token budget, prompt ordering, and formatting across short vs. long benchmarks.
- **Final Project Contribution:** Propose, evaluate, and ablate a **Budget-Aware, Diversity-Aware Demonstration Selection Algorithm** tailored to maximize performance under bounded prompt-token budgets.

---

## 🏗️ Experimental Framework & Setup

```mermaid
graph TD
    A[Train Pool / Candidate Bank] --> B{Demonstration Selection}
    B -->|Baseline 1| C1[Random Demonstration Selection]
    B -->|Baseline 2| C2[Embedding Similarity Search (k-NN)]
    B -->|Proposed| C3[Budget- & Diversity-Aware Selection]
    
    C1 --> D{Ordering & Formatting}
    C2 --> D
    C3 --> D
    
    D -->|Ordering| E1[Similar-First / Similar-Last / Random]
    D -->|Format| E2[Plain / Structured / Rationale-Augmented]
    
    E1 --> F[Prompt Assembler & Token Budgeter]
    E2 --> F
    
    F --> G[LLM Inference Engine]
    G --> H1[Llama-3-8B (Primary)]
    G --> H2[Llama-3.2-1B-Instruct (Secondary)]
    
    H1 --> I[Evaluation & Error Analysis]
    H2 --> I
```

### 1. Models Under Study
* **Primary Model:** `Meta-Llama-3-8B` / `Meta-Llama-3-8B-Instruct`
* **Secondary Model:** `Meta-Llama-3.2-1B-Instruct` (tests whether demonstration selection gains transfer across parameter scale and helps lower-capacity models more significantly).

### 2. Datasets & Task Categorization

To separate the effects of **input length** from **task/label complexity**, the benchmark suite balances standard short-sequence tasks with a dedicated long-input classification dataset:

| Dataset | Task Type | Classes | Input Characteristics | Role in Study |
| :--- | :--- | :--- | :--- | :--- |
| **SST-2** | Sentiment Classification | 2 (Binary) | Short single-sentence reviews | Standard baseline (Simple) |
| **AG News** | Topic Classification | 4 Topics | Sentence-to-paragraph news articles | Multi-class baseline |
| **TREC** | Question Classification | 6 Coarse | Short, query-style questions | Format-diversity baseline |
| **IMDb** | Sentiment Classification | 2 (Binary) | Full multi-paragraph movie reviews | **Primary Long-Input Dataset** (Decouples length from label space) |

---

## 🔬 Core Experiments (Mid-Project Phase)

The initial phase systematically unpacks *when*, *why*, and *how* demonstrations alter model predictions:

### Experiment 1: Baseline Characterization
Measure zero-shot and few-shot performance across all datasets:
* **Zero-Shot ($k=0$):** Inherent task prior of the model without demonstrations.
* **Random Demonstration Selection:** Uniformly sampled demonstrations across multiple fixed random seeds (to account for high variance).
* **Semantic Similarity Retrieval:** Retrieve $k$ nearest neighbors from the training set using cosine similarity with `BAAI/bge-m3` embeddings.

### Experiment 2: Demonstration Scaling ($k$-Shot Curves)
Evaluate model accuracy and Macro-F1 across demonstration counts:
$$k \in \{0, 2, 4, 8, 16\}$$
* Determine where performance plateaus or degrades.
* Analyze whether long-input datasets (IMDb) experience saturation earlier due to context limits.

### Experiment 3: Prompt Length & Token-Budget Stress Testing
Examine the exact trade-off between prompt length and ICL efficacy using the model's native tokenizer:
* **Protocol A (Fixed $k$, Variable Input Length):** Compare performance on short vs. long test inputs under a static $k$.
* **Protocol B (Fixed Token Budget):** Fix the prompt budget (e.g., 512, 1024, 2048, 4096 tokens) and determine whether packing more short examples outperforms including fewer, longer, highly-relevant examples.

### Experiment 4: Demonstration Ordering & Recency Bias
Hold the retrieved demonstration subset identical and evaluate permutations:
* **Similar-First:** Most similar example placed furthest from the test query.
* **Similar-Last:** Most similar example placed adjacent to the test query.
* **Random Shuffling:** Average performance over multiple permutation seeds.
* **Fixed Reference Ordering:** Predetermined ordering applied globally across all test inputs.

### Experiment 5: Demonstration Presentation & Formatting
Evaluate structural variants across input lengths:
* **Format A (Plain Text & Label):**
  ```text
  Review: "The film was excellent."
  Sentiment: Positive
  ```
* **Format B (Structured Fields):**
  ```text
  [Input] "The film was excellent."
  [Label] positive
  ```
* **Format C (Input + Brief Rationale + Label):**
  ```text
  [Input] "The film was excellent."
  [Rationale] "Excellent" expresses clear positive sentiment.
  [Label] positive
  ```
  *(Evaluated under controlled token budgets to prevent confounding token volume with explanatory power).*

---

## 💡 Novel Proposed Method: Budget- & Diversity-Aware Demonstration Selection (Final Project)

Standard similarity retrieval frequently selects near-duplicate candidates and exhausts prompt budgets quickly on long-input tasks. We propose a dynamic selection algorithm that optimizes relevance, diversity, and token cost under an explicit token budget constraint $B$.

### Formulation
For candidate demonstration $d_i$ and test input $x$, given the set of already selected demonstrations $\mathcal{D}$:

$$S(d_i, x \mid \mathcal{D}) = \alpha \cdot \operatorname{sim}(d_i, x) + \beta \cdot \operatorname{div}(d_i, \mathcal{D}) - \gamma \cdot \operatorname{cost}(d_i)$$

Where:
* $\operatorname{sim}(d_i, x)$: Semantic embedding similarity to test query $x$.
* $\operatorname{div}(d_i, \mathcal{D})$: Diversity penalty / novelty score relative to already-chosen demonstrations $\mathcal{D}$ (e.g., $1 - \max_{d_j \in \mathcal{D}} \operatorname{sim}(d_i, d_j)$).
* $\operatorname{cost}(d_i)$: Token footprint of candidate $d_i$ computed via the target LLM tokenizer.
* $\alpha, \beta, \gamma$: Hyperparameters tuned on held-out validation sets.

### Selection Procedure
1. **Candidate Retrieval:** Retrieve top-$M$ semantic candidates using dense embeddings.
2. **Greedy Knapsack Selection:** Iteratively add candidate $d^* = \arg\max_{d_i} S(d_i, x \mid \mathcal{D})$ provided $\text{length}(d^*) + \sum_{d \in \mathcal{D}} \text{length}(d) \le B$.
3. **Length-Adaptive Ordering:** Position demonstrations according to input complexity and length profile.

---

## 📊 Comparative Evaluation Matrix

Every selection strategy is benchmarked using identical splits and evaluation samples:

| Method | Selection Criterion | Complexity / Overhead | Evaluates |
| :--- | :--- | :--- | :--- |
| **Zero-Shot** | None ($k=0$) | $O(1)$ | No-demonstration baseline |
| **Random $k$-shot** | Uniform sampling | $O(1)$ | Basic ICL baseline across seeds |
| **$k$-NN Similarity** | Cosine similarity | $O(M)$ retrieval | Standard semantic retrieval baseline |
| **Max-Diversity (DPP / MMR)** | Semantic coverage | $O(M^2)$ | Diversity-only baseline |
| **Proposed Method** | Relevance + Diversity + Token Cost | $O(M \cdot k)$ greedy | Budget-constrained Pareto frontier |

### Evaluation Metrics
* **Task Performance:** Classification Accuracy, Macro-F1 (for class-imbalanced slices).
* **Efficiency:** Performance per $1\text{k}$ prompt tokens, inference latency, and retrieval overhead.
* **Robustness:** Standard deviation across 5 random seeds for sampling and ordering.

---

## 📅 Project Phases & Action Timeline

```
Phase 1: Setup & Data Split Freezing
├── Finalize dataset loaders (SST-2, AG News, TREC, IMDb)
├── Build tokenizer-aware prompt formatters
└── Set up inference pipelines for Llama-3-8B and Llama-3.2-1B

Phase 2: Mid-Project Baselines (Priority)
├── Zero-shot vs. Random vs. k-NN Retrieval benchmarks
├── Demonstration scaling sweeps (k ∈ {0, 2, 4, 8, 16})
├── Prompt-length & budget trade-off experiments
└── Prepare mid-project presentation slides & baseline plots

Phase 3: Diagnostic Experiments
├── Demonstration ordering analysis (similar-first vs. similar-last)
├── Formatting styles comparison (plain vs. structured vs. rationale)
└── Error analysis on short vs. long text inputs

Phase 4: Novel Method Implementation & Tuning
├── Implement budget-aware objective S(d_i, x | D)
├── Tune weights (α, β, γ) on validation splits
└── Benchmark on IMDb long-context inputs under tight token limits

Phase 5: Final Report & Ablation Studies
├── Component ablation (impact of diversity vs. cost penalty)
├── Cross-model transfer (8B vs. 1B)
└── Deliver final paper, code repository, and presentation
```

---

## Reviewer experiments

The [reviewer runner](docs/reviewer_experiments.md) uses **BGE-M3** for dense cosine retrieval across classification, Spider, BIRD, and CLAPNQ. BGE-M3 can embed up to 8,192 tokens per query or candidate. The runner counts the exact Qwen3 prompt tokens separately against the model's native 32,768-token window.

- **Classification** compares zero-shot, random demonstrations, BGE-M3 demonstrations, and label-balanced selection. The IMDb controls examine output parsing, label preference, and the location of sentiment cues.
- **Spider and BIRD** evaluate executable SQL using full schemas, retrieved tables, and tables derived from the reference query. BIRD also tests the effect of its domain evidence.
- **CLAPNQ** compares retrieved and benchmark passages for long-answer generation. The stress test varies prompt length and the position of the relevant passage.
- **Qualitative review** uses the saved prompts, predictions, scores, and blinded annotation packets from the run.

The [Colab launcher](ML_ICL_Reviewer_Experiments.ipynb) starts with a Qwen3-8B pilot. New classification summaries go to `results/bge_m3/`. The reviewer runner writes per-trial results and checkpoints to `ml_icl_review_runs/`. The other checked-in results, figures, slide deck, and report are historical artifacts.

---

## 📁 Repository Directory Structure

```plaintext
ML-ICL/
├── README.md                               # Project overview, research questions & review responses
├── Makefile                                # Automated targets for all experiments, slides, and reports
├── ML_ICL_Comprehensive_Experiment_Report.docx # Publication-grade comprehensive research report
├── ML_ICL_Experiment_Presentation.pptx     # 18-slide publication-ready PowerPoint presentation
├── docs.md                                 # Technical documentation & benchmark metrics
├── explanation.txt                         # Speaker notes for all 18 presentation slides
├── data/                                   # Processed JSONL & NPZ datasets
│   └── processed/                          # Formatted partitions (Yelp, Yahoo, TREC, IMDb, Spider, BIRD, etc.)
├── results/                                # Metrics, JSON archives, and publication plots
│   ├── figures/                            # High-resolution plots (selection, k-sweep, oracle, complex tasks)
│   ├── prompt_length_analysis.json         # Token statistics vs Qwen3 32K context window
│   ├── oracle_comparison_results.json      # Disentangled retriever vs LLM deficit metrics
│   ├── complex_tasks_results.json          # Spider, BIRD, and Long-Answer evaluation results
│   └── qualitative_case_studies.md         # Verbatim qualitative failure case studies
├── scripts/                                # Analysis and generation scripts
│   ├── run_oracle_study.py                 # Disentangles retriever from LLM via Oracle
│   ├── run_complex_tasks_study.py          # Benchmarks Spider, BIRD, and Long-Answer tasks
│   ├── qualitative_error_analysis.py       # Qualitative failure mode investigation
│   ├── token_profiler.py                   # Exact prompt token length profiling vs 32K window
│   ├── plot_comprehensive_comparisons.py   # Publication figure generation
│   ├── generate_ppt.py                     # 18-slide presentation compiler
│   └── generate_word_report.py             # Academic Word report compiler
└── src/                                    # Modular core library
    ├── data/                               # Data loaders (including Spider, BIRD, Long Answer)
    ├── selection/                          # Random, BGE-M3 kNN, Budget-Aware, and Oracle selectors
    ├── models/                             # Qwen3 HuggingFace inference runners
    └── evaluation/                         # Evaluation harness & classification metrics
```

---

## 🎯 Quickstart Commands

```bash
# Run unit test suite
make test

# Profile prompt token lengths vs Qwen3 32K context window
make profile

# Install the reviewer runner dependencies on a GPU runtime
make install_review

# Run the classification retrieval controls with BGE-M3
make oracle

# Run complex long-context benchmarks (Spider, BIRD, Long Answer QA)
make complex

# Export annotation packets from saved reviewer runs
make qualitative

# Plot saved BGE-M3 classification summaries
make plot
```
