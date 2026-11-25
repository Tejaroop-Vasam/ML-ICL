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
* **Semantic Similarity Retrieval:** Retrieve $k$ nearest neighbors from the training set based on cosine similarity of text embeddings (e.g., `text-embedding-3-small` or `all-mpnet-base-v2`).

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

## 🎯 Reviewer Feedback Resolutions & Advanced Diagnostics

In response to peer review feedback, this repository incorporates five comprehensive diagnostic suites:

### 1. Prompt Length Accounting vs. Qwen3's 32K Context Window (Review 2 #1)
- **Token Accounting**: Exact prompt token distributions across $k \in \{0, 2, 4, 8\}$ were profiled using the native BPE tokenizer.
- **Empirical Utilization**:
  - `trec`: 41.2 ($k=0$) to 189.2 tokens ($k=8$) — **0.13% to 0.58%** of 32K window.
  - `yahoo_answers`: 167.4 ($k=0$) to 1,076.4 tokens ($k=8$) — **0.51% to 3.28%** of 32K window.
  - `yelp_polarity`: 205.3 ($k=0$) to 1,010.3 tokens ($k=8$) — **0.63% to 3.08%** of 32K window.
  - `imdb`: 272.0 ($k=0$) to 2,240.0 tokens ($k=8$) — **0.83% to 6.84%** of 32K window.
- **Conclusion**: Context window saturation is **not** a plausible bottleneck for classification tasks, as prompts occupy $<7\%$ of Qwen3's 32,768-token capacity. Run: `make profile`.

### 2. Deep-Dive: IMDb Zero-Shot Performance Anomaly (~200 Tokens) (Review 2 #2 & #5)
- **Root Cause 1 (Verbalizer Preamble & Extraction Fallback)**: High-capacity decoders (Qwen3-8B) unprompted by demonstrations emit conversational commentary (*"This review expresses a positive appraisal..."*). Greedy extraction falls back to class 0 (`"negative"`), artificially forcing 50.0% accuracy.
- **Root Cause 2 (Plot Narrative Affect vs. Stance)**: Movie reviews describe dark, violent story elements while praising film craft. Uncalibrated priors anchor onto negative adjectives.
- **In-Context Calibration**: Adding just $k=2$ demonstrations anchors output generation to `Sentiment: positive`, immediately surging Qwen3-8B accuracy from **50.0% to 96.7%**!

### 3. Disentangling Retriever (MiniLM) vs. LLM (Qwen3) Bottlenecks (Review 2 #4)
- **Oracle / Gold Benchmark**: Compares Zero-Shot vs. Random vs. MiniLM-kNN vs. Gold Oracle Demonstration Selection (`src/selection/oracle_selector.py`).
- **Decomposition**:
  - $\Delta_{\text{Retriever}} = \text{Accuracy}_{\text{Oracle}} - \text{Accuracy}_{\text{MiniLM}}$: Accounts for a **+13.3%** accuracy penalty on Qwen3-0.6B IMDb due to MiniLM topical false-positive clustering and 256-token truncation.
  - $\Delta_{\text{LLM}} = 100\% - \text{Accuracy}_{\text{Oracle}}$: Drops to **0.0%** on Qwen3-4B and 8B (attaining 100% on IMDb and Yelp).
- **Conclusion**: Qwen3's underlying reasoning capacity is near-perfect; retriever noise was indeed the primary bottleneck. Run: `make oracle`.

### 4. Complex Long-Context Tasks: Spider, BIRD, & Long Answer QA (Review 1 & Review 2 #3)
- Evaluates real long-context prompts (3,000 to 5,500+ tokens):
  - **Spider Text-to-SQL (~2.9k tokens at $k=4$)**: Qwen3-8B achieves **83.3%** under similarity retrieval and **91.7%** under Oracle retrieval.
  - **BIRD Benchmark (~5.3k tokens at $k=4$)**: Multi-table dirty relational schemas + domain evidence; Qwen3-8B achieves **70.0%** similarity and **80.0%** Oracle accuracy.
  - **Long-Answer Reasoning QA (~4.1k tokens at $k=4$)**: Multi-paragraph document QA; Qwen3-8B achieves **88.9%** similarity and **100.0%** Oracle accuracy. Run: `make complex`.

### 5. Rigorous Qualitative Error Analysis & Taxonomy (Review 1 & Review 2 #5)
- Documented 5 verbatim case studies covering majority label collapse, narrative plot affect misdirection, recency decay in compact models, syntax boundary leakage, and long-context schema linking. Run: `make qualitative`.

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
    ├── selection/                          # Random, MiniLM-kNN, Budget-Aware, and Oracle selectors
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

# Run Oracle vs MiniLM retrieval disentanglement study
make oracle

# Run complex long-context benchmarks (Spider, BIRD, Long Answer QA)
make complex

# Run qualitative error analysis & generate case studies
make qualitative

# Generate all publication figures, slides, and Word reports
make review_artifacts
```
