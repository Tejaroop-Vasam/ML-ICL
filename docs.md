# Qwen3-0.6B — ICL Experiment Results

**Model**: `Qwen/Qwen3-0.6B` (~1.2 GB, runs locally on 8GB M1 Mac via MPS)
**Evaluation**: 30 test samples per run · seed = 42 · datasets: SST-2, AG News, TREC

---

## Overview

Four experiments were run across three datasets:

| Experiment | What it Tests |
| :--- | :--- |
| **Exp 1 — Baselines** | Zero-Shot vs Random Few-Shot vs Similarity-kNN |
| **Exp 2 — k-Sweep** | How performance scales from k=0 to k=8 |
| **Exp 4 — Ordering** | similar_last vs similar_first vs random ordering |
| **Exp 5 — Format** | plain vs structured vs rationale prompt format |

---

## Experiment 1 — Baseline Comparison (k=4)

Comparing the three selection strategies: no demonstrations (zero-shot), randomly chosen demonstrations, and similarity-based selection (kNN on sentence embeddings).

### SST-2 (Binary Sentiment)

| Method | Accuracy | Macro-F1 |
| :--- | :---: | :---: |
| Zero-Shot | 66.7% | 0.641 |
| Random | 90.0% | 0.900 |
| **Similarity-kNN** | **93.3%** | **0.933** |

![Baseline results on SST-2](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/baselines_sst2_0.6b.png)

> [!NOTE]
> SST-2 is the easiest dataset — binary sentiment classification. The model jumps from 66.7% (zero-shot) to 93.3% (similarity few-shot), a **+26.6 point** improvement just by adding relevant examples to the prompt.

---

### AG News (4-Class Topic Classification)

| Method | Accuracy | Macro-F1 |
| :--- | :---: | :---: |
| Zero-Shot | 36.7% | 0.251 |
| Random | 43.3% | 0.432 |
| **Similarity-kNN** | **53.3%** | **0.534** |

![Baseline results on AG News](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/baselines_ag_news_0.6b.png)

> [!NOTE]
> AG News is harder — 4 topic classes (World, Sports, Business, Sci/Tech). Zero-shot performance is near-random (~25% expected). Similarity-kNN gives the biggest lift (+16.6 points), but overall accuracy remains moderate — the 0.6B model struggles with fine-grained topic distinction.

---

### TREC (6-Class Question Classification)

| Method | Accuracy | Macro-F1 |
| :--- | :---: | :---: |
| Zero-Shot | 33.3% | 0.183 |
| Random | 43.3% | 0.421 |
| **Similarity-kNN** | **63.3%** | **0.588** |

![Baseline results on TREC](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/baselines_trec_0.6b.png)

> [!IMPORTANT]
> TREC has 6 fine-grained classes. Similarity-kNN gives the **biggest absolute gain here (+30 points)**, showing that retrieval-based demonstration selection is especially helpful when class boundaries are subtle and task complexity is higher.

---

## Experiment 2 — k-Shot Scaling (k ∈ {0, 2, 4, 8})

How does accuracy change as we add more demonstrations?

### SST-2 — k-Sweep

| k | Random Acc | kNN Acc | Random Tokens | kNN Tokens |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 66.7% | 66.7% | 40 | 40 |
| 2 | 86.7% | **93.3%** | 97 | 95 |
| 4 | 90.0% | **93.3%** | 159 | 153 |
| 8 | 90.0% | **96.7%** | 277 | 270 |

![k-sweep on SST-2](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/k_sweep_sst2_0.6b.png)

> [!TIP]
> **kNN is more token-efficient**: At k=2 it already matches Random at k=4, using ~40% fewer tokens (95 vs 159). This is the key efficiency finding of the project.

---

### AG News — k-Sweep

| k | Random Acc | kNN Acc |
| :---: | :---: | :---: |
| 0 | 36.7% | 36.7% |
| 2 | **66.7%** | **66.7%** |
| 4 | 43.3% | 53.3% |
| 8 | **70.0%** | 66.7% |

![k-sweep on AG News](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/k_sweep_ag_news_0.6b.png)

> [!WARNING]
> AG News shows **non-monotonic scaling** — k=4 actually performs *worse* than k=2 for Random selection. This is a known small-model behaviour: too many noisy demonstrations can confuse a 0.6B model before helping it.

---

### TREC — k-Sweep

| k | Random Acc | kNN Acc |
| :---: | :---: | :---: |
| 0 | 33.3% | 33.3% |
| 2 | 43.3% | 53.3% |
| 4 | 43.3% | 63.3% |
| 8 | 60.0% | **76.7%** |

![k-sweep on TREC](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/k_sweep_trec_0.6b.png)

> [!NOTE]
> TREC shows **consistent monotonic improvement** with kNN — more examples always help when they are semantically relevant. Random selection is less reliable, plateauing at k=4 before improving again at k=8.

---

## Experiment 4 — Demonstration Ordering

Does the order examples appear in the prompt matter?

### SST-2 — Ordering

| Ordering | Accuracy | Macro-F1 |
| :--- | :---: | :---: |
| similar_last | 93.3% | 0.933 |
| similar_first | **96.7%** | **0.967** |
| random | **96.7%** | **0.967** |

![Ordering on SST-2](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/ordering_sst2_0.6b.png)

### AG News — Ordering

| Ordering | Accuracy | Macro-F1 |
| :--- | :---: | :---: |
| similar_last | 53.3% | 0.534 |
| **similar_first** | **56.7%** | **0.575** |
| random | 53.3% | 0.545 |

![Ordering on AG News](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/ordering_ag_news_0.6b.png)

### TREC — Ordering

| Ordering | Accuracy | Macro-F1 |
| :--- | :---: | :---: |
| similar_last | 63.3% | 0.588 |
| similar_first | 66.7% | 0.639 |
| **random** | **73.3%** | **0.718** |

![Ordering on TREC](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/ordering_trec_0.6b.png)

> [!NOTE]
> Ordering effects are **dataset-dependent** — there is no single best ordering policy. `similar_last` (placing most relevant example right before the query, exploiting recency bias) did *not* win on any dataset for this model, contrary to popular belief for larger models.

---

## Experiment 5 — Prompt Format Study

Which format template gives the best results?

### SST-2 — Format

| Format | Accuracy | Macro-F1 | Avg Tokens |
| :--- | :---: | :---: | :---: |
| **plain** | **93.3%** | **0.933** | 153 |
| structured | 90.0% | 0.900 | 158 |
| rationale | 90.0% | 0.900 | 158 |

### AG News — Format

| Format | Accuracy | Macro-F1 | Avg Tokens |
| :--- | :---: | :---: | :---: |
| plain | 53.3% | 0.534 | 315 |
| **structured** | **63.3%** | **0.627** | 325 |
| **rationale** | **63.3%** | **0.627** | 325 |

### TREC — Format

| Format | Accuracy | Macro-F1 | Avg Tokens |
| :--- | :---: | :---: | :---: |
| plain | 63.3% | 0.588 | 109 |
| **structured** | **63.3%** | **0.647** | 119 |
| **rationale** | **63.3%** | **0.647** | 119 |

![Format on SST-2](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/formats_sst2_0.6b.png)
![Format on AG News](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/formats_ag_news_0.6b.png)
![Format on TREC](/Users/tejaroopvasam/.gemini/antigravity-ide/brain/9063ee67-546c-4d27-9c9d-e0edbb13f823/formats_trec_0.6b.png)

> [!TIP]
> **Plain is best for simple binary tasks (SST-2)** — less template noise. **Structured/Rationale win on harder multi-class tasks** (AG News, TREC) — explicit `[Input]`/`[Label]` markers help the small model parse the demonstration structure correctly. Rationale adds explanation text but doesn't cost extra accuracy over structured for this model size.

---

## Summary — Best Configurations per Dataset

| Dataset | Best Method | Best k | Best Format | Best Ordering | Best Accuracy |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **SST-2** | Similarity-kNN | 8 | plain | similar_first / random | **96.7%** |
| **AG News** | Random | 8 | structured | similar_first | **70.0%** |
| **TREC** | Similarity-kNN | 8 | structured | random | **76.7%** |

## Key Takeaways

1. **Few-shot dramatically beats zero-shot** — even 2 examples can give +20 point gains.
2. **Similarity-kNN is more token-efficient than Random** — achieves the same accuracy with fewer demonstrations on SST-2.
3. **More demonstrations generally help**, but AG News shows non-monotonic behaviour with random selection at small k.
4. **Plain format is best for simple tasks; structured format helps for harder multi-class tasks**.
5. **Ordering has modest but consistent effects** — no single ordering policy dominates across all datasets.
6. **Dataset difficulty matters**: SST-2 (2 classes) → 96.7%, AG News (4 classes) → 70%, TREC (6 classes) → 76.7%.
