# Source code guide

The code in `src/` builds prompts for the Qwen3 experiments, selects demonstrations or evidence with BGE-M3, scores responses, and saves the results. See the [experiment guide](../docs/reviewer_experiments.md) for setup and command-line options.

## Shared modules

| File | What it does |
| --- | --- |
| [`label_parsing.py`](label_parsing.py) | Defines the labels for each classification dataset. Scores a response by checking whether it mentions exactly one distinct candidate label and whether that label matches the reference. Also re-scores saved responses and preserves predictions based on label likelihoods. |
| [`retrieval.py`](retrieval.py) | Loads BGE-M3, converts text into normalized dense embeddings, and ranks candidates by cosine similarity. Measures input lengths with the BGE-M3 tokenizer so callers can track the retriever’s input limit. |

## Experiment package

| File | What it does |
| --- | --- |
| [`review_experiments/__init__.py`](review_experiments/__init__.py) | Marks `review_experiments` as a Python package and describes its purpose. |
| [`review_experiments/__main__.py`](review_experiments/__main__.py) | Defines the `run` and `export` commands. The `run` command reads CLI options, initializes BGE-M3, and starts each selected Qwen3 model. The `export` command packages an existing checkpoint. |
| [`review_experiments/config.py`](review_experiments/config.py) | Holds defaults for datasets, sample counts, random seeds, demonstration counts, output token budgets, and context stress-test lengths. Validates experiment settings before the run starts. |
| [`review_experiments/common.py`](review_experiments/common.py) | Defines `Job`, the record each task uses to pass a prompt, reference answer, token budget, and metadata to the runner. Also provides shared sampling, text normalization, hashing, and Hugging Face revision lookup functions. |
| [`review_experiments/classification.py`](review_experiments/classification.py) | Samples IMDb, Yelp Polarity, TREC, and Yahoo Answers and builds their classification prompts. Creates zero-shot conditions and comparisons across random, BGE-M3, and label-balanced demonstrations, plus IMDb controls for prompt format, output length, and review slices. |
| [`review_experiments/text_to_sql.py`](review_experiments/text_to_sql.py) | Prepares Spider and BIRD data and reads their SQLite schemas. Builds prompts using full schemas, random tables, BGE-M3-selected tables, or oracle tables identified from the reference SQL. Scores queries by comparing generated and reference results from the same read-only database. |
| [`review_experiments/long_answer.py`](review_experiments/long_answer.py) | Prepares CLAPNQ questions and builds prompts with no passage, a random passage, a BGE-M3-selected passage, or the annotated gold passage. Adds distractor passages to test prompt length and evidence position. Scores answers against the references with ROUGE-L and ROUGE-Lsum. |
| [`review_experiments/runner.py`](review_experiments/runner.py) | Renders raw or chat prompts, counts tokens with the Qwen3 tokenizer, and checks that each prompt and its reserved output fit within the configured limits. Runs generation or IMDb label-likelihood scoring, sends responses to the task scorer, and saves each trial. |
| [`review_experiments/artifacts.py`](review_experiments/artifacts.py) | Stores trials and rendered prompts in a resumable SQLite checkpoint. Exports the run manifest, trial records, compressed prompts, and annotation files into a ZIP alongside the checkpoint. |

## How a run works

1. Run `python -m src.review_experiments run` from the repository root. The CLI loads the settings, and the runner calls the task module for each selected dataset.
2. Task modules use BGE-M3 where a condition requires retrieval and yield one `Job` per evaluation item and condition.
3. The runner records actual Qwen3 prompt lengths, evaluates eligible jobs, and checkpoints the results. Use `--profile-only` to measure prompts without loading Qwen3 model weights.
4. The CLI exports a ZIP for each run by default. The [plotting script](../scripts/plot_reviewer_results.py) reads the saved results to produce charts.
