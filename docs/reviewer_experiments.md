# Qwen3 reviewer experiments

Run the experiments from the repository root with Python 3.11 or newer and CUDA-enabled PyTorch.

```bash
python -m pip install -r requirements-review.txt
python -m src.review_experiments run
```

The launcher notebook clones the branch or tag that contains this implementation. You can also run the same CLI in any CUDA-enabled terminal. Defaults use a small Qwen3-8B pilot with 4-bit NF4 weights.

```bash
python -m src.review_experiments run --help
python -m src.review_experiments run --datasets imdb spider --samples imdb=300 --samples spider=100
python -m src.review_experiments run --datasets clapnq --profile-only
python -m src.review_experiments export PATH_TO_RUN_DIRECTORY
```

The run manifest records the BGE-M3 checkpoint and effective token limit. Use the same repository revision and run configuration across Qwen3 models. The checked-in historical figures refer to earlier experiments.

Use `--samples DATASET=all` for the full eligible split. IMDb and the other classification pilots balance the labels. The SQL pilot samples across databases. CLAPNQ uses answerable dev items with at least one reference of 50 words or more by default. Use `--random-seeds 42 101 2024` for repeated random demonstrations.

Spider and BIRD download their dev releases and SQLite databases automatically. If an official download fails, supply the official release ZIP or extracted directory with `--spider-path` or `--bird-dev-path`. Spider uses official training examples for few-shot SQL. BIRD adds its few-shot arm when you also provide the official train release with `--bird-train-path`.

The runner commits completed trials to `checkpoint.sqlite` as it goes. Rerun the same configuration with the same GPU type to resume. Lower `--max-prompt-tokens` to fit a smaller GPU. Use `--skip-export` if you only need the checkpoint, or use `export` to package an existing checkpoint.

## Classification and IMDb diagnostics

- Rebuilds the project's plain prompt and its original extraction route, including the first-label fallback. Every generation is scored by the original parser, a strict exact-label parser, and a unique-label-mention diagnostic. All three consume the same response.
- Pairs raw continuation with Qwen3's official non-thinking chat template using the same instruction. It also tests the effect of asking for the label alone and, on IMDb, increasing the raw generation budget from 16 to 64 tokens. It records output cap hits, ambiguity, invalid outputs, and fallback.
- IMDb also compares teacher-forced label likelihoods, reversal of the order in the instruction, and prior adjustment using content-free inputs. Head and tail slices probe sensitivity to where review evidence appears. Those slices keep the full review's label as a reference for the intervention.
- Compares random, BGE-M3, and training-label-balanced BGE-M3 demonstrations at a fixed shot count. The BGE-M3 sweep uses the project's 2, 4, and 8-shot settings. IMDb additionally changes order and prompt format on the same chosen shots. IMDb's same-label oracle deliberately uses the test label to pick demonstrations and serves as a label-informed diagnostic.
- Logs BGE-M3 tokenizer lengths separately from Qwen3 lengths. Interpret comparisons on the paired example IDs. BGE-M3 uses normalized dense embeddings, cosine similarity, and a limit of 8,192 tokens per input. [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) · [Contextual calibration](https://proceedings.mlr.press/v139/zhao21c.html)

## Spider 1.0 and BIRD text-to-SQL

Qwen3 generates SQL against official dev schemas. The held-out reference SQL is stored for scoring. The oracle prompt exposes only the tables referenced by the gold query. Retrieved and random schemas receive the same number of tables when that count can be recovered, so these are controlled comparisons at a gold-informed table count. Full schema is the deployable zero-shot baseline. BIRD also tests removing its benchmark-provided domain evidence while keeping the full schema. The SQL few-shot arm uses BGE-M3 to retrieve examples from official training databases outside the dev set.

The runner executes one read-only SQL statement per prediction and compares its results to the reference. It records parse and execution errors, result shape, and schema coverage. Spider's single-database execution check preserves row order when the reference requests it and preserves multiplicity otherwise. Spider's official test-suite metric uses additional database instances. BIRD uses its row-set equality rule with local query timeout and row limits. Treat gold-query errors separately when evaluating accuracy. [Spider 1.0](https://yale-lily.github.io/spider) · [Spider test-suite evaluator](https://github.com/taoyds/test-suite-sql-eval) · [BIRD benchmark](https://bird-bench.github.io/) · [BIRD execution scorer](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py)

## Long-answer QA and long-context stress

CLAPNQ supplies natural questions, annotated multi-sentence references, gold passages, and supporting sentences. The closed-book, BGE-M3, random-passage, and supplied-gold-passage arms are paired by question. Retrieval uses a controlled shortlist containing the gold passage and seeded train distractors. The ICL comparisons keep their demonstration set fixed across retrieved and supplied-gold conditions.

The stress experiment adds real distractor passages to the same gold passage. Within each target length it holds the passage set and question fixed and moves the gold passage to the front, middle, or end. It fits whole passages up to approximately 2K, 8K, 16K, and 30K tokens, subject to the configured cap. Use the measured prompt length and evidence token position for analysis. Lexical overlap is recorded per reference for later review alongside human ratings of answer coverage and faithfulness. [CLAPNQ paper](https://aclanthology.org/2025.tacl-1.3/) · [CLAPNQ data](https://github.com/primeqa/clapnq/tree/main/annotated_data) · [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)

## Exact prompt lengths and GPU runner

The same Qwen3 input IDs are counted and passed into inference. Each trial stores prompt tokens, reserved generation tokens, native-window percentage, remaining headroom, actual generated tokens, stop reason, and a prompt hash. Jobs exceeding the configured prompt limit or the native 32,768-token budget are recorded with their status. Long stress prompts keep their gold evidence intact. [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)

Generation is greedy with Qwen3 thinking disabled in the chat arms. The legacy raw arms use the project's single tokenizer EOS. The runner records model revision, quantization, runtime versions, data revisions or checksums, and retrieval details for later interpretation.

## Export evidence for later review

The ZIP contains the raw `trials.jsonl` ledger, exact rendered prompts in `prompts.jsonl.gz`, run manifest, checkpoint, and annotation files. Join a trial to its prompt using `prompt_sha256`. A null `raw_output` on a score-only arm indicates a model likelihood comparison. Keep outcome-bearing `annotation_key.jsonl` separate from the blinded packets.

The annotation export draws a seeded random cohort and a separate targeted cohort of disagreements and diagnostic outcomes. Two reviewers can independently fill the annotation template before reconciliation.

1. **Verdict pass.** Read each case's input and response, then compare with the reference. Record `correct`, `partly_correct`, `incorrect`, `invalid`, or `uncertain`. For QA, give coverage 0 for missing, 1 for partial, or 2 for all requested parts.
2. **Evidence pass.** Use the prompt ledger to check what was available to the model. For QA, give faithfulness 0 for contradicted or unsupported main claims, 1 for mixed support, or 2 for claims supported by the supplied evidence. Leave it blank when no passage was supplied. Record a supporting input and output excerpt, a confidence rating, and a short rationale.
3. **Error codes.** For IMDb and other classification tasks, use `format_or_parser`, `ambiguous_output`, `plot_vs_review_stance`, `mixed_sentiment_or_negation`, or `wrong_class` when observable. For SQL, use `invalid_sql`, `missing_schema`, `wrong_table_or_column`, `join_or_aggregation`, or `wrong_predicate_or_value`. For QA, use `missing_evidence`, `missed_answer_part`, `unsupported_claim`, `contradiction`, or `inappropriate_abstention`. Across tasks use `output_cutoff`, `uncertain`, `other`, or `none` as appropriate. Treat parser and evidence availability as separately recorded observations when comparing matched variants.
