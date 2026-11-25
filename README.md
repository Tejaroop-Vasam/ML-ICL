# In-Context Learning with Qwen3

This project evaluates how BGE-M3 retrieval, response format, prompt length, and evidence placement affect Qwen3. The [CS725 report](output/pdf/CS725_Project_Report.pdf) uses saved runs for Qwen3-1.7B, 4B, and 8B.

| Evaluation | Datasets | Comparisons |
|---|---|---|
| Classification | IMDb, Yelp Polarity, TREC, Yahoo Answers | Zero-shot prompts, random and retrieved demonstrations, shot counts, and IMDb output-format and calibration controls |
| Text-to-SQL | Spider, BIRD | Full schema, random tables, BGE-M3 tables, and reference-derived tables, scored by execution |
| Long-answer QA | CLAPNQ | Random, retrieved, and annotated passages, plus a sweep over prompt lengths and evidence positions |

## Run the experiments

Open the [Colab notebook](ML_ICL_Reviewer_Experiments.ipynb) on a CUDA runtime and set the repository ref and model in its configuration cell. In a CUDA terminal with Python 3.11 or newer, the equivalent commands are:

```bash
make install_review
python -m src.review_experiments run --model Qwen/Qwen3-8B
```

The [experiment guide](docs/reviewer_experiments.md) covers dataset selection, prompt profiling, run directories, and checkpoint exports.
The [source code guide](src/README.md) explains each file involved in running and scoring the experiments.

## View saved results locally

```bash
make install
make plot
```

Open [the chart index](results/reviewer_plots/index.html) after plotting. The plotting tool reads the saved ZIPs, re-scores classification responses, and produces per-model and matched cross-model comparisons. Run `make test` for the local unit tests.

The [earlier project results](docs.md) document the original evaluation. The [editable LaTeX report](output/latex/CS725_Project_Report.tex) and [Overleaf bundle](output/latex/CS725_Project_Report_Overleaf.zip) cover the updated study.
