PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else which python3 2>/dev/null || echo python; fi)

# Configurable defaults (override with `make run_all MODELS="0.6b 1.7b 4b 8b" N_EVAL=100`)
MODELS   ?= 0.6b 1.7b
DATASETS ?= imdb
N_EVAL   ?= 30
K        ?= 4

.PHONY: help install install_review run_all run_all_models run_0.6b run_1.7b run_4b run_8b npz profile test exp1 exp2 exp4 exp5 proposed plot oracle complex qualitative report ppt review_artifacts clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install all required Python packages"
	@echo "  make install_review  - Install the reviewer runner dependencies on a GPU runtime"
	@echo "  make run_0.6b        - Run all experiments on Qwen3-0.6B"
	@echo "  make run_1.7b        - Run all experiments on Qwen3-1.7B"
	@echo "  make run_4b          - Run all experiments on Qwen3-4B"
	@echo "  make run_8b          - Run all experiments on Qwen3-8B"
	@echo "  make run_all         - Run experiments on default models (0.6b, 1.7b) and datasets"
	@echo "  make run_all_models  - Run ALL experiments on ALL models (0.6b, 1.7b, 4b, 8b) and ALL datasets"
	@echo "                         Usage: make run_all_models N_EVAL=30 K=4"
	@echo "  make npz             - Convert and cache all datasets into compressed NumPy .npz"
	@echo "  make profile         - Profile prompt token length vs Qwen3 32K context window"
	@echo "  make oracle          - Run classification retrieval controls with BGE-M3"
	@echo "  make complex         - Run Spider, BIRD, and CLAPNQ with BGE-M3"
	@echo "  make qualitative     - Export annotations from saved reviewer runs"
	@echo "  make test            - Run the complete unit test suite"
	@echo "  make exp1            - Run Experiment 1: Baselines (Zero-Shot vs Random vs Similarity)"
	@echo "  make exp2            - Run Experiment 2: Demonstration count (k-sweep)"
	@echo "  make exp4            - Run Experiment 4: Demonstration ordering study"
	@echo "  make exp5            - Run Experiment 5: Demonstration formatting study"
	@echo "  make proposed        - Run Proposed Method: Budget-aware selection on IMDb"
	@echo "  make plot            - Plot saved BGE-M3 classification results"
	@echo "  make ppt             - Rebuild the archived presentation"
	@echo "  make report          - Rebuild the archived Word report"
	@echo "  make review_artifacts- Rebuild archived figures and reports"
	@echo "  make clean           - Remove cached embeddings and Python bytecode"

install:
	$(PYTHON) -m pip install -r requirements.txt

install_review:
	$(PYTHON) -m pip install -r requirements-review.txt

run_all:
	$(PYTHON) scripts/run_all_experiments.py --models $(MODELS) --datasets $(DATASETS) --n_eval $(N_EVAL) --k $(K)

run_all_models:
	$(PYTHON) scripts/run_all_experiments.py --models 0.6b 1.7b 4b 8b --datasets yelp_polarity yahoo_answers trec imdb --n_eval $(N_EVAL) --k $(K)

# Model-by-model targets
run_0.6b:
	$(PYTHON) scripts/run_all_experiments.py --models 0.6b --datasets $(DATASETS) --n_eval $(N_EVAL) --k $(K)

run_1.7b:
	$(PYTHON) scripts/run_all_experiments.py --models 1.7b --datasets $(DATASETS) --n_eval $(N_EVAL) --k $(K)

run_4b:
	$(PYTHON) scripts/run_all_experiments.py --models 4b --datasets $(DATASETS) --n_eval $(N_EVAL) --k $(K)

run_8b:
	$(PYTHON) scripts/run_all_experiments.py --models 8b --datasets $(DATASETS) --n_eval $(N_EVAL) --k $(K)

npz:
	$(PYTHON) scripts/export_to_npz.py

profile:
	$(PYTHON) scripts/token_profiler.py

test:
	$(PYTHON) -m unittest discover tests

exp1:
	$(PYTHON) experiments/run_experiment.py --experiment exp1_baselines --dataset yelp_polarity --k 4 --n_eval 100

exp2:
	$(PYTHON) experiments/run_experiment.py --experiment exp2_k_sweep --dataset yelp_polarity --n_eval 100

exp4:
	$(PYTHON) experiments/run_experiment.py --experiment exp4_ordering --dataset yelp_polarity --k 4 --n_eval 100

exp5:
	$(PYTHON) experiments/run_experiment.py --experiment exp5_formatting --dataset yelp_polarity --k 4 --n_eval 100

proposed:
	$(PYTHON) experiments/run_experiment.py --experiment proposed_method --dataset imdb --token_budget 1500 --k 4 --n_eval 100

oracle:
	$(PYTHON) scripts/run_oracle_study.py

complex:
	$(PYTHON) scripts/run_complex_tasks_study.py

qualitative:
	$(PYTHON) scripts/qualitative_error_analysis.py

plot:
	$(PYTHON) scripts/run_all_experiments.py --plot_only

ppt:
	$(PYTHON) scripts/generate_ppt.py

report:
	$(PYTHON) scripts/generate_word_report.py

review_artifacts:
	$(PYTHON) scripts/run_all_experiments.py --plot_only --out_dir results
	$(PYTHON) scripts/plot_summary.py
	$(PYTHON) scripts/plot_comprehensive_comparisons.py
	$(MAKE) ppt report

clean:
	rm -rf data/cache/embeddings/*.npy
	find . -type d -name "__pycache__" -exec rm -rf {} +
