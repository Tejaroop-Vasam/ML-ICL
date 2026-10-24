PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else which python3 2>/dev/null || echo python; fi)

# Configurable defaults (override with `make run_all MODELS="0.6b 1.7b 4b 8b" N_EVAL=100`)
MODELS   ?= 0.6b 1.7b
DATASETS ?= yelp_polarity yahoo_answers trec
N_EVAL   ?= 30
K        ?= 4

.PHONY: help install run_all run_all_models npz profile test exp1 exp2 exp4 exp5 proposed plot clean

help:
	@echo "Available commands:"
	@echo "  make install         - Install all required Python packages"
	@echo "  make run_all         - Run experiments on default models (0.6b, 1.7b) and datasets"
	@echo "  make run_all_models  - Run ALL experiments on ALL models (0.6b, 1.7b, 4b, 8b) and ALL datasets"
	@echo "                         Usage: make run_all_models N_EVAL=30 K=4"
	@echo "  make npz             - Convert and cache all datasets into compressed NumPy .npz"
	@echo "  make profile         - Profile sequence length and word counts across datasets"
	@echo "  make test            - Run the complete unit test suite"
	@echo "  make exp1            - Run Experiment 1: Baselines (Zero-Shot vs Random vs Similarity)"
	@echo "  make exp2            - Run Experiment 2: Demonstration count (k-sweep)"
	@echo "  make exp4            - Run Experiment 4: Demonstration ordering study"
	@echo "  make exp5            - Run Experiment 5: Demonstration formatting study"
	@echo "  make proposed        - Run Proposed Method: Budget-aware selection on IMDb"
	@echo "  make plot            - Generate publication-ready figures for all results in results/"
	@echo "  make clean           - Remove cached embeddings and Python bytecode"

install:
	$(PYTHON) -m pip install -r requirements.txt

run_all:
	$(PYTHON) scripts/run_all_experiments.py --models $(MODELS) --datasets $(DATASETS) --n_eval $(N_EVAL) --k $(K)

run_all_models:
	$(PYTHON) scripts/run_all_experiments.py --models 0.6b 1.7b 4b 8b --datasets yelp_polarity yahoo_answers trec imdb --n_eval $(N_EVAL) --k $(K)

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

plot:
	$(PYTHON) src/visualization/plot_results.py --results_dir results --output_dir results/figures

clean:
	rm -rf data/cache/embeddings/*.npy
	find . -type d -name "__pycache__" -exec rm -rf {} +
