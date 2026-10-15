PYTHON := .venv/bin/python

.PHONY: help install data profile test exp1 exp2 exp4 exp5 proposed plot clean

help:
	@echo "Available commands (similar to npm scripts):"
	@echo "  make install     - Install all required Python packages into .venv"
	@echo "  make data        - Download and cache all 4 benchmark datasets"
	@echo "  make profile     - Profile sequence length and word counts across datasets"
	@echo "  make test        - Run the complete unit test suite"
	@echo "  make exp1        - Run Experiment 1: Baselines (Zero-Shot vs Random vs Similarity)"
	@echo "  make exp2        - Run Experiment 2: Demonstration count (k-sweep)"
	@echo "  make exp4        - Run Experiment 4: Demonstration ordering study"
	@echo "  make exp5        - Run Experiment 5: Demonstration formatting study"
	@echo "  make proposed    - Run Proposed Method: Budget-aware selection on IMDb"
	@echo "  make plot        - Generate publication-ready figures for all results in results/"
	@echo "  make clean       - Remove cached embeddings and Python bytecode"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) data/download_and_verify.py --dataset all

profile:
	$(PYTHON) data/token_profiler.py

test:
	$(PYTHON) -m unittest discover tests

exp1:
	$(PYTHON) experiments/run_experiment.py --experiment exp1_baselines --dataset sst2 --k 4 --n_eval 100

exp2:
	$(PYTHON) experiments/run_experiment.py --experiment exp2_k_sweep --dataset sst2 --n_eval 100

exp4:
	$(PYTHON) experiments/run_experiment.py --experiment exp4_ordering --dataset sst2 --k 4 --n_eval 100

exp5:
	$(PYTHON) experiments/run_experiment.py --experiment exp5_formatting --dataset sst2 --k 4 --n_eval 100

proposed:
	$(PYTHON) experiments/run_experiment.py --experiment proposed_method --dataset imdb --token_budget 1500 --k 4 --n_eval 100

plot:
	$(PYTHON) src/visualization/plot_results.py --results_dir results --output_dir results/figures

clean:
	rm -rf data/cache/embeddings/*.npy
	find . -type d -name "__pycache__" -exec rm -rf {} +
