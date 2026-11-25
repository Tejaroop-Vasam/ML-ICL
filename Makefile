PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else command -v python3; fi)

.PHONY: help install install_review profile oracle complex qualitative plot test

help:
	@echo "make install         Install laptop plotting dependencies"
	@echo "make install_review  Install reviewer runner dependencies on a GPU runtime"
	@echo "make profile         Count exact prompt tokens without running generation"
	@echo "make oracle          Run the classification experiments"
	@echo "make complex         Run Spider, BIRD, and CLAPNQ"
	@echo "make qualitative     Export annotations from saved checkpoints"
	@echo "make plot            Plot and re-score the saved run ZIPs"
	@echo "make test            Run local unit tests"

install:
	$(PYTHON) -m pip install -r requirements.txt

install_review:
	$(PYTHON) -m pip install -r requirements-review.txt

profile:
	$(PYTHON) -m src.review_experiments run --profile-only

oracle:
	$(PYTHON) scripts/run_oracle_study.py

complex:
	$(PYTHON) scripts/run_complex_tasks_study.py

qualitative:
	$(PYTHON) scripts/qualitative_error_analysis.py

plot:
	$(PYTHON) scripts/plot_reviewer_results.py Qwen_*.zip

test:
	$(PYTHON) -m unittest discover tests
