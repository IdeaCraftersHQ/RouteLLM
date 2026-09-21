.PHONY: help install build test lint fmt check clean install-hooks

PYTHON ?= python3
VENV   ?= .venv

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create the venv and install the package with dev extras
	@command -v uv >/dev/null 2>&1 || { echo "uv not installed. Run: brew install uv"; exit 1; }
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(VENV)/bin/python -e ".[dev,serve]"
	@echo "Installed. Activate with: source $(VENV)/bin/activate"

build: ## Build the sdist and wheel into dist/
	@command -v uv >/dev/null 2>&1 || { echo "uv not installed. Run: brew install uv"; exit 1; }
	uv build

test: ## Run the test suite
	@command -v uv >/dev/null 2>&1 || { echo "uv not installed. Run: brew install uv"; exit 1; }
	uv run --extra dev pytest routellm/tests -q

lint: ## Report lint findings without changing anything
	@command -v uv >/dev/null 2>&1 || { echo "uv not installed. Run: brew install uv"; exit 1; }
	uvx ruff check routellm
	uvx ruff format --check routellm

fmt: ## Format and apply ruff's safe fixes
	@command -v uv >/dev/null 2>&1 || { echo "uv not installed. Run: brew install uv"; exit 1; }
	uvx ruff format routellm
	uvx ruff check --fix routellm

check: lint test ## Run lint then the test suite

clean: ## Remove build artefacts and caches
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

install-hooks: ## Install pre-commit git hooks
	@command -v pre-commit >/dev/null 2>&1 || { echo "pre-commit not installed. Run: brew install pre-commit"; exit 1; }
	@if [ -n "$$(git config --get core.hooksPath)" ]; then \
		echo "core.hooksPath is set to $$(git config --get core.hooksPath), which shadows"; \
		echo "this repo's hooks. Install with: pre-commit install --hooks-path $$(git config --get core.hooksPath)"; \
		echo "or unset it for this repo: git config --unset core.hooksPath"; \
		exit 1; \
	fi
	pre-commit install
	@echo "Hooks installed. They run automatically on git commit."
