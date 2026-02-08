.PHONY: help install dev test lint format typecheck coverage clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

dev: ## Install with all dev dependencies
	pip install -e ".[all,dev]"

test: ## Run tests
	pytest tests/ -q

lint: ## Run linter (ruff)
	ruff check src/ tests/

format: ## Auto-format code (ruff)
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck: ## Run type checker (mypy)
	mypy src/datarecipe --ignore-missing-imports

coverage: ## Run tests with coverage report
	pytest tests/ --cov=datarecipe --cov-report=term-missing

ci: lint typecheck test ## Run full CI pipeline locally
