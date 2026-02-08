.PHONY: help install dev test lint format typecheck coverage cov clean build publish hooks hooks-run

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

cov: ## Run tests with coverage (fail under 80%)
	pytest tests/ --cov=datarecipe --cov-report=term-missing --cov-fail-under=80

ci: lint typecheck test ## Run full CI pipeline locally

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean ## Build wheel and sdist
	python -m build

publish: build ## Publish to PyPI
	twine upload dist/*

hooks: ## Install pre-commit hooks
	pre-commit install

hooks-run: ## Run all pre-commit hooks on all files
	pre-commit run --all-files
