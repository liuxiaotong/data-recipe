# Contributing to DataRecipe

## Development Setup

```bash
# Clone and install
git clone https://github.com/liuxiaotong/data-recipe.git
cd data-recipe
python -m venv .venv
source .venv/bin/activate
make dev          # Install with all dependencies

# Install pre-commit hooks
make hooks
```

## Development Workflow

```bash
make lint         # Run ruff linter
make format       # Auto-format code
make test         # Run tests (3294+ tests)
make cov          # Run tests with coverage (96%+, minimum 80%)
make typecheck    # Run mypy type checking
make ci           # Run full CI pipeline locally
```

## Testing

- Use `unittest.TestCase` style
- Mock all external dependencies (LLM APIs, network requests, file I/O)
- Place tests in `tests/test_<module>.py`
- Aim for 90%+ coverage on new code

```bash
# Run specific test file
pytest tests/test_analyzer.py -v

# Run with coverage for a specific module
pytest tests/ --cov=datarecipe.analyzer --cov-report=term-missing
```

## Code Style

- **Formatter**: ruff (line-length 100)
- **Target**: Python 3.10+ (`X | None` instead of `Optional[X]`)
- **Imports**: sorted by ruff (`I` rule)
- Pre-commit hooks enforce formatting on every commit

## Project Structure

```
src/datarecipe/
├── cli/                # CLI commands (7 modules)
├── core/               # Deep analysis engine
├── analyzers/          # Spec + LLM dataset analyzers
├── generators/         # Document generators (markdown/JSON)
├── extractors/         # Rubrics + prompt extraction
├── parsers/            # PDF/Word/image parsing
├── cost/               # Cost estimation models
├── knowledge/          # Knowledge base + dataset catalog
├── sources/            # Data sources (HuggingFace, GitHub, web)
├── providers/          # Deployment providers
├── constants.py        # Shared constants
└── schema.py           # Data models
```

## Commit Messages

Use conventional commit format in Chinese or English:

```
feat: 新增功能描述
fix: 修复问题描述
test: 测试相关
docs: 文档更新
chore: 构建/工具链
refactor: 重构
```

## Releasing

Releases are automated via GitHub Actions. To release:

1. Update version in `pyproject.toml`, `src/datarecipe/__init__.py`, `src/datarecipe/cli/__init__.py`
2. Update `CHANGELOG.md`
3. Commit and tag: `git tag -a v0.X.Y -m "v0.X.Y"`
4. Push: `git push origin main --tags`
5. GitHub Actions will auto-publish to PyPI and create a GitHub Release
