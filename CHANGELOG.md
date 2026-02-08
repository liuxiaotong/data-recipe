# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-08

### Added
- **Pre-commit hooks** (`.pre-commit-config.yaml`): ruff lint+format, trailing-whitespace, check-yaml, check-added-large-files
- **Test coverage 83%** (up from 20%): 3100+ tests across 30+ test files
  - Full coverage for deployer, pipeline, providers, sources (web/huggingface/github)
  - Full coverage for generators (executive_summary, annotation_spec, llm_enhancer)
  - CLI integration tests for all 27+ commands

### Changed
- **CLI refactored**: monolithic `cli.py` (3728 lines) split into `cli/` package with 7 focused modules
  - `_helpers.py`, `analyze.py`, `deep.py`, `spec.py`, `batch.py`, `tools.py`, `infra.py`
- **CI coverage threshold**: raised from 15% to 80%
- **CI Python matrix**: added Python 3.13

### Deprecated
- `deep_analyzer.py` (root module): use `datarecipe.analyzers.deep_analyzer_core` instead
- `llm_analyzer.py`: will be removed in v0.4.0

### Infrastructure
- Makefile with `lint`, `format`, `test`, `cov`, `hooks`, `hooks-run` targets
- Coverage config excludes legacy/untestable modules (deep_analyzer, llm_analyzer, mcp_server, watcher)

## [0.2.0] - 2025-06-01

### Added
- **LLM Enhancement Layer** (`llm_enhancer.py`): sits between analysis and generation for higher quality outputs
  - Three modes: `interactive` (Claude Code/App), `api` (standalone), `from-json` (precomputed)
  - `get_prompt()` + `enhance_from_response()` pattern for external LLM environments
- **Task profiles & quality gates** (`task_profiles.py`, `quality_metrics.py`): unified task type registry with configurable quality thresholds
- **Multi-stage pipeline** (`pipeline.py`): phase-based assembly with dependency resolution
- **Sample data generation** (`09_样例数据/`): Think-Po style realistic seed data
- **AI Agent friendly layer** (`08_AI_Agent/`): structured API surface for Claude Code/App integration
- **9 MCP tools** for Claude Desktop integration (expanded from 4)
- **CI/CD** with GitHub Actions: ruff linting, pytest, multi-Python matrix (3.10-3.12)

### Changed
- Renamed PyPI package from `datarecipe` to `knowlyr-datarecipe`
- CLI entry point renamed to `knowlyr-datarecipe`
- Output files organized into 9 categorized subdirectories (23+ files)
- DATA_SCHEMA uses actual fields instead of hardcoded templates

### Fixed
- Field generation defects and document quality improvements
- Italic Chinese text rendering issues
- MCP tool attribute reference errors
- CLI uses shared `DeepAnalyzerCore`, fills empty summary fields

## [0.1.0] - 2025-03-01

### Added
- **Core analysis engine**: `deep-analyze` command for HuggingFace dataset reverse engineering
- **Spec analyzer**: `analyze-spec` command for requirement document analysis
- **Executive summary generator** with value assessment and ROI analysis
- **Annotation spec generator** for production annotation guidelines
- **Milestone plan generator** with team composition and risk management
- **Industry benchmark comparison** with dataset catalog
- **Cost estimation engine** with human-machine work split analysis
- **Rubrics extraction** and pattern analysis from dataset samples
- **Analysis caching** system (`AnalysisCache`) with HuggingFace freshness checks
- **Multiple data sources**: HuggingFace Hub, GitHub repos, web URLs, PDF/Word documents
- **Knowledge base**: dataset catalog with similarity search
- **Radar integration** for external reporting
- **Production deployer** with provider plugin system
- **Annotator profiler** for team sizing

### Infrastructure
- Python 3.10+ with optional dependency groups (`[llm]`, `[pdf]`, `[quality]`, `[workflow]`, `[mcp]`)
- Rich console UI for interactive output
- Click-based CLI with comprehensive help

## [0.0.1] - 2025-01-01

### Added
- Initial release: basic dataset analysis with Markdown output
- HuggingFace dataset link parsing
- Chinese language output support
- GitHub and web URL source support
