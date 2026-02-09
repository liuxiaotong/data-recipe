# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-09

### Added
- 数据集相似度对比：SimilarityWeights/SimilarityBreakdown/SimilarityResult 评分体系
- 逐字段 FieldDiff 对比（12 个 Recipe 字段）
- SchemaComparison 列名/类型重叠分析（Jaccard 系数）
- compare 命令 --similarity 和 --schema 选项
- 本地文件分析支持：CSV、Parquet、JSONL
- LocalFileExtractor + detect_format() + 数据集类型自动检测
- analyze/deep-analyze/quality 命令均支持本地文件路径
- QualityAnalyzer.analyze_from_file() 方法

### Changed
- 迁移 deep_analyzer.py → analyzers/url_analyzer.py
- 迁移 llm_analyzer.py → analyzers/llm_url_analyzer.py
- 旧模块保留为 __getattr__ 兼容 shim（v0.5.0 移除）
- 消除所有 DeprecationWarning（3478 tests 零警告）

### Documentation
- 统一 knowlyr 生态表格格式
- 同步 README 副标题与 GitHub about
- 添加 GitHub Topics 行
- 统一尾部品牌标语

## [0.3.3] - 2026-02-08

### Changed
- 并行化分析器和生成器
- HuggingFace 元数据 TTL 缓存

## [0.3.2] - 2026-02-08

### Changed
- 测试覆盖率 97%
- CI 自动发布
- 添加 CONTRIBUTING.md

## [0.3.1] - 2026-02-08

### Changed
- 覆盖率 96%
- py.typed marker
- target-version 修复

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
