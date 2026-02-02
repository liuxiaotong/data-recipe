# DataRecipe - AI Dataset Analysis Tool

## Project Overview

DataRecipe is a CLI tool for analyzing AI training datasets. It provides:
- Dataset "nutrition label" analysis (provenance, methods, costs)
- Annotator profiling (team requirements, skills, labor costs)
- Production deployment (annotation guides, quality rules, timelines)

## Quick Start

```bash
# Install
uv pip install -e .

# Analyze a dataset
uv run datarecipe analyze <dataset_id>

# Generate annotator profile
uv run datarecipe profile <dataset_id> --region china

# Deploy annotation project
uv run datarecipe deploy <dataset_id>
```

## Available Slash Commands

- `/datarecipe <args>` - General DataRecipe operations
- `/analyze-dataset <dataset_id>` - Quick dataset analysis
- `/profile-annotators <dataset_id>` - Generate annotator requirements
- `/deploy-project <dataset_id>` - Generate full annotation project

## Common Tasks

### Analyze a HuggingFace Dataset
```bash
uv run datarecipe analyze Anthropic/hh-rlhf
uv run datarecipe analyze AI-MO/NuminaMath-CoT
uv run datarecipe analyze nguha/legalbench
```

### Generate Annotator Profile with Regional Costs
```bash
uv run datarecipe profile <dataset> --region china   # Chinese labor rates
uv run datarecipe profile <dataset> --region us      # US labor rates
uv run datarecipe profile <dataset> --region europe  # European rates
```

### Deploy Full Annotation Project
```bash
uv run datarecipe deploy <dataset>                    # Default: ./projects/<name>/
uv run datarecipe deploy <dataset> -o ./my_project   # Custom output
```

## Project Structure

```
src/datarecipe/
├── cli.py              # CLI commands
├── analyzer.py         # Dataset analysis
├── profiler.py         # Annotator profiling
├── deployer.py         # Production deployment
├── cost_calculator.py  # Cost estimation
├── schema.py           # Data models
└── providers/          # Deployment providers
    ├── __init__.py     # Plugin system
    └── local.py        # Local file provider
```

## High-Value Dataset Examples

| Dataset | Domain | Hourly Rate | Per-Example Cost |
|---------|--------|-------------|------------------|
| nguha/legalbench | Legal | $105 | $44.10 |
| openlifescienceai/MedMCQA | Medical | $105 | $34.65 |
| AI-MO/NuminaMath-CoT | Math Olympiad | $47.50 | $15.68 |
| Anthropic/hh-rlhf | RLHF | $27.50 | $2.20 |
