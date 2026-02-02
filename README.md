<div align="center">

# DataRecipe

**Reverse-engineer AI dataset construction pipelines**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](https://github.com/liuxiaotong/data-recipe)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## What is DataRecipe?

DataRecipe is a **"nutrition label" system for AI datasets**. Just as food products list their ingredients, AI datasets should disclose their construction methods, costs, and sources.

```
Dataset → DataRecipe → Recipe (provenance, methods, costs, reproducibility score)
```

**The Problem**: Model architectures are increasingly transparent, but dataset construction remains opaque. Researchers can't reproduce datasets, assess inherited biases, or estimate creation costs.

**The Solution**: DataRecipe analyzes datasets from HuggingFace, GitHub, and web sources to extract provenance information and generate actionable reproduction guides.

---

## Features

| Feature | Description | Command |
|---------|-------------|---------|
| **Analyze** | Extract metadata, detect teacher models, classify generation methods | `datarecipe analyze` |
| **Cost Estimate** | Calculate API, annotation, and compute costs | `datarecipe cost` |
| **Quality Metrics** | Diversity, consistency, complexity, AI detection | `datarecipe quality` |
| **Deep Analysis** | Parse papers, auto-discover arXiv, multi-source aggregation | `datarecipe deep-guide` |
| **Batch Process** | Parallel analysis of multiple datasets | `datarecipe batch` |
| **Compare** | Side-by-side dataset comparison with recommendations | `datarecipe compare` |
| **Workflow** | Generate complete reproduction projects with scripts | `datarecipe workflow` |
| **Annotator Profile** | Generate team requirements, skills, and labor cost estimates | `datarecipe profile` |
| **Production Deploy** | Create deployment packages with guidelines and quality rules | `datarecipe deploy` |
| **Provider System** | Pluggable deployment providers (local, custom integrations) | `datarecipe providers` |

---

## Installation

```bash
pip install datarecipe
```

**Optional dependencies:**

```bash
pip install datarecipe[pdf]      # PDF parsing
pip install datarecipe[llm]      # LLM-enhanced analysis
pip install datarecipe[quality]  # Quality metrics with embeddings
pip install datarecipe[all]      # Everything
```

**From source:**

```bash
git clone https://github.com/liuxiaotong/data-recipe.git
cd data-recipe
pip install -e .
```

---

## Quick Start

### Analyze a Dataset

```bash
# Basic analysis
datarecipe analyze Anthropic/hh-rlhf

# Export as YAML/JSON/Markdown
datarecipe analyze Anthropic/hh-rlhf -o report.md
datarecipe analyze Anthropic/hh-rlhf --yaml
datarecipe analyze Anthropic/hh-rlhf --json
```

### Estimate Production Cost

```bash
datarecipe cost Anthropic/hh-rlhf --model gpt-4o --examples 50000
```

### Generate Reproduction Guide

```bash
# Standard guide
datarecipe guide Anthropic/hh-rlhf -o guide.md

# Deep analysis (PDF parsing + paper discovery)
datarecipe deep-guide https://arxiv.org/abs/2506.07982 -o deep-guide.md

# With LLM enhancement
export ANTHROPIC_API_KEY="your-key"
datarecipe deep-guide https://example.com/dataset --llm -o guide.md
```

### Quality Analysis

```bash
datarecipe quality Anthropic/hh-rlhf --detect-ai --sample-size 1000
```

### Generate Workflow Project

```bash
datarecipe workflow Anthropic/hh-rlhf -o ./my_project
```

This creates:
```
my_project/
├── README.md
├── requirements.txt
├── config.yaml
├── checklist.md
├── timeline.md
└── scripts/
    ├── 01_seed_data.py
    ├── 02_llm_generation.py
    ├── 03_quality_filtering.py
    ├── 04_deduplication.py
    └── 05_validation.py
```

### Generate Annotator Profile

```bash
# Generate annotator requirements and labor cost estimate
datarecipe profile Anthropic/hh-rlhf

# Specify region for cost calculation
datarecipe profile Anthropic/hh-rlhf --region china

# Export as Markdown
datarecipe profile Anthropic/hh-rlhf -o profile.md
```

### Production Deployment

```bash
# Generate deployment package with guidelines and quality rules
datarecipe deploy Anthropic/hh-rlhf -o ./annotation_project

# Use specific provider
datarecipe deploy Anthropic/hh-rlhf -o ./project --provider local
```

This creates:
```
annotation_project/
├── README.md
├── recipe.yaml
├── annotator_profile.yaml
├── cost_estimate.yaml
├── annotation_guide.md
├── quality_rules.yaml
├── acceptance_criteria.yaml
├── timeline.md
└── scripts/
    └── validate.py
```

### Manage Providers

```bash
# List available deployment providers
datarecipe providers list
```

---

## Documentation

<details>
<summary><b>Recipe Schema</b></summary>

Analysis results use a structured YAML format:

```yaml
name: dataset-identifier
version: "1.0"

source:
  type: huggingface
  id: org/dataset-name

generation:
  synthetic_ratio: 0.85
  human_ratio: 0.15
  teacher_models:
    - GPT-4o
    - Claude 3.5
  methods:
    - type: distillation
      teacher_model: GPT-4o
      prompt_template: available

cost:
  estimated_total_usd: 75000
  breakdown:
    api_calls: 50000
    human_annotation: 25000
  confidence: medium

reproducibility:
  score: 7
  available:
    - source_data_references
    - teacher_model_names
  missing:
    - exact_prompts
    - filtering_criteria
```

</details>

<details>
<summary><b>Reproducibility Scoring (0-10)</b></summary>

| Criterion | Points |
|-----------|--------|
| Dataset description present | +1 |
| Detailed documentation (>500 chars) | +1 |
| Teacher model names disclosed | +1 |
| Prompt templates available | +1 |
| Source code referenced | +1 |
| Generation parameters specified | +1 |
| Paper reference available | +1 |
| Download available | +1 |
| License information | +1 |
| Quality thresholds documented | +1 |

</details>

<details>
<summary><b>Supported LLM Providers for Cost Estimation</b></summary>

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3 Opus/Sonnet/Haiku, Claude 3.5 Sonnet
- **Google**: Gemini Pro, Gemini 1.5 Pro
- **Open Source**: Llama 3, Mixtral (via API)

</details>

<details>
<summary><b>Generation Type Classification</b></summary>

| Category | Indicators |
|----------|------------|
| LLM Distillation | synthetic, generated, distill, teacher, api |
| Human Annotation | human, annotated, crowdsource, mturk, expert |
| Programmatic | procedural, compositional, rule-based, template |
| Simulation | simulator, environment, agent, POMDP |
| Benchmark | evaluation, test set, leaderboard, metrics |

</details>

<details>
<summary><b>All Commands</b></summary>

```bash
# Analysis
datarecipe analyze <dataset>           # Analyze dataset
datarecipe guide <dataset>             # Generate production guide
datarecipe deep-guide <url>            # Deep analysis with paper parsing
datarecipe cost <dataset>              # Estimate costs
datarecipe quality <dataset>           # Quality metrics

# Batch Operations
datarecipe batch <ds1> <ds2> ...       # Batch analysis
datarecipe compare <ds1> <ds2> ...     # Compare datasets

# Production
datarecipe workflow <dataset>          # Generate reproduction project
datarecipe profile <dataset>           # Generate annotator profile
datarecipe deploy <dataset>            # Deploy to annotation provider

# Provider Management
datarecipe providers list              # List available providers

# Utilities
datarecipe create                      # Interactive recipe creation
datarecipe show <file>                 # Display recipe
datarecipe list-sources                # List supported sources
```

</details>

<details>
<summary><b>Provider Plugin System</b></summary>

DataRecipe uses a plugin system for deployment providers. The built-in `local` provider generates files locally.

**Installing additional providers:**

```bash
pip install datarecipe-judgeguild    # Example: JudgeGuild integration
pip install datarecipe-labelstudio   # Example: Label Studio integration
```

**Creating a custom provider:**

Providers implement the `DeploymentProvider` protocol and register via entry points:

```python
# In your package's pyproject.toml
[project.entry-points."datarecipe.providers"]
myprovider = "mypackage.provider:MyProvider"
```

```python
# mypackage/provider.py
from datarecipe.schema import DeploymentProvider, ProductionConfig

class MyProvider(DeploymentProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    @property
    def description(self) -> str:
        return "My custom annotation provider"

    def submit(self, config: ProductionConfig) -> DeploymentResult:
        # Implementation
        ...
```

</details>

---

## Why DataRecipe?

### For Researchers
- **Reproduce datasets** with step-by-step guides
- **Estimate costs** before starting a project
- **Compare options** to find the best dataset for your needs

### For Organizations
- **Audit data provenance** for compliance
- **Assess quality** with standardized metrics
- **Track inherited biases** from teacher models

### For the Community
- **Standardize documentation** with Recipe format
- **Share best practices** through production guides
- **Improve transparency** in the ML data supply chain

---

## Roadmap

- [x] Multi-source metadata extraction
- [x] Deep analysis with PDF parsing
- [x] LLM-enhanced semantic analysis
- [x] Cost calculator with multi-provider pricing
- [x] Quality metrics and AI detection
- [x] Batch processing and comparison
- [x] Workflow generation with executable scripts
- [x] Annotator profiler with regional labor costs
- [x] Production deployer with quality rules
- [x] Plugin-based provider system
- [ ] Community recipe repository
- [ ] Web UI for interactive analysis
- [ ] API service

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{liu2026datarecipe,
  author  = {Liu, Kai},
  title   = {DataRecipe: A Framework for Dataset Provenance Analysis},
  year    = {2026},
  url     = {https://github.com/liuxiaotong/data-recipe},
  email   = {mrliukai@gmail.com}
}
```

---

<div align="center">

**[⬆ Back to Top](#datarecipe)**

Made with care for the AI research community

</div>
