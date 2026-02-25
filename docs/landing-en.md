<div align="right">

**English** | [中文](landing-zh.md)

</div>

<div align="center">

<h1>DataRecipe</h1>

<h3>Automated Dataset Reverse Engineering<br/>and Reproduction Cost Estimation</h3>

<p><em>Reverse-engineer any AI dataset: extract schemas, estimate costs, and generate production-ready documentation from samples or requirement docs</em></p>

<p>
<a href="https://github.com/liuxiaotong/data-recipe">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datarecipe/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>
</p>

</div>

## Why DataRecipe?

Reproducing an AI dataset requires answering three questions: **What does the data look like** (Schema), **How much will it cost** (Cost), and **How to build it** (Methodology). Today these answers come from manually reading papers, inspecting samples, and writing specs — a process that takes days and cannot be reused across datasets.

**DataRecipe automates the entire reverse engineering process.** Give it a HuggingFace dataset or a requirement document (PDF/Word/Image), and it will:

- **Infer Schema** — field types, constraints, distributions
- **Extract Rubrics & Prompts** — scoring criteria, annotation dimensions, prompt templates
- **Model Costs** — token-level analysis, phased cost breakdown, human-machine split ratios
- **Generate 23+ Production Documents** — for 6 stakeholder roles (executive, PM, annotators, engineers, finance, AI agents)
- **Enhance with LLM** — a single LLM call produces `EnhancedContext`, upgrading template outputs to domain-specific professional analyses

## Quick Start

```bash
pip install knowlyr-datarecipe

# Analyze a HuggingFace dataset (local, no API key needed)
knowlyr-datarecipe deep-analyze tencent/CL-bench

# Enable LLM enhancement for richer output
knowlyr-datarecipe deep-analyze tencent/CL-bench --use-llm

# Analyze a requirement document
knowlyr-datarecipe analyze-spec requirements.pdf
```

Optional extras: `pip install knowlyr-datarecipe[llm]` (Anthropic/OpenAI), `[pdf]`, `[mcp]`, or `[all]`.

## Six-Stage Analysis Pipeline

```mermaid
graph LR
    I["Input<br/>HF Dataset / PDF / Word"] --> A1["Schema<br/>Inference"]
    A1 --> A2["Rubric<br/>Extraction"]
    A2 --> A3["Prompt<br/>Extraction"]
    A3 --> A4["Cost<br/>Modeling"]
    A4 --> A5["Human-Machine<br/>Split"]
    A5 --> A6["Benchmark<br/>Comparison"]
    A6 --> E["LLM Enhancer<br/>EnhancedContext"]
    E --> G["Generators<br/>23+ Documents"]

    style A1 fill:#0969da,color:#fff,stroke:#0969da
    style E fill:#8b5cf6,color:#fff,stroke:#8b5cf6
    style G fill:#2da44e,color:#fff,stroke:#2da44e
    style I fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A2 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A3 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A4 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A5 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A6 fill:#1a1a2e,color:#e0e0e0,stroke:#444
```

Each stage outputs both human-readable (Markdown) and machine-parseable (JSON/YAML) formats. The LLM Enhancement Layer runs in three modes: `auto` (detect environment), `interactive` (host LLM handles it), or `api` (standalone Anthropic/OpenAI call).

## Core Features

| Feature | Description |
|:---|:---|
| **Multi-Source Input** | HuggingFace datasets, PDF, Word, images, plain text |
| **Token-Level Cost Analysis** | Phased cost model with human-machine split and industry benchmarks |
| **Stakeholder Documents** | 23+ docs for executives, PMs, annotators, engineers, finance, AI agents |
| **Agent-Ready Output** | Structured context, workflow state, reasoning traces, executable pipeline |
| **Radar Integration** | Batch-analyze datasets discovered by AI Dataset Radar |
| **12 MCP Tools** | Seamless AI IDE integration for analysis, enhancement, and comparison |
| **3572 Tests, 97% Coverage** | Production-grade reliability |

## Ecosystem

DataRecipe is part of the **knowlyr** data infrastructure:

| Layer | Project | Role |
|:---|:---|:---|
| Discovery | **AI Dataset Radar** | Dataset intelligence and trend analysis |
| Analysis | **DataRecipe** | Reverse engineering, schema inference, cost modeling |
| Production | **DataSynth** / **DataLabel** | LLM batch synthesis / lightweight annotation |
| Quality | **DataCheck** | Rule validation, anomaly detection, auto-fix |
| Audit | **ModelAudit** | Distillation detection, model fingerprinting |

```bash
# End-to-end workflow
knowlyr-datarecipe deep-analyze tencent/CL-bench --use-llm      # Analyze
knowlyr-datalabel generate ./projects/tencent_CL-bench/          # Annotate
knowlyr-datasynth generate ./projects/tencent_CL-bench/ -n 1000  # Synthesize
knowlyr-datacheck validate ./projects/tencent_CL-bench/          # Validate
```

<div align="center">
<br/>
<a href="https://github.com/liuxiaotong/data-recipe">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datarecipe/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>
<br/><br/>
<sub><a href="https://github.com/liuxiaotong">knowlyr</a> — automated dataset reverse engineering and reproduction cost estimation</sub>
</div>
