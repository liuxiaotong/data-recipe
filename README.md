<div align="center">

# DataRecipe

**Reverse engineering framework for AI datasets**

[![PyPI](https://img.shields.io/pypi/v/datarecipe?color=blue)](https://pypi.org/project/datarecipe/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Installation](#installation) · [Usage](#usage) · [Deep Analysis](#deep-analysis) · [Commands](#commands)

</div>

---

Analyze how any AI dataset was built. Extract patterns, generate production guides, and reproduce at scale.

## Installation

```bash
pip install datarecipe
```

## Usage

### Analyze a dataset

```bash
datarecipe analyze Anthropic/hh-rlhf
```

<details>
<summary>Output</summary>

```
╭──────────────────────── Dataset Recipe ────────────────────────╮
│  Anthropic/hh-rlhf                                             │
│                                                                │
│  Generation    Human 100%                                      │
│  Method        RLHF preference pairs                           │
│  Size          161K examples                                   │
│  Reproducibility  [7/10] ███████░░░                            │
│                                                                │
│  Missing: exact annotation guidelines, quality criteria        │
╰────────────────────────────────────────────────────────────────╯
```

</details>

### Get annotator profile & cost estimate

```bash
datarecipe profile nguha/legalbench --region china
```

<details>
<summary>Output</summary>

```
╭──────────────────── Annotator Profile ─────────────────────╮
│                                                            │
│  Required Skills                                           │
│  ├─ Domain: Legal (Expert level)                           │
│  ├─ Language: English (Native)                             │
│  └─ Certification: J.D. preferred                          │
│                                                            │
│  Cost Estimate (China)                                     │
│  ├─ Hourly Rate: ¥150-200                                  │
│  ├─ Per Example: ¥45                                       │
│  └─ Total (10K examples): ¥450,000                         │
│                                                            │
╰────────────────────────────────────────────────────────────╯
```

</details>

### Generate production materials

```bash
datarecipe deploy AI-MO/NuminaMath-CoT -o ./my_project
```

---

## Deep Analysis

Extract actionable patterns from any dataset for reproduction at scale.

### Extract rubrics patterns

```bash
datarecipe extract-rubrics tencent/CL-bench
```

<details>
<summary>Output</summary>

```
╭────────────────────── Rubrics Analysis ──────────────────────╮
│  Total Rubrics: 1173                                         │
│  Unique Patterns: 900                                        │
│                                                              │
│  Top Verbs:                                                  │
│    - include: 91 (7.8%)                                      │
│    - state: 86 (7.3%)                                        │
│    - not: 71 (6.1%)                                          │
│    - explain: 70 (6.0%)                                      │
│    - provide: 58 (4.9%)                                      │
╰──────────────────────────────────────────────────────────────╯
```

</details>

### Generate human-machine allocation

```bash
datarecipe allocate --size 10000 --region china
```

<details>
<summary>Output</summary>

```
╭─────────────────── Allocation Summary ───────────────────╮
│  Total Tasks: 5                                          │
│    - Human Only: 3                                       │
│    - Machine Only: 1                                     │
│    - Hybrid: 1                                           │
│                                                          │
│  COSTS:                                                  │
│    Human Labor: $43,620 (2222 hours)                     │
│    Machine/API: $498                                     │
│    Total: $44,118                                        │
│                                                          │
│  WORKLOAD SPLIT:                                         │
│    Human: 84%                                            │
│    Machine: 16%                                          │
╰──────────────────────────────────────────────────────────╯
```

</details>

### Generate data from patterns

```bash
datarecipe generate --type rubrics --context "game rules" --count 10
```

---

## Commands

### Core Analysis

| Command | Description |
|---------|-------------|
| `analyze` | Extract dataset "recipe" (methods, sources, reproducibility) |
| `profile` | Generate annotator requirements and cost estimates |
| `deploy` | Output production-ready project materials |
| `cost` | Estimate API costs for synthetic generation |
| `quality` | Analyze data quality distribution |

### Deep Reverse Engineering

| Command | Description |
|---------|-------------|
| `extract-rubrics` | Extract evaluation criteria patterns (verbs, templates) |
| `extract-prompts` | Extract and deduplicate system prompt templates |
| `detect-strategy` | Detect context construction strategy (synthetic/modified/niche) |
| `allocate` | Generate human-machine task allocation with costs |
| `enhanced-guide` | Generate production guide with discovered patterns |
| `generate` | Generate data based on extracted patterns |

### Batch Operations

| Command | Description |
|---------|-------------|
| `batch` | Analyze multiple datasets at once |
| `compare` | Compare multiple datasets side-by-side |
| `guide` | Generate reproduction guide |
| `workflow` | Generate complete reproduction workflow |

---

## MCP Server

Use DataRecipe directly in Claude Desktop.

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datarecipe": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-recipe", "run", "datarecipe-mcp"]
    }
  }
}
```

Then ask Claude: *"Analyze the Anthropic/hh-rlhf dataset"*

## License

[MIT](LICENSE)
