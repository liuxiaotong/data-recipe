<div align="center">

# DataRecipe

**Reverse engineering framework for AI datasets**

[![PyPI](https://img.shields.io/pypi/v/datarecipe?color=blue)](https://pypi.org/project/datarecipe/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Installation](#installation) · [Usage](#usage) · [Commands](#commands) · [MCP Server](#mcp-server)

</div>

---

Analyze how any AI dataset was built. Generate production-ready materials to reproduce it at scale.

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

<details>
<summary>Output</summary>

```
my_project/
├── annotation_guide.md       # Step-by-step labeling instructions
├── quality_rules.md          # QA checklist
├── acceptance_criteria.md    # Delivery standards
└── timeline.md               # Project schedule
```

</details>

## Commands

| Command | Description |
|---------|-------------|
| `analyze` | Extract dataset "recipe" (methods, sources, reproducibility) |
| `profile` | Generate annotator requirements and cost estimates |
| `deploy` | Output production-ready project materials |
| `cost` | Estimate API costs for synthetic generation |
| `quality` | Analyze data quality distribution |
| `compare` | Compare multiple datasets side-by-side |
| `batch` | Analyze multiple datasets at once |
| `guide` | Generate reproduction guide |

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
