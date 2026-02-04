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

### Comprehensive analysis (recommended)

Run all analyses at once and generate a human-readable report:

```bash
datarecipe deep-analyze tencent/CL-bench -o ./output --size 1899
```

<details>
<summary>Output</summary>

```
============================================================
  DataRecipe 深度逆向分析
============================================================

数据集: tencent/CL-bench
输出目录: ./output

📥 加载数据集...
✓ 加载完成: 300 样本

📊 分析评分标准...
✓ 评分标准: 4120 条, 2412 种模式
📝 提取 Prompt 模板...
✓ Prompt模板: 293 个独特模板
🔍 检测上下文策略...
✓ 策略检测: hybrid (置信度 40.1%)
⚙️ 计算人机分配...
✓ 人机分配: 人工 84%, 机器 16%

📄 生成综合报告...
✓ 综合报告已保存

============================================================
  分析完成
============================================================

生成的文件:
  📊 prompt_templates.json (6.4MB)
  📊 context_strategy.json (1.6KB)
  📊 allocation.json (2.5KB)
  📊 rubrics_analysis.json (63.2KB)
  📑 rubric_templates.yaml / rubric_templates.md  ← 结构化 Rubric 模板库
  📄 ANALYSIS_REPORT.md (4.6KB)   ← 人类可读报告
```

</details>

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
│                                                              │
│  Structured Templates (Top 3):                               │
│    1. [list] should include → key evidence (≥3 items)        │
│    2. [avoid] should not include → offensive language        │
│    3. [explain] should explain → reasoning steps             │
╰──────────────────────────────────────────────────────────────╯
```

同时使用 `-o rubrics.json` 可获得：

- `rubrics.json`：详细统计 + 模式列表
- `rubrics_templates.yaml`：可复用的结构化模板（action/target/condition）
- `rubrics_templates.md`：面向非技术干系人的 Markdown 说明

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
| `deep-analyze` | **Run all analyses and generate comprehensive report** |
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
