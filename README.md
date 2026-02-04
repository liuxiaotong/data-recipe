<div align="center">

# DataRecipe

**AI 数据集逆向工程框架**

[![PyPI](https://img.shields.io/pypi/v/datarecipe?color=blue)](https://pypi.org/project/datarecipe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[快速开始](#快速开始) · [核心功能](#核心功能) · [深度分析](#深度分析) · [命令参考](#命令参考)

</div>

---

解析任意 AI 数据集的构建方式，提取可复用的模式，生成生产级资产。

## 核心价值

```
数据集 → 深度分析 → 可复用模板 → 生产指南 → 项目脚手架
```

| 目标 | 产出物 |
|------|--------|
| 摸清数据集构成 | `analyze` / `deep-analyze` 生成完整分析报告 |
| 复用评测标准 | `rubric_templates.yaml` / `.md` 结构化模板 |
| 提取 Prompt 策略 | `prompt_templates.json` + `context_strategy.json` |
| 估算成本与分工 | `allocation.json` 人机比例、成本拆分 |
| 生成生产指南 | `guide` / `deploy` / `workflow` 输出 Markdown + 项目结构 |

## 安装

```bash
pip install datarecipe
```

可选依赖：

```bash
pip install datarecipe[llm]      # LLM 分析 (Anthropic/OpenAI)
pip install datarecipe[quality]  # 质量分析
pip install datarecipe[mcp]      # MCP 服务器
pip install datarecipe[all]      # 全部功能
```

## 快速开始

### 分析数据集

```bash
datarecipe analyze Anthropic/hh-rlhf
```

<details>
<summary>输出示例</summary>

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

### 获取标注画像与成本估算

```bash
datarecipe profile nguha/legalbench --region china
```

<details>
<summary>输出示例</summary>

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

### 生成项目脚手架

```bash
datarecipe deploy AI-MO/NuminaMath-CoT -o ./my_project
```

---

## 深度分析

从数据集中提取可复用的模式，支持规模化复现。

### 一键深度分析

```bash
datarecipe deep-analyze tencent/CL-bench -o ./output --size 1899
```

<details>
<summary>输出示例</summary>

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
  生成的文件
============================================================

  📊 prompt_templates.json      Prompt 模板库
  📊 context_strategy.json      上下文策略分析
  📊 allocation.json            人机分配方案
  📊 rubrics_analysis.json      评分标准分析
  📑 rubric_templates.yaml      结构化 Rubric 模板
  📑 rubric_templates.md        可读 Rubric 文档
  📄 ANALYSIS_REPORT.md         综合分析报告
```

</details>

### 提取评分标准

```bash
datarecipe extract-rubrics tencent/CL-bench -o rubrics.json
```

<details>
<summary>输出示例</summary>

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

生成文件：
- `rubrics.json` - 详细统计与模式列表
- `rubrics_templates.yaml` - 结构化模板 (action/target/condition)
- `rubrics_templates.md` - Markdown 格式说明文档

</details>

### 人机分配估算

```bash
datarecipe allocate --size 10000 --region china
```

<details>
<summary>输出示例</summary>

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

---

## 命令参考

### 基础分析

| 命令 | 功能 |
|------|------|
| `analyze <dataset>` | 提取数据集「配方」(来源、方法、可复现性) |
| `profile <dataset>` | 生成标注员画像与成本估算 |
| `cost <dataset>` | 估算 API 合成成本 |
| `quality <dataset>` | 分析数据质量分布 |

### 深度逆向

| 命令 | 功能 |
|------|------|
| `deep-analyze <dataset>` | 运行全部分析，生成综合报告 |
| `extract-rubrics <dataset>` | 提取评分标准模式 |
| `extract-prompts <dataset>` | 提取 Prompt 模板 |
| `detect-strategy <dataset>` | 检测上下文构造策略 |
| `allocate` | 生成人机分配方案与成本 |
| `generate` | 基于提取模式生成数据 |

### 生产输出

| 命令 | 功能 |
|------|------|
| `deploy <dataset>` | 输出生产级项目结构 |
| `guide <dataset>` | 生成复现指南 |
| `workflow <dataset>` | 生成完整复现工作流 |
| `enhanced-guide <dataset>` | 结合发现模式生成增强指南 |

### 批量操作

| 命令 | 功能 |
|------|------|
| `batch <datasets...>` | 批量分析多个数据集 |
| `compare <datasets...>` | 并排对比多个数据集 |

---

## MCP 服务器

在 Claude Desktop 中直接使用 DataRecipe。

添加到 `~/Library/Application Support/Claude/claude_desktop_config.json`：

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

然后询问 Claude：*「分析 Anthropic/hh-rlhf 数据集」*

---

## License

[MIT](LICENSE)
