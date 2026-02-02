<div align="center">

# DataRecipe

**AI 数据集的"营养成分表"** | **Nutrition Labels for AI Datasets**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MCP Server](https://img.shields.io/badge/MCP-Server-purple.svg)](https://modelcontextprotocol.io/)

分析数据集构建方式 · 估算标注成本 · 生成投产项目

[快速开始](#快速开始) · [Claude 集成](#claude-集成) · [命令参考](#命令参考) · [English](#english)

</div>

---

## 一分钟了解 DataRecipe

```bash
# 分析一个数据集
$ datarecipe analyze AI-MO/NuminaMath-CoT

╭─────────────────────────── Dataset Recipe ───────────────────────────╮
│  Name: AI-MO/NuminaMath-CoT                                          │
│  📊 Generation: Synthetic 100%                                       │
│  🤖 Teacher Models: None detected                                    │
│  🔄 Reproducibility: [8/10] ████████░░                               │
╰──────────────────────────────────────────────────────────────────────╯

# 生成标注团队画像
$ datarecipe profile nguha/legalbench --region us

Required Skills: 法律(expert), 法律从业资格(required)
Education: Professional (J.D.)
Hourly Rate: $105/hour
Estimated Cost: $262,500
```

---

## 安装

```bash
pip install datarecipe

# 或使用 uv (推荐)
uv pip install datarecipe
```

<details>
<summary>可选依赖</summary>

```bash
pip install datarecipe[mcp]      # MCP Server (Claude App 集成)
pip install datarecipe[llm]      # LLM 增强分析
pip install datarecipe[pdf]      # PDF 解析
pip install datarecipe[all]      # 全部功能
```
</details>

---

## 快速开始

### 1. 分析数据集

```bash
datarecipe analyze Anthropic/hh-rlhf
datarecipe analyze AI-MO/NuminaMath-CoT --json
```

### 2. 估算标注成本

```bash
# 生成标注团队画像（技能要求 + 成本估算）
datarecipe profile <dataset> --region china    # 中国人力成本
datarecipe profile <dataset> --region us       # 美国人力成本
```

**高价值数据集成本参考：**

| 数据集 | 领域 | 时薪 | 单条成本 |
|--------|------|------|----------|
| nguha/legalbench | 法律 | $105 | $44 |
| openlifescienceai/MedMCQA | 医疗 | $105 | $35 |
| AI-MO/NuminaMath-CoT | 数学 | $48 | $16 |
| tatsu-lab/alpaca | 通用 | $6 | $0.5 |

### 3. 生成投产项目

```bash
datarecipe deploy <dataset>                    # 默认输出到 ./projects/
datarecipe deploy <dataset> -o ./my_project    # 自定义目录
```

生成的项目包含：
```
my_project/
├── README.md                 # 项目概述
├── annotation_guide.md       # 标注指南
├── quality_rules.yaml        # 质检规则
├── acceptance_criteria.yaml  # 验收标准
├── timeline.md               # 时间线 + 甘特图
└── scripts/                  # 自动化脚本
```

---

## Claude 集成

DataRecipe 支持两种方式与 Claude 集成：

### 方式 1: MCP Server (Claude Desktop)

让 Claude 直接调用 DataRecipe 分析数据集。

**配置** `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datarecipe": {
      "command": "uvx",
      "args": ["--from", "datarecipe", "datarecipe-mcp"]
    }
  }
}
```

**然后在 Claude 中：**
- "分析 Anthropic/hh-rlhf 数据集"
- "nguha/legalbench 需要什么技能的标注员？成本多少？"
- "为 AI-MO/NuminaMath-CoT 创建标注项目"

### 方式 2: Claude Code (CLI)

在项目目录下使用 slash commands：

```
/datarecipe analyze Anthropic/hh-rlhf
/profile-annotators nguha/legalbench --region us
/deploy-project AI-MO/NuminaMath-CoT
```

---

## 命令参考

| 命令 | 功能 | 示例 |
|------|------|------|
| `analyze` | 分析数据集元数据 | `datarecipe analyze <dataset>` |
| `profile` | 生成标注团队画像 | `datarecipe profile <dataset> --region china` |
| `deploy` | 生成投产项目 | `datarecipe deploy <dataset>` |
| `cost` | 估算 API/计算成本 | `datarecipe cost <dataset> --model gpt-4o` |
| `quality` | 质量分析 | `datarecipe quality <dataset> --detect-ai` |
| `compare` | 对比多个数据集 | `datarecipe compare <ds1> <ds2>` |
| `providers list` | 列出可用 Provider | `datarecipe providers list` |

<details>
<summary><b>完整命令列表</b></summary>

```bash
# 分析
datarecipe analyze <dataset>           # 分析数据集
datarecipe guide <dataset>             # 生成复现指南
datarecipe deep-guide <url>            # 深度分析（解析论文）
datarecipe cost <dataset>              # 成本估算
datarecipe quality <dataset>           # 质量分析

# 批量操作
datarecipe batch <ds1> <ds2> ...       # 批量分析
datarecipe compare <ds1> <ds2>         # 对比数据集

# 投产
datarecipe profile <dataset>           # 标注团队画像
datarecipe deploy <dataset>            # 生成投产项目
datarecipe workflow <dataset>          # 生成复现工作流

# 工具
datarecipe providers list              # 列出 Provider
datarecipe create                      # 交互式创建配方
datarecipe list-sources                # 支持的数据源
```
</details>

---

## 项目架构

```
datarecipe/
├── analyzer.py         # 数据集分析
├── profiler.py         # 标注专家画像
├── deployer.py         # 投产部署
├── cost_calculator.py  # 成本估算
├── mcp_server.py       # MCP Server
└── providers/          # Provider 插件
    └── local.py        # 本地文件 Provider
```

<details>
<summary><b>Provider 插件系统</b></summary>

DataRecipe 使用插件系统管理部署 Provider。

**安装额外 Provider：**
```bash
pip install datarecipe-labelstudio   # Label Studio 集成
```

**创建自定义 Provider：**

```python
# pyproject.toml
[project.entry-points."datarecipe.providers"]
myprovider = "mypackage:MyProvider"
```

```python
from datarecipe.schema import DeploymentProvider

class MyProvider(DeploymentProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    def submit(self, config):
        # 实现部署逻辑
        ...
```
</details>

<details>
<summary><b>数据配方 Schema</b></summary>

```yaml
name: dataset-name
source:
  type: huggingface
  id: org/dataset

generation:
  synthetic_ratio: 0.85
  human_ratio: 0.15
  teacher_models: [GPT-4o, Claude-3]

cost:
  estimated_total_usd: 75000
  confidence: medium

reproducibility:
  score: 7
  available: [source_data, teacher_models]
  missing: [exact_prompts, filtering_criteria]
```
</details>

---

## English

DataRecipe is a "nutrition label" system for AI datasets - analyzing construction methods, estimating annotation costs, and generating production-ready annotation projects.

**Key Features:**
- Analyze dataset provenance and generation methods
- Estimate annotation costs by region (US, China, Europe)
- Generate complete annotation projects with guidelines and quality rules
- Integrate with Claude via MCP Server

**Quick Start:**
```bash
pip install datarecipe
datarecipe analyze Anthropic/hh-rlhf
datarecipe profile nguha/legalbench --region us
datarecipe deploy AI-MO/NuminaMath-CoT
```

---

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{datarecipe2026,
  title   = {DataRecipe: Nutrition Labels for AI Datasets},
  author  = {Liu, Kai},
  year    = {2026},
  url     = {https://github.com/liuxiaotong/data-recipe}
}
```

<div align="center">

---

**[GitHub](https://github.com/liuxiaotong/data-recipe)** · **[Issues](https://github.com/liuxiaotong/data-recipe/issues)**

</div>
