<div align="center">

# DataRecipe

**AI 数据集逆向工程框架** | **Reverse Engineering Framework for AI Datasets**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MCP Server](https://img.shields.io/badge/MCP-Server-purple.svg)](https://modelcontextprotocol.io/)

**逆向分析数据集构建方式 · 生成可复现的生产资料包 · 批量生产同类数据**

[快速开始](#快速开始) · [实战案例](#实战案例-cl-bench-逆向复现) · [命令参考](#命令参考) · [English](#english)

</div>

---

## 核心能力

DataRecipe 不只是分析数据集，而是帮你**完整逆向工程**一个数据集，输出可直接用于批量生产的全套资料。

```
输入: 任意 AI 数据集
      ↓
DataRecipe 逆向分析
      ↓
输出: 1. 数据集"配方"（构建方法、成本、来源）
      2. 标注团队画像（技能要求、薪资、招聘建议）
      3. 生产资料包（标注指南、质检规则、验收标准）
      4. 模式分析（Rubrics 模式、Prompt 模板）
      ↓
你可以: 批量生产同类高质量数据
```

---

## 安装

```bash
pip install datarecipe

# 或使用 uv (推荐)
uv pip install datarecipe
```

---

## 快速开始

### 1. 分析数据集

```bash
datarecipe analyze Anthropic/hh-rlhf
datarecipe analyze AI-MO/NuminaMath-CoT --json
```

```
╭─────────────────────────── Dataset Recipe ───────────────────────────╮
│  Name: AI-MO/NuminaMath-CoT                                          │
│  📊 Generation: Synthetic 100%                                       │
│  🤖 Teacher Models: None detected                                    │
│  🔄 Reproducibility: [8/10] ████████░░                               │
╰──────────────────────────────────────────────────────────────────────╯
```

### 2. 估算标注成本

```bash
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
├── quality_rules.md          # 质检规则
├── acceptance_criteria.md    # 验收标准
├── timeline.md               # 时间线 + 甘特图
└── scripts/                  # 自动化脚本
```

---

## 实战案例: CL-bench 逆向复现

我们用 DataRecipe 完整逆向了腾讯混元的 [CL-bench](https://github.com/Tencent-Hunyuan/CL-bench) 数据集（1,899 个任务，31,607 条 Rubrics），生成了可直接用于批量生产的完整资料包。

**查看完整案例: [`examples/cl-bench-reproduction/`](examples/cl-bench-reproduction/)**

### 案例亮点

| 产出 | 说明 |
|------|------|
| [PRODUCTION_GUIDE.md](examples/cl-bench-reproduction/PRODUCTION_GUIDE.md) | 512 行完整生产指南 |
| [system_prompt_templates.json](examples/cl-bench-reproduction/reproduction_kit/system_prompt_templates.json) | 495 个 System Prompt 模板 |
| [subcategory_analysis.json](examples/cl-bench-reproduction/reproduction_kit/subcategory_analysis.json) | 18 个子类别详细分析 |
| [batch_production_demo.py](examples/cl-bench-reproduction/scripts/batch_production_demo.py) | 无需 API 的批量生产脚本 |

### 逆向发现的 Rubrics 模式

```
句式: The response should [动词] [对象] [条件/细节]

Top 动词:
- not (3.2%) - 否定检查，如 "should not assume..."
- include (2.5%) - 包含检查，如 "should include all..."
- state (2.4%) - 陈述检查，如 "should state that..."
- provide (1.9%) - 提供检查，如 "should provide evidence..."
- explain (1.1%) - 解释检查，如 "should explain why..."
```

### 快速体验

```bash
cd examples/cl-bench-reproduction

# 运行批量生产演示（无需 API）
python scripts/batch_production_demo.py

# 查看生成的数据
cat production_output/batch_*.jsonl | head -1 | python -m json.tool
```

---

## 命令参考

| 命令 | 功能 | 示例 |
|------|------|------|
| `analyze` | 分析数据集元数据 | `datarecipe analyze <dataset>` |
| `profile` | 生成标注团队画像 | `datarecipe profile <dataset> --region china` |
| `deploy` | 生成投产项目 | `datarecipe deploy <dataset>` |
| `cost` | 估算 API/计算成本 | `datarecipe cost <dataset> --model gpt-4o` |
| `quality` | 质量分析 | `datarecipe quality <dataset>` |
| `compare` | 对比多个数据集 | `datarecipe compare <ds1> <ds2>` |

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

## Claude Desktop 集成 (MCP)

DataRecipe 提供 MCP Server，可在 Claude Desktop 中直接使用。

**配置方法:**

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

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

重启 Claude Desktop 后，可直接对话：
- "分析 Anthropic/hh-rlhf 数据集"
- "nguha/legalbench 需要什么技能的标注员？"
- "为 AI-MO/NuminaMath-CoT 创建标注项目"

---

## 项目结构

```
data-recipe/
├── src/datarecipe/           # 核心代码
│   ├── analyzer.py           # 数据集分析
│   ├── profiler.py           # 标注专家画像
│   ├── deployer.py           # 投产部署
│   ├── cost_calculator.py    # 成本估算
│   ├── mcp_server.py         # MCP Server
│   ├── sources/              # 数据源 (HuggingFace, GitHub, Web)
│   └── providers/            # 部署 Provider 插件
├── examples/                 # 实战案例
│   └── cl-bench-reproduction/  # CL-bench 逆向复现完整资料
├── pyproject.toml
└── README.md
```

---

## English

DataRecipe is a **reverse engineering framework for AI datasets**. It analyzes how datasets were built and generates production-ready materials for reproducing similar data at scale.

**Key Capabilities:**
- Reverse engineer dataset construction methods
- Extract patterns (rubrics, prompts, evaluation criteria)
- Generate complete production kits (annotation guides, quality rules, templates)
- Estimate annotation costs by region

**Example: CL-bench Reproduction**

We fully reverse-engineered Tencent's [CL-bench](https://github.com/Tencent-Hunyuan/CL-bench) dataset (1,899 tasks, 31,607 rubrics) and generated a complete production kit. See [`examples/cl-bench-reproduction/`](examples/cl-bench-reproduction/).

**Quick Start:**
```bash
pip install datarecipe
datarecipe analyze Anthropic/hh-rlhf
datarecipe deploy AI-MO/NuminaMath-CoT
```

---

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{datarecipe2026,
  title   = {DataRecipe: Reverse Engineering Framework for AI Datasets},
  author  = {Liu, Kai},
  year    = {2026},
  url     = {https://github.com/liuxiaotong/data-recipe}
}
```

<div align="center">

---

**[GitHub](https://github.com/liuxiaotong/data-recipe)** · **[Issues](https://github.com/liuxiaotong/data-recipe/issues)** · **[Examples](examples/)**

</div>
