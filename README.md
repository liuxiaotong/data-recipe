# DataRecipe

**AI 数据集逆向工程框架**

分析任意 AI 数据集的构建方式，生成可用于批量生产同类数据的完整资料包。

```bash
pip install datarecipe
```

## 它能做什么

```
datarecipe analyze Anthropic/hh-rlhf
```

```
╭──────────────────────── Dataset Recipe ────────────────────────╮
│  Anthropic/hh-rlhf                                             │
│                                                                │
│  Generation    Human 100%                                      │
│  Method        RLHF preference pairs                           │
│  Size          161K examples                                   │
│  Reproduce     [7/10] ███████░░░                               │
│                                                                │
│  Missing: exact annotation guidelines, quality criteria        │
╰────────────────────────────────────────────────────────────────╯
```

## 三个核心命令

### 1. analyze - 逆向分析数据集

提取数据集的"配方"：构建方法、数据来源、合成比例、可复现性评分。

```bash
datarecipe analyze <dataset>              # HuggingFace 数据集
datarecipe analyze ./local/data.jsonl     # 本地文件
datarecipe analyze https://github.com/... # GitHub 仓库
```

### 2. profile - 生成标注团队画像

估算复现该数据集需要的人力配置和成本。

```bash
datarecipe profile <dataset> --region china
```

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

### 3. deploy - 生成投产资料包

输出可直接交付给标注团队的完整项目。

```bash
datarecipe deploy <dataset> -o ./my_project
```

```
my_project/
├── annotation_guide.md       # 标注指南
├── quality_rules.md          # 质检规则
├── acceptance_criteria.md    # 验收标准
└── timeline.md               # 排期建议
```

## 更多命令

| 命令 | 功能 |
|------|------|
| `cost` | 估算 API 调用成本 |
| `quality` | 分析数据质量分布 |
| `compare` | 对比多个数据集 |
| `batch` | 批量分析 |
| `guide` | 生成复现指南 |
| `workflow` | 生成工作流模板 |

## MCP Server

支持 Claude Desktop 直接调用。

配置 `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

然后在 Claude 中直接对话："分析 Anthropic/hh-rlhf 数据集"

---

## English

DataRecipe is a reverse engineering framework for AI datasets. It analyzes how datasets were constructed and generates production-ready materials for reproducing similar data at scale.

**Quick Start:**

```bash
pip install datarecipe

# Analyze a dataset
datarecipe analyze Anthropic/hh-rlhf

# Get annotator requirements and cost estimate
datarecipe profile nguha/legalbench --region us

# Generate production materials
datarecipe deploy AI-MO/NuminaMath-CoT -o ./output
```

---

MIT License
