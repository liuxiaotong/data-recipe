<div align="center">

# DataRecipe

**AI 数据集逆向工程框架**

[![PyPI](https://img.shields.io/pypi/v/datarecipe?color=blue)](https://pypi.org/project/datarecipe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[快速开始](#快速开始) · [核心功能](#核心功能) · [深度分析](#深度分析) · [成本估算](#成本估算) · [命令参考](#命令参考)

</div>

---

解析任意 AI 数据集的构建方式，提取可复用的模式，生成生产级资产。

## 核心价值

```
数据集 → 深度分析 → 可复用模板 → 生产指南 → 标注规范 → 成本估算
```

| 目标 | 产出物 |
|------|--------|
| 摸清数据集构成 | `ANALYSIS_REPORT.md` 完整分析报告 |
| **复刻数据集** | `REPRODUCTION_GUIDE.md` 可操作的复刻指南 |
| **标注外包规范** | `ANNOTATION_SPEC.md` 前瞻性标注规范 ⭐ |
| **精准成本估算** | `COST_BREAKDOWN.md` 分阶段成本明细 ⭐ |
| 复用评测标准 | `rubric_templates.yaml` / `.md` 结构化模板 |
| 提取 Prompt 策略 | `prompt_templates.json` + `context_strategy.json` |
| 估算成本与分工 | `allocation.json` 人机比例、成本拆分 |

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

从数据集中提取可复用的模式，生成可操作的复刻指南。

### 一键深度分析

```bash
datarecipe deep-analyze tencent/CL-bench -o ./output
```

输出目录结构：

```
output/
└── tencent_CL-bench/
    ├── ANALYSIS_REPORT.md       # 统计分析报告
    ├── REPRODUCTION_GUIDE.md    # 复刻指南 ⭐
    ├── ANNOTATION_SPEC.md       # 标注规范 (外包交付用) ⭐
    ├── COST_BREAKDOWN.md        # 分阶段成本明细 ⭐
    ├── recipe_summary.json      # 标准化摘要 (Radar 兼容)
    ├── rubric_templates.yaml    # 评分标准模板
    ├── rubric_templates.md      # 评分标准文档
    ├── prompt_templates.json    # Prompt 模板库
    ├── context_strategy.json    # 上下文策略
    ├── allocation.json          # 人机分配方案
    ├── rubrics_analysis.json    # 原始分析数据
    ├── annotation_spec.json     # 标注规范 (JSON)
    ├── token_analysis.json      # Token 统计与 API 成本
    ├── cost_comparison.json     # 多模型成本对比
    ├── complexity_analysis.json # 内容复杂度分析
    ├── cost_calibration.json    # 历史基准校准
    ├── phased_cost.json         # 分阶段成本模型
    └── llm_analysis.json        # LLM 智能分析 (--use-llm)
```

### 复刻指南 (REPRODUCTION_GUIDE.md)

核心产出物，包含 8 个可操作部分：

| 部分 | 内容 |
|------|------|
| 数据结构规范 | 字段定义 + JSON Schema |
| 任务分类体系 | category / sub_category 完整列表 |
| System Prompt 模板库 | 按领域分类的真实示例 |
| Rubric 编写规范 | 句式模式 + 结构 + 完整示例 |
| 复刻 SOP | 3 阶段 9 步骤标准流程 |
| 完整数据示例 | JSON 格式参考 |
| 资源估算 | 人力配置 + 成本 |
| 检查清单 | 发布前质量检查 |

<details>
<summary>运行示例</summary>

```
============================================================
  DataRecipe 深度逆向分析
============================================================

数据集: tencent/CL-bench
输出目录: ./output/tencent_CL-bench

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
📋 生成复刻指南...
✓ 复刻指南已保存

============================================================
  分析完成
============================================================

核心产出:
  📄 分析报告: ./output/tencent_CL-bench/ANALYSIS_REPORT.md
  📋 复刻指南: ./output/tencent_CL-bench/REPRODUCTION_GUIDE.md
```

</details>

### LLM 智能分析（未知数据集类型）

当遇到无法识别的数据集类型时，使用 LLM 进行智能分析：

```bash
# 使用 Anthropic Claude (默认)
export ANTHROPIC_API_KEY=your_key
datarecipe deep-analyze unknown/dataset --use-llm

# 使用 OpenAI
export OPENAI_API_KEY=your_key
datarecipe deep-analyze unknown/dataset --use-llm --llm-provider openai
```

LLM 会自动识别数据集类型，并生成：
- 数据集类型和用途说明
- 关键字段分析
- 生产流程 SOP
- 质量标准
- 标注指南
- 团队配置建议
- 难度评估
- 相似数据集参考

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

## 成本估算

DataRecipe 采用多维度成本估算方法，从四个角度综合评估数据集生产成本。

### 估算方法论

```
Token 分析 ──┐
             ├──→ 综合成本估算 ──→ COST_BREAKDOWN.md
复杂度分析 ──┤
             │
历史校准 ────┤
             │
分阶段模型 ──┘
```

| 方法 | 说明 | 产出 |
|------|------|------|
| **Token 分析** | 基于实际 token 数量计算 API 成本 | `token_analysis.json` |
| **复杂度分析** | 领域难度、文本长度、结构复杂度 | `complexity_analysis.json` |
| **历史校准** | 与知识库中相似数据集对比 | `cost_calibration.json` |
| **分阶段模型** | 设计/生产/质量三阶段拆分 | `phased_cost.json` |

### Token 精确计算

基于真实模型定价，支持 15+ 主流模型对比：

```bash
# 深度分析会自动输出 token 分析和多模型对比
datarecipe deep-analyze dataset/id -o ./output
```

<details>
<summary>cost_comparison.json 示例</summary>

```json
{
  "model_comparison": [
    {"model": "deepseek-v3", "cost": "$2.74", "savings_vs_gpt4o": "95%"},
    {"model": "gpt-4o-mini", "cost": "$3.00", "savings_vs_gpt4o": "94%"},
    {"model": "claude-3-haiku", "cost": "$5.00", "savings_vs_gpt4o": "90%"},
    {"model": "gpt-4o", "cost": "$50.00", "baseline": true},
    {"model": "claude-opus-4", "cost": "$150.00"}
  ]
}
```

</details>

### 分阶段成本模型

成本按项目阶段拆分，便于预算规划：

| 阶段 | 内容 | 成本类型 |
|------|------|----------|
| **设计阶段** | Schema 设计、标注指南、试点测试、工具配置 | 固定成本 |
| **生产阶段** | 人工标注、API 生成、初审、基础设施 | 变动成本 |
| **质量阶段** | QA 抽检、返工修正、专家复核、终验 | 比例成本 |

<details>
<summary>COST_BREAKDOWN.md 示例</summary>

```markdown
# 分阶段成本估算

**数据集类型**: preference
**目标规模**: 50 条
**项目规模**: small

## 阶段一：设计阶段（固定成本）
| 项目 | 成本 |
|------|------|
| Schema 设计 | $400 |
| 标注指南编写 | $640 |
| 试点测试 | $120 |
| 工具配置 | $200 |
| **小计** | **$1,360** |

## 阶段二：生产阶段（变动成本）
| 项目 | 成本 | 单价 |
|------|------|------|
| 人工标注 | $17 | - |
| API 生成 | $0 | - |
| **小计** | **$68** | $1.37/条 |

## 汇总
| 阶段 | 成本 | 占比 |
|------|------|------|
| 设计阶段 | $1,360 | 94.8% |
| 生产阶段 | $68 | 4.8% |
| 质量阶段 | $6 | 0.4% |
| 风险预留 (15%) | $215 | - |
| **总计** | **$1,649** | - |
```

</details>

### 标注规范生成

`ANNOTATION_SPEC.md` 是面向外包团队的前瞻性标注规范，包含：

| 部分 | 内容 |
|------|------|
| 题目类型描述 | 任务名称、说明、认知要求、推理链 |
| 数据要求 | 字段数量、必填字段、长度要求、领域要求 |
| 质量约束 | 质量差异、选择依据、AI 内容限制、格式规范 |
| 格式要求 | Schema 定义、字段类型 |
| 例题 | 题目、答案、打分标准 |
| 打分标准 | 评分维度、部分得分规则 |

<details>
<summary>ANNOTATION_SPEC.md 示例</summary>

```markdown
# Anthropic/hh-rlhf 标注规范

> 生成时间: 2026-02-05
> 数据类型: 偏好对比数据
> 样本来源: 5 条

## 一、题目类型描述

**任务名称**: 偏好对比数据
**任务说明**: 人类偏好对比数据集，用于训练奖励模型或直接偏好优化...

## 二、数据要求

1. 每条数据需包含 2 个字段
2. 必填字段: chosen, rejected
3. 文本长度要求: 中等（500-2000 字符）
4. 领域要求: 需要 code 领域专业知识

## 三、质量约束

1. 两个回答必须有明显的质量差异
2. 交付数据中不能含有任何 AI 产出的内容
...
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
| `deep-analyze --use-llm` | 使用 LLM 智能分析未知类型数据集 |
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
| `batch-from-radar <report>` | 从 Radar 报告批量分析 |
| `batch-from-radar --incremental` | 增量模式，跳过已分析数据集 |
| `batch-from-radar --sort-by downloads` | 按下载量排序 |
| `compare <datasets...>` | 并排对比多个数据集 |

### 知识库

| 命令 | 功能 |
|------|------|
| `knowledge --report` | 生成知识库报告 |
| `knowledge --patterns` | 查看 Top 模式 |
| `knowledge --benchmarks` | 查看成本基准 |
| `knowledge --trends` | 查看近期趋势 |
| `knowledge --recommend <type>` | 获取类型推荐 |

### 缓存管理

| 命令 | 功能 |
|------|------|
| `cache --list` | 列出缓存的数据集 |
| `cache --stats` | 查看缓存统计 |
| `cache --clear-expired` | 清理过期缓存 |
| `cache --invalidate <id>` | 使特定缓存失效 |
| `deep-analyze --force` | 强制重新分析 |

### 自动监听

| 命令 | 功能 |
|------|------|
| `watch <dir>` | 监听目录，自动分析新报告 |
| `watch --once` | 单次检查模式 |
| `watch --config <yaml>` | 使用配置文件 |

### 整合报告

| 命令 | 功能 |
|------|------|
| `integrate-report` | 生成整合报告 |
| `integrate-report -r <radar.json>` | 整合 Radar 发现 |
| `integrate-report --recipe-dir <dir>` | 指定分析目录 |

---

## 与 ai-dataset-radar 联动

DataRecipe 与 [ai-dataset-radar](https://github.com/liuxiaotong/ai-dataset-radar) 构成完整的 AI native 工作流：

```
Radar (发现新数据集) → Recipe (逆向分析) → 复刻生产
```

### 从 Radar 报告批量分析

```bash
# 分析 Radar 周报中的所有数据集
datarecipe batch-from-radar ./intel_report_2024-01-01.json

# 按条件筛选
datarecipe batch-from-radar ./report.json \
  --orgs Anthropic,OpenAI \
  --categories preference,sft \
  --min-downloads 1000 \
  --limit 10

# 启用 LLM 分析未知类型
datarecipe batch-from-radar ./report.json --use-llm
```

### 标准化输出格式

每个分析结果都会生成 `recipe_summary.json`，格式与 Radar 兼容：

```json
{
  "dataset_id": "Anthropic/hh-rlhf",
  "dataset_type": "preference",
  "reproduction_cost": {"human": 5000, "api": 200, "total": 5200},
  "difficulty": "medium",
  "human_percentage": 84.0,
  "key_patterns": ["rubric:include", "rubric:explain"],
  "report_path": "./output/Anthropic_hh-rlhf/ANALYSIS_REPORT.md",
  "guide_path": "./output/Anthropic_hh-rlhf/REPRODUCTION_GUIDE.md"
}
```

### 批量分析输出

```
output/
├── batch_summary.json          # 汇总统计
├── Anthropic_hh-rlhf/
│   ├── recipe_summary.json     # 标准化摘要
│   └── ...
└── OpenAI_xxx/
    └── ...
```

---

## 自动化工作流

### 监听 Radar 输出自动分析

```bash
# 持续监听，每 5 分钟检查一次
datarecipe watch ./radar_reports/ --interval 300

# 带过滤条件
datarecipe watch ./reports --orgs Anthropic,OpenAI --min-downloads 1000

# 单次检查
datarecipe watch ./reports --once
```

### 整合报告

将 Radar 发现和 Recipe 分析整合成一份完整周报：

```bash
# 基于 Radar 报告生成
datarecipe integrate-report -r ./intel_report.json -o ./reports

# 仅基于已分析数据集
datarecipe integrate-report --recipe-dir ./analysis_output

# 指定时间范围
datarecipe integrate-report --start-date 2024-01-01 --end-date 2024-01-07
```

生成的报告包含：
- 执行摘要（发现数、分析数、总成本）
- 组织分布和类型分布
- 详细数据集列表（已分析/待分析）
- 成本分析（按类型）
- 关键洞察和趋势

### 配置文件 (triggers.yaml)

```yaml
triggers:
  orgs:
    - Anthropic
    - OpenAI
    - Google
  categories:
    - preference
    - sft
  min_downloads: 500
  max_datasets_per_report: 10
  sample_size: 200
  use_llm: false
  region: china
```

```bash
datarecipe watch ./reports --config ./triggers.yaml
```

### 缓存机制

分析结果自动缓存，避免重复计算：
- 缓存目录: `~/.datarecipe/cache/`
- 默认 TTL: 7 天
- 自动检测数据集更新（HuggingFace commit hash）

```bash
# 查看缓存
datarecipe cache --list

# 强制重新分析（忽略缓存）
datarecipe deep-analyze dataset/id --force

# 禁用缓存
datarecipe deep-analyze dataset/id --no-cache
```

---

## 知识库

分析结果自动积累到本地知识库 (`~/.datarecipe/knowledge/`)，用于：
- 跨数据集模式发现
- 成本基准比较
- 趋势分析
- 智能推荐

```bash
# 查看成本基准
datarecipe knowledge --benchmarks

# 输出示例:
# | 类型 | 平均成本 | 范围 | 人工% | 数据集数 |
# |------|----------|------|-------|----------|
# | preference | $5,200 | $800-$12,000 | 84% | 5 |
# | evaluation | $8,500 | $2,000-$15,000 | 78% | 8 |

# 获取推荐
datarecipe knowledge --recommend preference

# 生成完整报告
datarecipe knowledge --report -o ./knowledge_report.md
```

---

## MCP 服务器

在 Claude Desktop / Claude App 中直接使用 DataRecipe，**生成与 CLI 完全相同的完整产出物**。

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

### 可用工具

| 工具 | 功能 | 产出物 |
|------|------|--------|
| `deep_analyze` | 深度分析数据集 | 完整产出 ⭐ |
| `get_reproduction_guide` | 获取复刻指南 | 指南全文 |
| `compare_datasets` | 对比多个数据集 | 对比报告 |
| `batch_analyze_from_radar` | 从 Radar 报告批量分析 | 批量产出 |
| `find_similar_datasets` | 找相似数据集 | 相似度列表 |
| `analyze_dataset` | 基础分析 | JSON 摘要 |
| `profile_annotators` | 标注专家画像 | 画像报告 |
| `estimate_cost` | 估算生产成本 | 成本明细 |
| `deploy_project` | 生成投产项目 | 项目脚手架 |

### MCP 产出物

调用 `deep_analyze` 会在 `./analysis_output/<dataset>/` 生成完整文件：

```
analysis_output/
└── tencent_CL-bench/
    ├── REPRODUCTION_GUIDE.md    # 复刻指南 ⭐
    ├── ANALYSIS_REPORT.md       # 分析报告 ⭐
    ├── ANNOTATION_SPEC.md       # 标注规范 ⭐
    ├── COST_BREAKDOWN.md        # 成本明细 ⭐
    ├── recipe_summary.json      # 标准化摘要
    ├── rubric_templates.yaml    # 评分模板
    ├── rubric_templates.md      # 评分文档
    ├── prompt_templates.json    # Prompt 模板
    ├── context_strategy.json    # 上下文策略
    ├── allocation.json          # 人机分配
    ├── annotation_spec.json     # 标注规范 (JSON)
    ├── token_analysis.json      # Token 分析
    ├── cost_comparison.json     # 模型成本对比
    ├── complexity_analysis.json # 复杂度分析
    ├── cost_calibration.json    # 成本校准
    ├── phased_cost.json         # 分阶段成本
    └── llm_analysis.json        # LLM 分析 (可选)
```

### 使用示例

```
用户: 深度分析 tencent/CL-bench 数据集
Claude: [调用 deep_analyze]
        ✅ 已生成完整分析:
        - 类型: evaluation
        - 复刻成本: $5,200 (人工 84%)
        - 产出文件: 16 个 (见 ./analysis_output/tencent_CL-bench/)

用户: 给我复刻指南
Claude: [调用 get_reproduction_guide]
        📋 REPRODUCTION_GUIDE.md 内容:
        # tencent/CL-bench 复刻指南
        ...

用户: 对比 Anthropic/hh-rlhf 和 OpenAI/summarize_from_feedback
Claude: [调用 compare_datasets]
        两者都是偏好数据集:
        - hh-rlhf: $5,200, 人工 84%
        - summarize: $3,800, 人工 76%

用户: Radar 发现了新数据集，帮我分析前 5 个
Claude: [调用 batch_analyze_from_radar]
        已分析 5 个数据集，总复刻成本 $28,000
        每个数据集都已生成完整产出文件
```

---

## 项目架构

```
src/datarecipe/
├── core/                    # 核心分析引擎
│   ├── deep_analyzer.py     # 深度分析主逻辑
│   └── ...
├── cost/                    # 成本估算模块 ⭐
│   ├── token_analyzer.py    # Token 统计与 API 定价
│   ├── complexity_analyzer.py # 内容复杂度分析
│   ├── calibrator.py        # 历史数据校准
│   └── phased_model.py      # 分阶段成本模型
├── generators/              # 文档生成器
│   ├── annotation_spec.py   # 标注规范生成 ⭐
│   ├── reproduction_guide.py # 复刻指南生成
│   └── ...
├── extractors/              # 模式提取器
│   ├── rubric_extractor.py  # 评分标准提取
│   ├── prompt_extractor.py  # Prompt 模板提取
│   └── ...
├── sources/                 # 数据源适配
│   ├── huggingface.py       # HuggingFace 数据集
│   ├── github.py            # GitHub 数据集
│   └── ...
├── mcp_server.py            # MCP 服务器入口
└── cli.py                   # CLI 入口
```

---

## License

[MIT](LICENSE)
