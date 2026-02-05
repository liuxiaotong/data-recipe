<div align="center">

# DataRecipe

**面向人工智能数据集的逆向工程框架**

[![PyPI](https://img.shields.io/pypi/v/datarecipe?color=blue)](https://pypi.org/project/datarecipe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-9_Tools-purple.svg)](#mcp-server)

[快速开始](#快速开始) · [深度分析](#深度分析) · [需求文档分析](#需求文档分析) · [MCP Server](#mcp-server) · [与 Radar 联动](#与-radar-联动)

</div>

---

从数据集样本或需求规格中自动提取构建范式，生成可复用的标注规范与成本模型，支持人机协同的可解释性输出。

## 核心能力

```
输入源 (数据集样本 / 需求规格) → 逆向工程分析 → 结构化产出 (人类可读 + 机器可解析)
```

### 按角色快速导航

| 角色 | 目录 | 用途 |
|------|------|------|
| 👔 **决策层** | `01_决策参考/` | 价值评分、ROI、投资建议 |
| 📋 **项目经理** | `02_项目管理/` | 里程碑、验收标准、风险管理 |
| 📝 **标注团队** | `03_标注规范/` | 标注指南、培训手册、质检清单 |
| 🔧 **技术团队** | `04_复刻指南/` | 生产流程、数据结构、难度验证 |
| 💰 **财务/预算** | `05_成本分析/` | 分阶段成本、人机分配 |
| 🧪 **生产启动** | `09_样例数据/` | 样例数据、自动化评估、人工步骤说明 |

### 输出物一览

| 文件 | 用途 | 消费者 |
|------|------|--------|
| `EXECUTIVE_SUMMARY.md` | 决策摘要 (评分 + ROI) | 人类 |
| `MILESTONE_PLAN.md` | 里程碑计划 | 人类 |
| `ANNOTATION_SPEC.md` | 标注规范 | 人类 |
| `TRAINING_GUIDE.md` | 标注员培训手册 | 人类 |
| `QA_CHECKLIST.md` | 质量检查清单 | 人类 |
| `PRODUCTION_SOP.md` | 生产标准流程 | 人类 |
| `DATA_SCHEMA.json` | 数据格式定义 | 人类 + Agent |
| `DIFFICULTY_VALIDATION.md` | 难度验证流程 (按需) | 人类 |
| `COST_BREAKDOWN.md` | 成本明细 | 人类 |
| `data_template.json` | 数据录入模板 | 人类 + Agent |
| `agent_context.json` | 聚合入口 | Agent |
| `workflow_state.json` | 工作流状态 | Agent |
| `reasoning_traces.json` | 推理链 | Agent + 人类 |
| `pipeline.yaml` | 可执行流水线 | Agent |
| `samples.json` | 样例数据 (最多50条) | Agent + 人类 |
| `SAMPLE_GUIDE.md` | 样例指南与自动化评估 | 人类 |

## 安装

```bash
pip install datarecipe
```

可选依赖：

```bash
pip install datarecipe[llm]      # LLM 分析 (Anthropic/OpenAI)
pip install datarecipe[pdf]      # PDF 解析
pip install datarecipe[mcp]      # MCP 服务器
pip install datarecipe[all]      # 全部功能
```

## 快速开始

### 分析 HuggingFace 数据集

```bash
datarecipe deep-analyze tencent/CL-bench -o ./output
```

<details>
<summary>输出示例</summary>

```
============================================================
  DataRecipe 深度逆向分析
============================================================

数据集: tencent/CL-bench
📥 加载数据集...
✓ 加载完成: 300 样本

📊 分析评分标准...
✓ 评分标准: 4120 条, 2412 种模式
📝 提取 Prompt 模板...
✓ Prompt模板: 293 个独特模板
⚙️ 计算人机分配...
✓ 人机分配: 人工 84%, 机器 16%

============================================================
  分析完成
============================================================

核心产出:
  📄 执行摘要: ./output/tencent_CL-bench/01_决策参考/EXECUTIVE_SUMMARY.md
  📋 里程碑计划: ./output/tencent_CL-bench/02_项目管理/MILESTONE_PLAN.md
  📝 标注规范: ./output/tencent_CL-bench/03_标注规范/ANNOTATION_SPEC.md
```

</details>

### 分析需求文档 (PDF/Word)

```bash
# API 模式 (需要 ANTHROPIC_API_KEY)
datarecipe analyze-spec requirements.pdf -o ./output

# 交互模式 (在 Claude Code 中使用，无需 API key)
datarecipe analyze-spec requirements.pdf --interactive

# 从预计算 JSON 生成
datarecipe analyze-spec requirements.pdf --from-json analysis.json
```

<details>
<summary>输出示例</summary>

```
============================================================
  DataRecipe 需求文档分析
============================================================

文档: ICL需求和样例.pdf
📄 解析文档...
✓ 文档解析完成 (包含 6 张图片)
✓ 加载完成: ICL多模态复杂推理基准

📝 生成项目文档...
✓ 生成完成 (23 个文件)

核心产出:
  📄 执行摘要: ./output/ICL多模态复杂推理基准/01_决策参考/EXECUTIVE_SUMMARY.md
  📝 标注规范: ./output/ICL多模态复杂推理基准/03_标注规范/ANNOTATION_SPEC.md
  📖 培训手册: ./output/ICL多模态复杂推理基准/03_标注规范/TRAINING_GUIDE.md
  🔧 生产流程: ./output/ICL多模态复杂推理基准/04_复刻指南/PRODUCTION_SOP.md
  📋 难度验证: ./output/ICL多模态复杂推理基准/04_复刻指南/DIFFICULTY_VALIDATION.md
```

</details>

---

## 深度分析

从数据集样本中提取构建范式与质量标准，生成可复用的生产规范。

### 输出目录结构

```
output/
└── tencent_CL-bench/
    ├── README.md                        # 目录导航
    ├── recipe_summary.json              # 核心摘要 (Radar 兼容)
    │
    ├── 01_决策参考/                      # 👔 决策层
    │   └── EXECUTIVE_SUMMARY.md         # 执行摘要
    │
    ├── 02_项目管理/                      # 📋 项目经理
    │   ├── MILESTONE_PLAN.md            # 里程碑计划
    │   └── INDUSTRY_BENCHMARK.md        # 行业基准对比
    │
    ├── 03_标注规范/                      # 📝 标注团队
    │   ├── ANNOTATION_SPEC.md           # 标注规范
    │   ├── TRAINING_GUIDE.md            # 培训手册
    │   └── QA_CHECKLIST.md              # 质检清单
    │
    ├── 04_复刻指南/                      # 🔧 技术团队
    │   ├── PRODUCTION_SOP.md            # 生产流程
    │   ├── DATA_SCHEMA.json             # 数据格式
    │   ├── REPRODUCTION_GUIDE.md        # 复刻指南
    │   └── DIFFICULTY_VALIDATION.md     # 难度验证 (按需)
    │
    ├── 05_成本分析/                      # 💰 成本相关
    │   └── COST_BREAKDOWN.md            # 成本明细
    │
    ├── 06_原始数据/                      # 📊 分析数据
    │   └── spec_analysis.json
    │
    ├── 07_模板/                          # 📋 模板
    │   └── data_template.json           # 数据录入模板
    │
    ├── 08_AI_Agent/                      # 🤖 AI Agent 入口
    │   ├── agent_context.json           # 聚合上下文
    │   ├── workflow_state.json          # 工作流状态
    │   ├── reasoning_traces.json        # 推理链
    │   └── pipeline.yaml                # 可执行流水线
    │
    └── 09_样例数据/                       # 🧪 样例数据
        ├── samples.json                 # 样例数据 (最多50条)
        └── SAMPLE_GUIDE.md              # 样例指南
```

### 人机协同的可解释性输出

所有分析管道均生成双重格式输出，支持人类审阅与 AI Agent 自动化消费：

| 人类可读文档 | 机器可解析文件 | 语义映射 |
|--------------|----------------|----------|
| `EXECUTIVE_SUMMARY.md` | `reasoning_traces.json` | 决策依据与推理链 |
| `MILESTONE_PLAN.md` | `workflow_state.json` | 进度状态与阻塞项 |
| `PRODUCTION_SOP.md` | `pipeline.yaml` | 可执行工作流定义 |

机器可解析文件的设计原则：
- **推理可追溯**: 每个结论附带完整的证据链与推理步骤
- **置信度量化**: 明确标注不确定性范围与假设条件
- **引用而非复制**: 通过相对路径引用详细文档，避免信息冗余
- **声明式流水线**: pipeline.yaml 可被 AI Agent 直接解析执行

### 样例数据生成

分析完成后自动生成最多 50 条样例数据，帮助团队快速启动生产：

```
09_样例数据/
├── samples.json        # 机器可解析的样例数据
└── SAMPLE_GUIDE.md     # 人类可读的样例指南
```

**智能自动化评估**：系统根据任务特性自动判断自动化程度

| 自动化程度 | 特征 | 建议工作流 |
|------------|------|------------|
| 🟢 高 (80%+) | 规则明确、无需创意 | 自动生成 → 抽检审核 → 批量提交 |
| 🟡 中 (50%+) | 需要专业知识或难度验证 | 自动初稿 → 人工审核 → 质检 → 提交 |
| 🟠 低 (30%+) | 需要人类创意 | 人工创作 → AI辅助格式化 → 质检 → 提交 |
| 🔴 全人工 | 禁止AI参与 | 人工创作 → 交叉审核 → 质检 → 提交 |

**人工步骤说明**：当无法完全自动化时，样例文件会明确标注：
- 哪些步骤需要人工参与
- 为什么需要人工参与
- 预估工作量 (高/中/低)

### LLM 智能分析

遇到无法识别的数据集类型时，使用 LLM 进行智能分析：

```bash
export ANTHROPIC_API_KEY=your_key
datarecipe deep-analyze unknown/dataset --use-llm
```

---

## 需求文档分析

从需求规格文档直接提取数据构建范式，无需现有数据集样本。

### 支持格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| PDF | `.pdf` | 支持图片提取 |
| Word | `.docx` | 支持表格和图片 |
| 图片 | `.png`, `.jpg` | 多模态输入 |
| 文本 | `.txt`, `.md` | 纯文本 |

### 智能难度验证

当需求文档中包含难度验证要求时（如「用 doubao1.8 跑 3 次，最多 1 次正确」），系统会：

1. **自动提取验证配置**：模型名称、设置、测试次数、通过标准
2. **生成 DIFFICULTY_VALIDATION.md**：完整的验证流程和记录模板
3. **更新相关文档**：培训手册、质检清单、数据模板都会包含验证要求

如果文档中没有难度验证要求，则不生成该文件。

---

## MCP Server

在 Claude Desktop / Claude Code 中直接使用，无需 API key。

### 配置

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

| 工具 | 功能 |
|------|------|
| `parse_spec_document` | 解析需求文档 (PDF/Word/图片) |
| `generate_spec_output` | 生成项目文档 (23 个文件) |
| `analyze_huggingface_dataset` | 深度分析 HuggingFace 数据集 |
| `get_extraction_prompt` | 获取 LLM 提取模板 |
| `extract_rubrics` | 提取评分标准模式 |
| `extract_prompts` | 提取 Prompt 模板 |
| `compare_datasets` | 对比多个数据集 |
| `profile_dataset` | 数据集画像与成本估算 |
| `get_agent_context` | 获取 AI Agent 上下文 |

### 使用示例

```
用户: 帮我分析这个需求文档 /path/to/requirements.pdf

Claude: [调用 parse_spec_document]
        📄 文档解析完成 (包含 6 张图片)
        [分析文档，自动识别难度验证要求: doubao1.8 跑 3 次]

        [调用 generate_spec_output]
        ✅ 已生成 23 个文件:
        - 执行摘要、里程碑计划、标注规范
        - 培训手册、质检清单、生产流程
        - 难度验证、数据模板...
```

---

## 与 Radar 联动

联合 [AI Dataset Radar](https://github.com/liuxiaotong/ai-dataset-radar) 实现完整工作流：

```
Radar (发现数据集) → Recipe (逆向分析) → 复刻生产
```

### 双 MCP 配置

```json
{
  "mcpServers": {
    "ai-dataset-radar": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/ai-dataset-radar/mcp_server/server.py"]
    },
    "datarecipe": {
      "command": "uv",
      "args": ["--directory", "/path/to/data-recipe", "run", "datarecipe-mcp"]
    }
  }
}
```

### 工作流示例

```
用户: 扫描这周的数据集，找一个 SFT 类型的深度分析

Claude 自动执行:
  1. [radar_scan] → 获取 15 个数据集
  2. [radar_datasets category=sft] → allenai/Dolci-Instruct-SFT
  3. [datarecipe deep_analyze] → 生成逆向分析报告
  4. 返回：标注规范、成本估算、复刻指南
```

### 批量分析

```bash
# 从 Radar 报告批量分析
datarecipe batch-from-radar ./intel_report.json --limit 10

# 按条件筛选
datarecipe batch-from-radar ./report.json \
  --orgs Anthropic,OpenAI \
  --categories preference,sft \
  --min-downloads 1000
```

---

## 命令参考

| 命令 | 功能 |
|------|------|
| `analyze <dataset>` | 快速分析数据集 |
| `deep-analyze <dataset>` | 深度分析，生成完整报告 |
| `analyze-spec <file>` | 分析需求文档 |
| `analyze-spec <file> -i` | 交互模式 (Claude Code) |
| `profile <dataset>` | 标注员画像与成本估算 |
| `extract-rubrics <dataset>` | 提取评分标准 |
| `batch-from-radar <report>` | 从 Radar 报告批量分析 |
| `deploy <dataset>` | 输出生产级项目结构 |

---

## 项目架构

```
src/datarecipe/
├── analyzers/               # 分析器
│   ├── spec_analyzer.py     # 需求文档分析 (LLM 提取)
│   └── llm_dataset_analyzer.py
├── parsers/                 # 文档解析
│   └── document_parser.py   # PDF/Word/图片
├── generators/              # 文档生成
│   ├── spec_output.py       # 需求文档产出 (23 个文件)
│   ├── executive_summary.py
│   ├── milestone_plan.py
│   └── annotation_spec.py
├── cost/                    # 成本估算
│   ├── token_analyzer.py
│   └── phased_model.py
├── extractors/              # 模式提取
│   ├── rubric_extractor.py
│   └── prompt_extractor.py
├── mcp_server.py            # MCP Server (9 工具)
└── cli.py                   # CLI 入口
```

---

## License

[MIT](LICENSE)

---

<div align="center">
<sub>为数据工程团队、标注服务商及 AI 数据集研究者提供可复用的逆向工程方法论</sub>
</div>
