# CL-bench 数据生产完整指南

> 本文档供数据生产团队使用，包含从零开始批量生产「上下文学习评估数据」的全部信息。

---

## 一、项目概述

### 1.1 我们要生产什么

| 项目 | 规格 |
|------|------|
| **数据类型** | 上下文学习评估数据 (Context Learning Benchmark) |
| **核心特点** | 模型必须从提供的上下文中学习新知识才能回答 |
| **数据格式** | JSONL，每行一个完整任务 |
| **质量标准** | 无上下文时模型正确率 < 1% |

### 1.2 单条数据结构

```json
{
  "messages": [
    {"role": "system", "content": "系统指令（定义AI角色和行为规范）"},
    {"role": "user", "content": "上下文文档 + 具体问题"}
  ],
  "rubrics": [
    "The response should...",
    "The response should..."
  ],
  "metadata": {
    "task_id": "唯一ID",
    "context_category": "主类别",
    "sub_category": "子类别"
  }
}
```

### 1.3 目标产出

| 指标 | 目标值 |
|------|--------|
| 上下文数量 | 500 个 |
| 任务数量 | 1,899 个 |
| Rubrics 数量 | 31,607 条 |
| 平均 Context 长度 | 35,000 字符 |
| 平均 Rubrics/任务 | 16.6 条 |

---

## 二、人机分工总览

> **核心原则**：机器提供模板和框架，人类提供创意和专业判断。

### 2.1 人类必须完成的工作

| 任务 | 为什么必须人类做 | 工作量占比 | 技能要求 |
|------|------------------|-----------|----------|
| **1. Context 内容创作** | 需要原创性，确保不在模型预训练数据中 | 40% | 领域专家 |
| **2. 任务问题设计** | 需要教学设计思维，判断什么问题有区分度 | 25% | 测试设计经验 |
| **3. Rubrics 定制** | 需要理解 Context 细节，写出精确的验证标准 | 20% | 逻辑严谨 |
| **4. 质量审核** | 需要判断数据是否真的测试"上下文学习"能力 | 10% | 数据质量经验 |
| **5. 边界案例处理** | 需要判断模糊情况、解决标注冲突 | 5% | 经验丰富 |

### 2.2 机器/脚本可自动完成的工作

| 任务 | 自动化程度 | 使用工具 |
|------|-----------|----------|
| **System Prompt 生成** | 100% 自动 | 495 个模板直接复用 |
| **Rubrics 句式生成** | 80% 自动 | `generate_rubrics.py` 生成框架，人类填充细节 |
| **数据格式组装** | 100% 自动 | `batch_production_demo.py` |
| **统计分析** | 100% 自动 | `analyze_rubrics.py` |
| **批量导出** | 100% 自动 | 脚本自动输出 JSONL |
| **基础质检** | 70% 自动 | 格式检查、长度检查、重复检测 |

### 2.3 人机协作流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据生产流水线                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [机器] 提供模板           [人类] 创作内容           [机器] 组装输出   │
│  ─────────────            ─────────────            ─────────────    │
│                                                                      │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐   │
│  │ 领域模板    │ ──────▶ │ 专家撰写    │ ──────▶ │ 格式化      │   │
│  │ 子类别指南  │         │ Context     │         │ 验证        │   │
│  └─────────────┘         └─────────────┘         └─────────────┘   │
│         │                       │                       │           │
│         ▼                       ▼                       ▼           │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐   │
│  │ 问题类型库  │ ──────▶ │ 设计师出题  │ ──────▶ │ 难度标注    │   │
│  │ 难度框架    │         │ Task        │         │ 去重        │   │
│  └─────────────┘         └─────────────┘         └─────────────┘   │
│         │                       │                       │           │
│         ▼                       ▼                       ▼           │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐   │
│  │ Rubric 模板 │ ──────▶ │ 编写员定制  │ ──────▶ │ 批量导出    │   │
│  │ 动词库      │         │ 具体 Rubric │         │ JSONL       │   │
│  └─────────────┘         └─────────────┘         └─────────────┘   │
│                                                                      │
│  ◀──────── 自动化 ────────▶ ◀───── 人工 ─────▶ ◀──── 自动化 ────▶   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 人类工作量估算

以生产 **500 Context / 1,899 Task** 为例：

| 工作项 | 单位耗时 | 数量 | 总工时 | 人力配置 |
|--------|---------|------|--------|----------|
| Context 创作 | 20 小时/个 | 500 | 10,000 小时 | 10 人 × 4 个月 |
| Task 设计 | 4 小时/个 | 1,899 | 7,596 小时 | 6 人 × 5 个月 |
| Rubrics 编写 | 0.5 小时/条 | 31,607 | 15,804 小时 | 6 人 × 10 个月 |
| 质量审核 | 0.2 小时/Task | 1,899 | 380 小时 | 2 人 × 1 个月 |
| **总计** | | | **33,780 小时** | **~20 人 × 6 个月** |

> 💡 **关键洞察**：Rubrics 编写占总工时的 47%，是最耗人力的环节。这就是为什么我们提供了 Rubrics 模板生成器——它能将单条 Rubrics 编写时间从 30 分钟降到 5-10 分钟。

---

## 三、Context 构建指南

### 3.1 四大领域

| 领域 | 占比 | 描述 | 示例 |
|------|------|------|------|
| **Domain Knowledge Reasoning** | 35% | 专业领域知识 | 医疗、法律、金融文档 |
| **Rule System Application** | 30% | 规则系统 | 游戏规则、技术规范 |
| **Procedural Task Execution** | 25% | 流程执行 | 业务流程、操作手册 |
| **Empirical Discovery** | 10% | 经验发现 | 实验数据、观察记录 |

### 3.2 三种构建策略

#### 策略 A：虚构创作 (推荐用于规则系统)

**方法**：从零创建完全虚构的内容

**步骤**：
1. 确定系统类型（游戏/法规/组织制度）
2. 设计核心机制（至少 5 个相互关联的规则）
3. 添加例外情况（至少 3 个特殊规则）
4. 编写完整文档（规则书格式）
5. 内部自洽性检查

**示例**：虚构桌游「星际殖民者」的完整规则书

**优点**：确保知识不存在于模型预训练数据中

#### 策略 B：修改现实 (推荐用于专业知识)

**方法**：基于真实内容进行系统性修改

**步骤**：
1. 选择真实文档（技术手册、医疗指南等）
2. 识别可修改的核心参数
3. 系统性修改（保持内部一致性）
4. 验证修改后的逻辑完整性
5. 确保与原版明显不同

**示例**：修改后的药物使用指南（改变剂量、适应症）

**注意**：修改必须足够大，避免模型用预训练知识「猜中」

#### 策略 C：小众来源 (推荐用于经验发现)

**方法**：使用长尾/新兴/小众内容

**步骤**：
1. 寻找新发布的产品手册
2. 收集小众领域专业文档
3. 整理前沿研究数据
4. 结构化处理
5. 验证内容未被广泛传播

**示例**：新发布的企业内部系统操作手册

### 3.3 Context 质量标准

| 标准 | 要求 | 检查方法 |
|------|------|----------|
| **完整性** | 包含解决任务所需的全部信息 | 任务可答性测试 |
| **自洽性** | 内部规则不矛盾 | 交叉引用检查 |
| **独特性** | 不能从预训练知识推断 | 无上下文测试 < 1% |
| **复杂度** | 支持多层次推理 | 至少 3 层嵌套规则 |
| **长度** | 10,000 - 100,000 字符 | 字符计数 |

### 3.4 Context 模板

```markdown
# [文档标题]

## 1. 概述
[简要介绍本文档的目的和适用范围]

## 2. 术语定义
| 术语 | 定义 |
|------|------|
| [术语1] | [定义1] |
| [术语2] | [定义2] |
...（至少 10 个）

## 3. 核心内容
### 3.1 [主题1]
[详细说明，包含规则、条件、例外]

### 3.2 [主题2]
[详细说明]

...（至少 5 个主题）

## 4. 特殊规则/例外情况
[至少 5 个特殊情况的处理方式]

## 5. 示例/案例
[至少 3 个完整示例]

## 6. 常见问题
[至少 5 个 Q&A]
```

---

## 四、Task 设计指南

### 4.1 System Prompt 模板

```
You are an AI assistant specialized in [领域/系统名称].

Your role is to:
1. [职责1：如解释规则]
2. [职责2：如回答问题]
3. [职责3：如处理特殊情况]

Important guidelines:
- Only use information from the provided [context type]
- Do not invent or assume information not explicitly stated
- When uncertain, acknowledge the limitation
- [其他领域特定指南]

You should be [tone: helpful/precise/professional] in your responses.
```

### 4.2 问题类型

| 类型 | 占比 | 示例 |
|------|------|------|
| **定义型** | 15% | "What is X?" |
| **流程型** | 25% | "How does X work?" |
| **条件型** | 25% | "What happens when X?" |
| **判断型** | 20% | "Is X allowed in situation Y?" |
| **计算型** | 10% | "Calculate X given Y" |
| **综合型** | 5% | 多步骤推理问题 |

### 4.3 难度分级

| 级别 | 消息轮数 | Rubrics | 特点 |
|------|----------|---------|------|
| **简单** | 2 | 3-5 | 单一知识点 |
| **中等** | 3-4 | 8-15 | 多知识点组合 |
| **困难** | 5-8 | 20-40 | 多步推理 |
| **极难** | 9-12 | 50+ | 长程依赖 |

**分布建议**：简单 20%、中等 50%、困难 25%、极难 5%

### 4.4 多轮对话设计

```
Turn 1: 基础问题
Turn 2: 基于 Turn 1 答案的追问
Turn 3: 引入新条件
Turn 4: 综合判断
...
```

**51% 的任务应有序列依赖**（后续问题依赖前面的答案）

---

## 五、Rubrics 编写指南

### 5.1 基本格式

```
The response should [动词] [对象] [条件/细节].
```

### 5.2 动词选择

| 动词 | 用途 | 频率 |
|------|------|------|
| **include/contain** | 检查是否包含内容 | 2.5% |
| **state/mention** | 检查是否陈述事实 | 2.4% |
| **explain/describe** | 检查是否解释清楚 | 1.1% |
| **provide** | 检查是否提供信息 | 1.9% |
| **identify/name/list** | 检查是否识别/列举 | 1.5% |
| **not/avoid** | 检查是否避免错误 | 3.2% |

### 5.3 Rubric 类型与模板

#### 定义型
```
The response should define what [concept] is according to the provided [document type].
The response should clearly explain the meaning of [term] as specified in the context.
```

#### 列举型
```
The response should list all [N] [items], namely: [item1], [item2], [item3], [item4].
The response should identify at least [N] [items] from the context.
```

#### 解释型
```
The response should explain how [process] works based on the provided rules.
The response should describe the [procedure] step by step.
```

#### 条件型
```
The response should explain what happens when [condition].
The response should describe the correct procedure if [situation].
```

#### 否定型
```
The response should not [incorrect action].
The response should avoid [common mistake].
The response should not assume [unsupported assumption].
```

#### 包含型
```
The response should include [specific content].
The response should contain a [section type] that [description].
```

### 5.4 增强具体性

| 技巧 | 示例 |
|------|------|
| **namely 列举** | "...namely: A, B, C, D" |
| **at least 数量** | "...at least 3 examples" |
| **specifically 强调** | "...specifically mention X" |
| **引号术语** | "...the term 'X'" |
| **for example 举例** | "...for example, when X happens" |

### 5.5 Rubrics 数量指南

| 任务复杂度 | Rubrics 数量 |
|------------|-------------|
| 单一知识点 | 3-5 |
| 多知识点 | 8-15 |
| 复杂推理 | 20-40 |
| 完整流程 | 50-100 |

---

## 六、质量控制

### 6.1 三级审核流程

```
Level 1: 自检
  ↓
Level 2: 交叉审核（同级互审）
  ↓
Level 3: 质检员抽检（20%）
```

### 6.2 质检规则

| 规则ID | 名称 | 严重程度 | 检查方式 |
|--------|------|----------|----------|
| QR001 | 非空检查 | Error | 自动 |
| QR002 | 长度检查 | Warning | 自动 |
| QR003 | 重复检查 | Error | 自动 |
| QR004 | 格式规范 | Error | 自动 |
| QR005 | 事实性检查 | Error | 人工 |
| QR006 | 自洽性检查 | Error | 人工 |
| QR007 | 独特性检查 | Error | 测试 |

### 6.3 验收标准

| 指标 | 阈值 | 优先级 |
|------|------|--------|
| 完成率 | ≥ 98% | 必须 |
| 准确率（抽检） | ≥ 95% | 必须 |
| 一致性（Kappa） | ≥ 0.7 | 必须 |
| 格式合规 | 100% | 必须 |
| 独特性测试 | < 1% | 必须 |

### 6.4 独特性测试方法

```python
# 使用模型测试无上下文正确率
for task in tasks:
    # 移除 context，只保留问题
    question_only = extract_question(task)

    # 让模型回答
    response = model.generate(question_only)

    # 评估是否正确
    score = evaluate(response, task.rubrics)

# 通过标准：正确率 < 1%
```

---

## 七、生产流程

### 7.1 阶段划分

```
阶段 1: 准备 (1周)
├── 团队组建与培训
├── 工具配置
└── 样例数据准备

阶段 2: 试产 (2周)
├── 每类别生产 5 个 Context
├── 问题收集与解决
├── 指南修订
└── 质量基线建立

阶段 3: 正式生产 (8周)
├── 按计划批量生产
├── 每周质量报告
└── 持续优化

阶段 4: 验收 (2周)
├── 全量质检
├── 独特性测试
└── 最终交付
```

### 7.2 每日工作流

```
09:00 - 09:30  晨会，分配当日任务
09:30 - 12:00  Context 构建 / Task 设计
12:00 - 13:00  午休
13:00 - 17:00  Rubrics 编写 / 交叉审核
17:00 - 17:30  日报提交，问题汇总
```

### 7.3 产出追踪

| 周次 | Context | Task | Rubrics |
|------|---------|------|---------|
| Week 1 | 25 | 95 | 1,580 |
| Week 2 | 50 | 190 | 3,160 |
| Week 3 | 100 | 380 | 6,320 |
| ... | ... | ... | ... |
| Week 8 | 500 | 1,900 | 31,600 |

---

## 八、工具使用

### 8.1 数据生成脚本

```bash
# 批量生成数据
python scripts/batch_production_demo.py --count 100

# 使用 LLM 扩展 Context
python scripts/04_generate_benchmark.py --domain game_rules --num-contexts 50

# 分析 Rubrics 模式
python scripts/analyze_rubrics.py
```

### 8.2 验证脚本

```bash
# 格式验证
python scripts/03_validate.py --input data/produced.jsonl

# 独特性测试（需要 API Key）
python scripts/02_inference.py --model gpt-4o --input data/produced.jsonl --no-context
python scripts/03_evaluate.py --input data/responses.jsonl
```

### 8.3 输出格式验证

```python
import json

def validate_sample(sample):
    assert "messages" in sample
    assert len(sample["messages"]) >= 2
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["role"] == "user"
    assert "rubrics" in sample
    assert len(sample["rubrics"]) >= 3
    assert "metadata" in sample
    assert "task_id" in sample["metadata"]
    return True
```

---

## 九、常见问题

### Q1: Context 多长合适？
A: 建议 10,000-50,000 字符。太短信息量不足，太长增加生产成本。

### Q2: 如何保证独特性？
A: 使用虚构策略 + 无上下文测试。如果模型能猜对，说明知识不够独特。

### Q3: Rubrics 写多少条？
A: 每个任务 10-20 条。覆盖：定义、列举、解释、条件、否定。

### Q4: 发现数据有问题怎么办？
A: 标记问题类型，返回修改，重新提交审核。

### Q5: 如何处理模糊情况？
A: 记录问题，团队讨论，更新指南，保持一致性。

---

## 十、附录

### A. 文件清单

```
reproduction_kit/
├── sample_*.json              # 4个类别完整样本
├── system_prompt_templates.json  # System Prompt 模板库
├── subcategory_analysis.json  # 18个子类别分析
├── context_patterns.json      # Context 构建模式
├── judge_prompts.json         # 评估 Prompt
└── reproduction_checklist.md  # 检查清单

data/
├── cl_bench_full.jsonl        # 原始数据参考
├── rubrics_analysis.json      # Rubrics 模式分析
└── statistics.json            # 统计信息

scripts/
├── batch_production_demo.py   # 批量生产脚本
├── generate_rubrics.py        # Rubrics 生成器
└── analyze_rubrics.py         # Rubrics 分析器
```

### B. 参考资源

- [CL-bench GitHub](https://github.com/Tencent-Hunyuan/CL-bench)
- [CL-bench HuggingFace](https://huggingface.co/datasets/tencent/CL-bench)
- [官方排行榜](https://www.clbench.com)

---

*文档版本: 1.0*
*生成日期: 2026-02-04*
*由 DataRecipe 生成*
