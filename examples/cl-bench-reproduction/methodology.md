# CL-bench 复现方法论

> 基于论文解析和数据集逆向工程

## 1. 核心理念

CL-bench 的核心目标是测试 LLM 能否**从上下文中学习新知识**并正确应用，而不是依赖预训练知识。

### 关键设计原则

| 原则 | 说明 | 验证方法 |
|------|------|----------|
| **知识污染防护** | 任务不能用预训练知识解决 | GPT-5.1 无上下文正确率 < 1% |
| **新知识来源** | 专家构建或长尾来源 | 虚构系统 / 修改现实 / 小众领域 |
| **多维验证** | 每个任务多个验收标准 | 平均 16.6 条 rubrics/任务 |

---

## 2. 数据结构解析

### 2.1 整体结构

```
CL-bench Dataset
├── 1,899 个任务 (tasks)
├── 500 个复杂上下文 (contexts)
└── 31,607 条验证标准 (rubrics)
```

### 2.2 单条数据格式

```json
{
  "messages": [
    {"role": "system", "content": "系统指令：定义任务角色和行为规范"},
    {"role": "user", "content": "上下文 + 具体问题（可能非常长，最大 158K 字符）"},
    {"role": "assistant", "content": "期望回答（可选，用于few-shot）"}
  ],
  "rubrics": [
    "验证标准1：回答应包含...",
    "验证标准2：回答应正确解释...",
    "..."
  ],
  "metadata": {
    "task_id": "唯一标识",
    "context_category": "主类别",
    "sub_category": "子类别"
  }
}
```

### 2.3 统计特征

| 维度 | 最小 | 最大 | 平均 |
|------|------|------|------|
| 消息轮数 | 2 | 12 | 3.0 |
| Rubrics 数量 | 3 | 114 | 16.6 |
| User 内容长度 | ~1K | ~158K | ~30K |

---

## 3. 四大知识领域

### 3.1 领域分布

```
Domain Knowledge Reasoning    ████████████████████  34.9% (663)
Rule System Application       ████████████████      29.8% (566)
Procedural Task Execution     █████████████         24.8% (471)
Empirical Discovery           ██████                10.5% (199)
```

### 3.2 子领域 Top 10

| 子领域 | 数量 | 主领域 |
|--------|------|--------|
| Workflow Orchestration | 229 | Procedural Task |
| Technical Standards | 201 | Domain Knowledge |
| Operational Procedures | 185 | Procedural Task |
| Game Mechanics | 137 | Rule System |
| Humanities | 124 | Domain Knowledge |
| Management | 112 | Domain Knowledge |
| Healthcare | 105 | Domain Knowledge |
| Finance | 101 | Domain Knowledge |
| Observational Data | 95 | Empirical Discovery |
| Legal & Regulatory | 92 | Rule System |

---

## 4. 上下文构建方法

### 4.1 三种构建策略

#### 策略 A：虚构创作 (Fictional Creation)

**适用场景**：规则系统、游戏机制

**示例**：Twisted Cryptids 桌游
```
- 完全虚构的桌游规则
- 包含角色、卡牌、胜利条件等完整系统
- 确保预训练数据中不存在
```

**实施步骤**：
1. 确定系统类型（游戏/法律/组织）
2. 设计核心机制和规则
3. 编写完整文档（规则书形式）
4. 创建边界情况和特例

#### 策略 B：修改现实 (Modification)

**适用场景**：领域知识、技术标准

**示例**：修改后的医疗流程
```
- 基于真实医疗流程
- 修改关键参数、条件、步骤
- 保持逻辑一致性
- 确保与原版不同
```

**实施步骤**：
1. 选择现实知识源
2. 识别可修改的核心要素
3. 系统性修改（不是随机）
4. 验证内部一致性

#### 策略 C：小众来源 (Niche Sources)

**适用场景**：新产品、前沿研究

**示例**：新发布产品手册
```
- 刚发布的产品文档
- 小众领域专业知识
- 新兴研究发现
- 不太可能出现在训练数据
```

**实施步骤**：
1. 筛选长尾/新兴内容
2. 验证未被广泛传播
3. 结构化整理
4. 添加复杂场景

### 4.2 上下文质量标准

| 标准 | 要求 |
|------|------|
| 完整性 | 包含解决任务所需的所有信息 |
| 自洽性 | 内部规则不矛盾 |
| 独特性 | 不能从预训练知识推断 |
| 复杂度 | 支持多层次、多步骤推理 |
| 工作量 | 约 20 小时/上下文 |

---

## 5. 任务设计方法

### 5.1 任务结构

```
任务 = System Prompt + Context + Question + Rubrics
```

**System Prompt 模板**：
```
You are an AI designed to [任务描述].
Your purpose is to [职责列表].
You should [行为规范].
Do not [限制条件].
Support [受众类型].
If [条件], [默认行为].
```

### 5.2 任务依赖设计

**51.1% 的任务需要序列依赖**

```
任务链示例：
├── 任务 1: 理解规则基础
├── 任务 2: 应用规则到场景 A（依赖任务1）
├── 任务 3: 处理异常情况（依赖任务1,2）
└── 任务 4: 综合决策（依赖任务1,2,3）
```

### 5.3 难度分级

| 级别 | 消息轮数 | Rubrics | 特点 |
|------|----------|---------|------|
| 简单 | 2 | 3-5 | 单一规则应用 |
| 中等 | 3-4 | 10-20 | 多规则组合 |
| 困难 | 5-8 | 30-50 | 状态追踪 + 推理 |
| 极难 | 9-12 | 50-114 | 长程依赖 + 复杂决策 |

---

## 6. Rubrics 设计方法

### 6.1 Rubrics 结构

每条 Rubric 是一个**自然语言描述的验收标准**：

```
"The response should [动作] [对象] [条件/细节]"

示例：
- "The response should define what a Sighting card is and its role when playing Twisted Cryptids."
- "The response should name the four different sighting card types, namely: Decoys, Hoaxes, Silhouettes, and Genuine Sightings."
- "The response should state the requirements for when sighting cards can be revealed."
```

### 6.2 Rubrics 类型

| 类型 | 说明 | 示例 |
|------|------|------|
| 定义型 | 检查是否正确定义概念 | "should define X as..." |
| 列举型 | 检查是否列出所有项目 | "should name all four types..." |
| 条件型 | 检查是否说明条件 | "should state when X can be done..." |
| 推理型 | 检查推理过程 | "should explain why X leads to Y..." |
| 决策型 | 检查决策合理性 | "should recommend X because..." |

### 6.3 Rubrics 数量指南

| 任务复杂度 | 建议 Rubrics 数量 |
|------------|-------------------|
| 单一知识点 | 3-5 |
| 多知识点组合 | 8-15 |
| 复杂推理链 | 20-40 |
| 完整流程验证 | 50-100 |

---

## 7. 评估方法

### 7.1 评分机制

**二元评分**：
- Score 1: 满足所有 Rubric 要求
- Score 0: 任一 Rubric 未满足或输出为空

**通过率计算**：
```
Solving Rate = Score 1 的数量 / 总样本数
```

### 7.2 LM-based 评估

使用 LLM 作为评判者（Judge）：

```python
def evaluate(response, rubrics):
    for rubric in rubrics:
        prompt = f"""
        Response: {response}
        Rubric: {rubric}

        Does the response satisfy this rubric? (yes/no)
        """
        if judge_model(prompt) == "no":
            return 0
    return 1
```

### 7.3 基准结果

| 模型 | 通过率 |
|------|--------|
| GPT-5.1 (high) | 23.7% |
| 平均水平 | 17.2% |
| 无上下文基线 | < 1% |

---

## 8. 复现清单

### Phase 1: 准备 (2周)

- [ ] 组建领域专家团队（4个领域 × 2-3人）
- [ ] 确定知识构建策略
- [ ] 准备文档模板
- [ ] 设置标注工具

### Phase 2: 上下文构建 (8周)

- [ ] 每领域构建 125 个上下文
- [ ] 每上下文约 20 小时
- [ ] 多轮质量审核
- [ ] 验证知识独特性

### Phase 3: 任务设计 (4周)

- [ ] 每上下文设计 3-4 个任务
- [ ] 设计任务依赖关系
- [ ] 编写 System Prompt
- [ ] 设计多轮对话

### Phase 4: Rubrics 编写 (4周)

- [ ] 每任务 15-20 条 Rubrics
- [ ] 覆盖多个验证维度
- [ ] 确保可验证性
- [ ] 交叉审核

### Phase 5: 验证 (2周)

- [ ] 无上下文测试（应 < 1%）
- [ ] 有上下文基线测试
- [ ] 调整难度分布
- [ ] 最终质检

---

## 9. 资源估算

| 项目 | 数量 | 单价 | 总成本 |
|------|------|------|--------|
| 领域专家 | 10人 × 4周 | $80/小时 | $128,000 |
| 上下文构建 | 500 × 20小时 | $50/小时 | $500,000 |
| 任务设计 | 1,899 × 4小时 | $40/小时 | $303,840 |
| Rubrics 编写 | 31,607 × 0.5小时 | $30/小时 | $474,105 |
| 质检审核 | 20% 抽检 | - | $80,000 |
| **总计** | | | **~$1,500,000** |

> 注：这是完整复现的估算。如果使用 LLM 辅助生成+人工审核，成本可降至 20-30%。

---

## 参考资源

- [CL-bench GitHub](https://github.com/Tencent-Hunyuan/CL-bench)
- [CL-bench HuggingFace](https://huggingface.co/datasets/tencent/CL-bench)
- [官方排行榜](https://www.clbench.com)
- [36Kr 报道](https://eu.36kr.com/en/p/3667552328868488)

---
*由 DataRecipe 基于论文解析和数据逆向工程生成*
