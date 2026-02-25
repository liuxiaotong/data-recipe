<div align="right">

[English](landing-en.md) | **中文**

</div>

<div align="center">

<h1>DataRecipe</h1>

<h3>自动化数据集逆向工程<br/>与复刻成本估算</h3>

<p><em>逆向分析任意 AI 数据集：提取 Schema、估算成本、从样本或需求文档生成生产级文档</em></p>

<p>
<a href="https://github.com/liuxiaotong/data-recipe">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datarecipe/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>
</p>

</div>

## 为什么选择 DataRecipe？

复刻一个 AI 数据集需要回答三个问题：**数据长什么样**（Schema）、**要花多少钱**（Cost）、**怎么做**（Methodology）。现有方法依赖人工阅读论文、检查样本、编写规范——耗时数天，且无法跨数据集复用。

**DataRecipe 将整个逆向工程流程自动化。** 输入一个 HuggingFace 数据集或需求文档（PDF / Word / 图片），它会：

- **推断 Schema** —— 字段类型、约束、分布
- **提取评分标准与 Prompt** —— 标注维度、评分规则、Prompt 模板
- **建模成本** —— Token 级精确分析、分阶段成本明细、人机分配比例
- **生成 23+ 生产文档** —— 覆盖 6 类角色（决策层、项目经理、标注团队、技术团队、财务、AI Agent）
- **LLM 增强** —— 一次 LLM 调用生成 `EnhancedContext`，将模板化文档升级为具备领域洞察的专业分析

## 快速上手

```bash
pip install knowlyr-datarecipe

# 分析 HuggingFace 数据集（纯本地，无需 API key）
knowlyr-datarecipe deep-analyze tencent/CL-bench

# 启用 LLM 增强，获得更丰富的输出
knowlyr-datarecipe deep-analyze tencent/CL-bench --use-llm

# 分析需求文档
knowlyr-datarecipe analyze-spec requirements.pdf
```

可选依赖：`pip install knowlyr-datarecipe[llm]`（Anthropic/OpenAI）、`[pdf]`、`[mcp]` 或 `[all]`。

## 六阶段分析流水线

```mermaid
graph LR
    I["输入<br/>HF 数据集 / PDF / Word"] --> A1["Schema<br/>推断"]
    A1 --> A2["评分标准<br/>提取"]
    A2 --> A3["Prompt<br/>提取"]
    A3 --> A4["成本<br/>建模"]
    A4 --> A5["人机<br/>分配"]
    A5 --> A6["行业<br/>基准对比"]
    A6 --> E["LLM 增强器<br/>EnhancedContext"]
    E --> G["生成器<br/>23+ 文档"]

    style A1 fill:#0969da,color:#fff,stroke:#0969da
    style E fill:#8b5cf6,color:#fff,stroke:#8b5cf6
    style G fill:#2da44e,color:#fff,stroke:#2da44e
    style I fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A2 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A3 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A4 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A5 fill:#1a1a2e,color:#e0e0e0,stroke:#444
    style A6 fill:#1a1a2e,color:#e0e0e0,stroke:#444
```

每个阶段同时输出人类可读（Markdown）和机器可解析（JSON/YAML）格式。LLM 增强层支持三种模式：`auto`（自动检测环境）、`interactive`（宿主 LLM 处理）、`api`（独立调用 Anthropic / OpenAI）。

## 核心特性

| 特性 | 说明 |
|:---|:---|
| **多源输入** | HuggingFace 数据集、PDF、Word、图片、纯文本 |
| **Token 级成本分析** | 分阶段成本模型，含人机分配比例与行业基准对比 |
| **面向角色的文档** | 23+ 文档，覆盖决策层、项目经理、标注团队、技术、财务、AI Agent |
| **Agent 可消费的输出** | 结构化上下文、工作流状态、推理链、可执行流水线 |
| **Radar 集成** | 批量分析 AI Dataset Radar 发现的数据集 |
| **12 个 MCP 工具** | 无缝集成 AI IDE，分析、增强、对比一站完成 |
| **3572 测试，97% 覆盖率** | 生产级可靠性 |

## 生态系统

DataRecipe 是 **knowlyr** 数据基础设施的一部分：

| 层 | 项目 | 职责 |
|:---|:---|:---|
| 发现 | **AI Dataset Radar** | 数据集竞争情报、趋势分析 |
| 分析 | **DataRecipe** | 逆向工程、Schema 推断、成本建模 |
| 生产 | **DataSynth** / **DataLabel** | LLM 批量合成 / 轻量标注 |
| 质量 | **DataCheck** | 规则验证、异常检测、自动修复 |
| 审计 | **ModelAudit** | 蒸馏检测、模型指纹 |

```bash
# 端到端工作流
knowlyr-datarecipe deep-analyze tencent/CL-bench --use-llm      # 分析
knowlyr-datalabel generate ./projects/tencent_CL-bench/          # 标注
knowlyr-datasynth generate ./projects/tencent_CL-bench/ -n 1000  # 合成
knowlyr-datacheck validate ./projects/tencent_CL-bench/          # 质检
```

<div align="center">
<br/>
<a href="https://github.com/liuxiaotong/data-recipe">GitHub</a> ·
<a href="https://pypi.org/project/knowlyr-datarecipe/">PyPI</a> ·
<a href="https://knowlyr.com">knowlyr.com</a>
<br/><br/>
<sub><a href="https://github.com/liuxiaotong">knowlyr</a> — 自动化数据集逆向工程与复刻成本估算</sub>
</div>
