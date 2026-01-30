# DataRecipe: A Framework for Reverse-Engineering AI Dataset Construction Pipelines

---

## Abstract

The proliferation of large language models has created an unprecedented demand for high-quality training datasets. However, while model architectures and training procedures are increasingly transparent, the construction methodologies of these datasets remain opaque. This paper presents DataRecipe, a systematic framework for analyzing dataset provenance—analogous to "nutrition labels" for food products. Our tool extracts metadata from multiple sources (HuggingFace Hub, GitHub repositories, and web pages) to identify generation methodologies, teacher models used in distillation, estimated creation costs, and reproducibility scores. Beyond analysis, DataRecipe introduces a novel deep analysis capability that extracts detailed construction pipelines from academic papers and generates customized production guides. We demonstrate the framework's capabilities across diverse dataset types—including programmatic generation, simulation-based benchmarks, and LLM distillation—and discuss implications for AI governance, scientific reproducibility, and research accessibility.

**Keywords:** dataset provenance, synthetic data, knowledge distillation, reproducibility, AI transparency, production pipeline

---

## 1. Introduction

The rapid advancement of large language models (LLMs) has been fundamentally driven by the availability of large-scale training datasets. Models such as GPT-5.2, Claude 4.5, and Llama 4 owe their capabilities not only to architectural innovations but also to the quality and scale of their training corpora. However, a critical asymmetry exists in the current AI ecosystem: while model architectures, hyperparameters, and training procedures are increasingly well-documented through publications and technical reports, the construction methodologies of training datasets remain largely opaque.

This opacity presents significant challenges across multiple dimensions:

- **Reproducibility**: Researchers cannot reconstruct datasets without detailed provenance information
- **Bias Assessment**: Inherited biases from source data or teacher models cannot be evaluated
- **Cost Estimation**: Resource requirements for dataset creation remain hidden
- **Regulatory Compliance**: Emerging AI governance frameworks require data transparency
- **Knowledge Transfer**: Best practices in dataset construction are not systematically shared

DataRecipe addresses this gap by providing a systematic framework for reverse-engineering dataset construction pipelines. The core insight is that datasets, like food products, should carry "ingredient labels" that disclose their composition and manufacturing process. Furthermore, these labels should be actionable—enabling researchers to reproduce or adapt the construction methodology for their own purposes.

---

## 2. Background and Motivation

### 2.1 The Complexity of Modern AI Datasets

Modern AI datasets are complex assemblages comprising multiple data sources and generation methodologies:

**Human-Annotated Data.** Traditional datasets rely on human annotators, often recruited through crowdsourcing platforms such as Amazon Mechanical Turk, Scale AI, Surge AI, or Prolific. Quality control mechanisms vary significantly across platforms and projects.

**Synthetic Data.** Increasingly, datasets are generated through LLM distillation, where responses from large proprietary models serve as training signals for smaller models. This approach, while cost-effective, introduces dependencies on teacher model behaviors and potential policy violations.

**Programmatic Generation.** Emerging benchmark datasets employ compositional task generators that create diverse, verifiable tasks from atomic components. Examples include mathematical reasoning datasets and procedural content generation.

**Simulation-Based Datasets.** Interactive evaluation benchmarks utilize environment simulators where agents (both AI and human) interact with shared world states. These datasets require sophisticated modeling approaches such as Dec-POMDP (Decentralized Partially Observable Markov Decision Processes).

**Multi-Stage Pipelines.** Production datasets often combine multiple approaches with complex filtering criteria, quality thresholds, and iterative refinement processes.

### 2.2 Deficiencies in Current Documentation Practices

Our analysis of existing dataset documentation reveals several systematic deficiencies:

1. **Inconsistent Metadata Standards**: No unified schema exists for describing dataset provenance
2. **Missing Generation Details**: Prompts, filtering criteria, and quality thresholds are rarely disclosed
3. **Opaque Cost Structures**: True creation costs are almost never reported
4. **Reproducibility Barriers**: Critical information for reconstruction is frequently omitted
5. **No Actionable Guidance**: Even when methodologies are described, step-by-step production guides are absent

---

## 3. Methodology

### 3.1 System Architecture

DataRecipe employs a multi-stage analysis pipeline:

```
Data Source → Metadata Extraction → Pattern Matching → Deep Analysis → Structured Report → Production Guide
```

The system supports multiple data sources:
- **HuggingFace Hub**: Direct API access to dataset metadata and documentation
- **GitHub Repositories**: Repository analysis including README extraction
- **Web URLs**: General web page analysis for dataset documentation and papers
- **Local Files**: YAML-based recipe files for manual documentation

### 3.2 Teacher Model Detection

We identify teacher models through regular expression matching against known model name patterns:

```python
TEACHER_MODEL_PATTERNS = [
    (r"gpt-?5\.?2", "GPT-5.2"),
    (r"claude[-\s]?4\.?5", "Claude 4.5"),
    (r"llama[-\s]?4", "Llama 4"),
    (r"gemini[-\s]?2", "Gemini 2"),
    (r"qwen[-\s]?3", "Qwen 3"),
    (r"deepseek[-\s]?v3", "DeepSeek V3"),
]
```

Pattern matching is applied to dataset descriptions, README files, and associated tags.

### 3.3 Generation Type Classification

We classify datasets along multiple dimensions using keyword frequency analysis and structural pattern detection:

| Category | Indicators |
|----------|------------|
| LLM Distillation | synthetic, generated, distill, teacher, api |
| Human Annotation | human, annotated, crowdsource, mturk, expert |
| Programmatic | procedural, compositional, rule-based, template |
| Simulation | simulator, environment, agent, POMDP, interactive |
| Benchmark | evaluation, test set, leaderboard, metrics |

### 3.4 Deep Analysis

A key innovation in DataRecipe is the deep analysis capability, which extracts detailed methodology from academic papers and dataset documentation through multi-source aggregation:

```
URL Input
    ↓
┌─────────────────────────────────────┐
│  Multi-Source Content Aggregation   │
├─────────────────────────────────────┤
│  1. Website Content                 │
│  2. arXiv Paper (auto-discovered)   │
│     └─ PDF Full Text Extraction     │
│  3. GitHub README                   │
└─────────────────────────────────────┘
    ↓
Pattern Matching (30+ rules)
    ↓
Structured Analysis Result
```

**Key Features:**

- **PDF Full Text Parsing**: Extracts complete paper content using PyMuPDF, not just abstracts
- **Automatic Paper Discovery**: Searches arXiv for related papers when methodology is unclear
- **Multi-Source Aggregation**: Combines website, paper, and GitHub README for comprehensive analysis
- **LLM-Enhanced Analysis** (Optional): Uses Claude or GPT for deeper semantic understanding

The deep analyzer identifies:
- **Dataset Category**: LLM distillation, programmatic, simulation, benchmark, etc.
- **Key Innovations**: Novel techniques or approaches (30+ detection patterns)
- **Generation Steps**: Step-by-step construction pipeline with code examples
- **Quality Methods**: Validation and filtering approaches
- **Resource Requirements**: Code, data, and infrastructure needs

### 3.5 Production Guide Generation

Based on analysis results, DataRecipe generates customized production guides with:

1. **Pipeline Templates**: Pre-defined workflows for different dataset types
2. **Code Snippets**: Executable examples for each step
3. **Quality Criteria**: Checklist for validation
4. **Common Pitfalls**: Known issues and mitigation strategies
5. **Domain Adaptation**: Guidance for applying the methodology to other domains

Available pipeline templates:

| Template | Description |
|----------|-------------|
| distillation | LLM-based synthetic data generation |
| human_annotation | Crowdsourced or expert annotation |
| hybrid | Combined LLM generation with human verification |
| programmatic | Rule-based compositional task generation |
| simulation | Environment simulator-driven data generation |
| benchmark | Standardized evaluation dataset creation |

### 3.6 Reproducibility Scoring

Reproducibility is assessed on a 10-point scale based on information availability:

| Criterion | Points |
|-----------|--------|
| Dataset description present | +1 |
| Detailed documentation (>500 chars) | +1 |
| Teacher model names disclosed | +1 |
| Prompt templates available | +1 |
| Source code or scripts referenced | +1 |
| Generation parameters specified | +1 |
| Paper reference available | +1 |
| Download available | +1 |
| License information | +1 |
| Quality thresholds documented | +1 |

---

## 4. Output Specification

### 4.1 Recipe Format

Analysis results conform to a structured YAML schema:

```yaml
name: dataset-identifier
version: "1.0"

source:
  type: huggingface  # huggingface, github, web, local
  id: org/dataset-name

generation:
  synthetic_ratio: 0.85
  human_ratio: 0.15
  teacher_models:
    - GPT-5.2
    - Claude 4.5
  methods:
    - type: distillation
      teacher_model: GPT-5.2
      prompt_template: available
    - type: human_annotation
      platform: Scale AI

cost:
  estimated_total_usd: 75000
  breakdown:
    api_calls: 50000
    human_annotation: 25000
  confidence: medium

reproducibility:
  score: 7
  available:
    - source_data_references
    - teacher_model_names
  missing:
    - exact_prompts
    - filtering_criteria
```

### 4.2 Production Guide Format

Production guides include:
- Reference dataset analysis summary
- Step-by-step pipeline with code examples
- Environment and action specifications (for simulation-based datasets)
- Quality standards checklist
- Common pitfalls and mitigation strategies
- Domain adaptation guidance

---

## 5. Limitations and Future Work

### 5.1 Current Limitations

| Limitation | Description |
|------------|-------------|
| Metadata Dependency | Analysis quality depends on documentation completeness |
| Heuristic Accuracy | Keyword-based classification may produce false positives/negatives |
| Cost Calibration | Estimates require empirical validation |
| Language Support | Currently optimized for English documentation |

### 5.2 Future Directions

1. **Content-Level Analysis**: Implement classifiers to detect AI-generated text patterns within dataset samples
2. **Cost Model Calibration**: Develop empirically-grounded estimation based on API pricing and annotation market rates
3. ~~**LLM-Enhanced Analysis**~~: ✅ Implemented - Use `--llm` flag with Anthropic or OpenAI API
4. **Community Validation**: Enable crowdsourced recipe contribution and verification
5. **Automated Pipeline Execution**: Generate executable scripts for dataset reproduction

---

## 6. Broader Implications

The development of dataset transparency tools raises important considerations for the AI research community:

**Governance and Accountability.** As AI systems are deployed in high-stakes domains, understanding training data provenance becomes a regulatory imperative. Dataset ingredient labels may become mandatory under emerging AI governance frameworks.

**Scientific Reproducibility.** The reproducibility crisis in machine learning is partly attributable to dataset opacity. Standardized provenance documentation could significantly improve research verifiability.

**Economic Accessibility.** Hidden dataset creation costs represent barriers to entry for resource-constrained research groups. Transparent cost reporting would enable more informed resource allocation decisions.

**Ethical Assessment.** Identifying teacher models enables evaluation of inherited biases and potential intellectual property or policy violations in the data supply chain.

**Knowledge Transfer.** Production guides enable researchers to learn from and build upon successful dataset construction methodologies, accelerating progress in the field.

---

## 7. Conclusion

DataRecipe provides a comprehensive framework for analyzing AI dataset construction pipelines and generating actionable production guides. By treating datasets as products requiring ingredient labels, we establish a path toward greater transparency in the machine learning data supply chain. The deep analysis capability extends beyond simple metadata extraction to provide detailed, customized guidance for reproducing or adapting dataset construction methodologies. While current implementation relies primarily on metadata and documentation analysis, the framework demonstrates the feasibility and value of systematic dataset provenance analysis.

---

## Installation

```bash
pip install datarecipe
```

Or install from source:

```bash
git clone https://github.com/yourusername/data-recipe.git
cd data-recipe
pip install -e .
```

### Optional Dependencies

```bash
# For PDF parsing (recommended)
pip install datarecipe[pdf]

# For LLM-enhanced analysis
pip install datarecipe[llm]

# Install all optional dependencies
pip install datarecipe[all]
```

## Usage

### Basic Analysis

```bash
# Analyze HuggingFace dataset
datarecipe analyze Anthropic/hh-rlhf

# Analyze from URL
datarecipe analyze https://huggingface.co/datasets/org/dataset

# Export as different formats
datarecipe analyze <dataset_id> -o report.md    # Markdown
datarecipe analyze <dataset_id> --yaml          # YAML
datarecipe analyze <dataset_id> --json          # JSON
```

### Production Guide Generation

```bash
# Generate standard production guide
datarecipe guide Anthropic/hh-rlhf -o guide.md

# Deep analysis with customized guide (PDF parsing + multi-source)
datarecipe deep-guide https://arxiv.org/abs/2506.07982 -o deep-guide.md

# Deep analysis for dataset websites (auto-discovers papers)
datarecipe deep-guide https://arcprize.org/arc-agi/2/ -o arc-guide.md

# LLM-enhanced analysis (requires API key)
export ANTHROPIC_API_KEY="your-key"
datarecipe deep-guide https://example.com/dataset --llm -o guide.md

# Use OpenAI instead
export OPENAI_API_KEY="your-key"
datarecipe deep-guide https://example.com/dataset --llm --provider openai -o guide.md
```

### Other Commands

```bash
datarecipe create                    # Interactive recipe creation
datarecipe show recipes/example.yaml # Display recipe file
datarecipe list-sources              # List supported data sources
```

## License

MIT License

## Citation

```bibtex
@software{liu2026datarecipe,
  author       = {Liu, Kai},
  title        = {DataRecipe: A Framework for Dataset Provenance Analysis},
  year         = {2026},
  url          = {https://github.com/liuxiaotong/data-recipe},
  email        = {mrliukai@gmail.com}
}
```

---
---

# DataRecipe：一个用于逆向分析人工智能数据集构建流程的框架

---

## 摘要

大语言模型的快速发展催生了对高质量训练数据集的空前需求。然而，尽管模型架构和训练流程日益透明，这些数据集的构建方法却仍处于不透明状态。本文介绍 DataRecipe，一个用于分析数据集来源的系统性框架——其理念类似于食品行业的"营养成分标签"制度。该工具从多种来源（HuggingFace Hub、GitHub 仓库和网页）提取元数据，以识别生成方法、蒸馏过程中使用的教师模型、估算的创建成本以及可复现性评分。除了分析功能外，DataRecipe 还引入了一项创新的深度分析能力，可从学术论文中提取详细的构建流程并生成定制化的生产指南。我们在多种数据集类型上展示了该框架的能力——包括程序化生成、基于模拟器的基准测试和大语言模型蒸馏——并讨论其对人工智能治理、科学可复现性和研究可及性的影响。

**关键词：** 数据集溯源、合成数据、知识蒸馏、可复现性、人工智能透明度、生产流程

---

## 1. 引言

大语言模型（LLMs）的快速发展在根本上依赖于大规模训练数据集的支撑。GPT-5.2、Claude 4.5 和 Llama 4 等模型的能力不仅源于架构创新，更源于其训练语料的质量和规模。然而，当前人工智能生态系统中存在一个显著的不对称现象：模型架构、超参数和训练流程通过论文和技术报告日益透明化，而训练数据集的构建方法却大多处于不透明状态。

这种不透明性在多个维度带来重大挑战：

- **可复现性**：缺乏详细的溯源信息，研究人员无法重建数据集
- **偏差评估**：无法评估从源数据或教师模型继承的偏差
- **成本估算**：数据集创建的资源需求被隐藏
- **合规性**：新兴的人工智能治理框架要求数据透明
- **知识传递**：数据集构建的最佳实践未能得到系统性分享

DataRecipe 旨在填补这一空白，通过提供一套系统性框架对数据集构建流程进行逆向分析。其核心理念是：数据集如同食品一样，应当附带"成分标签"，披露其组成和制造过程。更重要的是，这些标签应当具有可操作性——使研究人员能够复现或改编构建方法以满足自身需求。

---

## 2. 背景与动机

### 2.1 现代人工智能数据集的复杂性

现代人工智能数据集是由多种数据来源和生成方法组成的复杂集合体：

**人工标注数据。** 传统数据集依赖人工标注者，通常通过 Amazon Mechanical Turk、Scale AI、Surge AI 或 Prolific 等众包平台招募。不同平台和项目的质量控制机制差异显著。

**合成数据。** 越来越多的数据集通过大语言模型蒸馏生成，即利用大型专有模型的输出作为训练较小模型的信号。这种方法虽然成本效益高，但引入了对教师模型行为的依赖以及潜在的政策违规风险。

**程序化生成。** 新兴的基准数据集采用组合式任务生成器，从原子组件创建多样化、可验证的任务。例如数学推理数据集和程序化内容生成。

**基于模拟器的数据集。** 交互式评估基准使用环境模拟器，其中代理（人工智能和人类）与共享的世界状态进行交互。这些数据集需要复杂的建模方法，如 Dec-POMDP（分布式部分可观测马尔可夫决策过程）。

**多阶段流程。** 生产级数据集通常结合多种方法，采用复杂的过滤标准、质量阈值和迭代优化过程。

### 2.2 当前文档实践的不足

我们对现有数据集文档的分析揭示了若干系统性缺陷：

1. **元数据标准不一致**：不存在描述数据集来源的统一模式
2. **生成细节缺失**：提示词、过滤标准和质量阈值很少被披露
3. **成本结构不透明**：真实的创建成本几乎从不报告
4. **可复现性障碍**：重建所需的关键信息经常被遗漏
5. **缺乏可操作指导**：即使描述了方法论，也缺少逐步的生产指南

---

## 3. 方法论

### 3.1 系统架构

DataRecipe 采用多阶段分析流程：

```
数据源 → 元数据提取 → 模式匹配 → 深度分析 → 结构化报告 → 生产指南
```

系统支持多种数据源：
- **HuggingFace Hub**：直接 API 访问数据集元数据和文档
- **GitHub 仓库**：仓库分析，包括 README 提取
- **Web URL**：对数据集文档和论文进行通用网页分析
- **本地文件**：基于 YAML 的配方文件，用于手动文档记录

### 3.2 教师模型检测

我们通过正则表达式匹配已知模型名称模式来识别教师模型：

```python
TEACHER_MODEL_PATTERNS = [
    (r"gpt-?5\.?2", "GPT-5.2"),
    (r"claude[-\s]?4\.?5", "Claude 4.5"),
    (r"llama[-\s]?4", "Llama 4"),
    (r"gemini[-\s]?2", "Gemini 2"),
    (r"qwen[-\s]?3", "Qwen 3"),
    (r"deepseek[-\s]?v3", "DeepSeek V3"),
]
```

模式匹配应用于数据集描述、README 文件和相关标签。

### 3.3 生成类型分类

我们使用关键词频率分析和结构模式检测在多个维度上对数据集进行分类：

| 类别 | 指示词 |
|------|--------|
| LLM 蒸馏 | synthetic, generated, distill, teacher, api |
| 人工标注 | human, annotated, crowdsource, mturk, expert |
| 程序化生成 | procedural, compositional, rule-based, template |
| 模拟器驱动 | simulator, environment, agent, POMDP, interactive |
| 评估基准 | evaluation, test set, leaderboard, metrics |

### 3.4 深度分析

DataRecipe 的一项关键创新是深度分析能力，通过多源聚合从学术论文和数据集文档中提取详细的方法论：

```
URL 输入
    ↓
┌─────────────────────────────────────┐
│       多源内容聚合                    │
├─────────────────────────────────────┤
│  1. 网站内容                         │
│  2. arXiv 论文（自动发现）            │
│     └─ PDF 全文提取                  │
│  3. GitHub README                   │
└─────────────────────────────────────┘
    ↓
模式匹配（30+ 规则）
    ↓
结构化分析结果
```

**核心特性：**

- **PDF 全文解析**：使用 PyMuPDF 提取完整论文内容，而非仅摘要
- **自动论文发现**：当方法论不明确时，自动搜索 arXiv 相关论文
- **多源聚合**：综合网站、论文和 GitHub README 进行全面分析
- **LLM 增强分析**（可选）：使用 Claude 或 GPT 进行更深层的语义理解

深度分析器识别：
- **数据集类别**：LLM 蒸馏、程序化、模拟器、基准测试等
- **核心创新**：新颖的技术或方法（30+ 检测规则）
- **生成步骤**：逐步的构建流程及代码示例
- **质量方法**：验证和过滤方法
- **资源需求**：代码、数据和基础设施需求

### 3.5 生产指南生成

基于分析结果，DataRecipe 生成定制化的生产指南，包含：

1. **流程模板**：针对不同数据集类型的预定义工作流
2. **代码片段**：每个步骤的可执行示例
3. **质量标准**：验证检查清单
4. **常见陷阱**：已知问题和缓解策略
5. **领域适配**：将方法论应用于其他领域的指导

可用的流程模板：

| 模板 | 描述 |
|------|------|
| distillation | 基于 LLM 的合成数据生成 |
| human_annotation | 众包或专家标注 |
| hybrid | LLM 生成与人工验证相结合 |
| programmatic | 基于规则的组合式任务生成 |
| simulation | 环境模拟器驱动的数据生成 |
| benchmark | 标准化评估数据集创建 |

### 3.6 可复现性评分

可复现性采用 10 分制评估，基于信息可用性：

| 评估标准 | 分值 |
|---------|-----|
| 存在数据集描述 | +1 |
| 详细文档（>500字符） | +1 |
| 披露教师模型名称 | +1 |
| 提供提示词模板 | +1 |
| 引用源代码或脚本 | +1 |
| 指定生成参数 | +1 |
| 有论文引用 | +1 |
| 可下载 | +1 |
| 许可证信息 | +1 |
| 记录质量阈值 | +1 |

---

## 4. 输出规范

### 4.1 配方格式

分析结果符合结构化 YAML 模式：

```yaml
name: 数据集标识符
version: "1.0"

source:
  type: huggingface  # huggingface, github, web, local
  id: org/dataset-name

generation:
  synthetic_ratio: 0.85
  human_ratio: 0.15
  teacher_models:
    - GPT-5.2
    - Claude 4.5
  methods:
    - type: distillation
      teacher_model: GPT-5.2
      prompt_template: available
    - type: human_annotation
      platform: Scale AI

cost:
  estimated_total_usd: 75000
  breakdown:
    api_calls: 50000
    human_annotation: 25000
  confidence: medium

reproducibility:
  score: 7
  available:
    - source_data_references
    - teacher_model_names
  missing:
    - exact_prompts
    - filtering_criteria
```

### 4.2 生产指南格式

生产指南包含：
- 参考数据集分析摘要
- 带代码示例的逐步流程
- 环境和动作规范（用于基于模拟器的数据集）
- 质量标准检查清单
- 常见陷阱和缓解策略
- 领域适配指导

---

## 5. 局限性与未来工作

### 5.1 当前局限性

| 局限性 | 描述 |
|-------|------|
| 元数据依赖 | 分析质量取决于文档完整性 |
| 启发式准确性 | 基于关键词的分类可能产生假阳性/假阴性 |
| 成本校准 | 估算需要经验验证 |
| 语言支持 | 目前针对英文文档优化 |

### 5.2 未来方向

1. **内容级分析**：实现分类器以检测数据集样本中的人工智能生成文本模式
2. **成本模型校准**：基于 API 定价和标注市场费率开发经验性估算
3. ~~**LLM 增强分析**~~：✅ 已实现 - 使用 `--llm` 参数配合 Anthropic 或 OpenAI API
4. **社区验证**：支持众包配方贡献和验证
5. **自动化流程执行**：生成用于数据集复现的可执行脚本

---

## 6. 更广泛的影响

数据集透明度工具的开发为人工智能研究社区带来重要启示：

**治理与问责。** 随着人工智能系统部署于高风险领域，理解训练数据来源成为监管必要条件。数据集成分标签可能在新兴的人工智能治理框架下成为强制要求。

**科学可复现性。** 机器学习中的可复现性危机部分归因于数据集不透明。标准化的溯源文档可以显著提高研究的可验证性。

**经济可及性。** 隐藏的数据集创建成本对资源有限的研究团队构成准入障碍。透明的成本报告将有助于更明智的资源分配决策。

**伦理评估。** 识别教师模型有助于评估数据供应链中继承的偏差以及潜在的知识产权或政策违规。

**知识传递。** 生产指南使研究人员能够学习和借鉴成功的数据集构建方法，加速领域进展。

---

## 7. 结论

DataRecipe 为分析人工智能数据集构建流程和生成可操作的生产指南提供了一个综合性框架。通过将数据集视为需要成分标签的产品，我们为提升机器学习数据供应链的透明度开辟了道路。深度分析能力超越了简单的元数据提取，为复现或改编数据集构建方法提供了详细的定制化指导。尽管当前实现主要依赖于元数据和文档分析，但该框架展示了系统性数据集溯源分析的可行性和价值。

---

## 安装

```bash
pip install datarecipe
```

或从源码安装：

```bash
git clone https://github.com/yourusername/data-recipe.git
cd data-recipe
pip install -e .
```

### 可选依赖

```bash
# PDF 解析（推荐）
pip install datarecipe[pdf]

# LLM 增强分析
pip install datarecipe[llm]

# 安装所有可选依赖
pip install datarecipe[all]
```

## 使用方法

### 基础分析

```bash
# 分析 HuggingFace 数据集
datarecipe analyze Anthropic/hh-rlhf

# 从 URL 分析
datarecipe analyze https://huggingface.co/datasets/org/dataset

# 导出为不同格式
datarecipe analyze <dataset_id> -o report.md    # Markdown
datarecipe analyze <dataset_id> --yaml          # YAML
datarecipe analyze <dataset_id> --json          # JSON
```

### 生产指南生成

```bash
# 生成标准生产指南
datarecipe guide Anthropic/hh-rlhf -o guide.md

# 深度分析并生成定制化指南（PDF 解析 + 多源聚合）
datarecipe deep-guide https://arxiv.org/abs/2506.07982 -o deep-guide.md

# 深度分析数据集网站（自动发现相关论文）
datarecipe deep-guide https://arcprize.org/arc-agi/2/ -o arc-guide.md

# LLM 增强分析（需要 API 密钥）
export ANTHROPIC_API_KEY="your-key"
datarecipe deep-guide https://example.com/dataset --llm -o guide.md

# 使用 OpenAI
export OPENAI_API_KEY="your-key"
datarecipe deep-guide https://example.com/dataset --llm --provider openai -o guide.md
```

### 其他命令

```bash
datarecipe create                    # 交互式创建配方
datarecipe show recipes/example.yaml # 显示配方文件
datarecipe list-sources              # 列出支持的数据源
```

## 许可证

MIT 许可证

## 引用

```bibtex
@software{liu2026datarecipe,
  author       = {Liu, Kai},
  title        = {DataRecipe: A Framework for Dataset Provenance Analysis},
  year         = {2026},
  url          = {https://github.com/liuxiaotong/data-recipe},
  email        = {mrliukai@gmail.com}
}
```
