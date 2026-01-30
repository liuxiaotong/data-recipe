# DataRecipe

## Abstract

The rapid advancement of large language models (LLMs) has been fundamentally driven by the availability of large-scale training datasets. However, a critical asymmetry exists in the current AI ecosystem: while model architectures and training procedures are increasingly well-documented, the construction methodologies of training datasets remain largely opaque. This opacity presents significant challenges for reproducibility, bias assessment, and regulatory compliance. DataRecipe addresses this gap by providing a systematic framework for reverse-engineering dataset construction pipelines—analogous to "nutrition labels" for food products. Modern AI datasets are complex assemblages comprising multiple data sources and generation methodologies, including human-annotated data collected through crowdsourcing platforms such as Amazon MTurk and Scale AI with varying quality control mechanisms, synthetic data generated via LLM distillation where responses from proprietary models like GPT-5.2 and Claude 4.5 are used to train smaller models, web-scraped corpora that undergo filtering and processing, and multi-stage pipelines that combine these approaches with complex filtering and quality thresholds. Understanding these "ingredients" is essential for several critical concerns: reproducibility, which asks whether researchers can reconstruct datasets given available documentation; cost transparency, which examines what financial, computational, and human labor resources were required; bias assessment, which investigates what systematic biases might be inherited from source data or teacher models; regulatory compliance, which determines whether datasets meet emerging AI governance requirements; and scientific integrity, which addresses whether evaluation benchmarks are contaminated by training data overlap. Existing dataset documentation practices exhibit several deficiencies, including inconsistent metadata standards with no unified schema for describing dataset provenance, missing generation details where prompts, filtering criteria, and quality thresholds are rarely disclosed, opaque cost structures where true creation costs are almost never reported, and reproducibility barriers where critical information for reconstruction is frequently omitted. DataRecipe employs a heuristic-based analysis pipeline that progresses from dataset identification through metadata extraction, pattern matching, and heuristic analysis to structured report generation. The tool extracts and analyzes dataset metadata to identify key attributes including data generation methodologies that distinguish between synthetic data produced via LLM distillation and human-annotated data, teacher model identification for distillation-based datasets through regular expression matching against known model name patterns, cost estimation based on inferred API usage and annotation expenses, and reproducibility scoring that quantifies the feasibility of dataset reconstruction. The framework currently supports HuggingFace Hub as a primary data source and employs keyword frequency analysis comparing synthetic-indicative terms such as "synthetic," "generated," "distill," and "teacher" against human-indicative terms such as "human," "annotated," "crowdsource," and "expert" for generation type classification. Analysis results are structured according to a defined schema encompassing source information, generation details including synthetic and human ratios along with teacher models and methods, cost breakdowns with confidence levels, and reproducibility scores with itemized available and missing information. The current implementation has several limitations: analysis relies on documentation quality such that undocumented attributes cannot be detected, keyword-based classification may produce false positives or negatives, cost estimates currently use placeholder values requiring empirical calibration, and only HuggingFace Hub is currently supported as a data source. Future directions include implementing content-level analysis with classifiers to detect AI-generated text patterns within dataset samples, developing empirically-grounded cost estimation based on API pricing and annotation market rates, adding connectors for OpenAI datasets, local files, and other repositories, and enabling community-driven recipe contribution and validation. The development of dataset transparency tools raises important broader implications: as AI systems are increasingly deployed in high-stakes domains, understanding training data provenance becomes a governance imperative; dataset reconstruction capability is fundamental to the reproducibility of machine learning research; the hidden costs of dataset creation represent a significant barrier to entry for resource-constrained research groups; and identifying teacher models enables assessment of inherited biases and potential policy violations. While the current implementation relies on metadata analysis rather than content-level inspection, it establishes a foundational approach toward greater transparency in the machine learning data supply chain.

---

## 摘要

大语言模型（LLMs）的快速发展在根本上依赖于大规模训练数据集的支撑。然而，当前人工智能生态系统中存在一个显著的不对称现象：模型架构与训练流程的文档化程度日益完善，而训练数据集的构建方法却大多处于不透明状态。这种不透明性为可复现性研究、偏差评估以及合规审查带来了重大挑战。DataRecipe 旨在填补这一空白，通过提供一套系统性框架对数据集构建流程进行逆向分析——其理念类似于食品行业的"营养成分标签"制度。现代人工智能数据集是由多种数据来源和生成方法组成的复杂集合体，包括：通过众包平台（如 Amazon MTurk 和 Scale AI）收集的人工标注数据，这些平台具有不同程度的质量控制机制；通过大语言模型蒸馏生成的合成数据，即利用 GPT-5.2、Claude 4.5 等专有模型的输出来训练较小的模型；经过过滤和处理的网络爬取语料库；以及结合上述方法并采用复杂过滤和质量阈值的多阶段处理流程。理解这些"成分"对于以下几个关键问题至关重要：可复现性，即研究人员能否根据现有文档重建数据集；成本透明度，即创建数据集需要哪些财务、计算和人力资源；偏差评估，即可能从源数据或教师模型中继承哪些系统性偏差；合规性，即数据集是否符合新兴的人工智能治理要求；以及科学诚信，即评估基准是否受到训练数据重叠的污染。现有的数据集文档实践存在若干不足：元数据标准不一致，缺乏描述数据集来源的统一模式；生成细节缺失，提示词、过滤标准和质量阈值很少被披露；成本结构不透明，真实的创建成本几乎从不报告；以及可复现性障碍，重建所需的关键信息经常被遗漏。DataRecipe 采用基于启发式规则的分析流程，从数据集标识开始，经过元数据提取、模式匹配和启发式分析，最终生成结构化报告。该工具通过提取和分析数据集元数据来识别关键属性，包括：区分通过大语言模型蒸馏产生的合成数据与人工标注数据的数据生成方式；通过正则表达式匹配已知模型名称模式来识别蒸馏型数据集所使用的教师模型；基于推断的 API 调用量和标注成本进行费用估算；以及量化数据集重建可行性的可复现性评分。本框架目前支持 HuggingFace Hub 作为主要数据源，并采用关键词频率分析方法，将"synthetic"、"generated"、"distill"、"teacher"等表示合成数据的术语与"human"、"annotated"、"crowdsource"、"expert"等表示人工标注的术语进行比较，以实现生成类型分类。分析结果按照预定义的模式进行结构化组织，涵盖来源信息、包含合成与人工比例及教师模型和方法的生成细节、带有置信度的成本明细，以及列出可用和缺失信息的可复现性评分。当前实现存在若干局限性：分析依赖于文档质量，未记录的属性无法被检测；基于关键词的分类可能产生假阳性或假阴性；成本估算目前使用占位值，需要经验校准；且目前仅支持 HuggingFace Hub 作为数据源。未来的发展方向包括：实现内容级分析，使用分类器检测数据集样本中的人工智能生成文本模式；基于 API 定价和标注市场费率开发经验性成本估算模型；添加对 OpenAI 数据集、本地文件和其他存储库的连接器支持；以及支持社区驱动的配方贡献和验证机制。数据集透明度工具的开发引发了重要的深层思考：随着人工智能系统越来越多地部署在高风险领域，理解训练数据的来源已成为治理的必要条件；数据集重建能力是机器学习研究可复现性的基础；数据集创建的隐性成本对资源有限的研究团队构成了重大准入障碍；识别教师模型有助于评估继承的偏差和潜在的政策违规。尽管当前实现依赖于元数据分析而非内容级检测，但它为提升机器学习数据供应链的透明度建立了基础性方法框架。

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

## Usage

```bash
# Analyze a HuggingFace dataset
datarecipe analyze <dataset_id>

# Export analysis as Markdown report
datarecipe analyze <dataset_id> -o report.md

# Export as structured YAML
datarecipe analyze <dataset_id> --yaml

# Export as JSON
datarecipe analyze <dataset_id> --json

# Display existing recipe file
datarecipe show recipes/example.yaml

# List supported data sources
datarecipe list-sources
```

## Output Schema

```yaml
name: dataset-identifier
version: "1.0"

source:
  type: huggingface
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

## Supported Sources

| Source | Status | Example Identifier |
|--------|--------|-------------------|
| HuggingFace Hub | Supported | `org/dataset-name` |
| OpenAI Datasets | Planned | - |
| Local Files | Planned | - |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{datarecipe2024,
  title={DataRecipe: A Framework for Dataset Provenance Analysis},
  year={2024},
  url={https://github.com/yourusername/data-recipe}
}
```
