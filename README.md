# DataRecipe

## Abstract

The rapid advancement of large language models (LLMs) has been fundamentally driven by the availability of large-scale training datasets. However, a critical asymmetry exists in the current AI ecosystem: while model architectures and training procedures are increasingly well-documented, the construction methodologies of training datasets remain largely opaque. This opacity presents significant challenges for reproducibility, bias assessment, and regulatory compliance. DataRecipe addresses this gap by providing a systematic framework for reverse-engineering dataset construction pipelines—analogous to "nutrition labels" for food products. The tool extracts and analyzes dataset metadata to identify key attributes including: (1) data generation methodologies, distinguishing between synthetic data produced via LLM distillation and human-annotated data; (2) teacher model identification for distillation-based datasets; (3) cost estimation based on inferred API usage and annotation expenses; and (4) reproducibility scoring that quantifies the feasibility of dataset reconstruction. Our framework currently supports HuggingFace Hub as a primary data source and employs heuristic-based detection methods including regular expression matching for model identification and keyword frequency analysis for generation type classification. While the current implementation relies on metadata analysis rather than content-level inspection, it establishes a foundational approach toward greater transparency in the machine learning data supply chain.

---

## 摘要

大语言模型（LLMs）的快速发展在根本上依赖于大规模训练数据集的支撑。然而，当前人工智能生态系统中存在一个显著的不对称现象：模型架构与训练流程的文档化程度日益完善，而训练数据集的构建方法却大多处于不透明状态。这种不透明性为可复现性研究、偏差评估以及合规审查带来了重大挑战。DataRecipe 旨在填补这一空白，通过提供一套系统性框架对数据集构建流程进行逆向分析——其理念类似于食品行业的"营养成分标签"制度。该工具通过提取和分析数据集元数据，识别以下关键属性：（1）数据生成方式，区分通过大语言模型蒸馏产生的合成数据与人工标注数据；（2）针对蒸馏型数据集的教师模型识别；（3）基于推断的 API 调用量和标注成本进行费用估算；（4）可复现性评分，量化数据集重建的可行性程度。本框架目前支持 HuggingFace Hub 作为主要数据源，并采用基于启发式规则的检测方法，包括用于模型识别的正则表达式匹配以及用于生成类型分类的关键词频率分析。尽管当前实现依赖于元数据分析而非内容级检测，但它为提升机器学习数据供应链的透明度建立了基础性方法框架。

---

## 1. Introduction

### 1.1 Problem Statement

Modern AI datasets are complex assemblages comprising multiple data sources and generation methodologies:

- **Human-annotated data**: Collected through crowdsourcing platforms (e.g., Amazon MTurk, Scale AI) with varying quality control mechanisms
- **Synthetic data**: Generated via LLM distillation, where responses from proprietary models (GPT-5.2, Claude 4.5) are used to train smaller models
- **Web-scraped corpora**: Filtered and processed text from internet sources
- **Multi-stage pipelines**: Combinations of the above with complex filtering and quality thresholds

Understanding these "ingredients" is essential for:

| Concern | Description |
|---------|-------------|
| **Reproducibility** | Can researchers reconstruct this dataset given the available documentation? |
| **Cost Transparency** | What resources (financial, computational, human labor) were required? |
| **Bias Assessment** | What systematic biases might be inherited from source data or teacher models? |
| **Regulatory Compliance** | Does the dataset meet emerging AI governance requirements? |
| **Scientific Integrity** | Are evaluation benchmarks contaminated by training data overlap? |

### 1.2 Current Limitations

Existing dataset documentation practices exhibit several deficiencies:

1. **Inconsistent metadata standards**: No unified schema for describing dataset provenance
2. **Missing generation details**: Prompts, filtering criteria, and quality thresholds are rarely disclosed
3. **Opaque cost structures**: True creation costs are almost never reported
4. **Reproducibility barriers**: Critical information for reconstruction is frequently omitted

## 2. Methodology

### 2.1 Detection Approach

DataRecipe employs a heuristic-based analysis pipeline:

```
Dataset ID → Metadata Extraction → Pattern Matching → Heuristic Analysis → Structured Report
```

**Teacher Model Detection**: Regular expression matching against known model name patterns:

```python
TEACHER_MODEL_PATTERNS = [
    (r"gpt-?5\.?2", "GPT-5.2"),
    (r"claude[-\s]?4\.?5", "Claude 4.5"),
    (r"llama[-\s]?4", "Llama 4"),
    # ... additional patterns
]
```

**Generation Type Classification**: Keyword frequency analysis comparing synthetic-indicative terms (`synthetic`, `generated`, `distill`, `teacher`) against human-indicative terms (`human`, `annotated`, `crowdsource`, `expert`).

### 2.2 Output Schema

Analysis results are structured according to a defined schema:

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

## 3. Installation

```bash
pip install datarecipe
```

Or install from source:

```bash
git clone https://github.com/yourusername/data-recipe.git
cd data-recipe
pip install -e .
```

## 4. Usage

### 4.1 Command Line Interface

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

### 4.2 Example Output

```bash
datarecipe analyze Anthropic/hh-rlhf -o report.md
```

Generates a structured report containing:

- **Basic Information**: Dataset name, source platform, license, language coverage
- **Generation Method**: Synthetic/human ratio with visual distribution indicators
- **Teacher Models**: Detected LLMs used in data generation pipeline
- **Generation Pipeline**: Step-by-step breakdown of data creation process
- **Cost Estimation**: Estimated creation costs with confidence intervals
- **Reproducibility Assessment**: Quantified score with itemized available/missing information

## 5. Limitations and Future Work

### 5.1 Current Limitations

| Limitation | Description |
|------------|-------------|
| **Metadata dependency** | Analysis relies on documentation quality; undocumented attributes cannot be detected |
| **Heuristic accuracy** | Keyword-based classification may produce false positives/negatives |
| **Cost estimation** | Current estimates use placeholder values; actual costs require empirical calibration |
| **Source coverage** | Only HuggingFace Hub is currently supported |

### 5.2 Future Directions

1. **Content-level analysis**: Implement classifiers to detect AI-generated text patterns within dataset samples
2. **Cost model calibration**: Develop empirically-grounded cost estimation based on API pricing and annotation market rates
3. **Extended source support**: Add connectors for OpenAI datasets, local files, and other repositories
4. **Community-driven recipes**: Enable users to contribute and validate dataset recipes

## 6. Broader Implications

The development of dataset transparency tools raises important considerations:

- **Accountability**: As AI systems are increasingly deployed in high-stakes domains, understanding training data provenance becomes a governance imperative
- **Scientific reproducibility**: Dataset reconstruction capability is fundamental to the reproducibility of machine learning research
- **Economic transparency**: The hidden costs of dataset creation represent a significant barrier to entry for resource-constrained research groups
- **Ethical considerations**: Identifying teacher models enables assessment of inherited biases and potential policy violations

## 7. Supported Sources

| Source | Status | Example Identifier |
|--------|--------|-------------------|
| HuggingFace Hub | Supported | `org/dataset-name` |
| OpenAI Datasets | Planned | - |
| Local Files | Planned | - |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use DataRecipe in your research, please cite:

```bibtex
@software{datarecipe2024,
  title={DataRecipe: A Framework for Dataset Provenance Analysis},
  year={2024},
  url={https://github.com/yourusername/data-recipe}
}
```
