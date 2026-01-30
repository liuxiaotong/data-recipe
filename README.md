# DataRecipe

## Abstract

The proliferation of large-scale datasets has become fundamental to the advancement of artificial intelligence systems, yet the provenance and construction methodologies of these datasets often remain opaque. DataRecipe addresses this critical gap by providing a systematic framework for reverse-engineering dataset construction pipelines. This tool analyzes dataset metadata to extract key attributes including data generation methodologies (synthetic versus human-annotated), teacher model identification for distillation-based datasets, cost estimation, and reproducibility assessment. By treating datasets analogously to food products requiring ingredient labels, DataRecipe enhances transparency in the machine learning ecosystem, enabling researchers and practitioners to make informed decisions regarding dataset selection, bias assessment, and regulatory compliance. The framework currently supports HuggingFace Hub as a primary data source and outputs structured reports in multiple formats including YAML, JSON, and Markdown.

---

## 摘要

大规模数据集的广泛应用已成为推动人工智能系统发展的重要基础，然而这些数据集的来源及其构建方法往往缺乏透明度。DataRecipe 旨在填补这一研究空白，通过提供一套系统性框架对数据集构建流程进行逆向分析。该工具通过解析数据集元数据，提取关键属性信息，包括数据生成方式（合成数据与人工标注数据的比例）、蒸馏型数据集所使用的教师模型识别、成本估算以及可复现性评估。DataRecipe 将数据集类比为需要成分标签的食品，旨在提升机器学习生态系统的透明度，使研究人员和从业者能够在数据集选择、偏差评估及合规审查等方面做出更为审慎的决策。本框架目前支持 HuggingFace Hub 作为主要数据源，并可输出 YAML、JSON 及 Markdown 等多种格式的结构化分析报告。

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
# Analyze a dataset
datarecipe analyze <dataset_id>

# Export as Markdown report
datarecipe analyze <dataset_id> -o report.md

# Export as YAML
datarecipe analyze <dataset_id> --yaml

# Display local recipe file
datarecipe show recipes/example.yaml
```

## Example Output

```
datarecipe analyze Anthropic/hh-rlhf -o report.md
```

Generates a structured report containing:

- **Basic Information**: Dataset name, source, license
- **Generation Method**: Synthetic/human ratio with distribution visualization
- **Teacher Models**: Detected LLMs used for data generation
- **Cost Estimation**: Estimated creation costs with confidence levels
- **Reproducibility Score**: Assessment of reconstruction feasibility

## License

MIT License
