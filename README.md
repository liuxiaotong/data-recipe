# DataRecipe 🧪

**Reverse-engineer how AI datasets are built.**

DataRecipe is an "ingredients label" analyzer for AI datasets. Just like food labels tell you what's inside your food, DataRecipe reveals what's inside your training data.

## Why DataRecipe?

Modern AI datasets are complex mixtures of:
- Human-annotated data
- Synthetic data from teacher models
- Web scrapes and filtered corpora
- Multi-stage processing pipelines

Understanding these "ingredients" is crucial for:
- **Reproducibility**: Can you rebuild this dataset?
- **Cost estimation**: How expensive was this to create?
- **Quality assessment**: What are the potential biases?
- **Compliance**: Does it meet your data governance requirements?

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

## Quick Start

Analyze any HuggingFace dataset:

```bash
datarecipe analyze allenai/Sera-4.6-Lite-T2
```

Output:
```
╭─────────────────────────────────────────────────────────╮
│                    Dataset Recipe                        │
├─────────────────────────────────────────────────────────┤
│ Name: allenai/Sera-4.6-Lite-T2                          │
│ Source: HuggingFace Hub                                  │
│                                                          │
│ 📊 Generation Method:                                    │
│    • Synthetic: 85%                                      │
│    • Human: 15%                                          │
│                                                          │
│ 🤖 Teacher Models:                                       │
│    • GPT-4                                               │
│    • Claude 3                                            │
│                                                          │
│ 💰 Estimated Cost: $50,000 - $100,000                   │
│                                                          │
│ 🔄 Reproducibility Score: 7/10                          │
│    Missing: exact prompts, filtering criteria            │
╰─────────────────────────────────────────────────────────╯
```

## Recipe Format

DataRecipe uses YAML files to document dataset recipes:

```yaml
name: my-dataset
version: "1.0"
source:
  type: huggingface
  id: org/dataset-name

generation:
  synthetic_ratio: 0.85
  methods:
    - type: distillation
      teacher_model: gpt-4
      prompt_template: available
    - type: human_annotation
      platform: scale-ai
      annotators: 50

cost:
  estimated_total_usd: 75000
  breakdown:
    api_calls: 50000
    human_annotation: 25000

reproducibility:
  score: 7
  available:
    - source_data
    - teacher_model_name
  missing:
    - exact_prompts
    - filtering_criteria
```

## Features

- **Auto-detect generation methods**: Identifies synthetic vs human data
- **Teacher model detection**: Finds which LLMs were used for distillation
- **Cost estimation**: Rough cost estimates based on dataset size and methods
- **Reproducibility scoring**: How easy is it to recreate this dataset?
- **Export recipes**: Generate YAML recipes for documentation

## Supported Sources

- [x] HuggingFace Hub
- [ ] OpenAI datasets (coming soon)
- [ ] Custom local datasets

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

## License

MIT License - see [LICENSE](LICENSE) for details.
