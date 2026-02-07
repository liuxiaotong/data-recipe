"""Production workflow generation for dataset reproduction."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datarecipe.schema import Recipe
from datarecipe.cost_calculator import CostCalculator


@dataclass
class WorkflowStep:
    """A single step in the production workflow."""

    name: str
    description: str
    script_content: str  # Python script content
    dependencies: list[str] = field(default_factory=list)  # pip packages
    env_vars: list[str] = field(default_factory=list)  # Required env variables
    estimated_cost: float = 0.0
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


@dataclass
class Milestone:
    """A project milestone."""

    name: str
    description: str
    deliverables: list[str]
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ResourceChecklist:
    """Checklist of required resources."""

    api_keys: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (name, signup_url, env_var)
    dependencies: list[str] = field(default_factory=list)  # pip packages
    compute_requirements: dict = field(default_factory=dict)


@dataclass
class ProductionWorkflow:
    """Complete production workflow for dataset reproduction."""

    name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)
    milestones: list[Milestone] = field(default_factory=list)
    resource_checklist: ResourceChecklist = field(default_factory=ResourceChecklist)
    timeline: list[tuple[str, str]] = field(default_factory=list)  # (phase, description)
    estimated_total_cost: float = 0.0
    target_size: int = 0

    def generate_scripts(self, output_dir: str) -> list[str]:
        """Generate Python scripts for each workflow step.

        Args:
            output_dir: Directory to write scripts

        Returns:
            List of created file paths
        """
        scripts_dir = Path(output_dir) / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        for i, step in enumerate(self.steps, 1):
            filename = f"{i:02d}_{step.name.lower().replace(' ', '_')}.py"
            filepath = scripts_dir / filename

            # Add header comment
            script = f'''#!/usr/bin/env python3
"""
Step {i}: {step.name}

{step.description}

Dependencies: {', '.join(step.dependencies) if step.dependencies else 'None'}
Inputs: {', '.join(step.inputs) if step.inputs else 'None'}
Outputs: {', '.join(step.outputs) if step.outputs else 'None'}
"""

import os
import json
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("../data")
OUTPUT_DIR.mkdir(exist_ok=True)

'''
            script += step.script_content

            filepath.write_text(script, encoding="utf-8")
            created_files.append(str(filepath))

        return created_files

    def export_project(self, output_dir: str) -> list[str]:
        """Export complete project structure.

        Args:
            output_dir: Directory to write project files

        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Generate scripts
        created_files.extend(self.generate_scripts(output_dir))

        # Generate README.md
        readme_content = self._generate_readme()
        readme_path = output_path / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        created_files.append(str(readme_path))

        # Generate requirements.txt
        all_deps = set()
        for step in self.steps:
            all_deps.update(step.dependencies)
        all_deps.add("tqdm")  # Common utility
        all_deps.add("datasets")  # HuggingFace datasets

        requirements_content = "\n".join(sorted(all_deps))
        requirements_path = output_path / "requirements.txt"
        requirements_path.write_text(requirements_content, encoding="utf-8")
        created_files.append(str(requirements_path))

        # Generate config.yaml
        config_content = self._generate_config()
        config_path = output_path / "config.yaml"
        config_path.write_text(config_content, encoding="utf-8")
        created_files.append(str(config_path))

        # Generate checklist.md
        checklist_content = self._generate_checklist()
        checklist_path = output_path / "checklist.md"
        checklist_path.write_text(checklist_content, encoding="utf-8")
        created_files.append(str(checklist_path))

        # Generate timeline.md
        timeline_content = self._generate_timeline()
        timeline_path = output_path / "timeline.md"
        timeline_path.write_text(timeline_content, encoding="utf-8")
        created_files.append(str(timeline_path))

        # Create data directory
        data_dir = output_path / "data"
        data_dir.mkdir(exist_ok=True)

        # Create .env.example
        env_vars = set()
        for step in self.steps:
            env_vars.update(step.env_vars)
        for _, _, env_var in self.resource_checklist.api_keys:
            env_vars.add(env_var)

        if env_vars:
            env_content = "\n".join(f"{var}=your_{var.lower()}_here" for var in sorted(env_vars))
            env_path = output_path / ".env.example"
            env_path.write_text(env_content, encoding="utf-8")
            created_files.append(str(env_path))

        return created_files

    def _generate_readme(self) -> str:
        """Generate README.md content."""
        lines = []

        lines.append(f"# {self.name}")
        lines.append("")
        lines.append(self.description)
        lines.append("")

        # Quick start
        lines.append("## Quick Start")
        lines.append("")
        lines.append("```bash")
        lines.append("# 1. Install dependencies")
        lines.append("pip install -r requirements.txt")
        lines.append("")
        lines.append("# 2. Configure environment")
        lines.append("cp .env.example .env")
        lines.append("# Edit .env with your API keys")
        lines.append("")
        lines.append("# 3. Run pipeline")
        lines.append("for script in scripts/*.py; do python $script; done")
        lines.append("```")
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Target Size**: {self.target_size:,} examples")
        lines.append(f"- **Estimated Cost**: ${self.estimated_total_cost:,.0f}")
        lines.append(f"- **Steps**: {len(self.steps)}")
        lines.append("")

        # Pipeline steps
        lines.append("## Pipeline Steps")
        lines.append("")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"### Step {i}: {step.name}")
            lines.append("")
            lines.append(step.description)
            lines.append("")
            if step.inputs:
                lines.append(f"**Inputs**: {', '.join(step.inputs)}")
            if step.outputs:
                lines.append(f"**Outputs**: {', '.join(step.outputs)}")
            if step.estimated_cost > 0:
                lines.append(f"**Estimated Cost**: ${step.estimated_cost:,.2f}")
            lines.append("")

        # Resources
        lines.append("## Required Resources")
        lines.append("")
        lines.append("See `checklist.md` for detailed requirements.")
        lines.append("")

        # Timeline
        lines.append("## Timeline")
        lines.append("")
        lines.append("See `timeline.md` for project milestones.")
        lines.append("")

        lines.append("---")
        lines.append("> Generated by DataRecipe")

        return "\n".join(lines)

    def _generate_config(self) -> str:
        """Generate config.yaml content."""
        config = {
            "project": {
                "name": self.name,
                "target_size": self.target_size,
                "estimated_cost_usd": self.estimated_total_cost,
            },
            "paths": {
                "data_dir": "./data",
                "output_dir": "./output",
                "scripts_dir": "./scripts",
            },
            "generation": {
                "batch_size": 100,
                "max_retries": 3,
                "save_interval": 1000,
            },
            "quality": {
                "min_length": 50,
                "max_length": 10000,
                "dedup_threshold": 0.9,
            },
        }

        import yaml

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _generate_checklist(self) -> str:
        """Generate checklist.md content."""
        lines = []

        lines.append("# Resource Checklist")
        lines.append("")
        lines.append("Complete these items before starting the pipeline.")
        lines.append("")

        # API Keys
        lines.append("## API Keys")
        lines.append("")
        if self.resource_checklist.api_keys:
            for name, url, env_var in self.resource_checklist.api_keys:
                lines.append(f"- [ ] **{name}**")
                lines.append(f"  - Sign up: {url}")
                lines.append(f"  - Set environment variable: `{env_var}`")
                lines.append("")
        else:
            lines.append("No API keys required.")
            lines.append("")

        # Dependencies
        lines.append("## Python Dependencies")
        lines.append("")
        lines.append("```bash")
        lines.append("pip install -r requirements.txt")
        lines.append("```")
        lines.append("")
        if self.resource_checklist.dependencies:
            lines.append("Key packages:")
            for dep in self.resource_checklist.dependencies:
                lines.append(f"- [ ] {dep}")
            lines.append("")

        # Compute
        lines.append("## Compute Requirements")
        lines.append("")
        if self.resource_checklist.compute_requirements:
            for key, value in self.resource_checklist.compute_requirements.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- Standard CPU is sufficient for most steps")
            lines.append("- GPU recommended for embedding/training steps")
        lines.append("")

        # Data
        lines.append("## Data Storage")
        lines.append("")
        lines.append(f"- [ ] Ensure ~{max(1, self.target_size // 10000)}GB disk space for intermediate files")
        lines.append("- [ ] Create `./data` directory")
        lines.append("")

        return "\n".join(lines)

    def _generate_timeline(self) -> str:
        """Generate timeline.md content."""
        lines = []

        lines.append("# Project Timeline")
        lines.append("")

        if self.milestones:
            lines.append("## Milestones")
            lines.append("")
            for i, milestone in enumerate(self.milestones, 1):
                lines.append(f"### M{i}: {milestone.name}")
                lines.append("")
                lines.append(milestone.description)
                lines.append("")
                lines.append("**Deliverables:**")
                for d in milestone.deliverables:
                    lines.append(f"- [ ] {d}")
                lines.append("")

        if self.timeline:
            lines.append("## Phases")
            lines.append("")
            lines.append("| Phase | Description |")
            lines.append("|-------|-------------|")
            for phase, desc in self.timeline:
                lines.append(f"| {phase} | {desc} |")
            lines.append("")

        # Step-by-step
        lines.append("## Step-by-Step Progress")
        lines.append("")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"- [ ] Step {i}: {step.name}")
        lines.append("")

        return "\n".join(lines)


class WorkflowGenerator:
    """Generator for production workflows."""

    def __init__(self):
        """Initialize the workflow generator."""
        self.cost_calculator = CostCalculator()

    def generate(
        self,
        recipe: Recipe,
        target_size: Optional[int] = None,
        output_format: str = "huggingface",
    ) -> ProductionWorkflow:
        """Generate a production workflow from a recipe.

        Args:
            recipe: The recipe to generate workflow for
            target_size: Target number of examples
            output_format: Output format ('huggingface', 'jsonl', 'parquet')

        Returns:
            ProductionWorkflow with steps and resources
        """
        target = target_size or recipe.num_examples or 10000

        # Determine workflow type based on recipe
        synthetic_ratio = recipe.synthetic_ratio or 0.0

        if synthetic_ratio >= 0.9:
            return self._generate_synthetic_workflow(recipe, target, output_format)
        elif synthetic_ratio <= 0.1:
            return self._generate_human_workflow(recipe, target, output_format)
        else:
            return self._generate_hybrid_workflow(recipe, target, output_format)

    def _generate_synthetic_workflow(
        self, recipe: Recipe, target_size: int, output_format: str
    ) -> ProductionWorkflow:
        """Generate workflow for synthetic data generation."""

        # Determine model to use
        model = "gpt-4o"
        if recipe.teacher_models:
            model = self.cost_calculator._match_model(recipe.teacher_models[0])

        # Cost estimation
        cost = self.cost_calculator.estimate_from_recipe(recipe, target_size, model)

        steps = [
            WorkflowStep(
                name="Seed Data",
                description="Prepare seed data for generation",
                dependencies=["pandas", "datasets"],
                inputs=["seed_examples.json (optional)"],
                outputs=["data/seed_data.jsonl"],
                script_content='''
# Seed data preparation
# Option 1: Use existing seed data
# Option 2: Generate seed topics/categories

SEED_TOPICS = [
    # Add your seed topics here
    "topic_1",
    "topic_2",
    "topic_3",
]

def generate_seeds(topics, examples_per_topic=10):
    """Generate seed examples from topics."""
    seeds = []
    for topic in topics:
        for i in range(examples_per_topic):
            seeds.append({
                "topic": topic,
                "seed_id": f"{topic}_{i}",
            })
    return seeds

if __name__ == "__main__":
    seeds = generate_seeds(SEED_TOPICS)

    with open(OUTPUT_DIR / "seed_data.jsonl", "w") as f:
        for seed in seeds:
            f.write(json.dumps(seed) + "\\n")

    print(f"Generated {len(seeds)} seed examples")
''',
            ),
            WorkflowStep(
                name="LLM Generation",
                description=f"Generate data using {model}",
                dependencies=["openai", "anthropic", "tqdm", "tenacity"],
                env_vars=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
                inputs=["data/seed_data.jsonl"],
                outputs=["data/generated_raw.jsonl"],
                estimated_cost=cost.api_cost.expected,
                script_content=f'''
import os
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
MODEL = "{model}"
TARGET_SIZE = {target_size}
BATCH_SIZE = 100

PROMPT_TEMPLATE = """
Generate a high-quality example based on the following seed:

Seed: {{seed}}

Requirements:
1. Be specific and detailed
2. Maintain consistency
3. Follow the expected format

Generate:
"""

def get_client():
    """Get the appropriate API client."""
    if "gpt" in MODEL or "openai" in MODEL:
        from openai import OpenAI
        return OpenAI(), "openai"
    else:
        import anthropic
        return anthropic.Anthropic(), "anthropic"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
def generate_single(client, client_type, prompt):
    """Generate a single example."""
    if client_type == "openai":
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{{"role": "user", "content": prompt}}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{{"role": "user", "content": prompt}}],
        )
        return response.content[0].text

def load_seeds():
    """Load seed data."""
    seeds = []
    with open(OUTPUT_DIR / "seed_data.jsonl", "r") as f:
        for line in f:
            seeds.append(json.loads(line))
    return seeds

if __name__ == "__main__":
    client, client_type = get_client()
    seeds = load_seeds()

    generated = []

    for seed in tqdm(seeds[:TARGET_SIZE], desc="Generating"):
        prompt = PROMPT_TEMPLATE.format(seed=json.dumps(seed))
        try:
            output = generate_single(client, client_type, prompt)
            generated.append({{
                "seed": seed,
                "generated": output,
            }})
        except Exception as e:
            print(f"Error: {{e}}")
            continue

        # Save periodically
        if len(generated) % BATCH_SIZE == 0:
            with open(OUTPUT_DIR / "generated_raw.jsonl", "w") as f:
                for item in generated:
                    f.write(json.dumps(item) + "\\n")

    # Final save
    with open(OUTPUT_DIR / "generated_raw.jsonl", "w") as f:
        for item in generated:
            f.write(json.dumps(item) + "\\n")

    print(f"Generated {{len(generated)}} examples")
''',
            ),
            WorkflowStep(
                name="Quality Filtering",
                description="Filter generated data for quality",
                dependencies=["pandas"],
                inputs=["data/generated_raw.jsonl"],
                outputs=["data/filtered.jsonl"],
                script_content='''
# Quality filtering
import re

MIN_LENGTH = 50
MAX_LENGTH = 10000

def quality_check(item):
    """Check if an item passes quality criteria."""
    text = item.get("generated", "")

    # Length check
    if len(text) < MIN_LENGTH or len(text) > MAX_LENGTH:
        return False

    # Basic quality checks
    if text.count("\\n\\n\\n") > 3:  # Too many blank lines
        return False

    if text.lower().startswith("i cannot") or text.lower().startswith("i'm sorry"):
        return False

    return True

if __name__ == "__main__":
    filtered = []
    total = 0

    with open(OUTPUT_DIR / "generated_raw.jsonl", "r") as f:
        for line in f:
            total += 1
            item = json.loads(line)
            if quality_check(item):
                filtered.append(item)

    with open(OUTPUT_DIR / "filtered.jsonl", "w") as f:
        for item in filtered:
            f.write(json.dumps(item) + "\\n")

    print(f"Filtered: {len(filtered)}/{total} passed ({len(filtered)/total*100:.1f}%)")
''',
            ),
            WorkflowStep(
                name="Deduplication",
                description="Remove duplicate or near-duplicate examples",
                dependencies=["pandas", "datasketch"],
                inputs=["data/filtered.jsonl"],
                outputs=["data/deduped.jsonl"],
                script_content='''
# Deduplication using MinHash
from datasketch import MinHash, MinHashLSH

THRESHOLD = 0.9  # Similarity threshold

def get_minhash(text, num_perm=128):
    """Get MinHash signature for text."""
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode('utf8'))
    return m

if __name__ == "__main__":
    # Load data
    items = []
    with open(OUTPUT_DIR / "filtered.jsonl", "r") as f:
        for line in f:
            items.append(json.loads(line))

    # Build LSH index
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=128)
    minhashes = []

    for i, item in enumerate(items):
        text = item.get("generated", "")
        mh = get_minhash(text)
        minhashes.append(mh)
        lsh.insert(f"doc_{i}", mh)

    # Find duplicates
    seen = set()
    deduped = []

    for i, item in enumerate(items):
        if i in seen:
            continue

        # Find similar items
        result = lsh.query(minhashes[i])
        for r in result:
            idx = int(r.split("_")[1])
            if idx != i:
                seen.add(idx)

        deduped.append(item)

    # Save
    with open(OUTPUT_DIR / "deduped.jsonl", "w") as f:
        for item in deduped:
            f.write(json.dumps(item) + "\\n")

    print(f"Deduplication: {len(deduped)}/{len(items)} unique ({len(items)-len(deduped)} removed)")
''',
            ),
            WorkflowStep(
                name="Validation",
                description="Validate and format final dataset",
                dependencies=["datasets", "pandas"],
                inputs=["data/deduped.jsonl"],
                outputs=["data/final_dataset/"],
                script_content=f'''
# Final validation and formatting
from datasets import Dataset

def validate_item(item):
    """Validate a single item."""
    required_fields = ["generated"]
    for field in required_fields:
        if field not in item:
            return False
    return True

if __name__ == "__main__":
    # Load and validate
    valid_items = []
    with open(OUTPUT_DIR / "deduped.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            if validate_item(item):
                # Restructure for final format
                valid_items.append({{
                    "text": item["generated"],
                    "metadata": item.get("seed", {{}}),
                }})

    print(f"Validated: {{len(valid_items)}} items")

    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(valid_items)

    # Save
    output_path = OUTPUT_DIR / "final_dataset"
    dataset.save_to_disk(str(output_path))

    # Also save as JSONL
    with open(OUTPUT_DIR / "final_dataset.jsonl", "w") as f:
        for item in valid_items:
            f.write(json.dumps(item) + "\\n")

    print(f"Dataset saved to {{output_path}}")
    print(f"Total examples: {{len(dataset)}}")
''',
            ),
        ]

        # Create resource checklist
        api_keys = []
        if "gpt" in model or "openai" in model.lower():
            api_keys.append(("OpenAI API", "https://platform.openai.com/api-keys", "OPENAI_API_KEY"))
        if "claude" in model.lower() or "anthropic" in model.lower():
            api_keys.append(("Anthropic API", "https://console.anthropic.com/", "ANTHROPIC_API_KEY"))

        resource_checklist = ResourceChecklist(
            api_keys=api_keys,
            dependencies=["openai", "anthropic", "datasets", "pandas", "tqdm", "datasketch"],
            compute_requirements={
                "CPU": "4+ cores recommended",
                "RAM": "8GB minimum",
                "Storage": f"{max(1, target_size // 5000)}GB",
            },
        )

        # Create milestones
        milestones = [
            Milestone(
                name="Setup Complete",
                description="Environment configured and ready",
                deliverables=["API keys configured", "Dependencies installed", "Seed data prepared"],
            ),
            Milestone(
                name="Generation Complete",
                description="Raw data generated from LLM",
                deliverables=[f"{target_size} raw examples generated", "Generation logs saved"],
            ),
            Milestone(
                name="Quality Assured",
                description="Data filtered and deduplicated",
                deliverables=["Quality filtering complete", "Duplicates removed", "Validation passed"],
            ),
            Milestone(
                name="Dataset Published",
                description="Final dataset ready for use",
                deliverables=["Dataset formatted", "Documentation complete", "Optional: Published to HuggingFace"],
            ),
        ]

        # Create timeline
        timeline = [
            ("Phase 1: Setup", "Configure environment and prepare seeds"),
            ("Phase 2: Generation", "Run LLM generation pipeline"),
            ("Phase 3: Quality", "Filter, dedupe, and validate"),
            ("Phase 4: Publish", "Format and publish dataset"),
        ]

        return ProductionWorkflow(
            name=f"Reproduce {recipe.name}",
            description=f"Production workflow to generate {target_size:,} examples similar to {recipe.name}",
            steps=steps,
            milestones=milestones,
            resource_checklist=resource_checklist,
            timeline=timeline,
            estimated_total_cost=cost.total.expected,
            target_size=target_size,
        )

    def _generate_human_workflow(
        self, recipe: Recipe, target_size: int, output_format: str
    ) -> ProductionWorkflow:
        """Generate workflow for human annotation."""
        cost = self.cost_calculator.estimate_from_recipe(recipe, target_size)

        steps = [
            WorkflowStep(
                name="Data Collection",
                description="Collect raw data for annotation",
                dependencies=["pandas", "requests"],
                inputs=["data_sources.txt"],
                outputs=["data/raw_data.jsonl"],
                script_content='''
# Data collection script
# Customize based on your data sources

def collect_from_source(source):
    """Collect data from a source."""
    # Implement your data collection logic
    return []

if __name__ == "__main__":
    sources = []  # Add your data sources

    collected = []
    for source in sources:
        collected.extend(collect_from_source(source))

    with open(OUTPUT_DIR / "raw_data.jsonl", "w") as f:
        for item in collected:
            f.write(json.dumps(item) + "\\n")

    print(f"Collected {len(collected)} items")
''',
            ),
            WorkflowStep(
                name="Annotation Guidelines",
                description="Prepare annotation guidelines and examples",
                dependencies=[],
                inputs=[],
                outputs=["annotation_guidelines.md", "examples.json"],
                script_content='''
# Generate annotation guidelines template

GUIDELINES_TEMPLATE = """
# Annotation Guidelines

## Task Description
[Describe the annotation task]

## Labels/Categories
[List the labels or categories]

## Instructions
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Examples

### Good Example
[Provide a good example]

### Bad Example
[Provide a bad example]

## Edge Cases
[Describe how to handle edge cases]
"""

if __name__ == "__main__":
    with open("annotation_guidelines.md", "w") as f:
        f.write(GUIDELINES_TEMPLATE)

    print("Guidelines template created. Please customize before annotation.")
''',
            ),
            WorkflowStep(
                name="Annotation Platform Setup",
                description="Set up annotation platform (Label Studio, etc.)",
                dependencies=["label-studio"],
                inputs=["data/raw_data.jsonl"],
                outputs=["annotation_project_config.json"],
                script_content='''
# Label Studio setup
# Alternative: Use Scale AI, Labelbox, or MTurk

LABEL_CONFIG = """
<View>
  <Header value="Annotation Task"/>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text">
    <Choice value="category_1"/>
    <Choice value="category_2"/>
    <Choice value="category_3"/>
  </Choices>
</View>
"""

if __name__ == "__main__":
    config = {
        "title": "Dataset Annotation",
        "label_config": LABEL_CONFIG,
    }

    with open("annotation_project_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Annotation config created.")
    print("To start Label Studio: label-studio start")
''',
            ),
            WorkflowStep(
                name="Quality Control",
                description="Review annotations and calculate agreement",
                dependencies=["pandas", "scikit-learn"],
                inputs=["data/annotations.jsonl"],
                outputs=["data/quality_report.json", "data/final_annotations.jsonl"],
                estimated_cost=cost.human_annotation_cost.expected,
                script_content='''
# Quality control and agreement calculation
from sklearn.metrics import cohen_kappa_score
import pandas as pd

def calculate_agreement(annotations):
    """Calculate inter-annotator agreement."""
    # Group by item and annotator
    # Calculate Cohen's Kappa
    return 0.0  # Implement based on your annotation format

if __name__ == "__main__":
    # Load annotations
    annotations = []
    with open(OUTPUT_DIR / "annotations.jsonl", "r") as f:
        for line in f:
            annotations.append(json.loads(line))

    # Calculate quality metrics
    # agreement = calculate_agreement(annotations)

    # Filter low-quality annotations
    quality_annotations = [a for a in annotations if True]  # Add your criteria

    # Save
    with open(OUTPUT_DIR / "final_annotations.jsonl", "w") as f:
        for item in quality_annotations:
            f.write(json.dumps(item) + "\\n")

    print(f"Quality check: {len(quality_annotations)}/{len(annotations)} passed")
''',
            ),
            WorkflowStep(
                name="Dataset Export",
                description="Export final annotated dataset",
                dependencies=["datasets"],
                inputs=["data/final_annotations.jsonl"],
                outputs=["data/final_dataset/"],
                script_content='''
from datasets import Dataset

if __name__ == "__main__":
    items = []
    with open(OUTPUT_DIR / "final_annotations.jsonl", "r") as f:
        for line in f:
            items.append(json.loads(line))

    dataset = Dataset.from_list(items)
    dataset.save_to_disk(str(OUTPUT_DIR / "final_dataset"))

    print(f"Dataset saved: {len(dataset)} examples")
''',
            ),
        ]

        resource_checklist = ResourceChecklist(
            api_keys=[
                ("Label Studio (optional)", "https://labelstud.io/", "LABEL_STUDIO_API_KEY"),
            ],
            dependencies=["pandas", "datasets", "label-studio", "scikit-learn"],
            compute_requirements={
                "CPU": "2+ cores",
                "RAM": "4GB minimum",
            },
        )

        milestones = [
            Milestone(
                name="Data Collected",
                description="Raw data collected and prepared",
                deliverables=["Raw data file", "Data statistics"],
            ),
            Milestone(
                name="Guidelines Ready",
                description="Annotation guidelines finalized",
                deliverables=["Guidelines document", "Example annotations"],
            ),
            Milestone(
                name="Annotation Complete",
                description="All items annotated",
                deliverables=[f"{target_size} annotated examples", "Quality metrics"],
            ),
            Milestone(
                name="Dataset Published",
                description="Final dataset exported",
                deliverables=["Final dataset", "Documentation"],
            ),
        ]

        return ProductionWorkflow(
            name=f"Reproduce {recipe.name}",
            description=f"Human annotation workflow to create {target_size:,} examples",
            steps=steps,
            milestones=milestones,
            resource_checklist=resource_checklist,
            timeline=[
                ("Phase 1: Collection", "Collect and prepare raw data"),
                ("Phase 2: Setup", "Prepare guidelines and platform"),
                ("Phase 3: Annotation", "Execute annotation tasks"),
                ("Phase 4: QC & Export", "Quality control and export"),
            ],
            estimated_total_cost=cost.total.expected,
            target_size=target_size,
        )

    def _generate_hybrid_workflow(
        self, recipe: Recipe, target_size: int, output_format: str
    ) -> ProductionWorkflow:
        """Generate workflow for hybrid (LLM + human) generation."""
        synthetic_workflow = self._generate_synthetic_workflow(recipe, target_size, output_format)

        # Add human verification step
        verification_step = WorkflowStep(
            name="Human Verification",
            description="Human review and correction of LLM-generated data",
            dependencies=["pandas", "label-studio"],
            inputs=["data/deduped.jsonl"],
            outputs=["data/verified.jsonl"],
            estimated_cost=self.cost_calculator.ANNOTATION_COSTS["quality_check"] * target_size * (recipe.human_ratio or 0.3),
            script_content='''
# Human verification of LLM-generated content

def needs_review(item):
    """Determine if an item needs human review."""
    # Add your criteria
    return True

if __name__ == "__main__":
    items = []
    with open(OUTPUT_DIR / "deduped.jsonl", "r") as f:
        for line in f:
            items.append(json.loads(line))

    # Mark items for review
    for_review = [i for i in items if needs_review(i)]
    auto_pass = [i for i in items if not needs_review(i)]

    print(f"Items for review: {len(for_review)}")
    print(f"Auto-pass: {len(auto_pass)}")

    # Export for annotation platform
    with open(OUTPUT_DIR / "for_review.jsonl", "w") as f:
        for item in for_review:
            f.write(json.dumps(item) + "\\n")

    # After human review, merge results
    # verified = auto_pass + reviewed_items
    # Save to verified.jsonl
''',
        )

        # Insert verification step before validation
        synthetic_workflow.steps.insert(-1, verification_step)
        synthetic_workflow.name = f"Reproduce {recipe.name} (Hybrid)"
        synthetic_workflow.description = f"Hybrid workflow combining LLM generation with human verification for {target_size:,} examples"

        return synthetic_workflow
