"""Command-line interface for DataRecipe."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from datarecipe.analyzer import DatasetAnalyzer
from datarecipe.schema import Recipe


console = Console()


def recipe_to_markdown(recipe: Recipe) -> str:
    """Generate a beautiful Markdown document for a recipe in Chinese."""
    lines = []

    # Title
    lines.append(f"# 📊 数据集配方分析：{recipe.name}")
    lines.append("")

    # Summary box
    lines.append("> **DataRecipe 数据集成分分析报告**")
    lines.append("> ")
    lines.append("> 深入分析该数据集的构建方式——数据来源、生成方法与可复现性评估。")
    lines.append("")

    # Basic Info
    lines.append("## 📋 基本信息")
    lines.append("")
    lines.append("| 属性 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| **数据集名称** | `{recipe.name}` |")
    lines.append(f"| **数据来源** | {recipe.source_type.value.title()} |")
    if recipe.license:
        lines.append(f"| **许可证** | {recipe.license} |")
    if recipe.languages:
        langs = [l for l in recipe.languages if l]
        if langs:
            lines.append(f"| **语言** | {', '.join(langs)} |")
    if recipe.num_examples:
        lines.append(f"| **样本数量** | {recipe.num_examples:,} |")
    lines.append("")

    # Generation Method
    lines.append("## 🧬 数据生成方式")
    lines.append("")

    if recipe.synthetic_ratio is not None or recipe.human_ratio is not None:
        synthetic_pct = (recipe.synthetic_ratio or 0) * 100
        human_pct = (recipe.human_ratio or 0) * 100

        # Progress bar visualization (PDF-safe format)
        synthetic_filled = int(synthetic_pct / 5)
        human_filled = int(human_pct / 5)
        synthetic_bar = "[" + "=" * synthetic_filled + "-" * (20 - synthetic_filled) + "]"
        human_bar = "[" + "=" * human_filled + "-" * (20 - human_filled) + "]"

        lines.append("| 类型 | 占比 | 分布 |")
        lines.append("|------|------|------|")
        lines.append(f"| 合成数据 | {synthetic_pct:.0f}% | `{synthetic_bar}` |")
        lines.append(f"| 人工标注 | {human_pct:.0f}% | `{human_bar}` |")
    else:
        lines.append("*无法从现有元数据中确定生成方式。*")
    lines.append("")

    # Teacher Models
    lines.append("## 🎓 教师模型")
    lines.append("")

    if recipe.teacher_models:
        lines.append("检测到以下 AI 模型被用于数据生成：")
        lines.append("")
        for model in recipe.teacher_models:
            lines.append(f"- **{model}**")
    else:
        lines.append("*未在数据集文档中检测到教师模型。*")
    lines.append("")

    # Generation Methods Detail
    if recipe.generation_methods:
        method_type_map = {
            "distillation": "知识蒸馏",
            "human_annotation": "人工标注",
            "web_scrape": "网页抓取",
            "red_teaming": "红队测试",
        }
        lines.append("### 生成流程")
        lines.append("")
        for i, method in enumerate(recipe.generation_methods, 1):
            method_name = method_type_map.get(method.method_type, method.method_type.replace('_', ' ').title())
            lines.append(f"**步骤 {i}：{method_name}**")
            if method.teacher_model:
                lines.append(f"- 教师模型：`{method.teacher_model}`")
            if method.platform:
                lines.append(f"- 标注平台：{method.platform}")
            if method.prompt_template_available:
                lines.append(f"- 提示词模板：✅ 可用")
            lines.append("")

    # Cost Estimation
    lines.append("## 💰 成本估算")
    lines.append("")

    if recipe.cost and recipe.cost.estimated_total_usd:
        if recipe.cost.confidence == "low":
            low = recipe.cost.estimated_total_usd * 0.5
            high = recipe.cost.estimated_total_usd * 1.5
            lines.append(f"**预估总成本：${low:,.0f} - ${high:,.0f}** *(低置信度)*")
        elif recipe.cost.confidence == "medium":
            low = recipe.cost.estimated_total_usd * 0.8
            high = recipe.cost.estimated_total_usd * 1.2
            lines.append(f"**预估总成本：${low:,.0f} - ${high:,.0f}** *(中置信度)*")
        else:
            lines.append(f"**预估总成本：${recipe.cost.estimated_total_usd:,.0f}**")
        lines.append("")

        lines.append("| 类别 | 成本 |")
        lines.append("|------|------|")
        if recipe.cost.api_calls_usd:
            lines.append(f"| API 调用 | ${recipe.cost.api_calls_usd:,.0f} |")
        if recipe.cost.human_annotation_usd:
            lines.append(f"| 人工标注 | ${recipe.cost.human_annotation_usd:,.0f} |")
        if recipe.cost.compute_usd:
            lines.append(f"| 计算资源 | ${recipe.cost.compute_usd:,.0f} |")
    else:
        lines.append("*暂无成本估算数据。*")
    lines.append("")

    # Reproducibility
    lines.append("## 🔄 可复现性评估")
    lines.append("")

    if recipe.reproducibility:
        score = recipe.reproducibility.score
        score_bar = "[" + "#" * score + "-" * (10 - score) + "]"
        lines.append(f"### 评分：{score}/10")
        lines.append("")
        lines.append(f"**{score_bar}**")
        lines.append("")

        # Translation map for reproducibility items
        item_translation = {
            "description": "数据集描述",
            "detailed_documentation": "详细文档",
            "source_code_reference": "源代码引用",
            "teacher_model_names": "教师模型名称",
            "teacher_model_info": "教师模型信息",
            "prompt_templates": "提示词模板",
            "exact_prompts": "精确提示词",
            "filtering_criteria": "过滤标准",
            "quality_thresholds": "质量阈值",
            "generation_scripts": "生成脚本",
            "source_data_references": "源数据引用",
            "general_methodology": "通用方法论",
            "dataset_statistics": "数据集统计",
        }

        if recipe.reproducibility.available:
            lines.append("#### ✅ 已提供的信息")
            lines.append("")
            for item in recipe.reproducibility.available:
                translated = item_translation.get(item, item.replace('_', ' ').title())
                lines.append(f"- {translated}")
            lines.append("")

        if recipe.reproducibility.missing:
            lines.append("#### ❌ 缺失的信息")
            lines.append("")
            for item in recipe.reproducibility.missing:
                translated = item_translation.get(item, item.replace('_', ' ').title())
                lines.append(f"- {translated}")
            lines.append("")

        if recipe.reproducibility.notes:
            lines.append("#### 📝 备注")
            lines.append("")
            lines.append(recipe.reproducibility.notes)
            lines.append("")
    else:
        lines.append("*暂无可复现性评估。*")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*由 [DataRecipe](https://github.com/yourusername/data-recipe) 生成 - AI 数据集成分分析器*")

    return "\n".join(lines)


def display_recipe(recipe: Recipe) -> None:
    """Display a recipe in a formatted panel."""
    # Build the content
    lines = []

    # Header info
    lines.append(f"[bold]Name:[/bold] {recipe.name}")
    lines.append(f"[bold]Source:[/bold] {recipe.source_type.value}")
    lines.append("")

    # Generation Method
    lines.append("[bold cyan]📊 Generation Method:[/bold cyan]")
    if recipe.synthetic_ratio is not None:
        lines.append(f"   • Synthetic: {recipe.synthetic_ratio * 100:.0f}%")
    if recipe.human_ratio is not None:
        lines.append(f"   • Human: {recipe.human_ratio * 100:.0f}%")
    if recipe.generation_type.value == "unknown":
        lines.append("   • [dim]Unable to determine[/dim]")
    lines.append("")

    # Teacher Models
    lines.append("[bold cyan]🤖 Teacher Models:[/bold cyan]")
    if recipe.teacher_models:
        for model in recipe.teacher_models:
            lines.append(f"   • {model}")
    else:
        lines.append("   • [dim]None detected[/dim]")
    lines.append("")

    # Cost Estimation
    lines.append("[bold cyan]💰 Estimated Cost:[/bold cyan]")
    if recipe.cost and recipe.cost.estimated_total_usd:
        # Show as a range for low confidence
        if recipe.cost.confidence == "low":
            low = recipe.cost.estimated_total_usd * 0.5
            high = recipe.cost.estimated_total_usd * 1.5
            lines.append(f"   ${low:,.0f} - ${high:,.0f} [dim](low confidence)[/dim]")
        else:
            lines.append(f"   ${recipe.cost.estimated_total_usd:,.0f}")

        if recipe.cost.api_calls_usd:
            lines.append(f"   [dim]├─ API calls: ${recipe.cost.api_calls_usd:,.0f}[/dim]")
        if recipe.cost.human_annotation_usd:
            lines.append(
                f"   [dim]└─ Human annotation: ${recipe.cost.human_annotation_usd:,.0f}[/dim]"
            )
    else:
        lines.append("   [dim]Unable to estimate[/dim]")
    lines.append("")

    # Reproducibility
    lines.append("[bold cyan]🔄 Reproducibility Score:[/bold cyan]")
    if recipe.reproducibility:
        score = recipe.reproducibility.score
        score_bar = "█" * score + "░" * (10 - score)
        lines.append(f"   [{score}/10] {score_bar}")

        if recipe.reproducibility.available:
            lines.append(f"   [green]✓ Available:[/green] {', '.join(recipe.reproducibility.available[:3])}")
        if recipe.reproducibility.missing:
            lines.append(f"   [red]✗ Missing:[/red] {', '.join(recipe.reproducibility.missing[:3])}")
    else:
        lines.append("   [dim]Not assessed[/dim]")

    # Create panel
    content = "\n".join(lines)
    panel = Panel(
        content,
        title="[bold white]Dataset Recipe[/bold white]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


@click.group()
@click.version_option(version="0.2.0", prog_name="datarecipe")
def main():
    """DataRecipe - Analyze AI dataset ingredients, estimate costs, and generate workflows."""
    pass


@main.command()
@click.argument("dataset_id")
@click.option("--output", "-o", type=click.Path(), help="Export recipe to file (auto-detect format by extension)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--yaml", "as_yaml", is_flag=True, help="Output as YAML")
@click.option("--markdown", "--md", "as_markdown", is_flag=True, help="Output as Markdown")
def analyze(dataset_id: str, output: str, as_json: bool, as_yaml: bool, as_markdown: bool):
    """Analyze a dataset and display its recipe.

    DATASET_ID is the identifier of the dataset to analyze.
    For HuggingFace datasets, use the format: org/dataset-name
    """
    analyzer = DatasetAnalyzer()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        try:
            recipe = analyzer.analyze(dataset_id)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error analyzing dataset:[/red] {e}")
            sys.exit(1)

    # Output format
    if as_json:
        import json

        console.print(json.dumps(recipe.to_dict(), indent=2))
    elif as_yaml:
        console.print(recipe.to_yaml())
    elif as_markdown:
        print(recipe_to_markdown(recipe))
    else:
        display_recipe(recipe)

    # Export if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output.endswith(".md"):
            output_path.write_text(recipe_to_markdown(recipe), encoding="utf-8")
            console.print(f"\n[green]Markdown exported to:[/green] {output}")
        elif output.endswith(".json"):
            import json
            output_path.write_text(json.dumps(recipe.to_dict(), indent=2), encoding="utf-8")
            console.print(f"\n[green]JSON exported to:[/green] {output}")
        else:
            analyzer.export_recipe(recipe, output)
            console.print(f"\n[green]Recipe exported to:[/green] {output}")


@main.command()
@click.argument("recipe_file", type=click.Path(exists=True))
def show(recipe_file: str):
    """Display a recipe from a YAML file.

    RECIPE_FILE is the path to the recipe YAML file.
    """
    analyzer = DatasetAnalyzer()

    try:
        recipe = analyzer.analyze_from_yaml(recipe_file)
        display_recipe(recipe)
    except Exception as e:
        console.print(f"[red]Error loading recipe:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("dataset_id")
@click.argument("output_file", type=click.Path())
def export(dataset_id: str, output_file: str):
    """Analyze a dataset and export recipe to YAML.

    DATASET_ID is the identifier of the dataset to analyze.
    OUTPUT_FILE is the path where the YAML recipe will be saved.
    """
    analyzer = DatasetAnalyzer()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        try:
            recipe = analyzer.analyze(dataset_id)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error analyzing dataset:[/red] {e}")
            sys.exit(1)

    analyzer.export_recipe(recipe, output_file)
    console.print(f"[green]Recipe exported to:[/green] {output_file}")


@main.command()
def list_sources():
    """List supported data sources."""
    table = Table(title="Supported Data Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Example Input")

    table.add_row("HuggingFace Hub", "✓ Supported", "org/dataset-name 或 URL")
    table.add_row("GitHub", "✓ Supported", "https://github.com/org/repo")
    table.add_row("Web URL", "✓ Supported", "https://example.com/dataset")
    table.add_row("Local files", "✓ Supported", "datarecipe create (交互式)")

    console.print(table)


@main.command()
@click.argument("dataset_id")
@click.option("--output", "-o", type=click.Path(), help="Output file path for production guide")
@click.option("--target-size", "-n", type=int, help="Target dataset size")
def guide(dataset_id: str, output: str, target_size: int):
    """Generate a production guide for recreating a dataset.

    Analyzes a dataset and outputs a step-by-step guide for producing
    similar data, including code snippets, tools, and best practices.

    DATASET_ID can be a HuggingFace ID, GitHub URL, or any web URL.
    """
    from datarecipe.pipeline import get_pipeline_template, pipeline_to_markdown

    analyzer = DatasetAnalyzer()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        try:
            recipe = analyzer.analyze(dataset_id)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error analyzing dataset:[/red] {e}")
            sys.exit(1)

    # Get appropriate pipeline template
    pipeline = get_pipeline_template(
        recipe.generation_type.value if recipe.generation_type else "unknown",
        recipe.synthetic_ratio
    )

    # Customize pipeline with dataset info
    if target_size:
        pipeline.target_size = target_size

    if recipe.cost and recipe.cost.estimated_total_usd:
        pipeline.estimated_total_cost = recipe.cost.estimated_total_usd

    # Generate guide
    guide_content = pipeline_to_markdown(pipeline, recipe.name)

    # Add dataset-specific info at the top
    header = f"""# 数据生产指南：{recipe.name}

## 参考数据集分析

| 属性 | 值 |
|------|-----|
| **数据集名称** | {recipe.name} |
| **来源** | {recipe.source_type.value} |
| **合成数据比例** | {recipe.synthetic_ratio * 100 if recipe.synthetic_ratio else 'N/A'}% |
| **人工数据比例** | {recipe.human_ratio * 100 if recipe.human_ratio else 'N/A'}% |
| **教师模型** | {', '.join(recipe.teacher_models) if recipe.teacher_models else '无'} |
| **可复现性评分** | {recipe.reproducibility.score}/10 |

---

"""
    full_guide = header + guide_content.split("# ", 1)[-1]  # Remove duplicate title

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_guide, encoding="utf-8")
        console.print(f"[green]✓ 生产指南已保存到:[/green] {output}")
    else:
        print(full_guide)

    # Also display summary
    console.print("\n[bold cyan]生产指南概要:[/bold cyan]")
    console.print(f"  流程类型: {pipeline.name}")
    console.print(f"  步骤数量: {len(pipeline.steps)}")
    if pipeline.estimated_total_cost:
        console.print(f"  预估成本: ${pipeline.estimated_total_cost:,.0f}")


@main.command("deep-guide")
@click.argument("url")
@click.option("--output", "-o", type=click.Path(), help="Output file path for production guide")
@click.option("--llm/--no-llm", default=False, help="Use LLM for enhanced analysis (requires API key)")
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default="anthropic", help="LLM provider")
def deep_guide(url: str, output: str, llm: bool, provider: str):
    """Generate a customized production guide using deep analysis.

    This command performs deep analysis on a paper or dataset page and
    generates a specialized production guide based on the methodology
    detected in the source.

    URL can be an arXiv paper, dataset page, or any web URL describing
    a dataset's construction methodology.

    Use --llm flag to enable LLM-enhanced analysis for better results.
    Requires ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.

    Examples:
        datarecipe deep-guide https://arxiv.org/abs/2506.07982
        datarecipe deep-guide https://arcprize.org/arc-agi/2/ --llm
    """
    from datarecipe.deep_analyzer import deep_analysis_to_markdown

    # Try to use LLMAnalyzer with PDF parsing (even without LLM)
    try:
        from datarecipe.llm_analyzer import LLMAnalyzer
        if llm:
            console.print(f"[cyan]使用 LLM 增强分析 (provider: {provider})...[/cyan]")
            analyzer = LLMAnalyzer(use_llm=True, llm_provider=provider, parse_pdf=True)
        else:
            console.print("[cyan]使用 PDF 解析和多源聚合分析...[/cyan]")
            analyzer = LLMAnalyzer(use_llm=False, parse_pdf=True)
    except ImportError as e:
        if llm:
            console.print(f"[yellow]Warning:[/yellow] {e}")
        console.print("[yellow]使用基础模式匹配分析...[/yellow]")
        from datarecipe.deep_analyzer import DeepAnalyzer
        analyzer = DeepAnalyzer()

    with console.status(f"[cyan]Performing deep analysis on {url}...[/cyan]"):
        try:
            result = analyzer.analyze(url)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error during analysis:[/red] {e}")
            sys.exit(1)

    # Generate customized guide
    guide_content = deep_analysis_to_markdown(result)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(guide_content, encoding="utf-8")
        console.print(f"[green]✓ 专项生产指南已保存到:[/green] {output}")
    else:
        print(guide_content)

    # Display summary
    console.print("\n[bold cyan]深度分析概要:[/bold cyan]")
    console.print(f"  数据集名称: {result.name}")
    console.print(f"  分类: {result.category.value}")
    console.print(f"  领域: {result.domain or '通用'}")
    if result.methodology:
        console.print(f"  方法论: {result.methodology}")
    if result.key_innovations:
        console.print(f"  核心创新: {len(result.key_innovations)} 项")
    if result.generation_steps:
        console.print(f"  生产步骤: {len(result.generation_steps)} 步")
    if result.code_available:
        console.print(f"  代码可用: ✓ {result.code_url or ''}")
    if result.data_available:
        console.print(f"  数据可用: ✓ {result.data_url or ''}")
    if hasattr(result, 'paper_url') and result.paper_url:
        console.print(f"  [green]自动发现论文:[/green] {result.paper_url}")


@main.command()
@click.option("--output", "-o", type=click.Path(), help="Output YAML file path")
def create(output: str):
    """Interactively create a dataset recipe.

    This command guides you through creating a recipe file step by step.
    """
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt

    console.print("\n[bold cyan]📝 创建数据集配方 / Create Dataset Recipe[/bold cyan]\n")

    # Basic info
    name = Prompt.ask("数据集名称 / Dataset name")
    version = Prompt.ask("版本 / Version", default="1.0")

    # Source
    console.print("\n[bold]数据来源 / Data Source[/bold]")
    source_type = Prompt.ask(
        "来源类型 / Source type",
        choices=["huggingface", "github", "web", "local"],
        default="local"
    )
    source_id = Prompt.ask("来源标识 / Source ID (URL or ID)", default="")

    # Generation
    console.print("\n[bold]生成方式 / Generation Method[/bold]")
    synthetic_ratio = FloatPrompt.ask(
        "合成数据比例 / Synthetic ratio (0.0-1.0)",
        default=0.0
    )
    human_ratio = 1.0 - synthetic_ratio

    teacher_models = []
    if synthetic_ratio > 0:
        models_input = Prompt.ask(
            "教师模型 / Teacher models (逗号分隔 / comma-separated)",
            default=""
        )
        if models_input:
            teacher_models = [m.strip() for m in models_input.split(",")]

    # Cost
    console.print("\n[bold]成本估算 / Cost Estimation[/bold]")
    has_cost = Confirm.ask("是否添加成本信息? / Add cost info?", default=False)
    cost_total = None
    cost_confidence = "low"
    if has_cost:
        cost_total = FloatPrompt.ask("预估总成本 (USD) / Estimated total cost", default=0)
        cost_confidence = Prompt.ask(
            "置信度 / Confidence",
            choices=["low", "medium", "high"],
            default="low"
        )

    # Reproducibility
    console.print("\n[bold]可复现性 / Reproducibility[/bold]")
    repro_score = IntPrompt.ask("可复现性评分 (1-10) / Score", default=5)

    available_input = Prompt.ask(
        "已提供的信息 / Available info (逗号分隔 / comma-separated)",
        default="description"
    )
    available = [a.strip() for a in available_input.split(",") if a.strip()]

    missing_input = Prompt.ask(
        "缺失的信息 / Missing info (逗号分隔 / comma-separated)",
        default="exact_prompts,filtering_criteria"
    )
    missing = [m.strip() for m in missing_input.split(",") if m.strip()]

    # Metadata
    console.print("\n[bold]元数据 / Metadata[/bold]")
    num_examples = IntPrompt.ask("样本数量 / Number of examples", default=0)
    languages_input = Prompt.ask("语言 / Languages (逗号分隔)", default="en")
    languages = [l.strip() for l in languages_input.split(",") if l.strip()]
    license_str = Prompt.ask("许可证 / License", default="unknown")

    tags_input = Prompt.ask("标签 / Tags (逗号分隔)", default="")
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]

    # Build YAML content
    yaml_content = f"""# Recipe for {name}
# Generated by DataRecipe

name: {name}
version: "{version}"

source:
  type: {source_type}
  id: {source_id or name}

generation:
  synthetic_ratio: {synthetic_ratio}
  human_ratio: {human_ratio}
  teacher_models: {teacher_models}
  methods:"""

    if teacher_models:
        for model in teacher_models:
            yaml_content += f"""
    - type: distillation
      teacher_model: {model}"""

    if human_ratio > 0:
        yaml_content += """
    - type: human_annotation"""

    yaml_content += f"""

cost:
  estimated_total_usd: {cost_total if cost_total else 'null'}
  confidence: {cost_confidence}

reproducibility:
  score: {repro_score}
  available: {available}
  missing: {missing}

metadata:
  num_examples: {num_examples if num_examples else 'null'}
  languages: {languages}
  license: {license_str}
  tags: {tags}
"""

    # Output
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_content, encoding="utf-8")
        console.print(f"\n[green]✓ 配方已保存到 / Recipe saved to:[/green] {output}")
    else:
        # Default output path
        safe_name = name.replace("/", "-").replace(" ", "-").lower()
        output_path = Path(f"recipes/{safe_name}.yaml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_content, encoding="utf-8")
        console.print(f"\n[green]✓ 配方已保存到 / Recipe saved to:[/green] {output_path}")

    # Show preview
    console.print("\n[bold]预览 / Preview:[/bold]")
    console.print(yaml_content)


@main.command()
@click.argument("dataset_id")
@click.option("--model", "-m", default="gpt-4o", help="LLM model for cost estimation")
@click.option("--examples", "-n", type=int, help="Target number of examples")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def cost(dataset_id: str, model: str, examples: int, as_json: bool):
    """Calculate production cost estimate for a dataset.

    DATASET_ID is the identifier of the dataset to analyze.
    """
    from datarecipe.cost_calculator import CostCalculator

    analyzer = DatasetAnalyzer()
    calculator = CostCalculator()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        try:
            recipe = analyzer.analyze(dataset_id)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    target_size = examples or recipe.num_examples or 10000

    with console.status("[cyan]Calculating costs...[/cyan]"):
        cost_breakdown = calculator.estimate_from_recipe(recipe, target_size, model)

    if as_json:
        import json
        console.print(json.dumps(cost_breakdown.to_dict(), indent=2))
    else:
        console.print(f"\n[bold cyan]Cost Estimate for {dataset_id}[/bold cyan]")
        console.print(f"Target size: {target_size:,} examples")
        console.print(f"Model: {model}")
        console.print("")

        table = Table(title="Cost Breakdown")
        table.add_column("Category", style="cyan")
        table.add_column("Low", justify="right")
        table.add_column("Expected", justify="right", style="green")
        table.add_column("High", justify="right")

        table.add_row(
            "API Calls",
            f"${cost_breakdown.api_cost.low:,.0f}",
            f"${cost_breakdown.api_cost.expected:,.0f}",
            f"${cost_breakdown.api_cost.high:,.0f}",
        )
        table.add_row(
            "Human Annotation",
            f"${cost_breakdown.human_annotation_cost.low:,.0f}",
            f"${cost_breakdown.human_annotation_cost.expected:,.0f}",
            f"${cost_breakdown.human_annotation_cost.high:,.0f}",
        )
        table.add_row(
            "Compute",
            f"${cost_breakdown.compute_cost.low:,.0f}",
            f"${cost_breakdown.compute_cost.expected:,.0f}",
            f"${cost_breakdown.compute_cost.high:,.0f}",
        )
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]${cost_breakdown.total.low:,.0f}[/bold]",
            f"[bold green]${cost_breakdown.total.expected:,.0f}[/bold green]",
            f"[bold]${cost_breakdown.total.high:,.0f}[/bold]",
        )

        console.print(table)

        if cost_breakdown.assumptions:
            console.print("\n[bold]Assumptions:[/bold]")
            for assumption in cost_breakdown.assumptions:
                console.print(f"  - {assumption}")


@main.command()
@click.argument("dataset_id")
@click.option("--sample-size", "-n", type=int, default=1000, help="Number of examples to sample")
@click.option("--text-field", "-f", default="text", help="Field containing text to analyze")
@click.option("--detect-ai", is_flag=True, help="Run AI content detection")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def quality(dataset_id: str, sample_size: int, text_field: str, detect_ai: bool, as_json: bool):
    """Analyze quality metrics for a dataset.

    DATASET_ID is the identifier of the dataset to analyze.
    """
    from datarecipe.quality_metrics import QualityAnalyzer

    quality_analyzer = QualityAnalyzer()

    with console.status(f"[cyan]Analyzing quality of {dataset_id}...[/cyan]"):
        try:
            report = quality_analyzer.analyze_from_huggingface(
                dataset_id,
                text_field=text_field,
                sample_size=sample_size,
                detect_ai=detect_ai,
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    if as_json:
        import json
        console.print(json.dumps(report.to_dict(), indent=2))
    else:
        console.print(f"\n[bold cyan]Quality Report for {dataset_id}[/bold cyan]")
        console.print(f"Sample size: {report.sample_size:,}")
        console.print("")

        # Overall score
        score = report.overall_score
        score_bar = "[" + "#" * int(score / 10) + "-" * (10 - int(score / 10)) + "]"
        console.print(f"[bold]Overall Score: {score:.0f}/100 {score_bar}[/bold]")
        console.print("")

        # Metrics tables
        table = Table(title="Diversity Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Unique Token Ratio", f"{report.diversity.unique_token_ratio:.4f}")
        table.add_row("Vocabulary Size", f"{report.diversity.vocabulary_size:,}")
        table.add_row("Semantic Diversity", f"{report.diversity.semantic_diversity:.4f}")
        console.print(table)
        console.print("")

        table = Table(title="Consistency Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Format Consistency", f"{report.consistency.format_consistency:.4f}")
        table.add_row("Structure Score", f"{report.consistency.structure_score:.4f}")
        table.add_row("Field Completeness", f"{report.consistency.field_completeness:.4f}")
        console.print(table)
        console.print("")

        table = Table(title="Complexity Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Avg Length", f"{report.complexity.avg_length:.0f} chars")
        table.add_row("Avg Tokens", f"{report.complexity.avg_tokens:.0f}")
        table.add_row("Vocabulary Richness", f"{report.complexity.vocabulary_richness:.4f}")
        table.add_row("Readability Score", f"{report.complexity.readability_score:.0f}")
        console.print(table)

        if detect_ai and report.ai_detection:
            console.print("")
            table = Table(title="AI Detection")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            table.add_row("AI Probability", f"{report.ai_detection.ai_probability:.2%}")
            table.add_row("Confidence", f"{report.ai_detection.confidence:.2%}")
            if report.ai_detection.indicators:
                table.add_row("Indicators", ", ".join(report.ai_detection.indicators[:3]))
            console.print(table)

        if report.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in report.recommendations:
                console.print(f"  - {rec}")

        if report.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in report.warnings:
                console.print(f"  - {warning}")


@main.command()
@click.argument("dataset_ids", nargs=-1)
@click.option("--file", "-f", type=click.Path(exists=True), help="File with dataset IDs")
@click.option("--parallel", "-p", type=int, default=4, help="Number of parallel workers")
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.option("--format", "fmt", type=click.Choice(["yaml", "json", "markdown"]), default="yaml", help="Output format")
def batch(dataset_ids: tuple, file: str, parallel: int, output: str, fmt: str):
    """Analyze multiple datasets in parallel.

    DATASET_IDS are the identifiers of datasets to analyze.
    Use -f to read dataset IDs from a file.
    """
    from datarecipe.batch_analyzer import BatchAnalyzer

    # Collect dataset IDs
    ids = list(dataset_ids)
    if file:
        batch_analyzer = BatchAnalyzer(max_workers=parallel)
        result = batch_analyzer.analyze_from_file(file)
    elif ids:
        batch_analyzer = BatchAnalyzer(max_workers=parallel)

        def progress_callback(dataset_id, completed, total):
            console.print(f"  [{completed}/{total}] Analyzed: {dataset_id}")

        batch_analyzer.progress_callback = progress_callback
        result = batch_analyzer.analyze_batch(ids)
    else:
        console.print("[red]Error:[/red] Provide dataset IDs or use -f to specify a file")
        sys.exit(1)

    console.print(f"\n[bold cyan]Batch Analysis Complete[/bold cyan]")
    console.print(f"  Total: {len(result.results)}")
    console.print(f"  [green]Successful: {result.successful}[/green]")
    console.print(f"  [red]Failed: {result.failed}[/red]")
    console.print(f"  Duration: {result.total_duration_seconds:.1f}s")

    if result.failed > 0:
        console.print("\n[yellow]Failed datasets:[/yellow]")
        for r in result.get_failed():
            console.print(f"  - {r.dataset_id}: {r.error}")

    if output:
        created = batch_analyzer.export_results(result, output, fmt)
        console.print(f"\n[green]Results exported to {output}[/green]")
        console.print(f"  Created {len(created)} files")


@main.command()
@click.argument("dataset_ids", nargs=-1, required=True)
@click.option("--format", "fmt", type=click.Choice(["table", "markdown"]), default="table", help="Output format")
@click.option("--include-quality", is_flag=True, help="Include quality analysis (slower)")
@click.option("--output", "-o", type=click.Path(), help="Output file")
def compare(dataset_ids: tuple, fmt: str, include_quality: bool, output: str):
    """Compare multiple datasets side by side.

    DATASET_IDS are 2 or more dataset identifiers to compare.
    """
    from datarecipe.comparator import DatasetComparator

    if len(dataset_ids) < 2:
        console.print("[red]Error:[/red] Please provide at least 2 datasets to compare")
        sys.exit(1)

    comparator = DatasetComparator(include_quality=include_quality)

    with console.status(f"[cyan]Comparing {len(dataset_ids)} datasets...[/cyan]"):
        try:
            report = comparator.compare_by_ids(list(dataset_ids))
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    if fmt == "markdown":
        content = report.to_markdown()
    else:
        content = report.to_table()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        console.print(f"[green]Report saved to {output}[/green]")
    else:
        print(content)

    # Show recommendations
    if report.recommendations and fmt == "table":
        console.print("\n[bold cyan]Recommendations:[/bold cyan]")
        for rec in report.recommendations:
            console.print(f"  - {rec}")


@main.command()
@click.argument("dataset_id")
@click.option("--output", "-o", type=click.Path(), help="Output file for profile")
@click.option("--region", "-r", default="china", help="Region for cost estimation (china, us, europe, india, sea)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--markdown", "--md", "as_markdown", is_flag=True, help="Output as Markdown")
def profile(dataset_id: str, output: str, region: str, as_json: bool, as_markdown: bool):
    """Generate annotator profile for a dataset.

    Analyzes a dataset and generates requirements for annotation team,
    including skills, experience level, education, and workload estimation.

    DATASET_ID is the identifier of the dataset to analyze.
    """
    from datarecipe.profiler import AnnotatorProfiler, profile_to_markdown

    analyzer = DatasetAnalyzer()
    profiler = AnnotatorProfiler()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        try:
            recipe = analyzer.analyze(dataset_id)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    with console.status("[cyan]Generating annotator profile...[/cyan]"):
        annotator_profile = profiler.generate_profile(recipe, region=region)

    if as_json:
        import json
        console.print(json.dumps(annotator_profile.to_dict(), indent=2))
    elif as_markdown:
        md_content = profile_to_markdown(annotator_profile, recipe.name)
        print(md_content)
    else:
        # Display as formatted table
        console.print(f"\n[bold cyan]Annotator Profile for {dataset_id}[/bold cyan]")
        console.print("")

        # Skills table
        table = Table(title="Required Skills")
        table.add_column("Skill", style="cyan")
        table.add_column("Level", justify="center")
        table.add_column("Priority", justify="center")

        for skill in annotator_profile.skill_requirements:
            priority = "required" if skill.required else "preferred"
            priority_color = {"required": "red", "preferred": "yellow"}.get(priority, "white")
            table.add_row(
                skill.name,
                skill.level,
                f"[{priority_color}]{priority}[/{priority_color}]"
            )
        console.print(table)
        console.print("")

        # Requirements summary
        console.print("[bold]Requirements:[/bold]")
        console.print(f"  Experience Level: {annotator_profile.experience_level.value}")
        console.print(f"  Education: {annotator_profile.education_level.value}")
        if annotator_profile.domain_knowledge:
            console.print(f"  Domain Expertise: {', '.join(annotator_profile.domain_knowledge)}")
        if annotator_profile.language_requirements:
            console.print(f"  Languages: {', '.join(annotator_profile.language_requirements)}")
        console.print("")

        # Workload estimation
        hourly_rate = (annotator_profile.hourly_rate_range.get("min", 15) + annotator_profile.hourly_rate_range.get("max", 45)) / 2
        estimated_labor_cost = annotator_profile.estimated_person_days * 8 * hourly_rate
        console.print("[bold]Workload Estimation:[/bold]")
        console.print(f"  Team Size: {annotator_profile.team_size} annotators")
        console.print(f"  Person-Days: {annotator_profile.estimated_person_days:.0f}")
        console.print(f"  Hours per Example: {annotator_profile.estimated_hours_per_example:.2f}")
        console.print(f"  Hourly Rate: ${hourly_rate:.2f}")
        console.print(f"  Estimated Labor Cost: ${estimated_labor_cost:,.0f}")

    # Export if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output.endswith(".md"):
            md_content = profile_to_markdown(annotator_profile, recipe.name)
            output_path.write_text(md_content, encoding="utf-8")
            console.print(f"\n[green]Profile exported to:[/green] {output}")
        elif output.endswith(".json"):
            import json
            output_path.write_text(json.dumps(annotator_profile.to_dict(), indent=2), encoding="utf-8")
            console.print(f"\n[green]Profile exported to:[/green] {output}")
        else:
            # Default to YAML
            import yaml
            output_path.write_text(yaml.dump(annotator_profile.to_dict(), allow_unicode=True, default_flow_style=False), encoding="utf-8")
            console.print(f"\n[green]Profile exported to:[/green] {output}")


@main.command()
@click.argument("dataset_id")
@click.option("--output", "-o", type=click.Path(), help="Output directory (default: ./projects/<dataset_name>)")
@click.option("--provider", "-p", default="local", help="Deployment provider (local, judgeguild, etc.)")
@click.option("--region", "-r", default="china", help="Region for cost estimation")
@click.option("--submit", is_flag=True, help="Submit to provider after generating config")
def deploy(dataset_id: str, output: str, provider: str, region: str, submit: bool):
    """Generate production deployment for a dataset.

    Creates a complete project structure with annotation guidelines,
    quality rules, acceptance criteria, and timeline for data production.

    DATASET_ID is the identifier of the dataset to analyze.

    If --output is not specified, files are saved to ./projects/<dataset_name>/
    """
    from datarecipe.deployer import ProductionDeployer
    from datarecipe.profiler import AnnotatorProfiler
    from datarecipe.schema import DataRecipe

    # 默认输出目录
    if not output:
        safe_name = dataset_id.replace("/", "_").replace(" ", "_").lower()
        output = f"./projects/{safe_name}"
        console.print(f"[dim]Output directory: {output}[/dim]")

    analyzer = DatasetAnalyzer()
    deployer = ProductionDeployer()
    profiler = AnnotatorProfiler()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        try:
            recipe = analyzer.analyze(dataset_id)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    with console.status("[cyan]Generating annotator profile...[/cyan]"):
        profile = profiler.generate_profile(recipe, region=region)

    # Convert Recipe to DataRecipe
    data_recipe = DataRecipe(
        name=recipe.name,
        version=recipe.version,
        source_type=recipe.source_type,
        source_id=recipe.source_id,
        num_examples=recipe.num_examples,
        languages=recipe.languages or [],
        license=recipe.license,
        description=recipe.description,
        generation_type=recipe.generation_type,
        synthetic_ratio=recipe.synthetic_ratio,
        human_ratio=recipe.human_ratio,
        generation_methods=recipe.generation_methods or [],
        teacher_models=recipe.teacher_models or [],
        tags=recipe.tags or [],
    )

    with console.status("[cyan]Generating production config...[/cyan]"):
        config = deployer.generate_config(data_recipe, profile=profile)

    # Deploy to provider
    with console.status(f"[cyan]Deploying to {provider}...[/cyan]"):
        result = deployer.deploy(data_recipe, output, provider=provider, config=config, profile=profile)

    if result.success:
        console.print(f"\n[bold green]Deployment successful![/bold green]")
        if result.project_handle:
            console.print(f"  Project ID: {result.project_handle.project_id}")
        console.print(f"  Output: {output}")
        if result.details:
            console.print(f"  Details: {result.details}")

        # Show created files
        output_path = Path(output)
        if output_path.exists():
            files = list(output_path.rglob("*"))
            files = [f for f in files if f.is_file()]
            console.print(f"\n[bold]Created files ({len(files)}):[/bold]")
            for f in files[:10]:
                console.print(f"  - {f.relative_to(output_path)}")
            if len(files) > 10:
                console.print(f"  ... and {len(files) - 10} more")

        console.print(f"\n[bold cyan]Next steps:[/bold cyan]")
        console.print(f"  1. cd {output}")
        console.print(f"  2. Review annotation_guide.md")
        console.print(f"  3. Review quality_rules.yaml")
        console.print(f"  4. See README.md for detailed instructions")
    else:
        console.print(f"\n[red]Deployment failed:[/red] {result.error}")
        sys.exit(1)


@main.group()
def providers():
    """Manage deployment providers."""
    pass


@providers.command("list")
def providers_list():
    """List available deployment providers."""
    from datarecipe.providers import list_providers

    provider_list = list_providers()

    table = Table(title="Available Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for p in provider_list:
        table.add_row(p["name"], p["description"])

    console.print(table)

    console.print("\n[dim]Install additional providers with: pip install datarecipe-<provider>[/dim]")


@main.command()
@click.argument("dataset_id")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory for project")
@click.option("--target-size", "-n", type=int, help="Target number of examples")
@click.option("--format", "fmt", type=click.Choice(["huggingface", "jsonl", "parquet"]), default="huggingface", help="Output format")
def workflow(dataset_id: str, output: str, target_size: int, fmt: str):
    """Generate a production workflow for reproducing a dataset.

    Creates a complete project structure with scripts, configuration,
    and documentation for producing a dataset similar to DATASET_ID.
    """
    from datarecipe.workflow import WorkflowGenerator

    analyzer = DatasetAnalyzer()
    generator = WorkflowGenerator()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        try:
            recipe = analyzer.analyze(dataset_id)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    with console.status("[cyan]Generating workflow...[/cyan]"):
        wf = generator.generate(recipe, target_size, fmt)

    # Export project
    created_files = wf.export_project(output)

    console.print(f"\n[bold green]Workflow generated successfully![/bold green]")
    console.print(f"  Project: {output}")
    console.print(f"  Target size: {wf.target_size:,} examples")
    console.print(f"  Estimated cost: ${wf.estimated_total_cost:,.0f}")
    console.print(f"  Steps: {len(wf.steps)}")

    console.print(f"\n[bold]Created files ({len(created_files)}):[/bold]")
    for f in created_files[:10]:
        console.print(f"  - {f}")
    if len(created_files) > 10:
        console.print(f"  ... and {len(created_files) - 10} more")

    console.print(f"\n[bold cyan]Next steps:[/bold cyan]")
    console.print(f"  1. cd {output}")
    console.print(f"  2. pip install -r requirements.txt")
    console.print(f"  3. cp .env.example .env && edit .env")
    console.print(f"  4. See README.md for detailed instructions")


if __name__ == "__main__":
    main()
