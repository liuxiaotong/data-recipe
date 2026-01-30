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

        # Progress bar visualization
        synthetic_bar = "█" * int(synthetic_pct / 5) + "░" * (20 - int(synthetic_pct / 5))
        human_bar = "█" * int(human_pct / 5) + "░" * (20 - int(human_pct / 5))

        lines.append("| 类型 | 占比 | 分布 |")
        lines.append("|------|------|------|")
        lines.append(f"| 🤖 合成数据 | {synthetic_pct:.0f}% | `{synthetic_bar}` |")
        lines.append(f"| 👤 人工标注 | {human_pct:.0f}% | `{human_bar}` |")
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
        score_bar = "🟩" * score + "⬜" * (10 - score)
        lines.append(f"### 评分：{score}/10")
        lines.append("")
        lines.append(f"`{score_bar}`")
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
@click.version_option(version="0.1.0", prog_name="datarecipe")
def main():
    """DataRecipe - Analyze AI dataset ingredients."""
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
    table.add_column("Example ID")

    table.add_row("HuggingFace Hub", "✓ Supported", "org/dataset-name")
    table.add_row("OpenAI", "Coming soon", "-")
    table.add_row("Local files", "Coming soon", "-")

    console.print(table)


if __name__ == "__main__":
    main()
