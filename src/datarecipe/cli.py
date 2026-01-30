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
    """Generate a beautiful Markdown document for a recipe."""
    lines = []

    # Title
    lines.append(f"# 📊 Dataset Recipe: {recipe.name}")
    lines.append("")

    # Summary box
    lines.append("> **DataRecipe Analysis Report**")
    lines.append(f"> ")
    lines.append(f"> Analyzing how this dataset was built - its ingredients, methods, and reproducibility.")
    lines.append("")

    # Basic Info
    lines.append("## 📋 Basic Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| **Dataset** | `{recipe.name}` |")
    lines.append(f"| **Source** | {recipe.source_type.value.title()} |")
    if recipe.license:
        lines.append(f"| **License** | {recipe.license} |")
    if recipe.languages:
        langs = [l for l in recipe.languages if l]
        if langs:
            lines.append(f"| **Languages** | {', '.join(langs)} |")
    if recipe.num_examples:
        lines.append(f"| **Examples** | {recipe.num_examples:,} |")
    lines.append("")

    # Generation Method
    lines.append("## 🧬 Generation Method")
    lines.append("")

    if recipe.synthetic_ratio is not None or recipe.human_ratio is not None:
        synthetic_pct = (recipe.synthetic_ratio or 0) * 100
        human_pct = (recipe.human_ratio or 0) * 100

        # Progress bar visualization
        synthetic_bar = "█" * int(synthetic_pct / 5) + "░" * (20 - int(synthetic_pct / 5))
        human_bar = "█" * int(human_pct / 5) + "░" * (20 - int(human_pct / 5))

        lines.append("| Type | Ratio | Distribution |")
        lines.append("|------|-------|--------------|")
        lines.append(f"| 🤖 Synthetic | {synthetic_pct:.0f}% | `{synthetic_bar}` |")
        lines.append(f"| 👤 Human | {human_pct:.0f}% | `{human_bar}` |")
    else:
        lines.append("*Generation method could not be determined from available metadata.*")
    lines.append("")

    # Teacher Models
    lines.append("## 🎓 Teacher Models")
    lines.append("")

    if recipe.teacher_models:
        lines.append("The following AI models were detected as potential teachers for data generation:")
        lines.append("")
        for model in recipe.teacher_models:
            lines.append(f"- **{model}**")
    else:
        lines.append("*No teacher models detected in the dataset documentation.*")
    lines.append("")

    # Generation Methods Detail
    if recipe.generation_methods:
        lines.append("### Generation Pipeline")
        lines.append("")
        for i, method in enumerate(recipe.generation_methods, 1):
            lines.append(f"**Step {i}: {method.method_type.replace('_', ' ').title()}**")
            if method.teacher_model:
                lines.append(f"- Teacher Model: `{method.teacher_model}`")
            if method.platform:
                lines.append(f"- Platform: {method.platform}")
            if method.prompt_template_available:
                lines.append(f"- Prompt Template: ✅ Available")
            lines.append("")

    # Cost Estimation
    lines.append("## 💰 Cost Estimation")
    lines.append("")

    if recipe.cost and recipe.cost.estimated_total_usd:
        if recipe.cost.confidence == "low":
            low = recipe.cost.estimated_total_usd * 0.5
            high = recipe.cost.estimated_total_usd * 1.5
            lines.append(f"**Estimated Total: ${low:,.0f} - ${high:,.0f}** *(low confidence)*")
        elif recipe.cost.confidence == "medium":
            low = recipe.cost.estimated_total_usd * 0.8
            high = recipe.cost.estimated_total_usd * 1.2
            lines.append(f"**Estimated Total: ${low:,.0f} - ${high:,.0f}** *(medium confidence)*")
        else:
            lines.append(f"**Estimated Total: ${recipe.cost.estimated_total_usd:,.0f}**")
        lines.append("")

        lines.append("| Category | Cost |")
        lines.append("|----------|------|")
        if recipe.cost.api_calls_usd:
            lines.append(f"| API Calls | ${recipe.cost.api_calls_usd:,.0f} |")
        if recipe.cost.human_annotation_usd:
            lines.append(f"| Human Annotation | ${recipe.cost.human_annotation_usd:,.0f} |")
        if recipe.cost.compute_usd:
            lines.append(f"| Compute | ${recipe.cost.compute_usd:,.0f} |")
    else:
        lines.append("*Cost estimation not available.*")
    lines.append("")

    # Reproducibility
    lines.append("## 🔄 Reproducibility Assessment")
    lines.append("")

    if recipe.reproducibility:
        score = recipe.reproducibility.score
        score_bar = "🟩" * score + "⬜" * (10 - score)
        lines.append(f"### Score: {score}/10")
        lines.append("")
        lines.append(f"`{score_bar}`")
        lines.append("")

        if recipe.reproducibility.available:
            lines.append("#### ✅ Available Information")
            lines.append("")
            for item in recipe.reproducibility.available:
                lines.append(f"- {item.replace('_', ' ').title()}")
            lines.append("")

        if recipe.reproducibility.missing:
            lines.append("#### ❌ Missing Information")
            lines.append("")
            for item in recipe.reproducibility.missing:
                lines.append(f"- {item.replace('_', ' ').title()}")
            lines.append("")

        if recipe.reproducibility.notes:
            lines.append("#### 📝 Notes")
            lines.append("")
            lines.append(recipe.reproducibility.notes)
            lines.append("")
    else:
        lines.append("*Reproducibility not assessed.*")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by [DataRecipe](https://github.com/yourusername/data-recipe) - AI Dataset Ingredients Analyzer*")

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
