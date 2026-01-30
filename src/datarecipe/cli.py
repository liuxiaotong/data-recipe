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
@click.option("--output", "-o", type=click.Path(), help="Export recipe to YAML file")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--yaml", "as_yaml", is_flag=True, help="Output as YAML")
def analyze(dataset_id: str, output: str, as_json: bool, as_yaml: bool):
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
    else:
        display_recipe(recipe)

    # Export if requested
    if output:
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
