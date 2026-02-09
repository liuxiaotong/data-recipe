"""Basic analysis commands."""

import sys
from pathlib import Path

import click
from rich.table import Table

from datarecipe.analyzer import DatasetAnalyzer
from datarecipe.cli._helpers import (
    console,
    display_recipe,
    recipe_to_markdown,
)


@click.command("analyze")
@click.argument("dataset_id")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Export recipe to file (auto-detect format by extension)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--yaml", "as_yaml", is_flag=True, help="Output as YAML")
@click.option("--markdown", "--md", "as_markdown", is_flag=True, help="Output as Markdown")
def analyze(dataset_id: str, output: str, as_json: bool, as_yaml: bool, as_markdown: bool):
    """Analyze a dataset and display its recipe.

    DATASET_ID is the identifier of the dataset to analyze.
    Supports HuggingFace dataset IDs and local files (CSV, Parquet, JSONL).

    Examples:
        datarecipe analyze org/dataset-name
        datarecipe analyze ./data/train.csv
        datarecipe analyze ./data/train.parquet
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
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
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


@click.command()
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


@click.command()
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
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            sys.exit(1)

    analyzer.export_recipe(recipe, output_file)
    console.print(f"[green]Recipe exported to:[/green] {output_file}")


@click.command("list-sources")
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


@click.command()
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
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            sys.exit(1)

    # Get appropriate pipeline template
    pipeline = get_pipeline_template(
        recipe.generation_type.value if recipe.generation_type else "unknown",
        recipe.synthetic_ratio,
    )

    # Customize pipeline with dataset info
    if target_size:
        pipeline.target_size = target_size

    if recipe.cost and recipe.cost.estimated_total_usd:
        pipeline.estimated_total_cost = recipe.cost.estimated_total_usd

    # Generate guide
    guide_content = pipeline_to_markdown(pipeline, recipe.name)

    # Add dataset-specific info at the top
    synthetic_pct = (
        f"{recipe.synthetic_ratio * 100:.0f}%" if recipe.synthetic_ratio is not None else "N/A"
    )
    human_pct = f"{recipe.human_ratio * 100:.0f}%" if recipe.human_ratio is not None else "N/A"
    repro_score = f"{recipe.reproducibility.score}/10" if recipe.reproducibility else "N/A"

    header = f"""# 数据生产指南：{recipe.name}

## 参考数据集分析

| 属性 | 值 |
|------|-----|
| **数据集名称** | {recipe.name} |
| **来源** | {recipe.source_type.value} |
| **合成数据比例** | {synthetic_pct} |
| **人工数据比例** | {human_pct} |
| **教师模型** | {", ".join(recipe.teacher_models) if recipe.teacher_models else "无"} |
| **可复现性评分** | {repro_score} |

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


@click.command("deep-guide")
@click.argument("url")
@click.option("--output", "-o", type=click.Path(), help="Output file path for production guide")
@click.option(
    "--llm/--no-llm", default=False, help="Use LLM for enhanced analysis (requires API key)"
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    help="LLM provider",
)
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
    from datarecipe.analyzers.url_analyzer import deep_analysis_to_markdown

    # Try to use LLMAnalyzer with PDF parsing (even without LLM)
    try:
        from datarecipe.analyzers.llm_url_analyzer import LLMAnalyzer

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
        from datarecipe.analyzers.url_analyzer import DeepAnalyzer

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
    if hasattr(result, "paper_url") and result.paper_url:
        console.print(f"  [green]自动发现论文:[/green] {result.paper_url}")
