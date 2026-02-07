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


def validate_output_path(output: str, base_dir: Path = None) -> Path:
    """Validate and resolve output path to prevent path traversal attacks.

    Args:
        output: User-provided output path
        base_dir: Optional base directory to restrict outputs to

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is invalid or attempts traversal outside base_dir
    """
    output_path = Path(output).resolve()

    # If base_dir specified, ensure output is within it
    if base_dir:
        base_resolved = base_dir.resolve()
        try:
            output_path.relative_to(base_resolved)
        except ValueError:
            raise ValueError(
                f"Output path '{output}' is outside allowed directory '{base_dir}'"
            )

    # Block obviously dangerous paths
    dangerous_patterns = ["/etc/", "/usr/", "/bin/", "/var/", "/root/"]
    output_str = str(output_path)
    for pattern in dangerous_patterns:
        if output_str.startswith(pattern):
            raise ValueError(f"Output path '{output}' is in a protected system directory")

    return output_path


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
    lines.append("> 由 [DataRecipe](https://github.com/yourusername/data-recipe) 生成 — AI 数据集成分分析器")

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
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
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
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
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
    synthetic_pct = (
        f"{recipe.synthetic_ratio * 100:.0f}%"
        if recipe.synthetic_ratio is not None
        else "N/A"
    )
    human_pct = (
        f"{recipe.human_ratio * 100:.0f}%"
        if recipe.human_ratio is not None
        else "N/A"
    )
    repro_score = (
        f"{recipe.reproducibility.score}/10"
        if recipe.reproducibility
        else "N/A"
    )

    header = f"""# 数据生产指南：{recipe.name}

## 参考数据集分析

| 属性 | 值 |
|------|-----|
| **数据集名称** | {recipe.name} |
| **来源** | {recipe.source_type.value} |
| **合成数据比例** | {synthetic_pct} |
| **人工数据比例** | {human_pct} |
| **教师模型** | {', '.join(recipe.teacher_models) if recipe.teacher_models else '无'} |
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
    submit_action = submit or provider == "local"
    status_msg = (
        f"[cyan]Deploying to {provider}...[/cyan]"
        if submit_action
        else f"[cyan]Generating deployment package for {provider} (no auto submission)...[/cyan]"
    )
    with console.status(status_msg):
        result = deployer.deploy(
            data_recipe,
            output,
            provider=provider,
            config=config,
            profile=profile,
            submit=submit,
        )

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
        if provider != "local" and not submit:
            console.print(
                "  5. 使用 provider 平台手动提交项目 (本次未自动提交，需确认配置后再执行)"
            )
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


# =============================================================================
# New Commands: Pattern Extraction & Generation
# =============================================================================

@main.command("extract-rubrics")
@click.argument("dataset_id")
@click.option("--output", "-o", default=None, help="Output file path (JSON)")
@click.option("--sample-size", "-n", default=1000, help="Number of samples to analyze")
def extract_rubrics(dataset_id: str, output: str, sample_size: int):
    """Extract rubrics/evaluation patterns from a dataset."""
    from datarecipe.extractors import RubricsAnalyzer

    console.print(f"\n[bold]Extracting rubrics patterns from {dataset_id}...[/bold]\n")

    try:
        # Load dataset
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train", streaming=True)

        # Collect rubrics
        rubrics = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            # Try common rubrics field names
            for field in ["rubrics", "rubric", "criteria", "evaluation"]:
                if field in item:
                    value = item[field]
                    if isinstance(value, list):
                        rubrics.extend(value)
                    elif isinstance(value, str):
                        rubrics.append(value)

        if not rubrics:
            console.print("[yellow]No rubrics found in dataset.[/yellow]")
            console.print("Tried fields: rubrics, rubric, criteria, evaluation")
            return

        # Analyze
        analyzer = RubricsAnalyzer()
        result = analyzer.analyze(rubrics, task_count=sample_size)

        # Display summary
        console.print(Panel(result.summary(), title="Rubrics Analysis"))
        console.print("\n[bold]Top Structured Templates:[/bold]")
        for entry in result.structured_templates[:5]:
            console.print(
                f"• [{entry.get('category', 'general')}] {entry.get('action') or ''} → {entry.get('target') or ''}" +
                (f" | 条件: {entry.get('condition')}" if entry.get('condition') else "")
            )

        # Export if requested
        if output:
            import json
            base = output
            if output.endswith(".json"):
                data_path = output
                yaml_path = output.replace(".json", "_templates.yaml")
                md_path = output.replace(".json", "_templates.md")
            else:
                data_path = f"{output}.json"
                yaml_path = f"{output}_templates.yaml"
                md_path = f"{output}_templates.md"

            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(analyzer.to_dict(result), f, indent=2, ensure_ascii=False)
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(analyzer.to_yaml_templates(result))
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(analyzer.to_markdown_templates(result))

            console.print(f"\n[green]Exported analysis to {data_path}[/green]")
            console.print(f"[green]Exported templates to {yaml_path} & {md_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command("extract-prompts")
@click.argument("dataset_id")
@click.option("--output", "-o", default=None, help="Output file path (JSON)")
@click.option("--sample-size", "-n", default=500, help="Number of samples to analyze")
def extract_prompts(dataset_id: str, output: str, sample_size: int):
    """Extract system prompt templates from a dataset."""
    from datarecipe.extractors import PromptExtractor

    console.print(f"\n[bold]Extracting prompt templates from {dataset_id}...[/bold]\n")

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train", streaming=True)

        # Collect messages with progress
        messages = []
        console.print(f"[dim]Collecting messages from {sample_size} samples...[/dim]")
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            if i > 0 and i % 100 == 0:
                console.print(f"[dim]  Processed {i}/{sample_size} samples ({len(messages)} messages)[/dim]")
            # Try common message field names
            for field in ["messages", "conversation", "turns"]:
                if field in item and isinstance(item[field], list):
                    messages.extend(item[field])

        if not messages:
            console.print("[yellow]No messages found in dataset.[/yellow]")
            return

        console.print(f"[dim]Collected {len(messages)} messages, deduplicating...[/dim]")

        # Extract
        extractor = PromptExtractor()
        library = extractor.extract(messages)
        console.print(f"[green]✓ Deduplication complete[/green]")

        # Display summary
        console.print(Panel(library.summary(), title="Prompt Library"))

        # Export if output specified
        if output:
            import json
            data = extractor.to_dict(library)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[green]Exported to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command("detect-strategy")
@click.argument("dataset_id")
@click.option("--output", "-o", default=None, help="Output file path (JSON)")
@click.option("--sample-size", "-n", default=100, help="Number of samples to analyze")
def detect_strategy(dataset_id: str, output: str, sample_size: int):
    """Detect context construction strategy in a dataset."""
    from datarecipe.analyzers import ContextStrategyDetector

    console.print(f"\n[bold]Detecting context strategy in {dataset_id}...[/bold]\n")

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split="train", streaming=True)

        # Collect contexts
        contexts = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            # Try common context field names
            for field in ["context", "input", "text", "content", "document"]:
                if field in item and isinstance(item[field], str):
                    contexts.append(item[field])
                    break
            # Also check messages
            if "messages" in item and isinstance(item["messages"], list):
                for msg in item["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        contexts.append(msg.get("content", ""))

        if not contexts:
            console.print("[yellow]No contexts found in dataset.[/yellow]")
            return

        # Detect
        detector = ContextStrategyDetector()
        result = detector.analyze(contexts)

        # Display summary
        console.print(Panel(result.summary(), title="Context Strategy"))

        # Export if output specified
        if output:
            import json
            data = detector.to_dict(result)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[green]Exported to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command("allocate")
@click.option("--size", "-s", default=10000, help="Target dataset size")
@click.option("--region", "-r", default="china", help="Region for cost calculation")
@click.option("--output", "-o", default=None, help="Output file path (JSON/Markdown)")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "markdown"]), default="table")
def allocate(size: int, region: str, output: str, fmt: str):
    """Generate human-machine task allocation."""
    from datarecipe.generators import HumanMachineSplitter, TaskType

    console.print(f"\n[bold]Generating human-machine allocation...[/bold]")
    console.print(f"Target size: {size:,} | Region: {region}\n")

    splitter = HumanMachineSplitter(region=region)
    result = splitter.analyze(
        dataset_size=size,
        task_types=[
            TaskType.CONTEXT_CREATION,
            TaskType.TASK_DESIGN,
            TaskType.RUBRICS_WRITING,
            TaskType.DATA_GENERATION,
            TaskType.QUALITY_REVIEW,
        ]
    )

    if fmt == "table":
        console.print(Panel(result.summary(), title="Allocation Summary"))
        console.print("\n" + result.to_markdown_table())
    elif fmt == "markdown":
        console.print(result.summary())
        console.print("\n" + result.to_markdown_table())
    else:
        import json
        data = splitter.to_dict(result)
        console.print(json.dumps(data, indent=2))

    if output:
        import json
        with open(output, "w", encoding="utf-8") as f:
            if output.endswith(".json"):
                json.dump(splitter.to_dict(result), f, indent=2, ensure_ascii=False)
            else:
                f.write(result.summary() + "\n\n" + result.to_markdown_table())
        console.print(f"\n[green]Exported to {output}[/green]")


@main.command("enhanced-guide")
@click.argument("dataset_id")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--size", "-s", default=10000, help="Target dataset size")
@click.option("--region", "-r", default="china", help="Region for cost calculation")
def enhanced_guide(dataset_id: str, output: str, size: int, region: str):
    """Generate enhanced production guide with patterns and allocation."""
    from datarecipe.generators import EnhancedGuideGenerator, HumanMachineSplitter, TaskType
    from datarecipe.extractors import RubricsAnalyzer, PromptExtractor
    from datarecipe.analyzers import ContextStrategyDetector

    console.print(f"\n[bold]Generating enhanced guide for {dataset_id}...[/bold]\n")

    try:
        # Try to load and analyze the dataset
        rubrics_result = None
        prompt_library = None
        strategy_result = None

        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split="train", streaming=True)

            rubrics = []
            messages = []
            contexts = []

            for i, item in enumerate(ds):
                if i >= 500:
                    break
                # Collect rubrics
                for field in ["rubrics", "rubric", "criteria"]:
                    if field in item:
                        value = item[field]
                        if isinstance(value, list):
                            rubrics.extend(value)
                        elif isinstance(value, str):
                            rubrics.append(value)
                # Collect messages
                if "messages" in item and isinstance(item["messages"], list):
                    messages.extend(item["messages"])
                # Collect contexts
                for field in ["context", "input", "text"]:
                    if field in item and isinstance(item[field], str):
                        contexts.append(item[field])
                        break

            if rubrics:
                analyzer = RubricsAnalyzer()
                rubrics_result = analyzer.analyze(rubrics)
                console.print(f"[green]✓ Analyzed {len(rubrics)} rubrics[/green]")

            if messages:
                console.print(f"[dim]  Deduplicating {len(messages)} messages...[/dim]")
                extractor = PromptExtractor()
                prompt_library = extractor.extract(messages)
                console.print(f"[green]✓ Extracted {prompt_library.unique_count} unique prompts[/green]")

            if contexts:
                detector = ContextStrategyDetector()
                strategy_result = detector.analyze(contexts[:100])
                console.print(f"[green]✓ Detected strategy: {strategy_result.primary_strategy.value}[/green]")

        except Exception as e:
            console.print(f"[yellow]Could not analyze dataset: {e}[/yellow]")

        # Generate allocation
        splitter = HumanMachineSplitter(region=region)
        allocation = splitter.analyze(
            dataset_size=size,
            task_types=[
                TaskType.CONTEXT_CREATION,
                TaskType.TASK_DESIGN,
                TaskType.RUBRICS_WRITING,
                TaskType.QUALITY_REVIEW,
            ]
        )

        # Generate guide
        generator = EnhancedGuideGenerator()
        guide = generator.generate(
            dataset_name=dataset_id,
            target_size=size,
            rubrics_analysis=rubrics_result,
            prompt_library=prompt_library,
            context_strategy=strategy_result,
            allocation=allocation,
            region=region,
        )

        # Output
        markdown = generator.to_markdown(guide)

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(markdown)
            console.print(f"\n[green]Guide saved to {output}[/green]")
        else:
            console.print("\n" + markdown)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


@main.command("generate")
@click.option("--type", "gen_type", type=click.Choice(["rubrics", "prompts", "contexts"]), default="rubrics")
@click.option("--count", "-n", default=10, help="Number of items to generate")
@click.option("--context", "-c", default="the topic", help="Context/topic for generation")
@click.option("--output", "-o", default=None, help="Output file path (JSONL)")
def generate(gen_type: str, count: int, context: str, output: str):
    """Generate data based on patterns."""
    from datarecipe.generators import PatternGenerator

    console.print(f"\n[bold]Generating {count} {gen_type}...[/bold]\n")

    generator = PatternGenerator()

    if gen_type == "rubrics":
        result = generator.generate_rubrics(context=context, count=count)
    elif gen_type == "prompts":
        result = generator.generate_prompts(domain=context, count=count)
    elif gen_type == "contexts":
        result = generator.generate_contexts(count=count)
    else:
        console.print(f"[red]Unknown type: {gen_type}[/red]")
        return

    # Display
    console.print(Panel(result.summary(), title="Generation Result"))
    console.print("")

    for item in result.items[:5]:
        console.print(f"[cyan]{item.data_type}[/cyan]: {item.content[:100]}...")
        console.print("")

    if len(result.items) > 5:
        console.print(f"... and {len(result.items) - 5} more")

    # Export
    if output:
        generator.export_jsonl(result, output)
        console.print(f"\n[green]Exported to {output}[/green]")


@main.command("deep-analyze")
@click.argument("dataset_id")
@click.option("--output-dir", "-o", default="./analysis_output", help="Output directory")
@click.option("--sample-size", "-n", default=500, help="Number of samples to analyze")
@click.option("--size", "-s", default=None, type=int, help="Target dataset size (for cost estimation)")
@click.option("--region", "-r", default="china", help="Region for cost calculation")
@click.option("--split", default=None, help="Dataset split (auto-detect if not specified)")
@click.option("--use-llm", is_flag=True, default=False, help="Use LLM for intelligent analysis of unknown dataset types")
@click.option("--llm-provider", default="anthropic", type=click.Choice(["anthropic", "openai"]), help="LLM provider for intelligent analysis")
@click.option("--enhance-mode", default="auto", type=click.Choice(["auto", "interactive", "api"]), help="LLM enhancement mode: auto (detect), interactive (Claude Code/App), api (standalone)")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis, ignore cache")
@click.option("--no-cache", is_flag=True, help="Don't use or update cache")
def deep_analyze(dataset_id: str, output_dir: str, sample_size: int, size: int, region: str, split: str, use_llm: bool, llm_provider: str, enhance_mode: str, force: bool, no_cache: bool):
    """
    Run comprehensive deep analysis on a dataset.

    Generates both JSON data files and a human-readable Markdown report.

    Example:
        datarecipe deep-analyze tencent/CL-bench -o ./output
    """
    import os
    from datarecipe.cache import AnalysisCache
    from datarecipe.core.deep_analyzer import DeepAnalyzerCore

    # Create output directory with dataset subdirectory
    safe_dataset_name = dataset_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    dataset_output_dir = os.path.join(output_dir, safe_dataset_name)

    # Check cache first (unless --force or --no-cache)
    cache = AnalysisCache() if not no_cache else None
    if cache and not force:
        cached = cache.get(dataset_id, check_freshness=True)
        if cached:
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]  DataRecipe 深度逆向分析 (缓存命中)[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
            console.print(f"数据集: [bold]{dataset_id}[/bold]")
            console.print(f"[green]✓ 使用缓存结果 (创建于 {cached.created_at[:10]})[/green]")
            console.print(f"  类型: {cached.dataset_type or 'unknown'}")
            console.print(f"  样本: {cached.sample_count}")

            if cached.output_dir != dataset_output_dir:
                os.makedirs(dataset_output_dir, exist_ok=True)
                cache.copy_to_output(dataset_id, dataset_output_dir)
                console.print(f"  输出: {dataset_output_dir}")
            else:
                console.print(f"  输出: {cached.output_dir}")

            console.print(f"\n[dim]使用 --force 强制重新分析[/dim]")
            return

    # Display header
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]  DataRecipe 深度逆向分析[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    console.print(f"数据集: [bold]{dataset_id}[/bold]")
    console.print(f"输出目录: [bold]{dataset_output_dir}[/bold]\n")

    try:
        # Use shared DeepAnalyzerCore
        analyzer = DeepAnalyzerCore(
            output_dir=output_dir,
            region=region,
            use_llm=use_llm,
            llm_provider=llm_provider,
            enhance_mode=enhance_mode,
        )

        console.print("[dim]📥 加载数据集...[/dim]")
        result = analyzer.analyze(
            dataset_id=dataset_id,
            sample_size=sample_size,
            split=split,
            target_size=size,
        )

        if not result.success:
            console.print(f"[red]错误: {result.error}[/red]")
            return

        console.print(f"[green]✓ 加载完成: {result.sample_count} 样本[/green]")

        # Display analysis results
        if result.dataset_type == "preference":
            console.print(f"\n[dim]🔄 分析偏好模式...[/dim]")
            console.print(f"[green]✓ 偏好分析: {result.sample_count} 对[/green]")
        elif result.dataset_type == "swe_bench":
            console.print(f"\n[dim]🔧 分析 SWE 任务...[/dim]")
            console.print(f"[green]✓ SWE 分析完成[/green]")
        elif result.rubric_patterns > 0:
            console.print(f"\n[dim]📊 分析评分标准...[/dim]")
            console.print(f"[green]✓ 评分标准: {result.rubric_patterns} 种模式[/green]")

        if result.prompt_templates > 0:
            console.print(f"[dim]📝 提取 Prompt 模板...[/dim]")
            console.print(f"[green]✓ Prompt模板: {result.prompt_templates} 个[/green]")

        console.print(f"[dim]⚙️ 计算人机分配...[/dim]")
        console.print(f"[green]✓ 人机分配: 人工 {result.human_percentage:.0f}%, 机器 {100-result.human_percentage:.0f}%[/green]")

        console.print(f"\n[dim]📄 生成综合报告...[/dim]")
        console.print(f"[green]✓ 综合报告已保存[/green]")
        console.print(f"[dim]📋 生成复刻指南...[/dim]")
        console.print(f"[green]✓ 复刻指南已保存[/green]")
        console.print(f"[dim]📦 生成标准化摘要...[/dim]")
        console.print(f"[green]✓ 标准化摘要已保存 (Radar 兼容)[/green]")
        console.print(f"[dim]📚 更新知识库...[/dim]")
        console.print(f"[green]✓ 知识库已更新[/green]")
        console.print(f"[dim]💾 更新缓存...[/dim]")
        console.print(f"[green]✓ 缓存已更新[/green]")

        # Display summary
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]  分析完成[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        console.print(f"[bold]生成的文件:[/bold]")
        for fname in result.files_generated:
            fpath = os.path.join(result.output_dir, fname)
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath)
                if fsize > 1024:
                    size_str = f"{fsize / 1024:.1f}KB"
                else:
                    size_str = f"{fsize}B"
                icon = "📊" if fname.endswith(".json") else "📄" if fname.endswith(".md") else "📑"
                console.print(f"  {icon} {fname} ({size_str})")

        report_path = os.path.join(result.output_dir, "ANALYSIS_REPORT.md")
        guide_path = os.path.join(result.output_dir, "REPRODUCTION_GUIDE.md")
        console.print(f"\n[bold]核心产出:[/bold]")
        console.print(f"  📄 分析报告: [cyan]{report_path}[/cyan]")
        console.print(f"  📋 复刻指南: [cyan]{guide_path}[/cyan]")

        # Display warnings if any
        if hasattr(result, "warnings") and result.warnings:
            console.print(f"\n[yellow]⚠ 部分步骤跳过 ({len(result.warnings)} 项):[/yellow]")
            for w in result.warnings:
                console.print(f"  [dim]· {w}[/dim]")

    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        import traceback
        traceback.print_exc()

def _generate_analysis_report(
    dataset_id: str,
    sample_count: int,
    actual_size: int,
    rubrics_result,
    prompt_library,
    strategy_result,
    allocation,
    region: str,
) -> str:
    """Generate a comprehensive Markdown analysis report."""
    from datetime import datetime

    lines = []
    lines.append(f"# 🔬 {dataset_id} 深度逆向分析报告")
    lines.append("")
    lines.append(f"> **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> **数据集**: {dataset_id}")
    lines.append(f"> **分析样本**: {sample_count} 条")
    lines.append(f"> **目标规模**: {actual_size:,} 条")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## 📊 执行摘要")
    lines.append("")
    lines.append("| 维度 | 发现 |")
    lines.append("|------|------|")

    if rubrics_result:
        lines.append(f"| **评分标准** | {rubrics_result.total_rubrics:,} 条，{rubrics_result.unique_patterns:,} 种独特模式 |")
    if prompt_library:
        lines.append(f"| **Prompt模板** | {prompt_library.unique_count} 个去重后的系统提示模板 |")
    if strategy_result:
        lines.append(f"| **数据来源** | 混合策略（合成 {strategy_result.synthetic_score*100:.0f}% + 改编 {strategy_result.modified_score*100:.0f}% + 专业 {strategy_result.niche_score*100:.0f}%） |")

    lines.append(f"| **复现成本** | 约 ${allocation.total_cost:,.0f}（人工 ${allocation.total_human_cost:,.0f} + API ${allocation.total_machine_cost:,.0f}） |")
    lines.append(f"| **人机分配** | 人工 {allocation.human_work_percentage:.0f}%，机器 {allocation.machine_work_percentage:.0f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Rubrics Analysis
    if rubrics_result:
        lines.append("## 1️⃣ 评分标准（Rubrics）模式分析")
        lines.append("")
        lines.append("### 1.1 总体统计")
        lines.append("")
        lines.append(f"- **总数**: {rubrics_result.total_rubrics:,} 条评分标准")
        lines.append(f"- **独特模式**: {rubrics_result.unique_patterns:,} 种")
        lines.append(f"- **平均每任务**: {rubrics_result.avg_rubrics_per_task:.1f} 条")
        lines.append("")

        lines.append("### 1.2 高频动词分布")
        lines.append("")
        lines.append("| 排名 | 动词 | 出现次数 | 占比 |")
        lines.append("|------|------|----------|------|")

        sorted_verbs = sorted(rubrics_result.verb_distribution.items(), key=lambda x: -x[1])[:10]
        for i, (verb, count) in enumerate(sorted_verbs, 1):
            pct = count / rubrics_result.total_rubrics * 100
            lines.append(f"| {i} | **{verb}** | {count:,} | {pct:.1f}% |")
        lines.append("")

        lines.append("### 1.3 评分类别分布")
        lines.append("")
        sorted_cats = sorted(rubrics_result.category_distribution.items(), key=lambda x: -x[1])
        for cat, count in sorted_cats[:5]:
            pct = count / rubrics_result.total_rubrics * 100
            bar_len = int(pct / 2.5)
            bar = "█" * bar_len
            lines.append(f"- **{cat}**: {bar} {pct:.1f}% ({count:,})")
        lines.append("")

        if rubrics_result.structured_templates:
            lines.append("### 1.4 模板化结构（Top 5）")
            lines.append("")
            lines.append("| 类别 | 动作 | 目标 | 条件 | 频次 |")
            lines.append("|------|------|------|------|------|")
            for entry in rubrics_result.structured_templates[:5]:
                action = entry.get("action") or "N/A"
                target = entry.get("target") or "N/A"
                condition = entry.get("condition") or "—"
                freq = entry.get("frequency", 0)
                lines.append(
                    f"| {entry.get('category', 'general')} | {action} | {target} | {condition} | {freq} |"
                )
            lines.append("")
        lines.append("---")
        lines.append("")

    # Prompt Templates
    if prompt_library:
        lines.append("## 2️⃣ 系统提示（System Prompt）模板分析")
        lines.append("")
        lines.append("### 2.1 提取统计")
        lines.append("")
        lines.append(f"- **原始数量**: {prompt_library.total_extracted} 条")
        lines.append(f"- **去重后**: {prompt_library.unique_count} 个独特模板")
        lines.append(f"- **去重率**: {prompt_library.deduplication_ratio:.1%}")
        lines.append(f"- **平均长度**: {prompt_library.avg_length:,.0f} 字符")
        lines.append("")

        lines.append("### 2.2 模板分类")
        lines.append("")
        lines.append("| 类别 | 数量 | 说明 |")
        lines.append("|------|------|------|")
        category_desc = {
            "system": "系统角色设定",
            "constraint": "约束条件",
            "task": "任务说明",
            "format": "格式要求",
            "example": "示例说明",
            "other": "其他类型",
        }
        for cat, count in sorted(prompt_library.category_counts.items(), key=lambda x: -x[1]):
            desc = category_desc.get(cat, cat)
            lines.append(f"| **{cat}** | {count} | {desc} |")
        lines.append("")

        if prompt_library.domain_counts:
            lines.append("### 2.3 领域分布")
            lines.append("")
            for domain, count in sorted(prompt_library.domain_counts.items(), key=lambda x: -x[1])[:5]:
                pct = count / prompt_library.unique_count * 100
                lines.append(f"- **{domain}**: {count} ({pct:.0f}%)")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Context Strategy
    if strategy_result:
        lines.append("## 3️⃣ 上下文构造策略分析")
        lines.append("")
        lines.append("### 3.1 策略识别")
        lines.append("")
        lines.append(f"**主要策略**: {strategy_result.primary_strategy.value}")
        lines.append(f"**置信度**: {strategy_result.confidence:.1%}")
        lines.append("")

        lines.append("### 3.2 策略得分")
        lines.append("")
        lines.append("| 策略 | 得分 | 说明 |")
        lines.append("|------|------|------|")
        lines.append(f"| 🔧 合成生成 | {strategy_result.synthetic_score*100:.1f}% | 使用 AI 模型生成虚构内容 |")
        lines.append(f"| 📝 改编修改 | {strategy_result.modified_score*100:.1f}% | 基于真实来源改编 |")
        lines.append(f"| 🔬 专业领域 | {strategy_result.niche_score*100:.1f}% | 专业/小众领域内容 |")
        lines.append("")

        lines.append("### 3.3 检测到的指标")
        lines.append("")
        if strategy_result.synthetic_indicators:
            lines.append("**🔧 合成生成**")
            for ind in strategy_result.synthetic_indicators[:5]:
                lines.append(f"- `{ind}`")
            lines.append("")
        if strategy_result.modified_indicators:
            lines.append("**📝 改编修改**")
            for ind in strategy_result.modified_indicators[:5]:
                lines.append(f"- `{ind}`")
            lines.append("")
        if strategy_result.niche_indicators:
            lines.append("**🔬 专业领域**")
            for ind in strategy_result.niche_indicators[:5]:
                lines.append(f"- `{ind}`")
            lines.append("")

        if strategy_result.recommendations:
            lines.append("### 3.4 复现建议")
            lines.append("")
            for rec in strategy_result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Human-Machine Allocation
    lines.append("## 4️⃣ 人机任务分配")
    lines.append("")
    lines.append("### 4.1 分配总览")
    lines.append("")
    human_pct = allocation.human_work_percentage
    machine_pct = allocation.machine_work_percentage
    human_bar = "█" * int(human_pct / 2.5)
    machine_bar = "█" * int(machine_pct / 2.5)
    lines.append(f"- 人工工作: {human_bar} **{human_pct:.0f}%**")
    lines.append(f"- 机器工作: {machine_bar} **{machine_pct:.0f}%**")
    lines.append("")

    lines.append("### 4.2 任务明细")
    lines.append("")
    lines.append("| 任务 | 分配方式 | 人工占比 | 人工时长 | 人工成本 | 机器成本 |")
    lines.append("|------|----------|----------|----------|----------|----------|")

    decision_zh = {
        "human_only": "纯人工",
        "machine_only": "纯机器",
        "human_primary": "人工为主",
        "machine_primary": "机器为主",
        "balanced": "均衡",
    }
    for task in allocation.tasks:
        dec = decision_zh.get(task.decision.value, task.decision.value)
        lines.append(f"| **{task.task_name}** | {dec} | {task.human_percentage:.0f}% | {task.human_hours:.1f}h | ${task.human_cost:,.0f} | ${task.machine_cost:.1f} |")
    lines.append("")

    lines.append("### 4.3 成本估算")
    lines.append("")
    lines.append("| 项目 | 金额 |")
    lines.append("|------|------|")
    lines.append(f"| 人工成本 | ${allocation.total_human_cost:,.0f} |")
    lines.append(f"| API/机器成本 | ${allocation.total_machine_cost:,.0f} |")
    lines.append(f"| **总计** | **${allocation.total_cost:,.0f}** |")
    lines.append(f"| 预估节省 | ${allocation.estimated_savings_vs_all_human:,.0f}（相比全人工） |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Recommendations
    lines.append("## 5️⃣ 复现建议")
    lines.append("")
    lines.append("### 5.1 团队配置")
    lines.append("")
    lines.append("| 角色 | 人数 | 职责 |")
    lines.append("|------|------|------|")
    lines.append("| 领域专家 | 4 | 创建和审核上下文内容 |")
    lines.append("| 任务设计师 | 2 | 设计评估任务和问题 |")
    lines.append("| 标注员 | 4 | 编写评分标准和标注 |")
    lines.append("| QA审核员 | 2 | 质量保证和验证 |")
    lines.append("| 项目经理 | 1 | 协调团队和进度跟踪 |")
    lines.append("")

    lines.append("### 5.2 质量检查点")
    lines.append("")
    lines.append("- [ ] 上下文内容是原创的（不在训练数据中）")
    lines.append("- [ ] 任务需要上下文才能回答")
    lines.append("- [ ] 评分标准遵循已发现的模式")
    lines.append("- [ ] 通过交叉验证审核")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> 报告由 DataRecipe 自动生成")

    return "\n".join(lines)


def _generate_reproduction_guide(
    dataset_id: str,
    schema_info: dict,
    category_set: set,
    sub_category_set: set,
    system_prompts_by_domain: dict,
    rubrics_examples: list,
    sample_items: list,
    rubrics_result,
    prompt_library,
    allocation,
    # RLHF preference dataset support
    is_preference_dataset: bool = False,
    preference_pairs: list = None,
    preference_topics: dict = None,
    preference_patterns: dict = None,
    # SWE-bench dataset support
    is_swe_dataset: bool = False,
    swe_stats: dict = None,
    # LLM analysis for unknown types
    llm_analysis=None,
) -> str:
    """Generate a practical reproduction guide for recreating a similar dataset."""
    import json
    from datarecipe.analyzers.llm_dataset_analyzer import generate_llm_guide_section

    preference_pairs = preference_pairs or []
    preference_topics = preference_topics or {}
    preference_patterns = preference_patterns or {}
    swe_stats = swe_stats or {}

    lines = []
    lines.append(f"# 📋 {dataset_id} 复刻指南")
    lines.append("")

    if is_swe_dataset:
        lines.append("> **这是一个软件工程评测数据集 (SWE-bench 风格)。本指南提供任务构建规范，帮助你构建类似的代码修复/功能实现评测集。**")
    elif is_preference_dataset:
        lines.append("> **这是一个 RLHF 偏好数据集。本指南提供偏好标注规范，帮助你构建类似的人类偏好数据。**")
    elif llm_analysis and llm_analysis.dataset_type != "unknown":
        lines.append(f"> **数据集类型: {llm_analysis.dataset_type}。{llm_analysis.purpose}**")
    else:
        lines.append("> **本指南提供可直接操作的模板和规范，帮助你从零开始构建类似风格的数据集。**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ==================== LLM Analysis Section (if available) ====================
    if llm_analysis and llm_analysis.dataset_type != "unknown":
        lines.append(generate_llm_guide_section(llm_analysis))
        lines.append("")

    # ==================== Section 1: Data Schema ====================
    lines.append("## 1️⃣ 数据结构规范 (Schema)")
    lines.append("")
    lines.append("### 1.1 字段定义")
    lines.append("")
    lines.append("| 字段名 | 类型 | 子类型 | 说明 |")
    lines.append("|--------|------|--------|------|")

    field_descriptions = {
        "messages": "对话消息列表，包含 system/user/assistant 角色",
        "rubrics": "评分标准列表，用于评估模型回答质量",
        "metadata": "元数据字典，包含任务分类等信息",
        "input": "用户输入/上下文",
        "output": "期望的模型输出",
        "instruction": "任务指令",
        "context": "上下文信息",
        "question": "问题内容",
        "answer": "参考答案",
    }

    for field, info in schema_info.items():
        ftype = info["type"]
        nested = info.get("nested_type", "")
        if isinstance(nested, list):
            nested = f"keys: {', '.join(nested[:3])}"
        desc = field_descriptions.get(field, "—")
        lines.append(f"| `{field}` | `{ftype}` | `{nested or '—'}` | {desc} |")
    lines.append("")

    # JSON Schema
    lines.append("### 1.2 JSON Schema")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    for i, (field, info) in enumerate(schema_info.items()):
        comma = "," if i < len(schema_info) - 1 else ""
        if info["type"] == "list":
            if info.get("nested_type") == "dict":
                lines.append(f'  "{field}": [{{...}}]{comma}')
            elif info.get("nested_type") == "str":
                lines.append(f'  "{field}": ["..."]' + comma)
            else:
                lines.append(f'  "{field}": []{comma}')
        elif info["type"] == "dict":
            lines.append(f'  "{field}": {{...}}{comma}')
        elif info["type"] == "str":
            lines.append(f'  "{field}": "..."{comma}')
        else:
            lines.append(f'  "{field}": ...{comma}')
    lines.append("}")
    lines.append("```")
    lines.append("")

    # ==================== Section 2: Category System ====================
    lines.append("## 2️⃣ 任务分类体系")
    lines.append("")

    if category_set:
        lines.append("### 2.1 主分类 (context_category)")
        lines.append("")
        for cat in sorted(category_set):
            lines.append(f"- `{cat}`")
        lines.append("")

    if sub_category_set:
        lines.append("### 2.2 子分类 (sub_category)")
        lines.append("")
        for sub in sorted(sub_category_set):
            lines.append(f"- `{sub}`")
        lines.append("")

    if not category_set and not sub_category_set and not is_preference_dataset:
        lines.append("*未检测到分类体系*")
        lines.append("")

    # For preference datasets, show topic distribution
    if is_preference_dataset and preference_topics:
        lines.append("### 话题分布")
        lines.append("")
        lines.append("| 话题 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        total = sum(preference_topics.values())
        for topic, count in sorted(preference_topics.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"| {topic} | {count} | {pct:.1f}% |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ==================== Section 2.5: Preference Dataset Guide (if applicable) ====================
    if is_preference_dataset:
        lines.append("## 🔄 偏好数据集专用指南")
        lines.append("")
        lines.append("这是一个 RLHF (Reinforcement Learning from Human Feedback) 偏好数据集。")
        lines.append("每条数据包含一对回复：`chosen`（被选中的更好回复）和 `rejected`（被拒绝的较差回复）。")
        lines.append("")

        # Preference patterns analysis
        lines.append("### 偏好模式分析")
        lines.append("")
        if preference_patterns:
            total_patterns = sum(preference_patterns.values())
            if total_patterns > 0:
                lines.append("| 模式 | 数量 | 占比 | 说明 |")
                lines.append("|------|------|------|------|")
                pattern_desc = {
                    "chosen_longer": "被选中回复更长",
                    "rejected_longer": "被拒绝回复更长",
                    "same_length": "长度相近",
                    "chosen_safer": "被选中回复更安全（rejected 含拒绝词）",
                }
                for pattern, count in sorted(preference_patterns.items(), key=lambda x: -x[1]):
                    if count > 0:
                        pct = count / total_patterns * 100
                        desc = pattern_desc.get(pattern, pattern)
                        lines.append(f"| {pattern} | {count} | {pct:.1f}% | {desc} |")
                lines.append("")

        # Preference labeling guidelines
        lines.append("### 偏好标注规范")
        lines.append("")
        lines.append("标注员需要比较两个回复，选择「更好」的那个。判断标准：")
        lines.append("")
        lines.append("| 维度 | 选择 chosen 的条件 |")
        lines.append("|------|-------------------|")
        lines.append("| **有用性** | 更直接地回答了问题，提供了更实用的信息 |")
        lines.append("| **准确性** | 信息更准确，没有事实错误 |")
        lines.append("| **安全性** | 不包含有害、违法、歧视性内容 |")
        lines.append("| **完整性** | 覆盖了问题的各个方面，不遗漏关键信息 |")
        lines.append("| **清晰度** | 表达更清晰，结构更好，易于理解 |")
        lines.append("| **诚实性** | 承认不确定性，不编造信息 |")
        lines.append("")

        # Preference pair examples
        if preference_pairs:
            lines.append("### 偏好对示例")
            lines.append("")
            for i, pair in enumerate(preference_pairs[:3], 1):
                lines.append(f"**示例 {i}** (话题: `{pair.get('topic', 'unknown')}`)")
                lines.append("")
                lines.append("**Human:**")
                lines.append("```")
                lines.append(pair.get("human_query", "")[:300] or "(无)")
                lines.append("```")
                lines.append("")
                lines.append("**Chosen (被选中):**")
                lines.append("```")
                chosen_resp = pair.get("chosen_response", "")[:400]
                lines.append(chosen_resp if chosen_resp else "(无)")
                lines.append("```")
                lines.append("")
                lines.append("**Rejected (被拒绝):**")
                lines.append("```")
                rejected_resp = pair.get("rejected_response", "")[:400]
                lines.append(rejected_resp if rejected_resp else "(无)")
                lines.append("```")
                lines.append("")

        # SOP for preference dataset
        lines.append("### 偏好数据生产 SOP")
        lines.append("")
        lines.append("```")
        lines.append("Phase 1: 准备阶段")
        lines.append("├─ 步骤 1.1: 收集用户问题（多样化话题）")
        lines.append("├─ 步骤 1.2: 使用 LLM 生成多个候选回复（通常 2-4 个）")
        lines.append("└─ 步骤 1.3: 准备标注界面和标注指南")
        lines.append("")
        lines.append("Phase 2: 标注阶段")
        lines.append("├─ 步骤 2.1: 标注员阅读问题和所有候选回复")
        lines.append("├─ 步骤 2.2: 根据标注规范选择最佳回复 (chosen)")
        lines.append("├─ 步骤 2.3: 选择最差回复 (rejected)")
        lines.append("└─ 步骤 2.4: 记录选择理由（可选，用于质检）")
        lines.append("")
        lines.append("Phase 3: 质量控制")
        lines.append("├─ 步骤 3.1: 双人标注，计算一致性 (Cohen's Kappa)")
        lines.append("├─ 步骤 3.2: 不一致样本由第三人仲裁")
        lines.append("└─ 步骤 3.3: 抽样审核，确保标注质量")
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ==================== Section 2.6: SWE-bench Dataset Guide (if applicable) ====================
    if is_swe_dataset and swe_stats:
        lines.append("## 🔧 软件工程评测数据集专用指南")
        lines.append("")
        lines.append("这是一个 SWE-bench 风格的软件工程评测数据集，用于评估 AI 代码修复和功能实现能力。")
        lines.append("")

        # Language distribution
        if swe_stats.get("languages"):
            lines.append("### 编程语言分布")
            lines.append("")
            lines.append("| 语言 | 数量 | 占比 |")
            lines.append("|------|------|------|")
            total = sum(swe_stats["languages"].values())
            for lang, count in sorted(swe_stats["languages"].items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"| {lang} | {count} | {pct:.1f}% |")
            lines.append("")

        # Repository distribution
        if swe_stats.get("repos"):
            lines.append("### 仓库分布 (Top 10)")
            lines.append("")
            lines.append("| 仓库 | 任务数 |")
            lines.append("|------|--------|")
            for repo, count in sorted(swe_stats["repos"].items(), key=lambda x: -x[1])[:10]:
                lines.append(f"| `{repo}` | {count} |")
            lines.append("")

        # Issue types
        if swe_stats.get("issue_types"):
            lines.append("### 问题类型分布")
            lines.append("")
            lines.append("| 类型 | 数量 |")
            lines.append("|------|------|")
            for itype, count in sorted(swe_stats["issue_types"].items(), key=lambda x: -x[1]):
                lines.append(f"| `{itype}` | {count} |")
            lines.append("")

        # Issue categories
        if swe_stats.get("issue_categories"):
            lines.append("### 所需知识领域")
            lines.append("")
            lines.append("| 领域 | 数量 |")
            lines.append("|------|------|")
            for cat, count in sorted(swe_stats["issue_categories"].items(), key=lambda x: -x[1]):
                lines.append(f"| `{cat}` | {count} |")
            lines.append("")

        # Patch complexity
        if swe_stats.get("patch_lines"):
            avg_lines = sum(swe_stats["patch_lines"]) / len(swe_stats["patch_lines"])
            max_lines = max(swe_stats["patch_lines"])
            min_lines = min(swe_stats["patch_lines"])
            lines.append("### 代码修改复杂度")
            lines.append("")
            lines.append(f"- **平均修改行数**: {avg_lines:.1f} 行")
            lines.append(f"- **最大修改**: {max_lines} 行")
            lines.append(f"- **最小修改**: {min_lines} 行")
            lines.append("")

        # Problem statement examples
        if swe_stats.get("examples"):
            lines.append("### 问题描述示例")
            lines.append("")
            for i, ex in enumerate(swe_stats["examples"][:2], 1):
                lines.append(f"**示例 {i}** (`{ex.get('repo', 'unknown')}` - {ex.get('language', 'unknown')})")
                lines.append("")
                lines.append("**Problem Statement:**")
                lines.append("```")
                lines.append(ex.get("problem_statement", "")[:600])
                lines.append("```")
                lines.append("")
                if ex.get("requirements"):
                    lines.append("**Requirements:**")
                    lines.append("```")
                    lines.append(ex.get("requirements", "")[:400])
                    lines.append("```")
                    lines.append("")

        # SOP for SWE-bench dataset
        lines.append("### SWE-bench 数据生产 SOP")
        lines.append("")
        lines.append("```")
        lines.append("Phase 1: 仓库筛选")
        lines.append("├─ 步骤 1.1: 选择活跃的开源仓库（GPL 等强 copyleft 许可优先）")
        lines.append("├─ 步骤 1.2: 确保有完善的测试套件")
        lines.append("└─ 步骤 1.3: 筛选有清晰 issue/PR 历史的仓库")
        lines.append("")
        lines.append("Phase 2: 任务挖掘")
        lines.append("├─ 步骤 2.1: 从已合并的 PR 中提取 bug fix / feature")
        lines.append("├─ 步骤 2.2: 提取 base_commit (修复前) 和 patch (修复内容)")
        lines.append("├─ 步骤 2.3: 识别 fail-to-pass 测试（修复后应通过）")
        lines.append("└─ 步骤 2.4: 识别 pass-to-pass 测试（确保无回归）")
        lines.append("")
        lines.append("Phase 3: 任务增强")
        lines.append("├─ 步骤 3.1: 撰写 problem_statement（问题描述）")
        lines.append("├─ 步骤 3.2: 撰写 requirements（功能需求）")
        lines.append("├─ 步骤 3.3: 标注 interface（涉及的 API/函数）")
        lines.append("└─ 步骤 3.4: 分类 issue_categories（所需知识领域）")
        lines.append("")
        lines.append("Phase 4: 质量验证")
        lines.append("├─ 步骤 4.1: 验证 patch 能通过所有测试")
        lines.append("├─ 步骤 4.2: 确保 problem_statement 不泄露解决方案")
        lines.append("└─ 步骤 4.3: 验证任务可由人类工程师独立完成")
        lines.append("```")
        lines.append("")

        # Quality criteria
        lines.append("### 数据质量标准")
        lines.append("")
        lines.append("| 维度 | 要求 |")
        lines.append("|------|------|")
        lines.append("| **问题描述** | 清晰描述 bug 现象或功能需求，不泄露解决方案 |")
        lines.append("| **测试覆盖** | 至少有 1 个 fail-to-pass 测试验证修复正确性 |")
        lines.append("| **无回归** | pass-to-pass 测试确保不引入新 bug |")
        lines.append("| **可复现** | 提供完整的环境设置命令 |")
        lines.append("| **合理复杂度** | 修改行数适中，不过于简单也不过于复杂 |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ==================== Section 3: System Prompt Templates ====================
    lines.append("## 3️⃣ System Prompt 模板库")
    lines.append("")
    lines.append("> 以下是从数据集中提取的真实 System Prompt 示例，可直接复用或改编。")
    lines.append("")

    if system_prompts_by_domain:
        for domain, prompts in list(system_prompts_by_domain.items())[:5]:
            lines.append(f"### 3.{list(system_prompts_by_domain.keys()).index(domain)+1} {domain}")
            lines.append("")
            for i, p in enumerate(prompts[:2], 1):
                content = p["content"]
                # Truncate if too long
                if len(content) > 1500:
                    content = content[:1500] + "\n\n... (截断，完整内容见 prompt_templates.json)"
                lines.append(f"**示例 {i}:**")
                lines.append("")
                lines.append("```")
                lines.append(content)
                lines.append("```")
                lines.append("")
    else:
        lines.append("*未提取到 System Prompt*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ==================== Section 4: Rubric Writing Guide ====================
    lines.append("## 4️⃣ 评分标准 (Rubric) 编写规范")
    lines.append("")

    if rubrics_result:
        lines.append("### 4.1 句式模式")
        lines.append("")
        lines.append("从数据集中发现的高频句式模式：")
        lines.append("")

        # Top verbs
        sorted_verbs = sorted(rubrics_result.verb_distribution.items(), key=lambda x: -x[1])[:8]
        lines.append("| 核心动词 | 频次 | 示例句式 |")
        lines.append("|----------|------|----------|")
        verb_examples = {
            "include": "The response should include [具体内容]",
            "state": "The response should state [具体事实]",
            "explain": "The response should explain [概念/原因]",
            "provide": "The response should provide [信息/示例]",
            "not": "The response should not [禁止行为]",
            "identify": "The response should identify [目标对象]",
            "use": "The response should use [指定方法/格式]",
            "define": "The response should define [术语/概念]",
            "list": "The response should list [条目/步骤]",
            "describe": "The response should describe [描述对象]",
        }
        for verb, count in sorted_verbs:
            example = verb_examples.get(verb, f"... should {verb} ...")
            lines.append(f"| **{verb}** | {count} | `{example}` |")
        lines.append("")

        lines.append("### 4.2 评分标准结构")
        lines.append("")
        lines.append("推荐采用以下结构编写评分标准：")
        lines.append("")
        lines.append("```")
        lines.append("[主语] should [动作] [目标]. [条件/例外]. Fail if [失败条件].")
        lines.append("```")
        lines.append("")
        lines.append("**结构说明：**")
        lines.append("")
        lines.append("| 组成部分 | 说明 | 示例 |")
        lines.append("|----------|------|------|")
        lines.append("| 主语 | 被评估对象 | The response / The model / The answer |")
        lines.append("| 动作 | 期望行为 | should include / should explain / should not |")
        lines.append("| 目标 | 具体内容 | the definition of X / at least 3 examples |")
        lines.append("| 条件 | 适用范围 | For example, ... / When X, ... |")
        lines.append("| 失败条件 | 扣分标准 | Fail if X is missing / Fail if incorrect |")
        lines.append("")

    # Real rubric examples
    if rubrics_examples:
        lines.append("### 4.3 完整示例")
        lines.append("")
        lines.append("> 以下是从数据集中提取的真实评分标准示例：")
        lines.append("")

        for i, ex in enumerate(rubrics_examples[:3], 1):
            meta = ex.get("metadata", {})
            cat = meta.get("context_category", meta.get("category", "unknown"))
            sub = meta.get("sub_category", "")

            lines.append(f"**示例 {i}** (`{cat}` / `{sub}`)")
            lines.append("")
            for j, r in enumerate(ex["rubrics"][:5], 1):
                lines.append(f"{j}. {r}")
            if len(ex["rubrics"]) > 5:
                lines.append(f"   ... (共 {len(ex['rubrics'])} 条)")
            lines.append("")

    lines.append("---")
    lines.append("")

    # ==================== Section 5: Step-by-Step SOP ====================
    lines.append("## 5️⃣ 复刻 SOP (标准操作流程)")
    lines.append("")
    lines.append("### Phase 1: 准备阶段")
    lines.append("")
    lines.append("```")
    lines.append("步骤 1.1: 确定目标领域和分类体系")
    lines.append("         ├─ 参考上方「任务分类体系」")
    lines.append("         └─ 确定要覆盖的 context_category 列表")
    lines.append("")
    lines.append("步骤 1.2: 收集原始上下文材料")
    lines.append("         ├─ 专业文档、手册、规范")
    lines.append("         ├─ 确保材料不在 LLM 训练数据中")
    lines.append("         └─ 每个分类准备 10-20 份材料")
    lines.append("")
    lines.append("步骤 1.3: 准备 System Prompt 模板")
    lines.append("         ├─ 参考上方「System Prompt 模板库」")
    lines.append("         └─ 按领域定制角色设定")
    lines.append("```")
    lines.append("")

    lines.append("### Phase 2: 数据生成阶段")
    lines.append("")
    lines.append("```")
    lines.append("步骤 2.1: 编写 System Prompt")
    lines.append("         ├─ 定义 AI 角色和能力边界")
    lines.append("         ├─ 设置输出格式约束")
    lines.append("         └─ 添加领域特定指令")
    lines.append("")
    lines.append("步骤 2.2: 构造 User Query")
    lines.append("         ├─ 嵌入上下文材料")
    lines.append("         ├─ 设计需要理解上下文才能回答的问题")
    lines.append("         └─ 问题应有明确的评估标准")
    lines.append("")
    lines.append("步骤 2.3: 编写评分标准 (Rubrics)")
    lines.append("         ├─ 遵循上方「评分标准编写规范」")
    lines.append("         ├─ 每个任务 8-15 条评分标准")
    lines.append("         ├─ 覆盖：正确性、完整性、格式、约束")
    lines.append("         └─ 使用 Fail if ... 明确失败条件")
    lines.append("```")
    lines.append("")

    lines.append("### Phase 3: 质量控制阶段")
    lines.append("")
    lines.append("```")
    lines.append("步骤 3.1: 自检")
    lines.append("         ├─ [ ] 问题必须依赖上下文才能回答")
    lines.append("         ├─ [ ] 评分标准可量化、可执行")
    lines.append("         └─ [ ] 数据格式符合 Schema 规范")
    lines.append("")
    lines.append("步骤 3.2: 交叉审核")
    lines.append("         ├─ 另一标注员独立评估")
    lines.append("         ├─ 检查评分标准是否遗漏")
    lines.append("         └─ 验证标准是否存在歧义")
    lines.append("")
    lines.append("步骤 3.3: 抽样测试")
    lines.append("         ├─ 用 LLM 生成回答")
    lines.append("         ├─ 按 Rubrics 评分")
    lines.append("         └─ 验证评分标准的区分度")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ==================== Section 6: Complete Example ====================
    lines.append("## 6️⃣ 完整数据示例")
    lines.append("")

    if sample_items:
        item = sample_items[0]
        lines.append("```json")
        # Create a clean version for display
        display_item = {}
        for k, v in item.items():
            if k == "messages" and isinstance(v, list):
                display_messages = []
                for msg in v:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if len(content) > 500:
                            msg = dict(msg)
                            msg["content"] = content[:500] + "... (truncated)"
                        display_messages.append(msg)
                display_item[k] = display_messages
            elif k == "rubrics" and isinstance(v, list):
                display_item[k] = v[:5] + ["... (truncated)"] if len(v) > 5 else v
            elif isinstance(v, str) and len(v) > 300:
                display_item[k] = v[:300] + "... (truncated)"
            else:
                display_item[k] = v
        lines.append(json.dumps(display_item, indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("*无可用示例*")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ==================== Section 7: Resource Estimation ====================
    lines.append("## 7️⃣ 资源估算")
    lines.append("")

    if allocation:
        lines.append("### 7.1 人力配置建议")
        lines.append("")
        lines.append("| 角色 | 人数 | 主要职责 |")
        lines.append("|------|------|----------|")
        lines.append("| 领域专家 | 2-4 | 提供上下文材料，审核专业性 |")
        lines.append("| 任务设计师 | 1-2 | 设计问题，确保评测效度 |")
        lines.append("| 标注员 | 3-5 | 编写评分标准，标注数据 |")
        lines.append("| QA | 1-2 | 质量抽检，一致性校验 |")
        lines.append("")

        lines.append("### 7.2 成本估算")
        lines.append("")
        lines.append(f"- **人工成本**: ${allocation.total_human_cost:,.0f}")
        lines.append(f"- **API 成本**: ${allocation.total_machine_cost:,.0f}")
        lines.append(f"- **总计**: ${allocation.total_cost:,.0f}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ==================== Section 8: Checklist ====================
    lines.append("## 8️⃣ 发布前检查清单")
    lines.append("")
    lines.append("### 数据质量")
    lines.append("")
    lines.append("- [ ] 所有字段符合 Schema 规范")
    lines.append("- [ ] 无空值或异常值")
    lines.append("- [ ] 上下文材料不在公开训练集中")
    lines.append("- [ ] 评分标准无歧义，可量化执行")
    lines.append("")
    lines.append("### 覆盖度")
    lines.append("")
    lines.append("- [ ] 各分类数据量均衡")
    lines.append("- [ ] 难度分布合理")
    lines.append("- [ ] 领域覆盖完整")
    lines.append("")
    lines.append("### 合规性")
    lines.append("")
    lines.append("- [ ] 无版权问题")
    lines.append("- [ ] 无隐私信息泄露")
    lines.append("- [ ] 标注许可证明确")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> 指南由 DataRecipe 自动生成")

    return "\n".join(lines)


@main.command("batch-from-radar")
@click.argument("radar_report")
@click.option("--output-dir", "-o", default="./analysis_output", help="Output directory")
@click.option("--sample-size", "-n", default=200, help="Number of samples per dataset")
@click.option("--limit", "-l", default=0, type=int, help="Max datasets to analyze (0 = all)")
@click.option("--orgs", help="Filter by orgs (comma-separated)")
@click.option("--categories", help="Filter by categories (comma-separated)")
@click.option("--min-downloads", default=0, type=int, help="Minimum downloads")
@click.option("--use-llm", is_flag=True, help="Use LLM for unknown types")
@click.option("--region", "-r", default="china", help="Region for cost calculation")
@click.option("--sort-by", type=click.Choice(["downloads", "name", "category"]), default="downloads", help="Sort datasets by")
@click.option("--incremental", "-i", is_flag=True, help="Skip already analyzed datasets")
@click.option("--parallel", "-p", default=1, type=int, help="Parallel workers (1=sequential)")
def batch_from_radar(
    radar_report: str,
    output_dir: str,
    sample_size: int,
    limit: int,
    orgs: str,
    categories: str,
    min_downloads: int,
    use_llm: bool,
    region: str,
    sort_by: str,
    incremental: bool,
    parallel: int,
):
    """
    Batch analyze datasets from an ai-dataset-radar report.

    Reads a radar intel_report JSON file and analyzes all (or filtered) datasets.

    Example:
        datarecipe batch-from-radar ./data/reports/intel_report_2024-01-01.json
        datarecipe batch-from-radar ./report.json --orgs Anthropic,OpenAI --limit 5
        datarecipe batch-from-radar ./report.json --incremental --parallel 3
    """
    import json
    import os
    from datarecipe.integrations.radar import RadarIntegration, RecipeSummary

    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]  DataRecipe 批量分析 (Radar 集成)[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    # Load radar report
    console.print(f"[dim]📂 加载 Radar 报告: {radar_report}[/dim]")
    try:
        integration = RadarIntegration()
        all_datasets = integration.load_radar_report(radar_report)
        console.print(f"[green]✓ 加载 {len(all_datasets)} 个数据集[/green]")
    except Exception as e:
        console.print(f"[red]错误: 无法加载 Radar 报告 - {e}[/red]")
        return

    # Filter datasets
    org_list = [o.strip() for o in orgs.split(",")] if orgs else None
    cat_list = [c.strip() for c in categories.split(",")] if categories else None

    datasets = integration.filter_datasets(
        orgs=org_list,
        categories=cat_list,
        min_downloads=min_downloads,
        limit=0,  # Apply limit after sorting
    )

    if not datasets:
        console.print("[yellow]⚠ 没有符合条件的数据集[/yellow]")
        return

    # Sort datasets
    if sort_by == "downloads":
        datasets.sort(key=lambda x: x.downloads, reverse=True)
    elif sort_by == "name":
        datasets.sort(key=lambda x: x.id.lower())
    elif sort_by == "category":
        datasets.sort(key=lambda x: (x.category or "zzz", -x.downloads))

    # Incremental mode: skip already analyzed
    skipped_count = 0
    if incremental:
        filtered = []
        for ds in datasets:
            safe_name = ds.id.replace("/", "_").replace("\\", "_")
            summary_path = os.path.join(output_dir, safe_name, "recipe_summary.json")
            if os.path.exists(summary_path):
                skipped_count += 1
            else:
                filtered.append(ds)
        datasets = filtered
        if skipped_count > 0:
            console.print(f"[dim]增量模式: 跳过 {skipped_count} 个已分析数据集[/dim]")

    # Apply limit after filtering
    if limit > 0:
        datasets = datasets[:limit]

    if not datasets:
        console.print("[green]✓ 所有数据集已分析完成[/green]")
        return

    console.print(f"[dim]待分析: {len(datasets)} 个数据集 (排序: {sort_by})[/dim]\n")

    # Show datasets to analyze
    console.print("[bold]待分析数据集:[/bold]")
    for i, ds in enumerate(datasets[:10], 1):
        console.print(f"  {i}. {ds.id} ({ds.category}, {ds.downloads:,} downloads)")
    if len(datasets) > 10:
        console.print(f"  ... 还有 {len(datasets) - 10} 个")
    console.print("")

    # Save progress file for resume capability
    progress_file = os.path.join(output_dir, ".batch_progress.json")

    # Analyze each dataset
    summaries = []
    success_count = 0
    fail_count = 0

    for i, ds in enumerate(datasets, 1):
        console.print(f"\n[bold]━━━ [{i}/{len(datasets)}] {ds.id} ━━━[/bold]")

        try:
            # Import here to avoid circular imports
            from datasets import load_dataset
            from datarecipe.extractors import RubricsAnalyzer, PromptExtractor
            from datarecipe.analyzers import ContextStrategyDetector
            from datarecipe.generators import HumanMachineSplitter, TaskType

            # Create output directory
            safe_name = ds.id.replace("/", "_").replace("\\", "_")
            dataset_output_dir = os.path.join(output_dir, safe_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            # Load dataset
            console.print("[dim]  📥 加载数据...[/dim]")
            try:
                dataset = load_dataset(ds.id, split="train", streaming=True)
            except ValueError:
                # Try test split
                try:
                    dataset = load_dataset(ds.id, split="test", streaming=True)
                except Exception:
                    raise ValueError("无法找到可用的 split")

            # Collect samples
            schema_info = {}
            sample_items = []
            rubrics = []
            messages = []

            for j, item in enumerate(dataset):
                if j >= sample_size:
                    break

                # Schema info
                if j < 5:
                    for field, value in item.items():
                        if field not in schema_info:
                            schema_info[field] = {
                                "type": type(value).__name__,
                                "nested_type": None
                            }
                    sample_items.append(item)

                # Collect rubrics/messages
                for field in ["rubrics", "rubric", "criteria"]:
                    if field in item:
                        v = item[field]
                        if isinstance(v, list):
                            rubrics.extend(v)
                        elif isinstance(v, str):
                            rubrics.append(v)

                if "messages" in item:
                    messages.extend(item.get("messages", []))

            sample_count = j + 1
            console.print(f"[dim]  ✓ 加载 {sample_count} 样本[/dim]")

            # Detect dataset type
            is_preference = "chosen" in schema_info and "rejected" in schema_info
            is_swe = "repo" in schema_info and "patch" in schema_info

            dataset_type = ds.category or ""
            if is_preference:
                dataset_type = "preference"
            elif is_swe:
                dataset_type = "swe_bench"
            elif rubrics:
                dataset_type = "evaluation"

            # Human-machine allocation
            console.print("[dim]  ⚙️ 计算成本...[/dim]")
            splitter = HumanMachineSplitter(region=region)
            allocation = splitter.analyze(
                dataset_size=sample_count,
                task_types=[
                    TaskType.CONTEXT_CREATION,
                    TaskType.TASK_DESIGN,
                    TaskType.RUBRICS_WRITING,
                    TaskType.DATA_GENERATION,
                    TaskType.QUALITY_REVIEW,
                ]
            )

            # Rubrics analysis
            rubrics_result = None
            if rubrics:
                analyzer = RubricsAnalyzer()
                rubrics_result = analyzer.analyze(rubrics, task_count=sample_count)

            # Prompt analysis
            prompt_library = None
            if messages:
                extractor = PromptExtractor()
                prompt_library = extractor.extract(messages)

            # LLM analysis for unknown types
            llm_analysis = None
            if use_llm and not dataset_type:
                console.print("[dim]  🤖 LLM 分析中...[/dim]")
                try:
                    from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer
                    llm_analyzer = LLMDatasetAnalyzer()
                    llm_analysis = llm_analyzer.analyze(
                        dataset_id=ds.id,
                        schema_info=schema_info,
                        sample_items=sample_items,
                        sample_count=sample_count,
                    )
                    dataset_type = llm_analysis.dataset_type
                except Exception as e:
                    console.print(f"[yellow]  ⚠ LLM 分析失败: {e}[/yellow]")

            # Create summary
            summary = RadarIntegration.create_summary(
                dataset_id=ds.id,
                dataset_type=dataset_type,
                category=ds.category,
                allocation=allocation,
                rubrics_result=rubrics_result,
                prompt_library=prompt_library,
                schema_info=schema_info,
                sample_count=sample_count,
                llm_analysis=llm_analysis,
                output_dir=dataset_output_dir,
            )

            # Save summary
            RadarIntegration.save_summary(summary, dataset_output_dir)
            summaries.append(summary)
            success_count += 1

            console.print(f"[green]  ✓ 完成: {dataset_type or 'unknown'}, ${allocation.total_cost:,.0f}[/green]")

            # Update progress file
            progress = {
                "total": len(datasets),
                "completed": success_count,
                "failed": fail_count,
                "last_dataset": ds.id,
                "summaries": [s.dataset_id for s in summaries],
            }
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            fail_count += 1
            console.print(f"[red]  ✗ 失败: {e}[/red]")

            # Log failed dataset
            failed_log = os.path.join(output_dir, ".batch_failed.log")
            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"{ds.id}: {e}\n")
            continue

    # Clean up progress file on completion
    if os.path.exists(progress_file):
        os.remove(progress_file)

    # Generate aggregated report
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold cyan]  批量分析完成[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    console.print(f"成功: [green]{success_count}[/green]")
    console.print(f"失败: [red]{fail_count}[/red]")
    if skipped_count > 0:
        console.print(f"跳过: [dim]{skipped_count}[/dim] (已分析)")

    if summaries:
        # Save aggregated summary
        aggregate = RadarIntegration.aggregate_summaries(summaries)
        aggregate_path = os.path.join(output_dir, "batch_summary.json")
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)

        console.print(f"\n[bold]汇总统计:[/bold]")
        console.print(f"  总复刻成本: ${aggregate['total_reproduction_cost']['total']:,.0f}")
        console.print(f"  平均人工占比: {aggregate['avg_human_percentage']:.0f}%")
        console.print(f"  类型分布: {aggregate['type_distribution']}")

        console.print(f"\n[bold]输出文件:[/bold]")
        console.print(f"  📊 汇总报告: [cyan]{aggregate_path}[/cyan]")
        console.print(f"  📁 各数据集: [cyan]{output_dir}/<dataset>/recipe_summary.json[/cyan]")


@main.command("integrate-report")
@click.option("--radar-report", "-r", help="Path to Radar intel report JSON")
@click.option("--output-dir", "-o", default="./reports", help="Output directory")
@click.option("--recipe-dir", default="./analysis_output", help="Recipe analysis directory")
@click.option("--start-date", help="Period start date (YYYY-MM-DD)")
@click.option("--end-date", help="Period end date (YYYY-MM-DD)")
@click.option("--format", "-f", "formats", multiple=True, default=["md", "json"], help="Output formats")
def integrate_report(
    radar_report: str,
    output_dir: str,
    recipe_dir: str,
    start_date: str,
    end_date: str,
    formats: tuple,
):
    """
    Generate integrated report combining Radar discoveries and Recipe analysis.

    Example:
        datarecipe integrate-report -r ./intel_report.json -o ./reports
        datarecipe integrate-report --recipe-dir ./analysis_output
    """
    from datarecipe.reports import IntegratedReportGenerator

    console.print(f"\n[bold cyan]生成整合报告[/bold cyan]\n")

    generator = IntegratedReportGenerator(
        recipe_output_dir=recipe_dir,
    )

    # Generate report
    report = generator.generate_weekly_report(
        radar_report_path=radar_report,
        start_date=start_date,
        end_date=end_date,
    )

    # Display summary
    console.print(f"周期: {report.period_start} ~ {report.period_end}")
    console.print(f"发现数据集: {report.total_discovered}")
    console.print(f"已分析: {report.total_analyzed}")
    console.print(f"总复刻成本: ${report.total_reproduction_cost:,.0f}")
    console.print("")

    if report.insights:
        console.print("[bold]洞察:[/bold]")
        for insight in report.insights:
            console.print(f"  • {insight}")
        console.print("")

    # Save report
    paths = generator.save_report(report, output_dir, list(formats))

    console.print("[bold]生成文件:[/bold]")
    for fmt, path in paths.items():
        console.print(f"  📄 {path}")


@main.command("watch")
@click.argument("watch_dir")
@click.option("--output-dir", "-o", default="./analysis_output", help="Output directory")
@click.option("--interval", "-i", default=60, type=int, help="Check interval in seconds")
@click.option("--config", "-c", help="Path to trigger config YAML")
@click.option("--orgs", help="Filter by orgs (comma-separated)")
@click.option("--categories", help="Filter by categories (comma-separated)")
@click.option("--min-downloads", default=0, type=int, help="Minimum downloads")
@click.option("--limit", "-l", default=10, type=int, help="Max datasets per report")
@click.option("--once", is_flag=True, help="Check once and exit")
def watch_cmd(
    watch_dir: str,
    output_dir: str,
    interval: int,
    config: str,
    orgs: str,
    categories: str,
    min_downloads: int,
    limit: int,
    once: bool,
):
    """
    Watch for new Radar reports and auto-analyze datasets.

    Monitors a directory for new intel_report_*.json files and
    automatically triggers analysis for matching datasets.

    Example:
        datarecipe watch ./radar_reports/
        datarecipe watch ./reports --orgs Anthropic,OpenAI --interval 300
        datarecipe watch ./reports --config ./triggers.yaml --once
    """
    from datarecipe.triggers import RadarWatcher, TriggerConfig

    # Build config
    if config:
        trigger_config = TriggerConfig.from_yaml(config)
    else:
        trigger_config = TriggerConfig(
            orgs=[o.strip() for o in orgs.split(",")] if orgs else [],
            categories=[c.strip() for c in categories.split(",")] if categories else [],
            min_downloads=min_downloads,
            max_datasets_per_report=limit,
        )

    console.print(f"\n[bold cyan]DataRecipe Radar Watcher[/bold cyan]\n")
    console.print(f"监听目录: {watch_dir}")
    console.print(f"输出目录: {output_dir}")
    console.print(f"检查间隔: {interval}s")

    if trigger_config.orgs:
        console.print(f"组织过滤: {', '.join(trigger_config.orgs)}")
    if trigger_config.categories:
        console.print(f"类型过滤: {', '.join(trigger_config.categories)}")
    if trigger_config.min_downloads:
        console.print(f"最小下载: {trigger_config.min_downloads}")

    console.print("")

    # Create watcher
    def on_complete(dataset_id: str, result: dict):
        if result.get("success"):
            console.print(f"[green]✓[/green] {dataset_id}: {result.get('type', 'unknown')}, ${result.get('cost', 0):,.0f}")
        else:
            console.print(f"[red]✗[/red] {dataset_id}: {result.get('error', 'Unknown error')}")

    watcher = RadarWatcher(
        watch_dir=watch_dir,
        output_dir=output_dir,
        config=trigger_config,
        callback=on_complete,
    )

    if once:
        console.print("[dim]单次检查模式[/dim]\n")
        results = watcher.check_once()

        if not results:
            console.print("[dim]没有发现新报告[/dim]")
        else:
            for r in results:
                console.print(f"处理: {r['report']}")
                console.print(f"  成功: {r['datasets_analyzed']}, 失败: {r['datasets_failed']}")
    else:
        try:
            watcher.watch(interval=interval)
        except KeyboardInterrupt:
            console.print("\n[dim]已停止[/dim]")


@main.command("cache")
@click.option("--list", "-l", "list_cache", is_flag=True, help="List cached datasets")
@click.option("--stats", "-s", is_flag=True, help="Show cache statistics")
@click.option("--clear", is_flag=True, help="Clear all cache")
@click.option("--clear-expired", is_flag=True, help="Clear only expired entries")
@click.option("--invalidate", help="Invalidate cache for specific dataset")
def cache_cmd(list_cache: bool, stats: bool, clear: bool, clear_expired: bool, invalidate: str):
    """
    Manage the analysis cache.

    Example:
        datarecipe cache --list
        datarecipe cache --stats
        datarecipe cache --clear-expired
        datarecipe cache --invalidate Anthropic/hh-rlhf
    """
    from datarecipe.cache import AnalysisCache

    cache = AnalysisCache()

    if list_cache:
        entries = cache.list_entries()
        if not entries:
            console.print("[dim]缓存为空[/dim]")
            return

        console.print("\n[bold]缓存的数据集[/bold]\n")
        console.print("| 数据集 | 类型 | 样本 | 创建时间 | 状态 |")
        console.print("|--------|------|------|----------|------|")
        for e in entries:
            status = "[red]过期[/red]" if e.is_expired() else "[green]有效[/green]"
            console.print(
                f"| {e.dataset_id} | {e.dataset_type or '-'} | {e.sample_count} | "
                f"{e.created_at[:10]} | {status} |"
            )
        return

    if stats:
        s = cache.get_stats()
        console.print("\n[bold]缓存统计[/bold]\n")
        console.print(f"总条目: {s['total_entries']}")
        console.print(f"有效: {s['valid_entries']}")
        console.print(f"过期: {s['expired_entries']}")
        console.print(f"总大小: {s['total_size_mb']} MB")
        console.print(f"缓存目录: {s['cache_dir']}")
        return

    if clear:
        cache.clear_all(delete_files=True)
        console.print("[green]✓ 缓存已清空[/green]")
        return

    if clear_expired:
        count = cache.clear_expired(delete_files=True)
        console.print(f"[green]✓ 清理了 {count} 个过期条目[/green]")
        return

    if invalidate:
        cache.invalidate(invalidate, delete_files=False)
        console.print(f"[green]✓ 已使 {invalidate} 的缓存失效[/green]")
        return

    # Default: show stats
    s = cache.get_stats()
    console.print("\n[bold]缓存概览[/bold]\n")
    console.print(f"缓存条目: {s['total_entries']} ({s['valid_entries']} 有效, {s['expired_entries']} 过期)")
    console.print(f"占用空间: {s['total_size_mb']} MB")
    console.print("\n使用 --help 查看更多选项")


@main.command("knowledge")
@click.option("--report", "-r", is_flag=True, help="Generate knowledge report")
@click.option("--patterns", "-p", is_flag=True, help="Show top patterns")
@click.option("--benchmarks", "-b", is_flag=True, help="Show cost benchmarks")
@click.option("--trends", "-t", is_flag=True, help="Show recent trends")
@click.option("--recommend", help="Get recommendations for a dataset type")
@click.option("--output", "-o", help="Output path for report")
def knowledge_cmd(report: bool, patterns: bool, benchmarks: bool, trends: bool, recommend: str, output: str):
    """
    Query the knowledge base for patterns, benchmarks, and trends.

    Example:
        datarecipe knowledge --report
        datarecipe knowledge --patterns
        datarecipe knowledge --benchmarks
        datarecipe knowledge --recommend preference
    """
    from datarecipe.knowledge import KnowledgeBase

    kb = KnowledgeBase()

    if report:
        output_path = kb.export_report(output)
        console.print(f"[green]✓ 知识库报告已生成: {output_path}[/green]")
        return

    if patterns:
        console.print("\n[bold]Top 模式[/bold]\n")
        stats = kb.patterns.get_pattern_stats()

        if not stats["top_patterns"]:
            console.print("[dim]暂无数据，请先运行 deep-analyze[/dim]")
            return

        console.print("| 模式 | 类型 | 出现次数 |")
        console.print("|------|------|----------|")
        for p in stats["top_patterns"]:
            console.print(f"| {p['key']} | {p['type']} | {p['frequency']} |")

        console.print(f"\n总模式数: {stats['total_patterns']}")
        return

    if benchmarks:
        console.print("\n[bold]成本基准[/bold]\n")
        all_benchmarks = kb.trends.get_all_benchmarks()

        if not all_benchmarks:
            console.print("[dim]暂无数据，请先运行 deep-analyze[/dim]")
            return

        console.print("| 类型 | 平均成本 | 范围 | 人工% | 数据集数 |")
        console.print("|------|----------|------|-------|----------|")
        for dtype, bench in all_benchmarks.items():
            console.print(
                f"| {dtype} | ${bench.avg_total_cost:,.0f} | "
                f"${bench.min_cost:,.0f}-${bench.max_cost:,.0f} | "
                f"{bench.avg_human_percentage:.0f}% | {len(bench.datasets)} |"
            )
        return

    if trends:
        console.print("\n[bold]近期趋势 (30天)[/bold]\n")
        summary = kb.trends.get_trend_summary(30)

        if summary.get("datasets_analyzed", 0) == 0:
            console.print("[dim]暂无数据，请先运行 deep-analyze[/dim]")
            return

        console.print(f"分析数据集: {summary['datasets_analyzed']}")
        console.print(f"总复刻成本: ${summary['total_cost']:,.0f}")
        console.print(f"平均成本: ${summary['avg_cost_per_dataset']:,.0f}/数据集")

        if summary.get("type_distribution"):
            console.print("\n类型分布:")
            for dtype, count in summary["type_distribution"].items():
                console.print(f"  - {dtype}: {count}")
        return

    if recommend:
        console.print(f"\n[bold]{recommend} 类型推荐[/bold]\n")
        recs = kb.get_recommendations(recommend)

        if recs.get("cost_estimate"):
            ce = recs["cost_estimate"]
            console.print(f"成本估算: ${ce['avg_total']:,.0f} (范围 ${ce['range'][0]:,.0f}-${ce['range'][1]:,.0f})")
            console.print(f"人工占比: {ce['avg_human_percentage']:.0f}%")
            console.print(f"基于: {ce['based_on']} 个数据集")

        if recs.get("common_patterns"):
            console.print("\n常见模式:")
            for p in recs["common_patterns"][:5]:
                console.print(f"  - {p['pattern']} ({p['type']})")

        if recs.get("suggested_fields"):
            console.print(f"\n建议字段: {', '.join(recs['suggested_fields'][:5])}")
        return

    # Default: show summary
    console.print("\n[bold]知识库概览[/bold]\n")
    stats = kb.patterns.get_pattern_stats()
    console.print(f"总模式数: {stats['total_patterns']}")

    all_benchmarks = kb.trends.get_all_benchmarks()
    console.print(f"成本基准: {len(all_benchmarks)} 种类型")

    console.print("\n使用 --help 查看更多选项")


@main.command("analyze-spec")
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.option("--output-dir", "-o", default="./spec_output", help="Output directory")
@click.option("--size", "-s", default=100, type=int, help="Target dataset size (for cost estimation)")
@click.option("--region", "-r", default="china", help="Region for cost calculation (china/us)")
@click.option("--provider", "-p", default="anthropic", type=click.Choice(["anthropic", "openai"]), help="LLM provider")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode: output prompt, wait for JSON input from stdin")
@click.option("--from-json", "from_json", type=click.Path(exists=True), help="Load analysis from JSON file instead of using LLM")
def analyze_spec(file_path: str, output_dir: str, size: int, region: str, provider: str, interactive: bool, from_json: str):
    """
    Analyze a specification/requirements document and generate project artifacts.

    Supports PDF, Word (docx), images (png/jpg), and text files.
    Uses LLM to extract structured information and generate:
    - Annotation specification
    - Executive summary
    - Milestone plan
    - Cost breakdown
    - Industry benchmark comparison

    Three modes of operation:

    \b
    1. API mode (default): Uses LLM API to analyze document
       datarecipe analyze-spec requirements.pdf

    \b
    2. Interactive mode: For use within Claude Code/Desktop
       datarecipe analyze-spec requirements.pdf --interactive
       (Outputs prompt, waits for JSON on stdin)

    \b
    3. From JSON: Load pre-computed analysis
       datarecipe analyze-spec requirements.pdf --from-json analysis.json
    """
    import json
    import os
    import sys
    from pathlib import Path

    from datarecipe.analyzers.spec_analyzer import SpecAnalyzer
    from datarecipe.generators.spec_output import SpecOutputGenerator

    # Validate arguments
    if not file_path and not from_json:
        console.print("[red]错误: 需要提供文档路径或 --from-json 参数[/red]")
        return

    # Display header (to stderr in interactive mode)
    output = console if not interactive else Console(file=sys.stderr)

    output.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    output.print(f"[bold cyan]  DataRecipe 需求文档分析[/bold cyan]")
    output.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    if file_path:
        file_name = Path(file_path).name
        output.print(f"文档: [bold]{file_name}[/bold]")
    output.print(f"目标规模: [bold]{size}[/bold] 条")
    output.print(f"区域: [bold]{region}[/bold]")

    if interactive:
        output.print(f"模式: [bold]交互模式[/bold] (等待 stdin 输入)\n")
    elif from_json:
        output.print(f"模式: [bold]从 JSON 加载[/bold]\n")
    else:
        output.print(f"LLM: [bold]{provider}[/bold]\n")

    try:
        analyzer = SpecAnalyzer(provider=provider)
        analysis = None

        # Mode 1: From JSON file
        if from_json:
            output.print("[dim]📄 从 JSON 加载分析结果...[/dim]")
            with open(from_json, "r", encoding="utf-8") as f:
                extracted = json.load(f)

            # Parse document if provided (for metadata)
            doc = None
            if file_path:
                doc = analyzer.parse_document(file_path)
                if doc.has_images():
                    output.print(f"[green]✓ 文档解析完成 (包含 {len(doc.images)} 张图片)[/green]")
                else:
                    output.print(f"[green]✓ 文档解析完成[/green]")

            analysis = analyzer.create_analysis_from_json(extracted, doc)
            output.print(f"[green]✓ 加载完成: {analysis.project_name or '未命名项目'}[/green]")

        # Mode 2: Interactive mode
        elif interactive:
            output.print("[dim]📄 解析文档...[/dim]")
            doc = analyzer.parse_document(file_path)

            if doc.has_images():
                output.print(f"[green]✓ 文档解析完成 (包含 {len(doc.images)} 张图片)[/green]")
            else:
                output.print(f"[green]✓ 文档解析完成[/green]")

            # Output prompt to stdout
            prompt = analyzer.get_extraction_prompt(doc)
            output.print("\n[bold yellow]=" * 60 + "[/bold yellow]")
            output.print("[bold yellow]请将以下内容交给 LLM 分析，然后输入 JSON 结果：[/bold yellow]")
            output.print("[bold yellow]=" * 60 + "[/bold yellow]\n")

            # Print prompt to stdout (for piping to LLM)
            print(prompt)

            output.print("\n[bold yellow]=" * 60 + "[/bold yellow]")
            output.print("[bold yellow]请输入 LLM 返回的 JSON (以空行结束)：[/bold yellow]")
            output.print("[bold yellow]=" * 60 + "[/bold yellow]\n")

            # Read JSON from stdin
            json_lines = []
            try:
                for line in sys.stdin:
                    if line.strip() == "":
                        break
                    json_lines.append(line)
            except EOFError:
                pass

            json_text = "".join(json_lines)
            if not json_text.strip():
                output.print("[red]错误: 未收到 JSON 输入[/red]")
                return

            # Parse JSON
            try:
                # Try to extract JSON from markdown code block
                import re
                json_match = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_match = re.search(r"\{.*\}", json_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)

                extracted = json.loads(json_text)
                analysis = analyzer.create_analysis_from_json(extracted, doc)
                output.print(f"[green]✓ JSON 解析成功: {analysis.project_name or '未命名项目'}[/green]")
            except json.JSONDecodeError as e:
                output.print(f"[red]错误: JSON 解析失败 - {e}[/red]")
                return

        # Mode 3: API mode (default)
        else:
            output.print("[dim]📄 解析文档...[/dim]")
            analysis = analyzer.analyze(file_path)

            if analysis.has_images:
                output.print(f"[green]✓ 文档解析完成 (包含 {analysis.image_count} 张图片)[/green]")
            else:
                output.print(f"[green]✓ 文档解析完成[/green]")

            output.print("[dim]🤖 使用 LLM 提取结构化信息...[/dim]")
            if analysis.project_name:
                output.print(f"[green]✓ 识别项目: {analysis.project_name}[/green]")
                output.print(f"  类型: {analysis.dataset_type or 'unknown'}")
                output.print(f"  难度: {analysis.estimated_difficulty or 'unknown'}")
                output.print(f"  人工占比: {analysis.estimated_human_percentage:.0f}%")
            else:
                output.print("[yellow]⚠ LLM 提取信息有限，将使用默认值[/yellow]")

        # Step 3: LLM Enhancement (optional, enriches document quality)
        enhanced_context = None
        try:
            from datarecipe.generators.llm_enhancer import LLMEnhancer
            enhance_mode = "api" if not interactive else "interactive"
            enhancer = LLMEnhancer(mode=enhance_mode, provider=provider)
            enhanced_context = enhancer.enhance(
                dataset_id=analysis.project_name or "spec_analysis",
                dataset_type=analysis.dataset_type or "unknown",
                domain=analysis.estimated_domain or "通用",
                difficulty=analysis.estimated_difficulty or "medium",
                human_percentage=analysis.estimated_human_percentage,
                total_cost=0,
            )
            if enhanced_context and enhanced_context.generated:
                output.print("[green]✓ LLM 增强完成[/green]")
        except Exception:
            pass

        # Step 4: Generate outputs
        output.print("[dim]📝 生成项目文档...[/dim]")
        generator = SpecOutputGenerator(output_dir=output_dir)
        result = generator.generate(
            analysis=analysis,
            target_size=size,
            region=region,
            enhanced_context=enhanced_context,
        )

        if not result.success:
            output.print(f"[red]错误: {result.error}[/red]")
            return

        output.print(f"[green]✓ 生成完成[/green]")

        # Display summary
        output.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        output.print(f"[bold cyan]  分析完成[/bold cyan]")
        output.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        output.print(f"[bold]生成的文件:[/bold]")
        for fname in result.files_generated:
            fpath = os.path.join(result.output_dir, fname)
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath)
                if fsize > 1024:
                    size_str = f"{fsize / 1024:.1f}KB"
                else:
                    size_str = f"{fsize}B"
                icon = "📊" if fname.endswith(".json") else "📄" if fname.endswith(".md") else "📑"
                output.print(f"  {icon} {fname} ({size_str})")

        output.print(f"\n[bold]输出目录:[/bold] [cyan]{result.output_dir}[/cyan]")

        # Key files
        output.print(f"\n[bold]核心产出:[/bold]")
        output.print(f"  📄 执行摘要: [cyan]{result.output_dir}/01_决策参考/EXECUTIVE_SUMMARY.md[/cyan]")
        output.print(f"  📋 里程碑计划: [cyan]{result.output_dir}/02_项目管理/MILESTONE_PLAN.md[/cyan]")
        output.print(f"  📝 标注规范: [cyan]{result.output_dir}/03_标注规范/ANNOTATION_SPEC.md[/cyan]")

    except FileNotFoundError as e:
        output.print(f"[red]错误: 文件未找到 - {e}[/red]")
    except ValueError as e:
        output.print(f"[red]错误: {e}[/red]")
    except ImportError as e:
        output.print(f"[red]错误: 缺少依赖 - {e}[/red]")
        output.print("[dim]请安装所需依赖: pip install anthropic pymupdf python-docx[/dim]")
    except Exception as e:
        output.print(f"[red]错误: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
