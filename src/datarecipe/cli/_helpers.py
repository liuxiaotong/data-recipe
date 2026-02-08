"""CLI shared helpers."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

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
            raise ValueError(f"Output path '{output}' is outside allowed directory '{base_dir}'")

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
        lines.append("无法从现有元数据中确定生成方式。")
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
        lines.append("未在数据集文档中检测到教师模型。")
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
            method_name = method_type_map.get(
                method.method_type, method.method_type.replace("_", " ").title()
            )
            lines.append(f"**步骤 {i}：{method_name}**")
            if method.teacher_model:
                lines.append(f"- 教师模型：`{method.teacher_model}`")
            if method.platform:
                lines.append(f"- 标注平台：{method.platform}")
            if method.prompt_template_available:
                lines.append("- 提示词模板：✅ 可用")
            lines.append("")

    # Cost Estimation
    lines.append("## 💰 成本估算")
    lines.append("")

    if recipe.cost and recipe.cost.estimated_total_usd:
        if recipe.cost.confidence == "low":
            low = recipe.cost.estimated_total_usd * 0.5
            high = recipe.cost.estimated_total_usd * 1.5
            lines.append(f"**预估总成本：${low:,.0f} - ${high:,.0f}** (低置信度)")
        elif recipe.cost.confidence == "medium":
            low = recipe.cost.estimated_total_usd * 0.8
            high = recipe.cost.estimated_total_usd * 1.2
            lines.append(f"**预估总成本：${low:,.0f} - ${high:,.0f}** (中置信度)")
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
        lines.append("暂无成本估算数据。")
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
                translated = item_translation.get(item, item.replace("_", " ").title())
                lines.append(f"- {translated}")
            lines.append("")

        if recipe.reproducibility.missing:
            lines.append("#### ❌ 缺失的信息")
            lines.append("")
            for item in recipe.reproducibility.missing:
                translated = item_translation.get(item, item.replace("_", " ").title())
                lines.append(f"- {translated}")
            lines.append("")

        if recipe.reproducibility.notes:
            lines.append("#### 📝 备注")
            lines.append("")
            lines.append(recipe.reproducibility.notes)
            lines.append("")
    else:
        lines.append("暂无可复现性评估。")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        "> 由 [DataRecipe](https://github.com/yourusername/data-recipe) 生成 — AI 数据集成分分析器"
    )

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
            lines.append(
                f"   [green]✓ Available:[/green] {', '.join(recipe.reproducibility.available[:3])}"
            )
        if recipe.reproducibility.missing:
            lines.append(
                f"   [red]✗ Missing:[/red] {', '.join(recipe.reproducibility.missing[:3])}"
            )
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
