"""Deep analysis command and report generators."""

import json

import click

from datarecipe.cli._helpers import console


@click.command("deep-analyze")
@click.argument("dataset_id")
@click.option("--output-dir", "-o", default="./projects", help="Output directory")
@click.option("--sample-size", "-n", default=500, help="Number of samples to analyze")
@click.option(
    "--size", "-s", default=None, type=int, help="Target dataset size (for cost estimation)"
)
@click.option("--region", "-r", default="china", help="Region for cost calculation")
@click.option("--split", default=None, help="Dataset split (auto-detect if not specified)")
@click.option(
    "--use-llm",
    is_flag=True,
    default=False,
    help="Use LLM for intelligent analysis of unknown dataset types",
)
@click.option(
    "--llm-provider",
    default="anthropic",
    type=click.Choice(["anthropic", "openai"]),
    help="LLM provider for intelligent analysis",
)
@click.option(
    "--enhance-mode",
    default="auto",
    type=click.Choice(["auto", "interactive", "api"]),
    help="LLM enhancement mode: auto (detect), interactive (Claude Code/App), api (standalone)",
)
@click.option("--force", "-f", is_flag=True, help="Force re-analysis, ignore cache")
@click.option("--no-cache", is_flag=True, help="Don't use or update cache")
def deep_analyze(
    dataset_id: str,
    output_dir: str,
    sample_size: int,
    size: int,
    region: str,
    split: str,
    use_llm: bool,
    llm_provider: str,
    enhance_mode: str,
    force: bool,
    no_cache: bool,
):
    """
    Run comprehensive deep analysis on a dataset.

    Generates both JSON data files and a human-readable Markdown report.
    Supports HuggingFace dataset IDs and local files (CSV, Parquet, JSONL).

    Examples:
        datarecipe deep-analyze tencent/CL-bench -o ./output
        datarecipe deep-analyze ./data/train.csv -n 100
        datarecipe deep-analyze ./data/train.jsonl
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
            console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
            console.print("[bold cyan]  DataRecipe 深度逆向分析 (缓存命中)[/bold cyan]")
            console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")
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

            console.print("\n[dim]使用 --force 强制重新分析[/dim]")
            return

    # Display header
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print("[bold cyan]  DataRecipe 深度逆向分析[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")
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
            console.print("\n[dim]🔄 分析偏好模式...[/dim]")
            console.print(f"[green]✓ 偏好分析: {result.sample_count} 对[/green]")
        elif result.dataset_type == "swe_bench":
            console.print("\n[dim]🔧 分析 SWE 任务...[/dim]")
            console.print("[green]✓ SWE 分析完成[/green]")
        elif result.rubric_patterns > 0:
            console.print("\n[dim]📊 分析评分标准...[/dim]")
            console.print(f"[green]✓ 评分标准: {result.rubric_patterns} 种模式[/green]")

        if result.prompt_templates > 0:
            console.print("[dim]📝 提取 Prompt 模板...[/dim]")
            console.print(f"[green]✓ Prompt模板: {result.prompt_templates} 个[/green]")

        console.print("[dim]⚙️ 计算人机分配...[/dim]")
        console.print(
            f"[green]✓ 人机分配: 人工 {result.human_percentage:.0f}%, 机器 {100 - result.human_percentage:.0f}%[/green]"
        )

        console.print("\n[dim]📄 生成综合报告...[/dim]")
        console.print("[green]✓ 综合报告已保存[/green]")
        console.print("[dim]📋 生成复刻指南...[/dim]")
        console.print("[green]✓ 复刻指南已保存[/green]")
        console.print("[dim]📦 生成标准化摘要...[/dim]")
        console.print("[green]✓ 标准化摘要已保存 (Radar 兼容)[/green]")
        console.print("[dim]📚 更新知识库...[/dim]")
        console.print("[green]✓ 知识库已更新[/green]")
        console.print("[dim]💾 更新缓存...[/dim]")
        console.print("[green]✓ 缓存已更新[/green]")

        # Display summary
        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print("[bold cyan]  分析完成[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

        console.print("[bold]生成的文件:[/bold]")
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
        console.print("\n[bold]核心产出:[/bold]")
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
        lines.append(
            f"| **评分标准** | {rubrics_result.total_rubrics:,} 条，{rubrics_result.unique_patterns:,} 种独特模式 |"
        )
    if prompt_library:
        lines.append(f"| **Prompt模板** | {prompt_library.unique_count} 个去重后的系统提示模板 |")
    if strategy_result:
        lines.append(
            f"| **数据来源** | 混合策略（合成 {strategy_result.synthetic_score * 100:.0f}% + 改编 {strategy_result.modified_score * 100:.0f}% + 专业 {strategy_result.niche_score * 100:.0f}%） |"
        )

    lines.append(
        f"| **复现成本** | 约 ${allocation.total_cost:,.0f}（人工 ${allocation.total_human_cost:,.0f} + API ${allocation.total_machine_cost:,.0f}） |"
    )
    lines.append(
        f"| **人机分配** | 人工 {allocation.human_work_percentage:.0f}%，机器 {allocation.machine_work_percentage:.0f}% |"
    )
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
            for domain, count in sorted(prompt_library.domain_counts.items(), key=lambda x: -x[1])[
                :5
            ]:
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
        lines.append(
            f"| 🔧 合成生成 | {strategy_result.synthetic_score * 100:.1f}% | 使用 AI 模型生成虚构内容 |"
        )
        lines.append(
            f"| 📝 改编修改 | {strategy_result.modified_score * 100:.1f}% | 基于真实来源改编 |"
        )
        lines.append(
            f"| 🔬 专业领域 | {strategy_result.niche_score * 100:.1f}% | 专业/小众领域内容 |"
        )
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
        lines.append(
            f"| **{task.task_name}** | {dec} | {task.human_percentage:.0f}% | {task.human_hours:.1f}h | ${task.human_cost:,.0f} | ${task.machine_cost:.1f} |"
        )
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

    from datarecipe.analyzers.llm_dataset_analyzer import generate_llm_guide_section

    preference_pairs = preference_pairs or []
    preference_topics = preference_topics or {}
    preference_patterns = preference_patterns or {}
    swe_stats = swe_stats or {}

    lines = []
    lines.append(f"# 📋 {dataset_id} 复刻指南")
    lines.append("")

    if is_swe_dataset:
        lines.append(
            "> **这是一个软件工程评测数据集 (SWE-bench 风格)。本指南提供任务构建规范，帮助你构建类似的代码修复/功能实现评测集。**"
        )
    elif is_preference_dataset:
        lines.append(
            "> **这是一个 RLHF 偏好数据集。本指南提供偏好标注规范，帮助你构建类似的人类偏好数据。**"
        )
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
        lines.append("未检测到分类体系")
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
        lines.append(
            "每条数据包含一对回复：`chosen`（被选中的更好回复）和 `rejected`（被拒绝的较差回复）。"
        )
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
        lines.append(
            "这是一个 SWE-bench 风格的软件工程评测数据集，用于评估 AI 代码修复和功能实现能力。"
        )
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
                lines.append(
                    f"**示例 {i}** (`{ex.get('repo', 'unknown')}` - {ex.get('language', 'unknown')})"
                )
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
            lines.append(
                f"### 3.{list(system_prompts_by_domain.keys()).index(domain) + 1} {domain}"
            )
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
        lines.append("未提取到 System Prompt")
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
        lines.append("无可用示例")
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
