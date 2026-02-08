"""Batch analysis commands."""

import json

import click

from datarecipe.cli._helpers import console


@click.command("batch-from-radar")
@click.argument("radar_report")
@click.option("--output-dir", "-o", default="./projects", help="Output directory")
@click.option("--sample-size", "-n", default=200, help="Number of samples per dataset")
@click.option("--limit", "-l", default=0, type=int, help="Max datasets to analyze (0 = all)")
@click.option("--orgs", help="Filter by orgs (comma-separated)")
@click.option("--categories", help="Filter by categories (comma-separated)")
@click.option("--min-downloads", default=0, type=int, help="Minimum downloads")
@click.option("--use-llm", is_flag=True, help="Use LLM for unknown types")
@click.option("--region", "-r", default="china", help="Region for cost calculation")
@click.option(
    "--sort-by",
    type=click.Choice(["downloads", "name", "category"]),
    default="downloads",
    help="Sort datasets by",
)
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
    import os

    from datarecipe.integrations.radar import RadarIntegration

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print("[bold cyan]  DataRecipe 批量分析 (Radar 集成)[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

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

            from datarecipe.extractors import PromptExtractor, RubricsAnalyzer
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
                            schema_info[field] = {"type": type(value).__name__, "nested_type": None}
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
                ],
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

            console.print(
                f"[green]  ✓ 完成: {dataset_type or 'unknown'}, ${allocation.total_cost:,.0f}[/green]"
            )

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
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print("[bold cyan]  批量分析完成[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

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

        console.print("\n[bold]汇总统计:[/bold]")
        console.print(f"  总复刻成本: ${aggregate['total_reproduction_cost']['total']:,.0f}")
        console.print(f"  平均人工占比: {aggregate['avg_human_percentage']:.0f}%")
        console.print(f"  类型分布: {aggregate['type_distribution']}")

        console.print("\n[bold]输出文件:[/bold]")
        console.print(f"  📊 汇总报告: [cyan]{aggregate_path}[/cyan]")
        console.print(f"  📁 各数据集: [cyan]{output_dir}/<dataset>/recipe_summary.json[/cyan]")


@click.command("integrate-report")
@click.option("--radar-report", "-r", help="Path to Radar intel report JSON")
@click.option("--output-dir", "-o", default="./reports", help="Output directory")
@click.option("--recipe-dir", default="./projects", help="Recipe analysis directory")
@click.option("--start-date", help="Period start date (YYYY-MM-DD)")
@click.option("--end-date", help="Period end date (YYYY-MM-DD)")
@click.option(
    "--format", "-f", "formats", multiple=True, default=["md", "json"], help="Output formats"
)
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
        datarecipe integrate-report --recipe-dir ./projects
    """
    from datarecipe.reports import IntegratedReportGenerator

    console.print("\n[bold cyan]生成整合报告[/bold cyan]\n")

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
    for _fmt, path in paths.items():
        console.print(f"  📄 {path}")
