"""Infrastructure commands: watch, cache, knowledge."""

import click

from datarecipe.cli._helpers import console


@click.command("watch")
@click.argument("watch_dir")
@click.option("--output-dir", "-o", default="./projects", help="Output directory")
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

    console.print("\n[bold cyan]DataRecipe Radar Watcher[/bold cyan]\n")
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
            console.print(
                f"[green]✓[/green] {dataset_id}: {result.get('type', 'unknown')}, ${result.get('cost', 0):,.0f}"
            )
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


@click.command("cache")
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
    console.print(
        f"缓存条目: {s['total_entries']} ({s['valid_entries']} 有效, {s['expired_entries']} 过期)"
    )
    console.print(f"占用空间: {s['total_size_mb']} MB")
    console.print("\n使用 --help 查看更多选项")


@click.command("knowledge")
@click.option("--report", "-r", is_flag=True, help="Generate knowledge report")
@click.option("--patterns", "-p", is_flag=True, help="Show top patterns")
@click.option("--benchmarks", "-b", is_flag=True, help="Show cost benchmarks")
@click.option("--trends", "-t", is_flag=True, help="Show recent trends")
@click.option("--recommend", help="Get recommendations for a dataset type")
@click.option("--output", "-o", help="Output path for report")
def knowledge_cmd(
    report: bool, patterns: bool, benchmarks: bool, trends: bool, recommend: str, output: str
):
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
            console.print(
                f"成本估算: ${ce['avg_total']:,.0f} (范围 ${ce['range'][0]:,.0f}-${ce['range'][1]:,.0f})"
            )
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
