"""Integrated report generator combining Radar and Recipe analysis."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class DatasetEntry:
    """A dataset entry for the report."""

    dataset_id: str
    org: str = ""
    category: str = ""
    downloads: int = 0
    discovered_date: str = ""

    # Recipe analysis results
    analyzed: bool = False
    dataset_type: str = ""
    reproduction_cost: float = 0.0
    human_percentage: float = 0.0
    difficulty: str = ""
    sample_count: int = 0

    # Paths
    guide_path: str = ""
    report_path: str = ""


@dataclass
class WeeklyReport:
    """Weekly integrated report data."""

    period_start: str
    period_end: str
    generated_at: str = ""

    # Radar discoveries
    total_discovered: int = 0
    discoveries_by_org: dict[str, int] = field(default_factory=dict)
    discoveries_by_category: dict[str, int] = field(default_factory=dict)

    # Recipe analysis
    total_analyzed: int = 0
    analysis_by_type: dict[str, int] = field(default_factory=dict)
    total_reproduction_cost: float = 0.0
    avg_human_percentage: float = 0.0

    # Datasets
    datasets: list[DatasetEntry] = field(default_factory=list)

    # Insights
    insights: list[str] = field(default_factory=list)
    trends: list[str] = field(default_factory=list)


class IntegratedReportGenerator:
    """Generate integrated reports from Radar and Recipe data."""

    def __init__(
        self,
        radar_reports_dir: str = None,
        recipe_output_dir: str = "./projects",
    ):
        """Initialize generator.

        Args:
            radar_reports_dir: Directory containing Radar reports
            recipe_output_dir: Directory containing Recipe analysis outputs
        """
        self.radar_reports_dir = radar_reports_dir
        self.recipe_output_dir = recipe_output_dir

    def load_radar_report(self, report_path: str) -> dict[str, Any]:
        """Load a Radar intel report.

        Args:
            report_path: Path to intel_report_*.json

        Returns:
            Report data dict
        """
        with open(report_path, encoding="utf-8") as f:
            return json.load(f)

    def load_recipe_summary(self, dataset_id: str) -> dict[str, Any] | None:
        """Load Recipe analysis summary for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Summary data or None if not found
        """
        safe_name = dataset_id.replace("/", "_").replace("\\", "_")
        summary_path = os.path.join(self.recipe_output_dir, safe_name, "recipe_summary.json")

        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def generate_weekly_report(
        self,
        radar_report_path: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> WeeklyReport:
        """Generate a weekly integrated report.

        Args:
            radar_report_path: Path to Radar report (optional)
            start_date: Period start date (YYYY-MM-DD)
            end_date: Period end date (YYYY-MM-DD)

        Returns:
            WeeklyReport object
        """
        # Determine period
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start = datetime.now() - timedelta(days=7)
            start_date = start.strftime("%Y-%m-%d")

        report = WeeklyReport(
            period_start=start_date,
            period_end=end_date,
            generated_at=datetime.now().isoformat(),
        )

        datasets = []

        # Load Radar discoveries
        if radar_report_path and os.path.exists(radar_report_path):
            radar_data = self.load_radar_report(radar_report_path)

            for ds_data in radar_data.get("datasets", []):
                dataset_id = ds_data.get("id", "")
                org = dataset_id.split("/")[0] if "/" in dataset_id else ""

                entry = DatasetEntry(
                    dataset_id=dataset_id,
                    org=org,
                    category=ds_data.get("category", ""),
                    downloads=ds_data.get("downloads", 0),
                    discovered_date=ds_data.get("discovered_date", ""),
                )

                # Try to load Recipe analysis
                recipe_summary = self.load_recipe_summary(dataset_id)
                if recipe_summary:
                    entry.analyzed = True
                    entry.dataset_type = recipe_summary.get("dataset_type", "")
                    entry.reproduction_cost = recipe_summary.get("reproduction_cost", {}).get(
                        "total", 0
                    )
                    entry.human_percentage = recipe_summary.get("human_percentage", 0)
                    entry.difficulty = recipe_summary.get("difficulty", "")
                    entry.sample_count = recipe_summary.get("sample_count", 0)
                    entry.guide_path = recipe_summary.get("guide_path", "")
                    entry.report_path = recipe_summary.get("report_path", "")

                datasets.append(entry)

                # Update aggregates
                report.total_discovered += 1
                report.discoveries_by_org[org] = report.discoveries_by_org.get(org, 0) + 1
                if entry.category:
                    report.discoveries_by_category[entry.category] = (
                        report.discoveries_by_category.get(entry.category, 0) + 1
                    )

                if entry.analyzed:
                    report.total_analyzed += 1
                    report.total_reproduction_cost += entry.reproduction_cost
                    if entry.dataset_type:
                        report.analysis_by_type[entry.dataset_type] = (
                            report.analysis_by_type.get(entry.dataset_type, 0) + 1
                        )

        # Also scan recipe output directory for additional analyses
        if os.path.exists(self.recipe_output_dir):
            analyzed_ids = {d.dataset_id for d in datasets if d.analyzed}

            for name in os.listdir(self.recipe_output_dir):
                summary_path = os.path.join(self.recipe_output_dir, name, "recipe_summary.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, encoding="utf-8") as f:
                            summary = json.load(f)
                        dataset_id = summary.get("dataset_id", name.replace("_", "/", 1))

                        if dataset_id not in analyzed_ids:
                            org = dataset_id.split("/")[0] if "/" in dataset_id else ""
                            entry = DatasetEntry(
                                dataset_id=dataset_id,
                                org=org,
                                analyzed=True,
                                dataset_type=summary.get("dataset_type", ""),
                                reproduction_cost=summary.get("reproduction_cost", {}).get(
                                    "total", 0
                                ),
                                human_percentage=summary.get("human_percentage", 0),
                                difficulty=summary.get("difficulty", ""),
                                sample_count=summary.get("sample_count", 0),
                                guide_path=summary.get("guide_path", ""),
                                report_path=summary.get("report_path", ""),
                            )
                            datasets.append(entry)

                            report.total_analyzed += 1
                            report.total_reproduction_cost += entry.reproduction_cost
                            if entry.dataset_type:
                                report.analysis_by_type[entry.dataset_type] = (
                                    report.analysis_by_type.get(entry.dataset_type, 0) + 1
                                )

                    except Exception:
                        continue

        # Calculate averages
        analyzed_entries = [d for d in datasets if d.analyzed]
        if analyzed_entries:
            report.avg_human_percentage = sum(d.human_percentage for d in analyzed_entries) / len(
                analyzed_entries
            )

        # Sort datasets by downloads (descending)
        datasets.sort(key=lambda x: x.downloads, reverse=True)
        report.datasets = datasets

        # Generate insights
        report.insights = self._generate_insights(report)
        report.trends = self._generate_trends(report)

        return report

    def _generate_insights(self, report: WeeklyReport) -> list[str]:
        """Generate insights from report data."""
        insights = []

        # Top organizations
        if report.discoveries_by_org:
            top_org = max(report.discoveries_by_org.items(), key=lambda x: x[1])
            insights.append(f"本周最活跃组织: {top_org[0]} ({top_org[1]} 个数据集)")

        # Most common category
        if report.discoveries_by_category:
            top_cat = max(report.discoveries_by_category.items(), key=lambda x: x[1])
            insights.append(f"热门数据集类型: {top_cat[0]} ({top_cat[1]} 个)")

        # Analysis coverage
        if report.total_discovered > 0:
            coverage = report.total_analyzed / report.total_discovered * 100
            insights.append(
                f"分析覆盖率: {coverage:.0f}% ({report.total_analyzed}/{report.total_discovered})"
            )

        # Cost insights
        if report.total_analyzed > 0:
            avg_cost = report.total_reproduction_cost / report.total_analyzed
            insights.append(f"平均复刻成本: ${avg_cost:,.0f}")

        # Human percentage
        if report.avg_human_percentage > 0:
            insights.append(f"平均人工占比: {report.avg_human_percentage:.0f}%")

        return insights

    def _generate_trends(self, report: WeeklyReport) -> list[str]:
        """Generate trend observations."""
        trends = []

        # Type distribution trends
        if report.analysis_by_type:
            total = sum(report.analysis_by_type.values())
            for dtype, count in sorted(report.analysis_by_type.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                trends.append(f"{dtype}: {count} 个 ({pct:.0f}%)")

        return trends

    def to_markdown(self, report: WeeklyReport) -> str:
        """Convert report to Markdown format.

        Args:
            report: WeeklyReport object

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append("# AI 数据集周报")
        lines.append("")
        lines.append(f"> **周期**: {report.period_start} ~ {report.period_end}")
        lines.append(">")
        lines.append(f"> **生成时间**: {report.generated_at[:16].replace('T', ' ')}")
        lines.append(">")
        lines.append("> 由 DataRecipe + ai-dataset-radar 联合生成")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary
        lines.append("## 📊 执行摘要")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 发现数据集 | {report.total_discovered} |")
        lines.append(f"| 已分析 | {report.total_analyzed} |")
        lines.append(f"| 总复刻成本 | ${report.total_reproduction_cost:,.0f} |")
        lines.append(f"| 平均人工占比 | {report.avg_human_percentage:.0f}% |")
        lines.append("")

        # Insights
        if report.insights:
            lines.append("### 关键洞察")
            lines.append("")
            for insight in report.insights:
                lines.append(f"- {insight}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Discoveries by Organization
        if report.discoveries_by_org:
            lines.append("## 🏢 组织分布")
            lines.append("")
            lines.append("| 组织 | 数据集数 |")
            lines.append("|------|----------|")
            for org, count in sorted(report.discoveries_by_org.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"| {org} | {count} |")
            lines.append("")

        # Type Distribution
        if report.analysis_by_type:
            lines.append("## 📁 类型分布")
            lines.append("")
            lines.append("| 类型 | 数量 | 占比 |")
            lines.append("|------|------|------|")
            total = sum(report.analysis_by_type.values())
            for dtype, count in sorted(report.analysis_by_type.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"| {dtype} | {count} | {pct:.0f}% |")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Dataset Details
        lines.append("## 📋 数据集详情")
        lines.append("")

        # Analyzed datasets
        analyzed = [d for d in report.datasets if d.analyzed]
        if analyzed:
            lines.append("### 已分析")
            lines.append("")
            lines.append("| 数据集 | 类型 | 复刻成本 | 人工% | 难度 |")
            lines.append("|--------|------|----------|-------|------|")
            for d in analyzed[:20]:
                difficulty = d.difficulty.split()[0] if d.difficulty else "-"
                lines.append(
                    f"| [{d.dataset_id}]({d.guide_path}) | {d.dataset_type or '-'} | "
                    f"${d.reproduction_cost:,.0f} | {d.human_percentage:.0f}% | {difficulty} |"
                )
            if len(analyzed) > 20:
                lines.append(f"| ... | | | | 还有 {len(analyzed) - 20} 个 |")
            lines.append("")

        # Not analyzed
        not_analyzed = [d for d in report.datasets if not d.analyzed]
        if not_analyzed:
            lines.append("### 待分析")
            lines.append("")
            lines.append("| 数据集 | 组织 | 类别 | 下载量 |")
            lines.append("|--------|------|------|--------|")
            for d in not_analyzed[:10]:
                lines.append(
                    f"| {d.dataset_id} | {d.org} | {d.category or '-'} | {d.downloads:,} |"
                )
            if len(not_analyzed) > 10:
                lines.append(f"| ... | | | 还有 {len(not_analyzed) - 10} 个 |")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Cost Breakdown
        if analyzed:
            lines.append("## 💰 成本分析")
            lines.append("")

            # By type
            cost_by_type: dict[str, list[float]] = {}
            for d in analyzed:
                dtype = d.dataset_type or "unknown"
                if dtype not in cost_by_type:
                    cost_by_type[dtype] = []
                cost_by_type[dtype].append(d.reproduction_cost)

            lines.append("| 类型 | 平均成本 | 最低 | 最高 | 数量 |")
            lines.append("|------|----------|------|------|------|")
            for dtype, costs in sorted(cost_by_type.items(), key=lambda x: -sum(x[1])):
                avg = sum(costs) / len(costs)
                lines.append(
                    f"| {dtype} | ${avg:,.0f} | ${min(costs):,.0f} | ${max(costs):,.0f} | {len(costs)} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 报告由 DataRecipe 自动生成")

        return "\n".join(lines)

    def to_json(self, report: WeeklyReport) -> str:
        """Convert report to JSON format.

        Args:
            report: WeeklyReport object

        Returns:
            JSON string
        """
        data = {
            "period": {
                "start": report.period_start,
                "end": report.period_end,
            },
            "generated_at": report.generated_at,
            "summary": {
                "total_discovered": report.total_discovered,
                "total_analyzed": report.total_analyzed,
                "total_reproduction_cost": report.total_reproduction_cost,
                "avg_human_percentage": report.avg_human_percentage,
            },
            "distributions": {
                "by_org": report.discoveries_by_org,
                "by_category": report.discoveries_by_category,
                "by_type": report.analysis_by_type,
            },
            "insights": report.insights,
            "trends": report.trends,
            "datasets": [
                {
                    "id": d.dataset_id,
                    "org": d.org,
                    "category": d.category,
                    "downloads": d.downloads,
                    "analyzed": d.analyzed,
                    "type": d.dataset_type,
                    "cost": d.reproduction_cost,
                    "human_pct": d.human_percentage,
                    "difficulty": d.difficulty,
                }
                for d in report.datasets
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save_report(
        self,
        report: WeeklyReport,
        output_dir: str,
        formats: list[str] = None,
    ) -> dict[str, str]:
        """Save report to files.

        Args:
            report: WeeklyReport object
            output_dir: Output directory
            formats: List of formats ("md", "json")

        Returns:
            Dict of format -> file path
        """
        formats = formats or ["md", "json"]
        os.makedirs(output_dir, exist_ok=True)

        paths = {}
        date_str = report.period_end.replace("-", "")

        if "md" in formats:
            md_path = os.path.join(output_dir, f"weekly_report_{date_str}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.to_markdown(report))
            paths["md"] = md_path

        if "json" in formats:
            json_path = os.path.join(output_dir, f"weekly_report_{date_str}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(self.to_json(report))
            paths["json"] = json_path

        return paths
