"""Knowledge base for storing and querying analysis patterns."""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class PatternEntry:
    """A discovered pattern from dataset analysis."""

    pattern_type: str  # "rubric", "prompt", "schema", "workflow"
    pattern_key: str  # Unique identifier for the pattern
    frequency: int = 0  # How many times this pattern was seen
    datasets: List[str] = field(default_factory=list)  # Datasets containing this pattern
    examples: List[str] = field(default_factory=list)  # Example instances
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBenchmark:
    """Cost benchmark data for a dataset type."""

    dataset_type: str
    sample_count: int = 0
    avg_human_cost: float = 0.0
    avg_api_cost: float = 0.0
    avg_total_cost: float = 0.0
    avg_human_percentage: float = 0.0
    min_cost: float = 0.0
    max_cost: float = 0.0
    datasets: List[str] = field(default_factory=list)


class PatternStore:
    """Store for discovered patterns across datasets."""

    def __init__(self, store_path: str = None):
        self.store_path = store_path or os.path.expanduser("~/.datarecipe/knowledge/patterns.json")
        self.patterns: Dict[str, PatternEntry] = {}
        self._load()

    def _load(self):
        """Load patterns from disk."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, entry in data.items():
                    self.patterns[key] = PatternEntry(**entry)
            except Exception:
                self.patterns = {}

    def _save(self):
        """Save patterns to disk."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: asdict(v) for k, v in self.patterns.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def add_pattern(
        self,
        pattern_type: str,
        pattern_key: str,
        dataset_id: str,
        example: str = None,
        metadata: Dict = None,
    ):
        """Add or update a pattern entry."""
        full_key = f"{pattern_type}:{pattern_key}"

        if full_key not in self.patterns:
            self.patterns[full_key] = PatternEntry(
                pattern_type=pattern_type,
                pattern_key=pattern_key,
            )

        entry = self.patterns[full_key]
        entry.frequency += 1

        if dataset_id not in entry.datasets:
            entry.datasets.append(dataset_id)

        if example and len(entry.examples) < 5:
            entry.examples.append(example)

        if metadata:
            entry.metadata.update(metadata)

        self._save()

    def get_top_patterns(
        self,
        pattern_type: str = None,
        limit: int = 20,
    ) -> List[PatternEntry]:
        """Get most common patterns."""
        patterns = list(self.patterns.values())

        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]

        patterns.sort(key=lambda x: x.frequency, reverse=True)
        return patterns[:limit]

    def find_patterns_for_dataset(self, dataset_id: str) -> List[PatternEntry]:
        """Find all patterns associated with a dataset."""
        return [p for p in self.patterns.values() if dataset_id in p.datasets]

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get overall pattern statistics."""
        stats = {
            "total_patterns": len(self.patterns),
            "by_type": defaultdict(int),
            "top_patterns": [],
        }

        for entry in self.patterns.values():
            stats["by_type"][entry.pattern_type] += 1

        top = self.get_top_patterns(limit=10)
        stats["top_patterns"] = [
            {"key": p.pattern_key, "type": p.pattern_type, "frequency": p.frequency}
            for p in top
        ]

        return dict(stats)


class TrendAnalyzer:
    """Analyze trends across analyzed datasets."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.expanduser("~/.datarecipe/knowledge")
        self.trends_path = os.path.join(self.data_dir, "trends.json")
        self.benchmarks_path = os.path.join(self.data_dir, "cost_benchmarks.json")
        self._load()

    def _load(self):
        """Load trend data."""
        self.trends = {}
        self.benchmarks: Dict[str, CostBenchmark] = {}

        if os.path.exists(self.trends_path):
            try:
                with open(self.trends_path, "r", encoding="utf-8") as f:
                    self.trends = json.load(f)
            except Exception:
                pass

        if os.path.exists(self.benchmarks_path):
            try:
                with open(self.benchmarks_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    self.benchmarks[k] = CostBenchmark(**v)
            except Exception:
                pass

    def _save(self):
        """Save trend data."""
        os.makedirs(self.data_dir, exist_ok=True)

        with open(self.trends_path, "w", encoding="utf-8") as f:
            json.dump(self.trends, f, indent=2, ensure_ascii=False)

        with open(self.benchmarks_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: asdict(v) for k, v in self.benchmarks.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def record_analysis(
        self,
        dataset_id: str,
        dataset_type: str,
        human_cost: float,
        api_cost: float,
        human_percentage: float,
        sample_count: int,
    ):
        """Record an analysis result for trend tracking."""
        # Update cost benchmarks
        if dataset_type not in self.benchmarks:
            self.benchmarks[dataset_type] = CostBenchmark(dataset_type=dataset_type)

        bench = self.benchmarks[dataset_type]
        total_cost = human_cost + api_cost

        # Incremental average update
        n = len(bench.datasets)
        if n == 0:
            bench.avg_human_cost = human_cost
            bench.avg_api_cost = api_cost
            bench.avg_total_cost = total_cost
            bench.avg_human_percentage = human_percentage
            bench.min_cost = total_cost
            bench.max_cost = total_cost
        else:
            bench.avg_human_cost = (bench.avg_human_cost * n + human_cost) / (n + 1)
            bench.avg_api_cost = (bench.avg_api_cost * n + api_cost) / (n + 1)
            bench.avg_total_cost = (bench.avg_total_cost * n + total_cost) / (n + 1)
            bench.avg_human_percentage = (bench.avg_human_percentage * n + human_percentage) / (n + 1)
            bench.min_cost = min(bench.min_cost, total_cost)
            bench.max_cost = max(bench.max_cost, total_cost)

        bench.sample_count += sample_count
        if dataset_id not in bench.datasets:
            bench.datasets.append(dataset_id)

        # Record trend data point
        date_key = datetime.now().strftime("%Y-%m-%d")
        if date_key not in self.trends:
            self.trends[date_key] = {
                "datasets_analyzed": 0,
                "types": defaultdict(int),
                "total_cost": 0,
            }

        self.trends[date_key]["datasets_analyzed"] += 1
        self.trends[date_key]["types"][dataset_type] = self.trends[date_key].get("types", {}).get(dataset_type, 0) + 1
        self.trends[date_key]["total_cost"] += total_cost

        self._save()

    def get_cost_benchmark(self, dataset_type: str) -> Optional[CostBenchmark]:
        """Get cost benchmark for a dataset type."""
        return self.benchmarks.get(dataset_type)

    def get_all_benchmarks(self) -> Dict[str, CostBenchmark]:
        """Get all cost benchmarks."""
        return self.benchmarks.copy()

    def get_trend_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get trend summary for recent days."""
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        recent_trends = {k: v for k, v in self.trends.items() if k >= cutoff}

        if not recent_trends:
            return {"period": f"last {days} days", "data": []}

        total_datasets = sum(t["datasets_analyzed"] for t in recent_trends.values())
        total_cost = sum(t["total_cost"] for t in recent_trends.values())
        type_counts = defaultdict(int)
        for t in recent_trends.values():
            for dtype, count in t.get("types", {}).items():
                type_counts[dtype] += count

        return {
            "period": f"last {days} days",
            "datasets_analyzed": total_datasets,
            "total_cost": round(total_cost, 2),
            "avg_cost_per_dataset": round(total_cost / total_datasets, 2) if total_datasets > 0 else 0,
            "type_distribution": dict(type_counts),
            "daily_data": [
                {"date": k, **v}
                for k, v in sorted(recent_trends.items())
            ],
        }


class KnowledgeBase:
    """Main knowledge base interface combining patterns and trends."""

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.expanduser("~/.datarecipe/knowledge")
        os.makedirs(self.base_dir, exist_ok=True)

        self.patterns = PatternStore(os.path.join(self.base_dir, "patterns.json"))
        self.trends = TrendAnalyzer(self.base_dir)

    def ingest_analysis(
        self,
        dataset_id: str,
        summary: "RecipeSummary",
        rubrics_result: Any = None,
        prompt_library: Any = None,
    ):
        """Ingest analysis results into the knowledge base.

        Args:
            dataset_id: Dataset identifier
            summary: RecipeSummary from analysis
            rubrics_result: Optional RubricsAnalysis result
            prompt_library: Optional PromptLibrary result
        """
        # Record trend data
        self.trends.record_analysis(
            dataset_id=dataset_id,
            dataset_type=summary.dataset_type or "unknown",
            human_cost=summary.reproduction_cost.get("human", 0),
            api_cost=summary.reproduction_cost.get("api", 0),
            human_percentage=summary.human_percentage,
            sample_count=summary.sample_count,
        )

        # Extract rubric patterns
        if rubrics_result and hasattr(rubrics_result, "verb_distribution"):
            for verb, count in rubrics_result.verb_distribution.items():
                self.patterns.add_pattern(
                    pattern_type="rubric_verb",
                    pattern_key=verb,
                    dataset_id=dataset_id,
                    metadata={"count_in_dataset": count},
                )

        # Extract schema patterns
        for field in summary.fields:
            self.patterns.add_pattern(
                pattern_type="schema_field",
                pattern_key=field,
                dataset_id=dataset_id,
            )

        # Record dataset type as pattern
        if summary.dataset_type:
            self.patterns.add_pattern(
                pattern_type="dataset_type",
                pattern_key=summary.dataset_type,
                dataset_id=dataset_id,
            )

    def get_similar_patterns(self, dataset_id: str, limit: int = 10) -> List[str]:
        """Find datasets with similar patterns."""
        target_patterns = set(
            p.pattern_key for p in self.patterns.find_patterns_for_dataset(dataset_id)
        )

        if not target_patterns:
            return []

        # Find datasets with overlapping patterns
        dataset_scores = defaultdict(int)
        for entry in self.patterns.patterns.values():
            if entry.pattern_key in target_patterns:
                for ds in entry.datasets:
                    if ds != dataset_id:
                        dataset_scores[ds] += 1

        # Sort by score
        sorted_datasets = sorted(dataset_scores.items(), key=lambda x: -x[1])
        return [ds for ds, _ in sorted_datasets[:limit]]

    def get_recommendations(self, dataset_type: str) -> Dict[str, Any]:
        """Get recommendations based on accumulated knowledge."""
        benchmark = self.trends.get_cost_benchmark(dataset_type)
        top_patterns = self.patterns.get_top_patterns(limit=10)

        recommendations = {
            "dataset_type": dataset_type,
            "cost_estimate": None,
            "common_patterns": [],
            "suggested_fields": [],
        }

        if benchmark:
            recommendations["cost_estimate"] = {
                "avg_total": round(benchmark.avg_total_cost, 2),
                "range": [round(benchmark.min_cost, 2), round(benchmark.max_cost, 2)],
                "avg_human_percentage": round(benchmark.avg_human_percentage, 1),
                "based_on": len(benchmark.datasets),
            }

        # Find patterns common for this dataset type
        type_datasets = benchmark.datasets if benchmark else []
        for pattern in self.patterns.patterns.values():
            overlap = set(pattern.datasets) & set(type_datasets)
            if len(overlap) > len(type_datasets) * 0.5:  # Pattern in >50% of datasets
                recommendations["common_patterns"].append({
                    "pattern": pattern.pattern_key,
                    "type": pattern.pattern_type,
                    "frequency": len(overlap),
                })

        # Suggest common fields
        field_patterns = [
            p for p in self.patterns.patterns.values()
            if p.pattern_type == "schema_field"
        ]
        field_patterns.sort(key=lambda x: x.frequency, reverse=True)
        recommendations["suggested_fields"] = [
            p.pattern_key for p in field_patterns[:10]
        ]

        return recommendations

    def export_report(self, output_path: str = None) -> str:
        """Export knowledge base report as Markdown."""
        output_path = output_path or os.path.join(self.base_dir, "KNOWLEDGE_REPORT.md")

        lines = []
        lines.append("# DataRecipe 知识库报告")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Pattern stats
        pattern_stats = self.patterns.get_pattern_stats()
        lines.append("## 模式统计")
        lines.append("")
        lines.append(f"- 总模式数: {pattern_stats['total_patterns']}")
        lines.append("")
        lines.append("### 按类型分布")
        lines.append("")
        for ptype, count in pattern_stats["by_type"].items():
            lines.append(f"- {ptype}: {count}")
        lines.append("")

        lines.append("### Top 模式")
        lines.append("")
        lines.append("| 模式 | 类型 | 出现次数 |")
        lines.append("|------|------|----------|")
        for p in pattern_stats["top_patterns"]:
            lines.append(f"| {p['key']} | {p['type']} | {p['frequency']} |")
        lines.append("")

        # Cost benchmarks
        lines.append("## 成本基准")
        lines.append("")
        benchmarks = self.trends.get_all_benchmarks()
        if benchmarks:
            lines.append("| 数据集类型 | 平均成本 | 成本范围 | 人工占比 | 样本数 |")
            lines.append("|------------|----------|----------|----------|--------|")
            for dtype, bench in benchmarks.items():
                lines.append(
                    f"| {dtype} | ${bench.avg_total_cost:,.0f} | "
                    f"${bench.min_cost:,.0f}-${bench.max_cost:,.0f} | "
                    f"{bench.avg_human_percentage:.0f}% | {len(bench.datasets)} |"
                )
        else:
            lines.append("*暂无数据*")
        lines.append("")

        # Trends
        lines.append("## 近期趋势")
        lines.append("")
        trend_summary = self.trends.get_trend_summary(30)
        lines.append(f"- 分析周期: {trend_summary['period']}")
        lines.append(f"- 分析数据集数: {trend_summary.get('datasets_analyzed', 0)}")
        lines.append(f"- 总复刻成本: ${trend_summary.get('total_cost', 0):,.0f}")
        lines.append(f"- 平均成本/数据集: ${trend_summary.get('avg_cost_per_dataset', 0):,.0f}")
        lines.append("")

        if trend_summary.get("type_distribution"):
            lines.append("### 类型分布")
            lines.append("")
            for dtype, count in trend_summary["type_distribution"].items():
                lines.append(f"- {dtype}: {count}")
        lines.append("")

        lines.append("---")
        lines.append("*由 DataRecipe 知识库自动生成*")

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
