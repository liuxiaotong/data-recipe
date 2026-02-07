"""Cost calibration using historical knowledge base data.

Uses accumulated cost benchmarks to calibrate and improve
the accuracy of cost estimates for new datasets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CalibrationResult:
    """Result of cost calibration."""

    # Original estimates
    original_human_cost: float
    original_api_cost: float
    original_total: float

    # Calibrated estimates
    calibrated_human_cost: float
    calibrated_api_cost: float
    calibrated_total: float

    # Calibration factors
    human_calibration_factor: float = 1.0
    api_calibration_factor: float = 1.0

    # Confidence metrics
    confidence: float = 0.0  # 0-1, based on historical data availability
    based_on_datasets: int = 0
    similar_datasets: List[str] = field(default_factory=list)

    # Range estimates
    cost_range_low: float = 0.0
    cost_range_high: float = 0.0

    # Explanation
    calibration_method: str = "none"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original": {
                "human": round(self.original_human_cost, 2),
                "api": round(self.original_api_cost, 2),
                "total": round(self.original_total, 2),
            },
            "calibrated": {
                "human": round(self.calibrated_human_cost, 2),
                "api": round(self.calibrated_api_cost, 2),
                "total": round(self.calibrated_total, 2),
            },
            "factors": {
                "human": round(self.human_calibration_factor, 3),
                "api": round(self.api_calibration_factor, 3),
            },
            "confidence": round(self.confidence, 2),
            "based_on_datasets": self.based_on_datasets,
            "similar_datasets": self.similar_datasets[:5],  # Top 5
            "range": {
                "low": round(self.cost_range_low, 2),
                "high": round(self.cost_range_high, 2),
            },
            "method": self.calibration_method,
            "notes": self.notes,
        }


class CostCalibrator:
    """Calibrates cost estimates using historical data from knowledge base."""

    def __init__(self, knowledge_base=None):
        """Initialize calibrator.

        Args:
            knowledge_base: KnowledgeBase instance. If None, creates new one.
        """
        self.kb = knowledge_base
        if self.kb is None:
            try:
                from datarecipe.knowledge import KnowledgeBase

                self.kb = KnowledgeBase()
            except Exception:
                self.kb = None

    def calibrate(
        self,
        dataset_type: str,
        human_cost: float,
        api_cost: float,
        complexity_metrics: Optional[Any] = None,
        sample_count: int = 0,
    ) -> CalibrationResult:
        """Calibrate cost estimates using historical benchmarks.

        Args:
            dataset_type: Type of dataset (e.g., "preference", "sft", "evaluation")
            human_cost: Original human cost estimate
            api_cost: Original API cost estimate
            complexity_metrics: Optional ComplexityMetrics from analysis
            sample_count: Number of samples in dataset

        Returns:
            CalibrationResult with calibrated costs and confidence
        """
        total = human_cost + api_cost

        result = CalibrationResult(
            original_human_cost=human_cost,
            original_api_cost=api_cost,
            original_total=total,
            calibrated_human_cost=human_cost,
            calibrated_api_cost=api_cost,
            calibrated_total=total,
            calibration_method="none",
        )

        if self.kb is None:
            result.notes.append("知识库不可用，使用原始估算")
            result.cost_range_low = total * 0.7
            result.cost_range_high = total * 1.5
            return result

        # Get benchmark for this dataset type
        benchmark = self.kb.trends.get_cost_benchmark(dataset_type)

        if benchmark and len(benchmark.datasets) >= 3:
            # We have enough data to calibrate
            result = self._calibrate_with_benchmark(result, benchmark, human_cost, api_cost)
        elif benchmark and len(benchmark.datasets) >= 1:
            # Limited data, partial calibration
            result = self._calibrate_limited(result, benchmark, human_cost, api_cost)
        else:
            # No data for this type, try to find similar types
            result = self._calibrate_from_similar(result, dataset_type, human_cost, api_cost)

        return result

    def _calibrate_with_benchmark(
        self,
        result: CalibrationResult,
        benchmark,
        human_cost: float,
        api_cost: float,
    ) -> CalibrationResult:
        """Calibrate using sufficient benchmark data."""
        # Calculate calibration factors based on historical averages
        # We compare our base model's estimates with actual recorded costs

        # Use the benchmark's average costs to calibrate
        avg_total = benchmark.avg_total_cost
        avg_human = benchmark.avg_human_cost
        avg_api = benchmark.avg_api_cost
        n = len(benchmark.datasets)

        # Calculate factors (how much historical data differs from model predictions)
        # We assume the first recorded dataset used similar estimation methods
        # This is a simplification; ideally we'd track actual vs estimated costs

        # For now, use the historical average as the calibration target
        # and adjust proportionally based on the range
        result.calibrated_human_cost = human_cost
        result.calibrated_api_cost = api_cost
        result.calibrated_total = human_cost + api_cost

        # Use historical range for confidence interval
        result.cost_range_low = benchmark.min_cost
        result.cost_range_high = benchmark.max_cost

        # Confidence based on sample size
        result.confidence = min(0.3 + (n * 0.1), 0.9)
        result.based_on_datasets = n
        result.similar_datasets = benchmark.datasets[:5]

        result.calibration_method = "benchmark"
        result.notes = [
            f"基于 {n} 个 {benchmark.dataset_type} 类型数据集校准",
            f"历史平均成本: ${avg_total:.2f}",
            f"历史成本范围: ${benchmark.min_cost:.2f} - ${benchmark.max_cost:.2f}",
        ]

        # If current estimate is outside historical range, adjust
        current_total = human_cost + api_cost
        if current_total < benchmark.min_cost * 0.5:
            # Estimate seems too low
            adjustment = benchmark.min_cost / current_total if current_total > 0 else 1
            result.calibrated_human_cost = human_cost * adjustment
            result.calibrated_api_cost = api_cost * adjustment
            result.calibrated_total = result.calibrated_human_cost + result.calibrated_api_cost
            result.human_calibration_factor = adjustment
            result.api_calibration_factor = adjustment
            result.notes.append(f"估算偏低，调整系数: {adjustment:.2f}x")
        elif current_total > benchmark.max_cost * 2:
            # Estimate seems too high
            adjustment = benchmark.max_cost / current_total if current_total > 0 else 1
            result.calibrated_human_cost = human_cost * adjustment
            result.calibrated_api_cost = api_cost * adjustment
            result.calibrated_total = result.calibrated_human_cost + result.calibrated_api_cost
            result.human_calibration_factor = adjustment
            result.api_calibration_factor = adjustment
            result.notes.append(f"估算偏高，调整系数: {adjustment:.2f}x")

        return result

    def _calibrate_limited(
        self,
        result: CalibrationResult,
        benchmark,
        human_cost: float,
        api_cost: float,
    ) -> CalibrationResult:
        """Calibrate with limited benchmark data."""
        n = len(benchmark.datasets)

        result.cost_range_low = (human_cost + api_cost) * 0.6
        result.cost_range_high = (human_cost + api_cost) * 1.8

        # Low confidence due to limited data
        result.confidence = 0.2 + (n * 0.1)
        result.based_on_datasets = n
        result.similar_datasets = benchmark.datasets

        result.calibration_method = "limited"
        result.notes = [
            f"数据有限 ({n} 个数据集)，校准置信度较低",
            f"历史参考: ${benchmark.avg_total_cost:.2f}",
        ]

        return result

    def _calibrate_from_similar(
        self,
        result: CalibrationResult,
        dataset_type: str,
        human_cost: float,
        api_cost: float,
    ) -> CalibrationResult:
        """Try to calibrate using similar dataset types."""
        # Define type similarities
        type_similarities = {
            "preference": ["rlhf", "reward", "dpo"],
            "evaluation": ["benchmark", "test", "eval"],
            "sft": ["instruction", "chat", "finetune"],
            "code": ["programming", "coding", "software"],
        }

        # Check if any similar types have benchmarks
        all_benchmarks = self.kb.trends.get_all_benchmarks()
        similar_found = []

        for bench_type, bench in all_benchmarks.items():
            # Check direct similarity
            for base_type, similar_types in type_similarities.items():
                if dataset_type.lower() in similar_types or base_type == dataset_type.lower():
                    if bench_type.lower() in similar_types or bench_type.lower() == base_type:
                        similar_found.append((bench_type, bench))

        if similar_found:
            # Use the most relevant similar benchmark
            bench_type, bench = similar_found[0]
            result.cost_range_low = bench.min_cost * 0.8
            result.cost_range_high = bench.max_cost * 1.5
            result.confidence = 0.3
            result.based_on_datasets = len(bench.datasets)
            result.calibration_method = "similar_type"
            result.notes = [
                f"未找到 '{dataset_type}' 类型的历史数据",
                f"使用相似类型 '{bench_type}' 的数据参考",
            ]
        else:
            # No calibration possible
            result.cost_range_low = (human_cost + api_cost) * 0.5
            result.cost_range_high = (human_cost + api_cost) * 2.0
            result.confidence = 0.1
            result.calibration_method = "none"
            result.notes = [
                "无历史数据可用于校准",
                "建议：分析完成后将结果加入知识库以改善未来估算",
            ]

        return result

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get statistics about available calibration data."""
        if self.kb is None:
            return {"available": False, "reason": "知识库不可用"}

        benchmarks = self.kb.trends.get_all_benchmarks()
        if not benchmarks:
            return {
                "available": False,
                "reason": "无历史数据",
                "suggestion": "运行更多分析以积累校准数据",
            }

        stats = {
            "available": True,
            "total_datasets": sum(len(b.datasets) for b in benchmarks.values()),
            "types_covered": len(benchmarks),
            "benchmarks": {},
        }

        for dtype, bench in benchmarks.items():
            stats["benchmarks"][dtype] = {
                "count": len(bench.datasets),
                "avg_cost": round(bench.avg_total_cost, 2),
                "range": [round(bench.min_cost, 2), round(bench.max_cost, 2)],
                "calibration_ready": len(bench.datasets) >= 3,
            }

        return stats

    def suggest_next_datasets(self) -> List[str]:
        """Suggest dataset types that need more calibration data."""
        if self.kb is None:
            return []

        benchmarks = self.kb.trends.get_all_benchmarks()
        suggestions = []

        # Types with insufficient data
        for dtype, bench in benchmarks.items():
            if len(bench.datasets) < 3:
                suggestions.append(f"{dtype} (当前 {len(bench.datasets)} 个)")

        # Common types we don't have data for
        common_types = ["preference", "sft", "evaluation", "code", "math", "chat"]
        for ctype in common_types:
            if ctype not in benchmarks:
                suggestions.append(f"{ctype} (无数据)")

        return suggestions
