"""Dataset catalog with known datasets and their relationships.

This catalog provides:
- Pre-populated list of well-known datasets
- Similarity relationships between datasets
- Industry benchmark data for cost comparisons
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DatasetCategory(Enum):
    """Dataset categories."""

    PREFERENCE = "preference"
    EVALUATION = "evaluation"
    SFT = "sft"
    SWE_BENCH = "swe_bench"
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    SAFETY = "safety"


@dataclass
class DatasetInfo:
    """Information about a known dataset."""

    dataset_id: str
    category: DatasetCategory
    description: str = ""
    org: str = ""
    size: int = 0  # Approximate sample count
    year: int = 2024

    # Relationships
    similar_to: list[str] = field(default_factory=list)  # Dataset IDs
    derived_from: list[str] = field(default_factory=list)

    # Cost benchmark (if known)
    estimated_cost: float = 0  # USD to reproduce
    estimated_hours: float = 0  # Human hours

    # Tags for search
    tags: list[str] = field(default_factory=list)

    # Quality indicators
    citation_count: int = 0
    download_count: int = 0


@dataclass
class IndustryBenchmark:
    """Industry benchmark for cost comparison."""

    category: str
    description: str

    # Cost per sample benchmarks
    min_cost_per_sample: float = 0.0
    avg_cost_per_sample: float = 0.0
    max_cost_per_sample: float = 0.0

    # Total project benchmarks
    typical_project_size: int = 1000  # samples
    typical_total_cost: float = 0.0
    typical_duration_days: int = 30

    # Human work ratio
    avg_human_percentage: float = 70.0

    # Source reference
    source: str = "Industry estimate"
    year: int = 2024


# ============================================================
# KNOWN DATASETS CATALOG
# ============================================================

KNOWN_DATASETS: dict[str, DatasetInfo] = {
    # ----- PREFERENCE / RLHF -----
    "Anthropic/hh-rlhf": DatasetInfo(
        dataset_id="Anthropic/hh-rlhf",
        category=DatasetCategory.PREFERENCE,
        description="Human preference data for helpful and harmless AI assistants",
        org="Anthropic",
        size=170000,
        year=2022,
        similar_to=["OpenAI/summarize_from_feedback", "stanfordnlp/SHP", "Dahoas/rm-static"],
        tags=["rlhf", "preference", "safety", "alignment", "helpfulness"],
        citation_count=500,
        download_count=100000,
        estimated_cost=500000,  # Industry estimate
        estimated_hours=10000,
    ),
    "OpenAI/summarize_from_feedback": DatasetInfo(
        dataset_id="OpenAI/summarize_from_feedback",
        category=DatasetCategory.PREFERENCE,
        description="Human feedback for summarization quality",
        org="OpenAI",
        size=93000,
        year=2020,
        similar_to=["Anthropic/hh-rlhf", "CarperAI/openai_summarize_comparisons"],
        tags=["rlhf", "preference", "summarization"],
        citation_count=800,
    ),
    "stanfordnlp/SHP": DatasetInfo(
        dataset_id="stanfordnlp/SHP",
        category=DatasetCategory.PREFERENCE,
        description="Stanford Human Preferences dataset from Reddit",
        org="Stanford",
        size=385000,
        year=2023,
        similar_to=["Anthropic/hh-rlhf", "OpenAI/summarize_from_feedback"],
        tags=["rlhf", "preference", "reddit", "natural"],
        citation_count=100,
    ),
    "Dahoas/rm-static": DatasetInfo(
        dataset_id="Dahoas/rm-static",
        category=DatasetCategory.PREFERENCE,
        description="Reward model training dataset",
        org="Dahoas",
        size=76000,
        year=2023,
        similar_to=["Anthropic/hh-rlhf"],
        tags=["rlhf", "reward-model"],
    ),
    "argilla/ultrafeedback-binarized-preferences": DatasetInfo(
        dataset_id="argilla/ultrafeedback-binarized-preferences",
        category=DatasetCategory.PREFERENCE,
        description="UltraFeedback dataset with binarized preferences",
        org="Argilla",
        size=60000,
        year=2023,
        similar_to=["Anthropic/hh-rlhf", "openbmb/UltraFeedback"],
        tags=["rlhf", "dpo", "preference"],
    ),
    # ----- EVALUATION / BENCHMARK -----
    "cais/mmlu": DatasetInfo(
        dataset_id="cais/mmlu",
        category=DatasetCategory.EVALUATION,
        description="Massive Multitask Language Understanding benchmark",
        org="CAIS",
        size=15000,
        year=2021,
        similar_to=["lukaemon/mmlu", "tasksource/mmlu"],
        tags=["evaluation", "benchmark", "knowledge", "multitask"],
        citation_count=2000,
        download_count=500000,
        estimated_cost=200000,
        estimated_hours=5000,
    ),
    "Rowan/hellaswag": DatasetInfo(
        dataset_id="Rowan/hellaswag",
        category=DatasetCategory.EVALUATION,
        description="Commonsense reasoning benchmark",
        org="AI2",
        size=40000,
        year=2019,
        similar_to=["winogrande", "piqa"],
        tags=["evaluation", "commonsense", "reasoning"],
        citation_count=1000,
    ),
    "truthful_qa": DatasetInfo(
        dataset_id="truthful_qa",
        category=DatasetCategory.EVALUATION,
        description="Truthfulness evaluation benchmark",
        org="Various",
        size=817,
        year=2022,
        similar_to=["cais/mmlu"],
        tags=["evaluation", "truthfulness", "safety"],
        citation_count=500,
    ),
    "gsm8k": DatasetInfo(
        dataset_id="gsm8k",
        category=DatasetCategory.EVALUATION,
        description="Grade school math benchmark",
        org="OpenAI",
        size=8500,
        year=2021,
        similar_to=["MATH", "competition_math"],
        tags=["evaluation", "math", "reasoning"],
        citation_count=800,
    ),
    "HuggingFaceH4/mt_bench_prompts": DatasetInfo(
        dataset_id="HuggingFaceH4/mt_bench_prompts",
        category=DatasetCategory.EVALUATION,
        description="Multi-turn conversation benchmark prompts",
        org="HuggingFace",
        size=80,
        year=2023,
        similar_to=["lmsys/chatbot_arena_conversations"],
        tags=["evaluation", "chat", "multi-turn"],
    ),
    # ----- SFT / INSTRUCTION -----
    "tatsu-lab/alpaca": DatasetInfo(
        dataset_id="tatsu-lab/alpaca",
        category=DatasetCategory.SFT,
        description="Stanford Alpaca instruction-following dataset",
        org="Stanford",
        size=52000,
        year=2023,
        similar_to=["databricks/dolly-15k", "yahma/alpaca-cleaned"],
        tags=["sft", "instruction", "synthetic"],
        citation_count=1500,
        download_count=300000,
        estimated_cost=1000,  # Synthetic, low cost
        estimated_hours=100,
    ),
    "databricks/dolly-15k": DatasetInfo(
        dataset_id="databricks/dolly-15k",
        category=DatasetCategory.SFT,
        description="Human-written instruction-following dataset",
        org="Databricks",
        size=15000,
        year=2023,
        similar_to=["tatsu-lab/alpaca", "OpenAssistant/oasst1"],
        tags=["sft", "instruction", "human-written"],
        citation_count=300,
        estimated_cost=100000,
        estimated_hours=3000,
    ),
    "OpenAssistant/oasst1": DatasetInfo(
        dataset_id="OpenAssistant/oasst1",
        category=DatasetCategory.SFT,
        description="Open Assistant conversation dataset",
        org="LAION",
        size=161000,
        year=2023,
        similar_to=["databricks/dolly-15k", "sharegpt"],
        tags=["sft", "chat", "multilingual", "crowd-sourced"],
        citation_count=200,
        estimated_cost=300000,
        estimated_hours=8000,
    ),
    "HuggingFaceH4/ultrachat_200k": DatasetInfo(
        dataset_id="HuggingFaceH4/ultrachat_200k",
        category=DatasetCategory.SFT,
        description="High-quality subset of UltraChat",
        org="HuggingFace",
        size=200000,
        year=2023,
        similar_to=["stingning/ultrachat", "OpenAssistant/oasst1"],
        tags=["sft", "chat", "synthetic"],
    ),
    # ----- CODE / SWE -----
    "princeton-nlp/SWE-bench": DatasetInfo(
        dataset_id="princeton-nlp/SWE-bench",
        category=DatasetCategory.SWE_BENCH,
        description="Software engineering benchmark from GitHub issues",
        org="Princeton",
        size=2300,
        year=2023,
        similar_to=["bigcode/humanevalpack", "codeparrot/apps"],
        tags=["code", "software-engineering", "github", "evaluation"],
        citation_count=100,
        download_count=50000,
        estimated_cost=150000,
        estimated_hours=2000,
    ),
    "bigcode/the-stack": DatasetInfo(
        dataset_id="bigcode/the-stack",
        category=DatasetCategory.CODE,
        description="Large-scale source code dataset",
        org="BigCode",
        size=6400000000,  # 6.4TB of code
        year=2022,
        similar_to=["codeparrot/github-code"],
        tags=["code", "pretraining", "multilingual"],
        citation_count=300,
    ),
    "openai_humaneval": DatasetInfo(
        dataset_id="openai_humaneval",
        category=DatasetCategory.CODE,
        description="Python programming benchmark",
        org="OpenAI",
        size=164,
        year=2021,
        similar_to=["mbpp", "bigcode/humanevalpack"],
        tags=["code", "evaluation", "python"],
        citation_count=1500,
    ),
    # ----- REASONING -----
    "allenai/ai2_arc": DatasetInfo(
        dataset_id="allenai/ai2_arc",
        category=DatasetCategory.REASONING,
        description="AI2 Reasoning Challenge",
        org="AI2",
        size=7800,
        year=2018,
        similar_to=["cais/mmlu", "Rowan/hellaswag"],
        tags=["reasoning", "science", "evaluation"],
        citation_count=600,
    ),
    "microsoft/orca-math-word-problems-200k": DatasetInfo(
        dataset_id="microsoft/orca-math-word-problems-200k",
        category=DatasetCategory.REASONING,
        description="Math word problems for training",
        org="Microsoft",
        size=200000,
        year=2024,
        similar_to=["gsm8k", "MATH"],
        tags=["math", "reasoning", "synthetic"],
    ),
    # ----- SAFETY -----
    "PKU-Alignment/PKU-SafeRLHF": DatasetInfo(
        dataset_id="PKU-Alignment/PKU-SafeRLHF",
        category=DatasetCategory.SAFETY,
        description="Safe RLHF dataset with safety labels",
        org="PKU",
        size=330000,
        year=2023,
        similar_to=["Anthropic/hh-rlhf"],
        tags=["safety", "rlhf", "alignment"],
    ),
}


# ============================================================
# INDUSTRY BENCHMARKS
# ============================================================

INDUSTRY_BENCHMARKS: dict[str, IndustryBenchmark] = {
    "preference": IndustryBenchmark(
        category="preference",
        description="RLHF/DPO 偏好标注数据",
        min_cost_per_sample=1.0,
        avg_cost_per_sample=3.0,
        max_cost_per_sample=10.0,
        typical_project_size=10000,
        typical_total_cost=30000,
        typical_duration_days=60,
        avg_human_percentage=85.0,
        source="Based on Anthropic hh-rlhf, SHP, and industry reports",
    ),
    "evaluation": IndustryBenchmark(
        category="evaluation",
        description="模型评测基准数据",
        min_cost_per_sample=5.0,
        avg_cost_per_sample=15.0,
        max_cost_per_sample=50.0,
        typical_project_size=1000,
        typical_total_cost=15000,
        typical_duration_days=45,
        avg_human_percentage=90.0,
        source="Based on MMLU, HellaSwag creation costs",
    ),
    "sft": IndustryBenchmark(
        category="sft",
        description="监督微调指令数据",
        min_cost_per_sample=0.5,
        avg_cost_per_sample=2.0,
        max_cost_per_sample=8.0,
        typical_project_size=15000,
        typical_total_cost=30000,
        typical_duration_days=45,
        avg_human_percentage=70.0,
        source="Based on Dolly-15k, OASST1 creation costs",
    ),
    "swe_bench": IndustryBenchmark(
        category="swe_bench",
        description="软件工程评测数据",
        min_cost_per_sample=20.0,
        avg_cost_per_sample=60.0,
        max_cost_per_sample=150.0,
        typical_project_size=500,
        typical_total_cost=30000,
        typical_duration_days=90,
        avg_human_percentage=95.0,
        source="Based on SWE-bench creation costs",
    ),
    "chat": IndustryBenchmark(
        category="chat",
        description="对话数据",
        min_cost_per_sample=0.3,
        avg_cost_per_sample=1.5,
        max_cost_per_sample=5.0,
        typical_project_size=50000,
        typical_total_cost=75000,
        typical_duration_days=60,
        avg_human_percentage=75.0,
        source="Based on OpenAssistant, ShareGPT estimates",
    ),
    "code": IndustryBenchmark(
        category="code",
        description="代码生成/理解数据",
        min_cost_per_sample=2.0,
        avg_cost_per_sample=8.0,
        max_cost_per_sample=30.0,
        typical_project_size=5000,
        typical_total_cost=40000,
        typical_duration_days=60,
        avg_human_percentage=80.0,
        source="Based on HumanEval, CodeContests estimates",
    ),
    "reasoning": IndustryBenchmark(
        category="reasoning",
        description="推理能力评测数据",
        min_cost_per_sample=3.0,
        avg_cost_per_sample=10.0,
        max_cost_per_sample=40.0,
        typical_project_size=2000,
        typical_total_cost=20000,
        typical_duration_days=45,
        avg_human_percentage=85.0,
        source="Based on ARC, GSM8K creation costs",
    ),
    "safety": IndustryBenchmark(
        category="safety",
        description="安全对齐数据",
        min_cost_per_sample=2.0,
        avg_cost_per_sample=5.0,
        max_cost_per_sample=15.0,
        typical_project_size=10000,
        typical_total_cost=50000,
        typical_duration_days=75,
        avg_human_percentage=90.0,
        source="Based on SafeRLHF, red-teaming estimates",
    ),
}


class DatasetCatalog:
    """Catalog of known datasets with search and similarity functions."""

    def __init__(self):
        self.datasets = KNOWN_DATASETS.copy()
        self.benchmarks = INDUSTRY_BENCHMARKS.copy()

    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get dataset info by ID."""
        # Normalize ID
        normalized = dataset_id.lower().replace("_", "-")

        # Direct lookup
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]

        # Try normalized lookup
        for key, info in self.datasets.items():
            if key.lower().replace("_", "-") == normalized:
                return info

        return None

    def find_similar_datasets(
        self,
        dataset_id: str = None,
        category: str = None,
        tags: list[str] = None,
        limit: int = 5,
    ) -> list[DatasetInfo]:
        """Find similar datasets.

        Args:
            dataset_id: Find datasets similar to this one
            category: Filter by category
            tags: Filter by tags
            limit: Maximum results

        Returns:
            List of similar DatasetInfo objects
        """
        results = []

        # If dataset_id provided, use its similar_to list first
        if dataset_id:
            dataset = self.get_dataset(dataset_id)
            if dataset and dataset.similar_to:
                for similar_id in dataset.similar_to:
                    similar = self.get_dataset(similar_id)
                    if similar:
                        results.append(similar)

        # Filter by category
        if category:
            category_enum = None
            try:
                category_enum = DatasetCategory(category.lower())
            except ValueError:
                pass

            if category_enum:
                for info in self.datasets.values():
                    if info.category == category_enum and info not in results:
                        if dataset_id is None or info.dataset_id != dataset_id:
                            results.append(info)

        # Filter by tags
        if tags:
            tags_lower = {t.lower() for t in tags}
            for info in self.datasets.values():
                info_tags = {t.lower() for t in info.tags}
                if tags_lower & info_tags and info not in results:
                    if dataset_id is None or info.dataset_id != dataset_id:
                        results.append(info)

        # Sort by citation count and download count
        results.sort(key=lambda x: x.citation_count + x.download_count // 1000, reverse=True)

        return results[:limit]

    def find_by_category(self, category: str, limit: int = 10) -> list[DatasetInfo]:
        """Find datasets by category."""
        try:
            category_enum = DatasetCategory(category.lower())
        except ValueError:
            return []

        results = [info for info in self.datasets.values() if info.category == category_enum]

        results.sort(key=lambda x: x.citation_count, reverse=True)
        return results[:limit]

    def find_by_tags(self, tags: list[str], limit: int = 10) -> list[DatasetInfo]:
        """Find datasets by tags."""
        tags_lower = {t.lower() for t in tags}

        results = []
        for info in self.datasets.values():
            info_tags = {t.lower() for t in info.tags}
            overlap = len(tags_lower & info_tags)
            if overlap > 0:
                results.append((info, overlap))

        results.sort(key=lambda x: (x[1], x[0].citation_count), reverse=True)
        return [info for info, _ in results[:limit]]

    def get_benchmark(self, category: str) -> Optional[IndustryBenchmark]:
        """Get industry benchmark for a category."""
        return self.benchmarks.get(category.lower())

    def get_all_benchmarks(self) -> dict[str, IndustryBenchmark]:
        """Get all industry benchmarks."""
        return self.benchmarks.copy()

    def compare_with_benchmark(
        self,
        category: str,
        sample_count: int,
        total_cost: float,
        human_percentage: float,
    ) -> dict[str, Any]:
        """Compare a project with industry benchmarks.

        Returns comparison analysis.
        """
        benchmark = self.get_benchmark(category)
        if not benchmark:
            return {"available": False, "reason": f"No benchmark for category: {category}"}

        cost_per_sample = total_cost / sample_count if sample_count > 0 else 0

        # Cost comparison
        if cost_per_sample < benchmark.min_cost_per_sample:
            cost_rating = "below_average"
            cost_explanation = "成本低于行业平均水平，可能存在质量风险"
        elif cost_per_sample <= benchmark.avg_cost_per_sample:
            cost_rating = "average"
            cost_explanation = "成本处于行业正常范围"
        elif cost_per_sample <= benchmark.max_cost_per_sample:
            cost_rating = "above_average"
            cost_explanation = "成本略高于行业平均，但在合理范围内"
        else:
            cost_rating = "high"
            cost_explanation = "成本显著高于行业基准，建议评估优化空间"

        # Human percentage comparison
        human_diff = human_percentage - benchmark.avg_human_percentage
        if abs(human_diff) < 10:
            human_rating = "typical"
            human_explanation = "人工比例符合行业惯例"
        elif human_diff > 0:
            human_rating = "more_human"
            human_explanation = "人工比例高于行业平均，质量有保障但成本较高"
        else:
            human_rating = "more_automated"
            human_explanation = "自动化程度高于行业平均，成本效率好"

        return {
            "available": True,
            "benchmark": {
                "category": category,
                "description": benchmark.description,
                "typical_cost_per_sample": {
                    "min": benchmark.min_cost_per_sample,
                    "avg": benchmark.avg_cost_per_sample,
                    "max": benchmark.max_cost_per_sample,
                },
                "typical_project": {
                    "size": benchmark.typical_project_size,
                    "cost": benchmark.typical_total_cost,
                    "duration_days": benchmark.typical_duration_days,
                },
                "avg_human_percentage": benchmark.avg_human_percentage,
                "source": benchmark.source,
            },
            "your_project": {
                "sample_count": sample_count,
                "total_cost": total_cost,
                "cost_per_sample": round(cost_per_sample, 2),
                "human_percentage": human_percentage,
            },
            "comparison": {
                "cost_rating": cost_rating,
                "cost_explanation": cost_explanation,
                "cost_vs_avg": f"{((cost_per_sample / benchmark.avg_cost_per_sample) - 1) * 100:+.0f}%"
                if benchmark.avg_cost_per_sample > 0
                else "N/A",
                "human_rating": human_rating,
                "human_explanation": human_explanation,
            },
            "similar_projects": [
                {
                    "name": ds.dataset_id,
                    "size": ds.size,
                    "estimated_cost": ds.estimated_cost,
                }
                for ds in self.find_by_category(category, limit=3)
                if ds.estimated_cost > 0
            ],
        }
