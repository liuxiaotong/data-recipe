"""Phased cost model for dataset production.

Separates costs into three phases:
1. Design Phase (fixed costs)
2. Production Phase (variable costs)
3. Quality Phase (proportional costs)
"""

from dataclasses import dataclass, field
from enum import Enum

from datarecipe.constants import REGION_COST_MULTIPLIERS


class ProjectScale(Enum):
    """Project scale categories."""

    SMALL = "small"  # < 1,000 examples
    MEDIUM = "medium"  # 1,000 - 10,000 examples
    LARGE = "large"  # 10,000 - 100,000 examples
    ENTERPRISE = "enterprise"  # > 100,000 examples


@dataclass
class DesignPhaseCost:
    """Design phase costs (mostly fixed)."""

    schema_design: float = 0.0  # Schema and data model design
    guideline_writing: float = 0.0  # Annotation guidelines
    pilot_testing: float = 0.0  # Initial pilot with small sample
    tool_setup: float = 0.0  # Tooling and infrastructure

    @property
    def total(self) -> float:
        return self.schema_design + self.guideline_writing + self.pilot_testing + self.tool_setup

    def to_dict(self) -> dict:
        return {
            "schema_design": round(self.schema_design, 2),
            "guideline_writing": round(self.guideline_writing, 2),
            "pilot_testing": round(self.pilot_testing, 2),
            "tool_setup": round(self.tool_setup, 2),
            "total": round(self.total, 2),
        }


@dataclass
class ProductionPhaseCost:
    """Production phase costs (variable, per-sample)."""

    annotation_cost: float = 0.0  # Human annotation
    generation_cost: float = 0.0  # LLM generation (API)
    review_cost: float = 0.0  # Initial review
    infrastructure: float = 0.0  # Compute/storage

    # Per-sample metrics
    cost_per_sample: float = 0.0
    samples_count: int = 0

    @property
    def total(self) -> float:
        return self.annotation_cost + self.generation_cost + self.review_cost + self.infrastructure

    def to_dict(self) -> dict:
        return {
            "annotation": round(self.annotation_cost, 2),
            "generation": round(self.generation_cost, 2),
            "review": round(self.review_cost, 2),
            "infrastructure": round(self.infrastructure, 2),
            "total": round(self.total, 2),
            "cost_per_sample": round(self.cost_per_sample, 4),
            "samples": self.samples_count,
        }


@dataclass
class QualityPhaseCost:
    """Quality phase costs (proportional to production)."""

    qa_sampling: float = 0.0  # Quality assurance sampling
    rework: float = 0.0  # Rework/corrections
    expert_review: float = 0.0  # Expert spot checks
    final_validation: float = 0.0  # Final validation pass

    # Rates
    qa_rate: float = 0.2  # % of samples QA'd
    expected_rework_rate: float = 0.1  # % needing rework

    @property
    def total(self) -> float:
        return self.qa_sampling + self.rework + self.expert_review + self.final_validation

    def to_dict(self) -> dict:
        return {
            "qa_sampling": round(self.qa_sampling, 2),
            "rework": round(self.rework, 2),
            "expert_review": round(self.expert_review, 2),
            "final_validation": round(self.final_validation, 2),
            "total": round(self.total, 2),
            "qa_rate": self.qa_rate,
            "rework_rate": self.expected_rework_rate,
        }


@dataclass
class PhasedCostBreakdown:
    """Complete phased cost breakdown."""

    # Phase costs
    design: DesignPhaseCost = field(default_factory=DesignPhaseCost)
    production: ProductionPhaseCost = field(default_factory=ProductionPhaseCost)
    quality: QualityPhaseCost = field(default_factory=QualityPhaseCost)

    # Summary
    total_fixed: float = 0.0  # Design phase
    total_variable: float = 0.0  # Production phase
    total_proportional: float = 0.0  # Quality phase
    grand_total: float = 0.0

    # Project info
    scale: ProjectScale = ProjectScale.MEDIUM
    target_size: int = 0
    dataset_type: str = ""

    # Risk buffer
    contingency_rate: float = 0.15
    contingency_amount: float = 0.0

    def to_dict(self) -> dict:
        return {
            "phases": {
                "design": self.design.to_dict(),
                "production": self.production.to_dict(),
                "quality": self.quality.to_dict(),
            },
            "summary": {
                "fixed_costs": round(self.total_fixed, 2),
                "variable_costs": round(self.total_variable, 2),
                "proportional_costs": round(self.total_proportional, 2),
                "subtotal": round(
                    self.total_fixed + self.total_variable + self.total_proportional, 2
                ),
                "contingency": {
                    "rate": self.contingency_rate,
                    "amount": round(self.contingency_amount, 2),
                },
                "grand_total": round(self.grand_total, 2),
            },
            "project": {
                "scale": self.scale.value,
                "target_size": self.target_size,
                "dataset_type": self.dataset_type,
            },
        }


# Base costs by project scale
DESIGN_PHASE_BASE_COSTS = {
    ProjectScale.SMALL: {
        "schema_design": 500,
        "guideline_writing": 800,
        "pilot_testing": 300,
        "tool_setup": 200,
    },
    ProjectScale.MEDIUM: {
        "schema_design": 1000,
        "guideline_writing": 1500,
        "pilot_testing": 800,
        "tool_setup": 500,
    },
    ProjectScale.LARGE: {
        "schema_design": 2000,
        "guideline_writing": 3000,
        "pilot_testing": 1500,
        "tool_setup": 1000,
    },
    ProjectScale.ENTERPRISE: {
        "schema_design": 5000,
        "guideline_writing": 8000,
        "pilot_testing": 3000,
        "tool_setup": 3000,
    },
}

# Quality rates by quality requirement
QUALITY_RATES = {
    "basic": {"qa_rate": 0.10, "rework_rate": 0.05},
    "standard": {"qa_rate": 0.20, "rework_rate": 0.10},
    "high": {"qa_rate": 0.30, "rework_rate": 0.15},
    "expert": {"qa_rate": 0.50, "rework_rate": 0.20},
}


class PhasedCostModel:
    """Calculates costs broken down by project phase."""

    def __init__(self, region: str = "china"):
        """Initialize with region for labor costs.

        Args:
            region: Region for labor cost calculation
        """
        self.region = region

        self.labor_multipliers = REGION_COST_MULTIPLIERS
        self.labor_mult = self.labor_multipliers.get(region, 1.0)

    def calculate(
        self,
        target_size: int,
        dataset_type: str = "unknown",
        human_percentage: float = 50.0,
        api_cost_per_sample: float = 0.01,
        complexity_multiplier: float = 1.0,
        quality_requirement: str = "standard",
    ) -> PhasedCostBreakdown:
        """Calculate phased cost breakdown.

        Args:
            target_size: Target number of samples
            dataset_type: Type of dataset
            human_percentage: % of work done by humans (0-100)
            api_cost_per_sample: API cost per sample
            complexity_multiplier: Multiplier from complexity analysis
            quality_requirement: Quality level (basic/standard/high/expert)

        Returns:
            PhasedCostBreakdown with complete cost analysis
        """
        # Determine scale
        scale = self._determine_scale(target_size)

        # Create breakdown
        breakdown = PhasedCostBreakdown(
            scale=scale,
            target_size=target_size,
            dataset_type=dataset_type,
        )

        # 1. Design Phase (fixed costs)
        self._calculate_design_phase(breakdown, scale, complexity_multiplier)

        # 2. Production Phase (variable costs)
        self._calculate_production_phase(
            breakdown, target_size, human_percentage, api_cost_per_sample, complexity_multiplier
        )

        # 3. Quality Phase (proportional costs)
        self._calculate_quality_phase(
            breakdown, target_size, quality_requirement, human_percentage, complexity_multiplier
        )

        # Calculate totals
        breakdown.total_fixed = breakdown.design.total
        breakdown.total_variable = breakdown.production.total
        breakdown.total_proportional = breakdown.quality.total

        subtotal = breakdown.total_fixed + breakdown.total_variable + breakdown.total_proportional
        breakdown.contingency_amount = subtotal * breakdown.contingency_rate
        breakdown.grand_total = subtotal + breakdown.contingency_amount

        return breakdown

    def _determine_scale(self, target_size: int) -> ProjectScale:
        """Determine project scale from target size."""
        if target_size < 1000:
            return ProjectScale.SMALL
        elif target_size < 10000:
            return ProjectScale.MEDIUM
        elif target_size < 100000:
            return ProjectScale.LARGE
        else:
            return ProjectScale.ENTERPRISE

    def _calculate_design_phase(
        self,
        breakdown: PhasedCostBreakdown,
        scale: ProjectScale,
        complexity_mult: float,
    ) -> None:
        """Calculate design phase costs."""
        base = DESIGN_PHASE_BASE_COSTS[scale]

        breakdown.design.schema_design = base["schema_design"] * self.labor_mult * complexity_mult
        breakdown.design.guideline_writing = (
            base["guideline_writing"] * self.labor_mult * complexity_mult
        )
        breakdown.design.pilot_testing = base["pilot_testing"] * self.labor_mult
        breakdown.design.tool_setup = base["tool_setup"]  # Tool costs are region-independent

    def _calculate_production_phase(
        self,
        breakdown: PhasedCostBreakdown,
        target_size: int,
        human_percentage: float,
        api_cost_per_sample: float,
        complexity_mult: float,
    ) -> None:
        """Calculate production phase costs."""
        human_ratio = human_percentage / 100

        # Base annotation cost per sample (varies by region and complexity)
        base_annotation_per_sample = 0.50 * self.labor_mult * complexity_mult

        # Annotation cost (human work)
        breakdown.production.annotation_cost = (
            target_size * base_annotation_per_sample * human_ratio
        )

        # Generation cost (API)
        breakdown.production.generation_cost = target_size * api_cost_per_sample * (1 - human_ratio)

        # Review cost (10% of annotation)
        breakdown.production.review_cost = breakdown.production.annotation_cost * 0.1

        # Infrastructure (minimal for most projects)
        if target_size > 10000:
            breakdown.production.infrastructure = 100 + (target_size / 10000) * 50
        else:
            breakdown.production.infrastructure = 50

        # Per-sample metrics
        breakdown.production.samples_count = target_size
        breakdown.production.cost_per_sample = (
            breakdown.production.total / target_size if target_size > 0 else 0
        )

    def _calculate_quality_phase(
        self,
        breakdown: PhasedCostBreakdown,
        target_size: int,
        quality_requirement: str,
        human_percentage: float,
        complexity_mult: float,
    ) -> None:
        """Calculate quality phase costs."""
        rates = QUALITY_RATES.get(quality_requirement, QUALITY_RATES["standard"])

        breakdown.quality.qa_rate = rates["qa_rate"]
        breakdown.quality.expected_rework_rate = rates["rework_rate"]

        # QA sampling cost
        qa_samples = int(target_size * rates["qa_rate"])
        base_qa_cost_per_sample = 0.30 * self.labor_mult * complexity_mult
        breakdown.quality.qa_sampling = qa_samples * base_qa_cost_per_sample

        # Rework cost (re-annotation of failed samples)
        rework_samples = int(target_size * rates["rework_rate"])
        base_annotation_per_sample = 0.50 * self.labor_mult * complexity_mult
        breakdown.quality.rework = (
            rework_samples * base_annotation_per_sample * 1.5
        )  # Rework is 1.5x

        # Expert review (for high/expert quality)
        if quality_requirement in ["high", "expert"]:
            expert_samples = int(target_size * 0.05)  # 5% expert review
            expert_rate = 2.0 * self.labor_mult * complexity_mult
            breakdown.quality.expert_review = expert_samples * expert_rate

        # Final validation
        breakdown.quality.final_validation = min(target_size * 0.01, 500) * self.labor_mult

    def format_report(self, breakdown: PhasedCostBreakdown) -> str:
        """Format breakdown as markdown report."""
        lines = []
        lines.append("# 分阶段成本估算")
        lines.append("")
        lines.append(f"**数据集类型**: {breakdown.dataset_type}")
        lines.append(f"**目标规模**: {breakdown.target_size:,} 条")
        lines.append(f"**项目规模**: {breakdown.scale.value}")
        lines.append("")

        # Design Phase
        lines.append("## 阶段一：设计阶段（固定成本）")
        lines.append("")
        lines.append("| 项目 | 成本 |")
        lines.append("|------|------|")
        lines.append(f"| Schema 设计 | ${breakdown.design.schema_design:,.0f} |")
        lines.append(f"| 标注指南编写 | ${breakdown.design.guideline_writing:,.0f} |")
        lines.append(f"| 试点测试 | ${breakdown.design.pilot_testing:,.0f} |")
        lines.append(f"| 工具配置 | ${breakdown.design.tool_setup:,.0f} |")
        lines.append(f"| **小计** | **${breakdown.design.total:,.0f}** |")
        lines.append("")

        # Production Phase
        lines.append("## 阶段二：生产阶段（变动成本）")
        lines.append("")
        lines.append("| 项目 | 成本 | 单价 |")
        lines.append("|------|------|------|")
        lines.append(f"| 人工标注 | ${breakdown.production.annotation_cost:,.0f} | - |")
        lines.append(f"| API 生成 | ${breakdown.production.generation_cost:,.0f} | - |")
        lines.append(f"| 初审 | ${breakdown.production.review_cost:,.0f} | - |")
        lines.append(f"| 基础设施 | ${breakdown.production.infrastructure:,.0f} | - |")
        lines.append(
            f"| **小计** | **${breakdown.production.total:,.0f}** | ${breakdown.production.cost_per_sample:.4f}/条 |"
        )
        lines.append("")

        # Quality Phase
        lines.append("## 阶段三：质量阶段（比例成本）")
        lines.append("")
        lines.append("| 项目 | 成本 | 说明 |")
        lines.append("|------|------|------|")
        lines.append(
            f"| QA 抽检 | ${breakdown.quality.qa_sampling:,.0f} | {breakdown.quality.qa_rate * 100:.0f}% 抽检率 |"
        )
        lines.append(
            f"| 返工修正 | ${breakdown.quality.rework:,.0f} | {breakdown.quality.expected_rework_rate * 100:.0f}% 预期返工 |"
        )
        lines.append(f"| 专家复核 | ${breakdown.quality.expert_review:,.0f} | - |")
        lines.append(f"| 终验 | ${breakdown.quality.final_validation:,.0f} | - |")
        lines.append(f"| **小计** | **${breakdown.quality.total:,.0f}** | - |")
        lines.append("")

        # Summary
        lines.append("## 汇总")
        lines.append("")
        lines.append("| 阶段 | 成本 | 占比 |")
        lines.append("|------|------|------|")
        subtotal = breakdown.total_fixed + breakdown.total_variable + breakdown.total_proportional
        lines.append(
            f"| 设计阶段 | ${breakdown.total_fixed:,.0f} | {breakdown.total_fixed / subtotal * 100:.1f}% |"
        )
        lines.append(
            f"| 生产阶段 | ${breakdown.total_variable:,.0f} | {breakdown.total_variable / subtotal * 100:.1f}% |"
        )
        lines.append(
            f"| 质量阶段 | ${breakdown.total_proportional:,.0f} | {breakdown.total_proportional / subtotal * 100:.1f}% |"
        )
        lines.append(f"| 小计 | ${subtotal:,.0f} | 100% |")
        lines.append(
            f"| 风险预留 ({breakdown.contingency_rate * 100:.0f}%) | ${breakdown.contingency_amount:,.0f} | - |"
        )
        lines.append(f"| **总计** | **${breakdown.grand_total:,.0f}** | - |")
        lines.append("")

        return "\n".join(lines)
