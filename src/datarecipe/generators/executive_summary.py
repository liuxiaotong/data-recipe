"""Executive summary generator for decision makers.

Generates a 1-page executive summary with:
- Recommendation (go/no-go)
- Value assessment (1-10 score)
- Use cases and expected outcomes
- ROI analysis
- Key risks
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Recommendation(Enum):
    """Recommendation levels."""

    HIGHLY_RECOMMENDED = "highly_recommended"
    RECOMMENDED = "recommended"
    CONDITIONAL = "conditional"
    NOT_RECOMMENDED = "not_recommended"


@dataclass
class ValueAssessment:
    """Value assessment for a dataset."""

    # Overall score (1-10)
    score: float = 5.0

    # Recommendation
    recommendation: Recommendation = Recommendation.CONDITIONAL
    recommendation_reason: str = ""

    # Use cases
    primary_use_case: str = ""
    secondary_use_cases: list[str] = field(default_factory=list)

    # Expected outcomes
    expected_outcomes: list[str] = field(default_factory=list)

    # ROI analysis
    roi_ratio: float = 1.0  # Expected value / Cost
    roi_explanation: str = ""
    payback_scenarios: list[str] = field(default_factory=list)

    # Risks
    risks: list[dict[str, str]] = field(default_factory=list)  # [{level, description, mitigation}]

    # Competitive analysis
    alternatives: list[str] = field(default_factory=list)
    competitive_advantage: str = ""


# Dataset type configurations
DATASET_TYPE_CONFIG = {
    "preference": {
        "primary_use_case": "训练奖励模型 (Reward Model) 或直接偏好优化 (DPO)",
        "secondary_use_cases": [
            "RLHF 训练流程",
            "模型对齐 (Alignment)",
            "响应质量评估",
        ],
        "expected_outcomes": [
            "提升模型响应质量和用户满意度",
            "减少有害/不当输出",
            "增强模型遵循人类偏好的能力",
        ],
        "value_multiplier": 1.5,  # High value for alignment
    },
    "evaluation": {
        "primary_use_case": "模型能力评测和基准测试",
        "secondary_use_cases": [
            "模型选型依据",
            "训练效果验证",
            "能力差距分析",
        ],
        "expected_outcomes": [
            "量化模型在特定任务上的表现",
            "识别模型能力短板",
            "支持模型迭代决策",
        ],
        "value_multiplier": 1.2,
    },
    "sft": {
        "primary_use_case": "监督微调 (Supervised Fine-Tuning)",
        "secondary_use_cases": [
            "领域适配",
            "指令遵循能力提升",
            "特定任务优化",
        ],
        "expected_outcomes": [
            "提升模型在目标领域的表现",
            "增强指令理解和执行能力",
            "定制化模型行为",
        ],
        "value_multiplier": 1.3,
    },
    "swe_bench": {
        "primary_use_case": "代码生成和软件工程能力评测",
        "secondary_use_cases": [
            "代码助手模型评估",
            "自动化代码修复研究",
            "软件工程 AI 基准",
        ],
        "expected_outcomes": [
            "评估模型代码理解和生成能力",
            "识别代码任务中的能力差距",
            "支持代码模型选型",
        ],
        "value_multiplier": 1.4,
    },
}

# Risk templates by factor
RISK_TEMPLATES = {
    "high_cost": {
        "level": "高",
        "description": "复刻成本较高，需要充足预算",
        "mitigation": "分阶段实施，优先完成核心子集",
    },
    "expert_required": {
        "level": "中",
        "description": "需要领域专家参与，人才获取可能困难",
        "mitigation": "提前储备人才，或考虑外包合作",
    },
    "quality_variance": {
        "level": "中",
        "description": "标注质量可能存在波动",
        "mitigation": "建立严格 QA 流程，设置质量门槛",
    },
    "time_intensive": {
        "level": "中",
        "description": "项目周期较长",
        "mitigation": "合理规划里程碑，设置阶段性交付",
    },
    "data_freshness": {
        "level": "低",
        "description": "数据可能随时间过时",
        "mitigation": "建立持续更新机制",
    },
}


class ExecutiveSummaryGenerator:
    """Generate executive summary for decision makers."""

    def __init__(self):
        pass

    def generate(
        self,
        dataset_id: str,
        dataset_type: str,
        sample_count: int,
        reproduction_cost: dict[str, float],
        human_percentage: float,
        complexity_metrics: Any | None = None,
        phased_breakdown: Any | None = None,
        llm_analysis: Any | None = None,
        enhanced_context: Any | None = None,
    ) -> ValueAssessment:
        """Generate value assessment for a dataset.

        Args:
            dataset_id: Dataset identifier
            dataset_type: Type of dataset (preference, evaluation, etc.)
            sample_count: Number of samples
            reproduction_cost: Cost breakdown dict
            human_percentage: Human work percentage
            complexity_metrics: Complexity analysis result
            phased_breakdown: Phased cost breakdown
            llm_analysis: LLM analysis result
            enhanced_context: LLM-enhanced context (optional)

        Returns:
            ValueAssessment object
        """
        assessment = ValueAssessment()

        # Get type config
        config = DATASET_TYPE_CONFIG.get(dataset_type, {})

        # Set use cases - prefer LLM-enhanced if available
        ec = enhanced_context
        if ec and getattr(ec, "generated", False) and ec.tailored_use_cases:
            assessment.primary_use_case = ec.tailored_use_cases[0]
            assessment.secondary_use_cases = ec.tailored_use_cases[1:]
        else:
            assessment.primary_use_case = config.get("primary_use_case", "通用数据集，用途待定")
            assessment.secondary_use_cases = config.get("secondary_use_cases", [])
        assessment.expected_outcomes = config.get("expected_outcomes", [])

        # Calculate value score (1-10)
        score = self._calculate_value_score(
            dataset_type=dataset_type,
            sample_count=sample_count,
            reproduction_cost=reproduction_cost,
            human_percentage=human_percentage,
            complexity_metrics=complexity_metrics,
            config=config,
        )
        assessment.score = score

        # Determine recommendation
        assessment.recommendation, assessment.recommendation_reason = self._get_recommendation(
            score=score,
            reproduction_cost=reproduction_cost,
            dataset_type=dataset_type,
        )

        # Calculate ROI
        assessment.roi_ratio, assessment.roi_explanation = self._calculate_roi(
            dataset_type=dataset_type,
            reproduction_cost=reproduction_cost,
            sample_count=sample_count,
            config=config,
        )

        # Generate payback scenarios - prefer LLM-enhanced
        if ec and getattr(ec, "generated", False) and ec.tailored_roi_scenarios:
            assessment.payback_scenarios = ec.tailored_roi_scenarios
        else:
            assessment.payback_scenarios = self._generate_payback_scenarios(
                dataset_type=dataset_type,
                reproduction_cost=reproduction_cost,
            )

        # Assess risks - prefer LLM-enhanced
        if ec and getattr(ec, "generated", False) and ec.tailored_risks:
            assessment.risks = ec.tailored_risks
        else:
            assessment.risks = self._assess_risks(
                reproduction_cost=reproduction_cost,
                human_percentage=human_percentage,
                complexity_metrics=complexity_metrics,
            )

        # Find alternatives
        assessment.alternatives = self._find_alternatives(dataset_id, dataset_type)
        if ec and getattr(ec, "generated", False) and ec.competitive_positioning:
            assessment.competitive_advantage = ec.competitive_positioning
        else:
            assessment.competitive_advantage = self._get_competitive_advantage(
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                sample_count=sample_count,
            )

        return assessment

    def _calculate_value_score(
        self,
        dataset_type: str,
        sample_count: int,
        reproduction_cost: dict[str, float],
        human_percentage: float,
        complexity_metrics: Any | None,
        config: dict,
    ) -> float:
        """Calculate value score (1-10)."""
        score = 5.0  # Base score

        # Type multiplier
        type_multiplier = config.get("value_multiplier", 1.0)
        score *= type_multiplier

        # Size factor (larger is generally better)
        if sample_count >= 10000:
            score += 1.5
        elif sample_count >= 1000:
            score += 1.0
        elif sample_count >= 100:
            score += 0.5
        elif sample_count < 50:
            score -= 1.0

        # Cost efficiency factor
        total_cost = reproduction_cost.get("total", 0)
        cost_per_sample = total_cost / sample_count if sample_count > 0 else 0

        if cost_per_sample < 1:
            score += 1.0  # Very cost efficient
        elif cost_per_sample < 5:
            score += 0.5
        elif cost_per_sample > 50:
            score -= 1.0  # Expensive

        # Complexity factor (moderate complexity is good)
        if complexity_metrics:
            difficulty = getattr(complexity_metrics, "difficulty_score", 2.0)
            if 1.5 <= difficulty <= 3.0:
                score += 0.5  # Good complexity range
            elif difficulty > 4.0:
                score -= 0.5  # Too complex

        # Normalize to 1-10 range
        score = max(1.0, min(10.0, score))

        return round(score, 1)

    def _get_recommendation(
        self,
        score: float,
        reproduction_cost: dict[str, float],
        dataset_type: str,
    ) -> tuple[Recommendation, str]:
        """Determine recommendation based on score and factors."""
        total_cost = reproduction_cost.get("total", 0)

        if score >= 8.0:
            return (
                Recommendation.HIGHLY_RECOMMENDED,
                f"数据集价值高 (评分 {score}/10)，强烈建议复刻",
            )
        elif score >= 6.0:
            return (Recommendation.RECOMMENDED, f"数据集价值良好 (评分 {score}/10)，建议复刻")
        elif score >= 4.0:
            if total_cost > 10000:
                return (
                    Recommendation.CONDITIONAL,
                    f"数据集有一定价值 (评分 {score}/10)，但成本较高，建议评估预算后决定",
                )
            else:
                return (
                    Recommendation.CONDITIONAL,
                    f"数据集有一定价值 (评分 {score}/10)，建议根据具体需求决定",
                )
        else:
            return (
                Recommendation.NOT_RECOMMENDED,
                f"数据集价值较低 (评分 {score}/10)，建议寻找替代方案",
            )

    def _calculate_roi(
        self,
        dataset_type: str,
        reproduction_cost: dict[str, float],
        sample_count: int,
        config: dict,
    ) -> tuple[float, str]:
        """Calculate ROI ratio and explanation."""
        total_cost = reproduction_cost.get("total", 0)
        if total_cost == 0:
            return (0, "无法计算 ROI（成本为零）")

        # Estimate value based on type and size
        # These are rough estimates for illustration
        value_multiplier = config.get("value_multiplier", 1.0)

        # Base value per sample (in USD equivalent)
        if dataset_type == "preference":
            base_value_per_sample = 2.0  # High value for alignment
        elif dataset_type == "evaluation":
            base_value_per_sample = 1.5
        elif dataset_type == "sft":
            base_value_per_sample = 1.0
        elif dataset_type == "swe_bench":
            base_value_per_sample = 3.0  # High value for code
        else:
            base_value_per_sample = 0.5

        estimated_value = sample_count * base_value_per_sample * value_multiplier
        roi_ratio = estimated_value / total_cost

        if roi_ratio >= 3.0:
            explanation = f"预计投资回报率 {roi_ratio:.1f}x，投资价值高"
        elif roi_ratio >= 1.5:
            explanation = f"预计投资回报率 {roi_ratio:.1f}x，投资价值良好"
        elif roi_ratio >= 1.0:
            explanation = f"预计投资回报率 {roi_ratio:.1f}x，接近收支平衡"
        else:
            explanation = f"预计投资回报率 {roi_ratio:.1f}x，需要谨慎评估"

        return (round(roi_ratio, 2), explanation)

    def _generate_payback_scenarios(
        self,
        dataset_type: str,
        reproduction_cost: dict[str, float],
    ) -> list[str]:
        """Generate payback scenarios."""
        reproduction_cost.get("total", 0)

        scenarios = []

        if dataset_type == "preference":
            scenarios = [
                "场景 A: 用于内部模型对齐，避免 1 次重大 PR 危机即可收回成本",
                "场景 B: 提升用户满意度 5%，按用户价值计算可在 3 个月内回本",
                "场景 C: 作为数据资产出售/授权给其他团队",
            ]
        elif dataset_type == "evaluation":
            scenarios = [
                "场景 A: 避免选择错误模型导致的返工成本",
                "场景 B: 缩短模型选型周期，节省团队时间",
                "场景 C: 作为标准化评测基准持续复用",
            ]
        elif dataset_type == "sft":
            scenarios = [
                "场景 A: 微调后模型性能提升带来的业务价值",
                "场景 B: 减少对昂贵 API 的依赖",
                "场景 C: 构建差异化能力形成竞争壁垒",
            ]
        elif dataset_type == "swe_bench":
            scenarios = [
                "场景 A: 评估代码助手投资决策",
                "场景 B: 提升开发效率带来的人力成本节省",
                "场景 C: 技术领先带来的品牌价值",
            ]
        else:
            scenarios = [
                "场景 A: 直接业务应用价值",
                "场景 B: 研究和技术储备价值",
                "场景 C: 数据资产长期价值",
            ]

        return scenarios

    def _assess_risks(
        self,
        reproduction_cost: dict[str, float],
        human_percentage: float,
        complexity_metrics: Any | None,
    ) -> list[dict[str, str]]:
        """Assess project risks."""
        risks = []

        total_cost = reproduction_cost.get("total", 0)

        # Cost risk
        if total_cost > 50000:
            risks.append(RISK_TEMPLATES["high_cost"])

        # Expert requirement risk
        if complexity_metrics:
            domain = getattr(complexity_metrics, "primary_domain", None)
            if domain:
                domain_value = domain.value if hasattr(domain, "value") else str(domain)
                if domain_value in ["medical", "legal", "finance"]:
                    risks.append(RISK_TEMPLATES["expert_required"])

        # Human-heavy risk
        if human_percentage > 85:
            risks.append(RISK_TEMPLATES["quality_variance"])

        # Time risk (estimate based on cost)
        if total_cost > 20000:
            risks.append(RISK_TEMPLATES["time_intensive"])

        # Always include data freshness as low risk
        risks.append(RISK_TEMPLATES["data_freshness"])

        return risks

    def _find_alternatives(self, dataset_id: str, dataset_type: str) -> list[str]:
        """Find alternative datasets."""
        # Try to get from knowledge base
        try:
            from datarecipe.knowledge import KnowledgeBase

            kb = KnowledgeBase()
            similar = kb.find_similar_datasets(dataset_type, limit=5)
            return [s.dataset_id for s in similar if s.dataset_id != dataset_id][:3]
        except (ImportError, AttributeError, TypeError):
            pass

        # Fallback to known alternatives
        alternatives_map = {
            "preference": [
                "Anthropic/hh-rlhf",
                "OpenAI/summarize_from_feedback",
                "stanfordnlp/SHP",
            ],
            "evaluation": [
                "MMLU",
                "HellaSwag",
                "TruthfulQA",
            ],
            "sft": [
                "OpenAssistant/oasst1",
                "databricks/dolly-15k",
                "tatsu-lab/alpaca",
            ],
            "swe_bench": [
                "princeton-nlp/SWE-bench",
                "bigcode/the-stack",
                "codeparrot/github-code",
            ],
        }

        alts = alternatives_map.get(dataset_type, [])
        return [a for a in alts if a.lower() != dataset_id.lower()][:3]

    def _get_competitive_advantage(
        self,
        dataset_id: str,
        dataset_type: str,
        sample_count: int,
    ) -> str:
        """Describe competitive advantage of this dataset."""
        org = dataset_id.split("/")[0] if "/" in dataset_id else ""

        advantages = []

        # Org reputation
        top_orgs = ["anthropic", "openai", "google", "meta", "microsoft", "tencent", "alibaba"]
        if org.lower() in top_orgs:
            advantages.append(f"来自知名机构 ({org})，数据质量有保障")

        # Size advantage
        if sample_count >= 10000:
            advantages.append("数据规模大，覆盖面广")
        elif sample_count >= 1000:
            advantages.append("数据规模适中，适合微调")

        # Type-specific advantages
        if dataset_type == "preference":
            advantages.append("支持 RLHF/DPO 训练，符合当前对齐研究趋势")
        elif dataset_type == "swe_bench":
            advantages.append("面向代码能力评测，市场上稀缺")

        return "；".join(advantages) if advantages else "标准数据集，无特殊优势"

    def to_markdown(
        self,
        assessment: ValueAssessment,
        dataset_id: str,
        dataset_type: str,
        reproduction_cost: dict[str, float],
        phased_breakdown: Any | None = None,
    ) -> str:
        """Generate executive summary markdown."""
        lines = []

        # Header
        lines.append(f"# {dataset_id} 执行摘要")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {dataset_type}")
        lines.append("")

        # Decision box
        lines.append("---")
        lines.append("")

        # Recommendation with visual indicator
        rec = assessment.recommendation
        if rec == Recommendation.HIGHLY_RECOMMENDED:
            rec_icon = "🟢"
            rec_text = "强烈推荐"
        elif rec == Recommendation.RECOMMENDED:
            rec_icon = "🟢"
            rec_text = "推荐"
        elif rec == Recommendation.CONDITIONAL:
            rec_icon = "🟡"
            rec_text = "有条件推荐"
        else:
            rec_icon = "🔴"
            rec_text = "不推荐"

        lines.append(f"## {rec_icon} 决策建议: {rec_text}")
        lines.append("")
        lines.append(f"**评分**: {assessment.score}/10")
        lines.append("")
        lines.append(f"**理由**: {assessment.recommendation_reason}")
        lines.append("")

        # Quick stats
        lines.append("### 关键指标")
        lines.append("")
        total_cost = reproduction_cost.get("total", 0)
        human_cost = reproduction_cost.get("human", 0)
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 总成本 | ${total_cost:,.0f} |")
        lines.append(
            f"| 人工成本 | ${human_cost:,.0f} ({human_cost / total_cost * 100:.0f}%) |"
            if total_cost > 0
            else f"| 人工成本 | ${human_cost:,.0f} |"
        )
        lines.append(f"| 投资回报率 | {assessment.roi_ratio:.1f}x |")
        lines.append("")

        # Use cases
        lines.append("---")
        lines.append("")
        lines.append("## 用途与价值")
        lines.append("")
        lines.append("### 主要用途")
        lines.append(f"**{assessment.primary_use_case}**")
        lines.append("")

        if assessment.secondary_use_cases:
            lines.append("### 其他用途")
            for use_case in assessment.secondary_use_cases:
                lines.append(f"- {use_case}")
            lines.append("")

        if assessment.expected_outcomes:
            lines.append("### 预期成果")
            for outcome in assessment.expected_outcomes:
                lines.append(f"- {outcome}")
            lines.append("")

        # ROI Analysis
        lines.append("---")
        lines.append("")
        lines.append("## ROI 分析")
        lines.append("")
        lines.append(f"**{assessment.roi_explanation}**")
        lines.append("")

        if assessment.payback_scenarios:
            lines.append("### 回报场景")
            for scenario in assessment.payback_scenarios:
                lines.append(f"- {scenario}")
            lines.append("")

        # Risks
        lines.append("---")
        lines.append("")
        lines.append("## 风险评估")
        lines.append("")

        if assessment.risks:
            lines.append("| 风险等级 | 描述 | 缓解措施 |")
            lines.append("|----------|------|----------|")
            for risk in assessment.risks:
                level = risk.get("level", "中")
                desc = risk.get("description", "")
                mitigation = risk.get("mitigation", "")
                lines.append(f"| {level} | {desc} | {mitigation} |")
            lines.append("")

        # Alternatives
        lines.append("---")
        lines.append("")
        lines.append("## 替代方案")
        lines.append("")

        if assessment.alternatives:
            lines.append("可考虑的替代数据集:")
            for alt in assessment.alternatives:
                lines.append(f"- {alt}")
            lines.append("")
        else:
            lines.append("暂无已知替代方案。")
            lines.append("")

        if assessment.competitive_advantage:
            lines.append(f"**本数据集优势**: {assessment.competitive_advantage}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("> 本摘要由 DataRecipe 自动生成，供决策参考")

        return "\n".join(lines)

    def to_dict(self, assessment: ValueAssessment) -> dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "score": assessment.score,
            "recommendation": assessment.recommendation.value,
            "recommendation_reason": assessment.recommendation_reason,
            "primary_use_case": assessment.primary_use_case,
            "secondary_use_cases": assessment.secondary_use_cases,
            "expected_outcomes": assessment.expected_outcomes,
            "roi_ratio": assessment.roi_ratio,
            "roi_explanation": assessment.roi_explanation,
            "payback_scenarios": assessment.payback_scenarios,
            "risks": assessment.risks,
            "alternatives": assessment.alternatives,
            "competitive_advantage": assessment.competitive_advantage,
        }

    # ------------------------------------------------------------------
    # analyze-spec pipeline: generate from SpecificationAnalysis
    # ------------------------------------------------------------------

    def spec_to_markdown(
        self,
        analysis: Any,
        target_size: int,
        region: str,
        cost_per_item: float,
        enhanced_context: Any = None,
    ) -> str:
        """Generate EXECUTIVE_SUMMARY.md from a SpecificationAnalysis object.

        Used by the analyze-spec pipeline (vs to_markdown which is for deep-analyze).
        """
        total_cost = cost_per_item * target_size
        human_cost = total_cost * (analysis.estimated_human_percentage / 100)

        # Determine recommendation
        if analysis.estimated_difficulty == "expert":
            recommendation = "有条件推荐"
            rec_icon = "🟡"
            score = 5.5
        elif analysis.estimated_difficulty == "hard":
            recommendation = "推荐"
            rec_icon = "🟢"
            score = 6.5
        else:
            recommendation = "强烈推荐"
            rec_icon = "🟢"
            score = 7.5

        lines = []
        lines.append(f"# {analysis.project_name} 执行摘要")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {analysis.dataset_type}")
        lines.append(f"> 目标规模: {target_size} 条")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Decision box
        lines.append(f"## {rec_icon} 决策建议: {recommendation}")
        lines.append("")
        lines.append(f"**评分**: {score}/10")
        lines.append("")
        lines.append(f"**理由**: 数据集价值良好 (评分 {score}/10)，{recommendation}")
        lines.append("")

        # Key metrics
        lines.append("### 关键指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 总成本 | ${total_cost:,.0f} |")
        lines.append(
            f"| 人工成本 | ${human_cost:,.0f} ({analysis.estimated_human_percentage:.0f}%) |"
        )
        lines.append(f"| 难度 | {analysis.estimated_difficulty} |")
        lines.append(f"| 领域 | {analysis.estimated_domain} |")
        lines.append("")

        # Use cases
        ec = enhanced_context
        lines.append("---")
        lines.append("")
        lines.append("## 用途与价值")
        lines.append("")
        if ec and ec.generated and ec.dataset_purpose_summary:
            lines.append(f"**主要用途**: {ec.dataset_purpose_summary}")
        else:
            lines.append(f"**主要用途**: {analysis.description or analysis.task_description}")
        lines.append("")

        if ec and ec.generated and ec.tailored_use_cases:
            lines.append("### 具体应用场景")
            lines.append("")
            for i, uc in enumerate(ec.tailored_use_cases, 1):
                lines.append(f"{i}. {uc}")
            lines.append("")

        if ec and ec.generated and ec.tailored_roi_scenarios:
            lines.append("### 投资回报分析")
            lines.append("")
            for i, roi in enumerate(ec.tailored_roi_scenarios, 1):
                lines.append(f"{i}. {roi}")
            lines.append("")

        if ec and ec.generated and ec.competitive_positioning:
            lines.append("### 竞争定位")
            lines.append("")
            lines.append(ec.competitive_positioning)
            lines.append("")

        # Risks
        lines.append("---")
        lines.append("")
        lines.append("## 风险评估")
        lines.append("")
        lines.append("| 风险等级 | 描述 | 缓解措施 |")
        lines.append("|----------|------|----------|")

        if ec and ec.generated and ec.tailored_risks:
            for risk in ec.tailored_risks:
                level = risk.get("level", "中")
                desc = risk.get("description", "")
                mit = risk.get("mitigation", "")
                lines.append(f"| {level} | {desc} | {mit} |")
        else:
            if (
                "AI" in str(analysis.forbidden_items)
                or "ai" in str(analysis.forbidden_items).lower()
            ):
                lines.append(
                    "| 高 | 禁止使用AI生成内容，全人工成本高 | 严格审核流程，确保数据原创性 |"
                )

            if analysis.estimated_difficulty in ["hard", "expert"]:
                lines.append("| 中 | 难度较高，需要专业人员 | 提前储备人才，加强培训 |")

            if analysis.has_images:
                lines.append("| 中 | 包含图片，制作成本较高 | 建立图片素材库，规范制作流程 |")

            lines.append("| 低 | 标注质量可能波动 | 建立QA流程，定期校准 |")
        lines.append("")

        # Similar datasets
        if analysis.similar_datasets:
            lines.append("---")
            lines.append("")
            lines.append("## 类似数据集")
            lines.append("")
            for ds in analysis.similar_datasets:
                lines.append(f"- {ds}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本摘要由 DataRecipe 从需求文档自动生成")

        return "\n".join(lines)

    def spec_to_dict(
        self,
        analysis: Any,
        target_size: int,
        cost_per_item: float,
    ) -> dict:
        """Convert SpecificationAnalysis to executive summary dict."""
        total_cost = cost_per_item * target_size
        human_cost = total_cost * (analysis.estimated_human_percentage / 100)
        api_cost = total_cost - human_cost

        if analysis.estimated_difficulty == "expert":
            recommendation, score = "有条件推荐", 5.5
        elif analysis.estimated_difficulty == "hard":
            recommendation, score = "推荐", 6.5
        else:
            recommendation, score = "强烈推荐", 7.5

        return {
            "project_name": analysis.project_name,
            "recommendation": recommendation,
            "score": score,
            "total_cost": total_cost,
            "human_cost": human_cost,
            "api_cost": api_cost,
            "human_percentage": analysis.estimated_human_percentage,
            "difficulty": analysis.estimated_difficulty,
            "domain": analysis.estimated_domain,
        }
