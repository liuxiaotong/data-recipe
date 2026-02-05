"""Milestone plan generator for project management.

Generates a phased project plan with:
- Milestones and deliverables
- Team assignments
- Quality gates
- Risk mitigation actions
- Acceptance criteria
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class MilestoneStatus(Enum):
    """Milestone status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Milestone:
    """A project milestone."""
    id: str
    name: str
    description: str
    deliverables: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    team: List[str] = field(default_factory=list)
    effort_percentage: float = 0.0  # % of total project effort
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED


@dataclass
class RiskItem:
    """A project risk with mitigation."""
    id: str
    category: str  # technical, resource, quality, schedule
    description: str
    probability: str  # high, medium, low
    impact: str  # high, medium, low
    mitigation: str
    contingency: str
    owner: str = ""


@dataclass
class AcceptanceCriteria:
    """Acceptance criteria for quality gate."""
    category: str
    criterion: str
    metric: str
    threshold: str
    verification_method: str


@dataclass
class MilestonePlan:
    """Complete milestone plan."""
    dataset_id: str
    dataset_type: str
    target_size: int

    # Milestones
    milestones: List[Milestone] = field(default_factory=list)

    # Risks
    risks: List[RiskItem] = field(default_factory=list)

    # Acceptance criteria
    acceptance_criteria: List[AcceptanceCriteria] = field(default_factory=list)

    # Team summary
    team_composition: Dict[str, int] = field(default_factory=dict)

    # Estimated duration (in work days)
    estimated_days: int = 0


# Milestone templates by dataset type
MILESTONE_TEMPLATES = {
    "preference": [
        Milestone(
            id="M1",
            name="项目启动与规范制定",
            description="完成项目初始化、制定标注规范和质量标准",
            deliverables=[
                "标注指南文档 v1.0",
                "Schema 定义与示例",
                "标注工具配置完成",
                "团队培训材料",
            ],
            acceptance_criteria=[
                "标注指南通过专家评审",
                "3 名标注员完成培训并通过测试",
                "标注工具功能验证通过",
            ],
            team=["项目经理", "领域专家", "工具工程师"],
            effort_percentage=15,
        ),
        Milestone(
            id="M2",
            name="试点标注与标准校准",
            description="完成试点批次，验证标注流程和质量标准",
            deliverables=[
                "试点数据 (目标的 5%)",
                "标注一致性报告",
                "流程问题清单与解决方案",
            ],
            acceptance_criteria=[
                "标注员间一致性 (Cohen's Kappa) ≥ 0.7",
                "专家抽检合格率 ≥ 90%",
                "平均标注速度达到预期的 80%",
            ],
            dependencies=["M1"],
            team=["标注员", "QA", "领域专家"],
            effort_percentage=10,
        ),
        Milestone(
            id="M3",
            name="主体标注 - 第一批次",
            description="完成 40% 的标注量",
            deliverables=[
                "已标注数据 (目标的 40%)",
                "质量周报",
            ],
            acceptance_criteria=[
                "数据完成率 ≥ 40%",
                "抽检合格率 ≥ 95%",
                "无重大质量问题",
            ],
            dependencies=["M2"],
            team=["标注员", "QA"],
            effort_percentage=30,
        ),
        Milestone(
            id="M4",
            name="主体标注 - 第二批次",
            description="完成剩余 60% 的标注量",
            deliverables=[
                "已标注数据 (目标的 100%)",
                "质量周报",
            ],
            acceptance_criteria=[
                "数据完成率 = 100%",
                "抽检合格率 ≥ 95%",
                "所有标注员产出稳定",
            ],
            dependencies=["M3"],
            team=["标注员", "QA"],
            effort_percentage=30,
        ),
        Milestone(
            id="M5",
            name="质量审核与交付",
            description="完成最终质量审核和数据交付",
            deliverables=[
                "最终数据集",
                "质量报告",
                "数据文档",
            ],
            acceptance_criteria=[
                "全量抽检合格率 ≥ 95%",
                "数据格式验证通过",
                "文档完整性检查通过",
            ],
            dependencies=["M4"],
            team=["QA", "领域专家", "项目经理"],
            effort_percentage=15,
        ),
    ],
    "evaluation": [
        Milestone(
            id="M1",
            name="评测框架设计",
            description="设计评测维度、题型和评分标准",
            deliverables=[
                "评测维度定义",
                "题型分类和比例",
                "评分标准模板",
            ],
            acceptance_criteria=[
                "评测维度覆盖目标能力",
                "题型分布合理",
                "评分标准可操作",
            ],
            team=["领域专家", "评测设计师"],
            effort_percentage=20,
        ),
        Milestone(
            id="M2",
            name="题目编写与审核",
            description="完成评测题目的编写和专家审核",
            deliverables=[
                "评测题目 (目标的 100%)",
                "答案和评分标准",
            ],
            acceptance_criteria=[
                "题目覆盖所有评测维度",
                "专家审核通过率 ≥ 90%",
                "难度分布符合设计",
            ],
            dependencies=["M1"],
            team=["题目编写员", "领域专家", "QA"],
            effort_percentage=50,
        ),
        Milestone(
            id="M3",
            name="难度校准与验证",
            description="使用基准模型验证题目难度和区分度",
            deliverables=[
                "模型测试结果",
                "难度调整记录",
                "最终题目集",
            ],
            acceptance_criteria=[
                "题目区分度 ≥ 0.3",
                "难度分布符合预期",
                "无明显 bug 题目",
            ],
            dependencies=["M2"],
            team=["评测工程师", "QA"],
            effort_percentage=20,
        ),
        Milestone(
            id="M4",
            name="交付与文档",
            description="完成数据集打包和文档编写",
            deliverables=[
                "最终数据集",
                "使用说明文档",
                "评测报告模板",
            ],
            acceptance_criteria=[
                "数据格式正确",
                "文档完整清晰",
                "示例代码可运行",
            ],
            dependencies=["M3"],
            team=["工程师", "技术写作"],
            effort_percentage=10,
        ),
    ],
    "sft": [
        Milestone(
            id="M1",
            name="数据规范与种子数据",
            description="制定数据格式规范，准备种子数据",
            deliverables=[
                "数据格式规范",
                "种子数据 (100-200 条)",
                "质量标准文档",
            ],
            acceptance_criteria=[
                "格式规范清晰完整",
                "种子数据质量优秀",
                "覆盖主要场景",
            ],
            team=["数据架构师", "领域专家"],
            effort_percentage=15,
        ),
        Milestone(
            id="M2",
            name="数据生成与筛选",
            description="批量生成候选数据并进行质量筛选",
            deliverables=[
                "候选数据集",
                "筛选规则和脚本",
            ],
            acceptance_criteria=[
                "候选数据量达到目标的 150%",
                "自动筛选规则有效",
            ],
            dependencies=["M1"],
            team=["数据工程师", "ML 工程师"],
            effort_percentage=25,
        ),
        Milestone(
            id="M3",
            name="人工审核与优化",
            description="人工审核筛选后的数据，进行优化调整",
            deliverables=[
                "审核后数据集 (目标的 100%)",
                "常见问题总结",
            ],
            acceptance_criteria=[
                "审核通过率记录",
                "数据质量达标",
            ],
            dependencies=["M2"],
            team=["标注员", "QA", "领域专家"],
            effort_percentage=40,
        ),
        Milestone(
            id="M4",
            name="验证与交付",
            description="训练验证和最终交付",
            deliverables=[
                "验证实验结果",
                "最终数据集",
                "数据卡片",
            ],
            acceptance_criteria=[
                "微调效果达到预期",
                "数据无隐私泄露",
                "格式验证通过",
            ],
            dependencies=["M3"],
            team=["ML 工程师", "QA"],
            effort_percentage=20,
        ),
    ],
}

# Risk templates
RISK_TEMPLATES = {
    "quality_variance": RiskItem(
        id="R1",
        category="quality",
        description="标注质量不稳定，不同标注员之间差异大",
        probability="medium",
        impact="high",
        mitigation="1) 加强培训 2) 建立详细的标注指南 3) 定期校准会议",
        contingency="增加 QA 抽检比例，问题严重时暂停并重新培训",
    ),
    "schedule_delay": RiskItem(
        id="R2",
        category="schedule",
        description="项目进度延误，无法按时交付",
        probability="medium",
        impact="medium",
        mitigation="1) 预留缓冲时间 2) 设置中间检查点 3) 提前识别瓶颈",
        contingency="增加人力或调整范围，与客户沟通调整计划",
    ),
    "resource_shortage": RiskItem(
        id="R3",
        category="resource",
        description="关键人员离职或不可用",
        probability="low",
        impact="high",
        mitigation="1) 知识文档化 2) 交叉培训 3) 保持备选人员",
        contingency="紧急招聘或外包部分工作",
    ),
    "requirement_change": RiskItem(
        id="R4",
        category="technical",
        description="需求变更导致返工",
        probability="medium",
        impact="medium",
        mitigation="1) 充分的需求确认 2) 变更控制流程 3) 灵活的架构设计",
        contingency="评估变更影响，协商调整范围或时间",
    ),
    "data_quality_issue": RiskItem(
        id="R5",
        category="quality",
        description="原始数据质量问题导致标注困难",
        probability="low",
        impact="medium",
        mitigation="1) 数据预处理 2) 制定异常处理规则 3) 建立问题反馈机制",
        contingency="跳过问题数据或请求替换",
    ),
}

# Acceptance criteria templates
ACCEPTANCE_CRITERIA_TEMPLATES = {
    "preference": [
        AcceptanceCriteria(
            category="一致性",
            criterion="标注员间一致性",
            metric="Cohen's Kappa",
            threshold="≥ 0.7",
            verification_method="随机抽取 100 条由两人独立标注，计算 Kappa 值",
        ),
        AcceptanceCriteria(
            category="准确性",
            criterion="专家审核通过率",
            metric="通过率",
            threshold="≥ 95%",
            verification_method="领域专家随机抽检 5% 的数据",
        ),
        AcceptanceCriteria(
            category="完整性",
            criterion="数据完整性",
            metric="空值率",
            threshold="= 0%",
            verification_method="自动脚本检查所有必填字段",
        ),
        AcceptanceCriteria(
            category="格式",
            criterion="格式合规性",
            metric="Schema 验证通过率",
            threshold="= 100%",
            verification_method="JSON Schema 自动验证",
        ),
        AcceptanceCriteria(
            category="偏好",
            criterion="偏好区分度",
            metric="chosen 与 rejected 质量差异",
            threshold="两者有明显质量差异",
            verification_method="抽检确认 chosen 明显优于 rejected",
        ),
    ],
    "evaluation": [
        AcceptanceCriteria(
            category="覆盖度",
            criterion="评测维度覆盖",
            metric="覆盖率",
            threshold="= 100%",
            verification_method="检查每个维度都有足够题目",
        ),
        AcceptanceCriteria(
            category="难度",
            criterion="难度分布",
            metric="easy/medium/hard 比例",
            threshold="符合设计比例 ±10%",
            verification_method="基准模型测试验证",
        ),
        AcceptanceCriteria(
            category="区分度",
            criterion="题目区分度",
            metric="区分度系数",
            threshold="≥ 0.3",
            verification_method="多模型测试计算区分度",
        ),
        AcceptanceCriteria(
            category="准确性",
            criterion="答案正确性",
            metric="专家验证通过率",
            threshold="= 100%",
            verification_method="专家审核所有题目答案",
        ),
    ],
    "sft": [
        AcceptanceCriteria(
            category="多样性",
            criterion="指令多样性",
            metric="指令类型覆盖",
            threshold="≥ 设计的 90%",
            verification_method="分类统计验证",
        ),
        AcceptanceCriteria(
            category="质量",
            criterion="回答质量",
            metric="抽检合格率",
            threshold="≥ 95%",
            verification_method="人工抽检 5%",
        ),
        AcceptanceCriteria(
            category="安全",
            criterion="安全性",
            metric="有害内容检出率",
            threshold="= 0%",
            verification_method="安全分类器扫描 + 人工复核",
        ),
        AcceptanceCriteria(
            category="效果",
            criterion="微调效果",
            metric="验证集性能",
            threshold="相对基线提升",
            verification_method="在验证集上评测",
        ),
    ],
}


class MilestonePlanGenerator:
    """Generate milestone plan for project management."""

    def __init__(self):
        pass

    def generate(
        self,
        dataset_id: str,
        dataset_type: str,
        target_size: int,
        reproduction_cost: Dict[str, float],
        human_percentage: float,
        complexity_metrics: Optional[Any] = None,
        phased_breakdown: Optional[Any] = None,
    ) -> MilestonePlan:
        """Generate milestone plan.

        Args:
            dataset_id: Dataset identifier
            dataset_type: Type of dataset
            target_size: Target dataset size
            reproduction_cost: Cost breakdown
            human_percentage: Human work percentage
            complexity_metrics: Complexity analysis
            phased_breakdown: Phased cost breakdown

        Returns:
            MilestonePlan object
        """
        plan = MilestonePlan(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            target_size=target_size,
        )

        # Get milestones template
        milestones = self._get_milestones(dataset_type, target_size)
        plan.milestones = milestones

        # Get risks
        plan.risks = self._get_risks(dataset_type, complexity_metrics)

        # Get acceptance criteria
        plan.acceptance_criteria = self._get_acceptance_criteria(dataset_type)

        # Calculate team composition
        plan.team_composition = self._calculate_team(
            dataset_type, target_size, human_percentage
        )

        # Estimate duration
        plan.estimated_days = self._estimate_duration(
            target_size, human_percentage, complexity_metrics
        )

        return plan

    def _get_milestones(self, dataset_type: str, target_size: int) -> List[Milestone]:
        """Get milestones for dataset type."""
        templates = MILESTONE_TEMPLATES.get(
            dataset_type, MILESTONE_TEMPLATES.get("preference", [])
        )

        # Deep copy and customize
        milestones = []
        for template in templates:
            milestone = Milestone(
                id=template.id,
                name=template.name,
                description=template.description,
                deliverables=template.deliverables.copy(),
                acceptance_criteria=template.acceptance_criteria.copy(),
                dependencies=template.dependencies.copy(),
                team=template.team.copy(),
                effort_percentage=template.effort_percentage,
                status=MilestoneStatus.NOT_STARTED,
            )

            # Customize deliverables with actual numbers
            if "目标的" in str(milestone.deliverables):
                milestone.deliverables = [
                    d.replace("目标的 5%", f"{int(target_size * 0.05)} 条")
                    .replace("目标的 40%", f"{int(target_size * 0.4)} 条")
                    .replace("目标的 100%", f"{target_size} 条")
                    .replace("目标的 60%", f"{int(target_size * 0.6)} 条")
                    for d in milestone.deliverables
                ]

            milestones.append(milestone)

        return milestones

    def _get_risks(
        self, dataset_type: str, complexity_metrics: Optional[Any]
    ) -> List[RiskItem]:
        """Get risks for project."""
        risks = []

        # Always include common risks
        risks.append(RISK_TEMPLATES["quality_variance"])
        risks.append(RISK_TEMPLATES["schedule_delay"])

        # Add based on complexity
        if complexity_metrics:
            domain = getattr(complexity_metrics, 'primary_domain', None)
            if domain:
                domain_value = domain.value if hasattr(domain, 'value') else str(domain)
                if domain_value in ['medical', 'legal', 'finance', 'code']:
                    risks.append(RISK_TEMPLATES["resource_shortage"])

            difficulty = getattr(complexity_metrics, 'difficulty_score', 2.0)
            if difficulty > 3.0:
                risks.append(RISK_TEMPLATES["requirement_change"])

        risks.append(RISK_TEMPLATES["data_quality_issue"])

        return risks

    def _get_acceptance_criteria(self, dataset_type: str) -> List[AcceptanceCriteria]:
        """Get acceptance criteria for dataset type."""
        return ACCEPTANCE_CRITERIA_TEMPLATES.get(
            dataset_type, ACCEPTANCE_CRITERIA_TEMPLATES.get("preference", [])
        )

    def _calculate_team(
        self, dataset_type: str, target_size: int, human_percentage: float
    ) -> Dict[str, int]:
        """Calculate recommended team composition."""
        # Base team
        team = {
            "项目经理": 1,
            "领域专家": 2,
            "QA": 1,
        }

        # Scale annotators based on size and human percentage
        if human_percentage > 50:
            if target_size >= 10000:
                team["标注员"] = 8
            elif target_size >= 1000:
                team["标注员"] = 4
            else:
                team["标注员"] = 2
        else:
            team["标注员"] = 2

        # Type-specific roles
        if dataset_type == "evaluation":
            team["评测设计师"] = 1
            team["评测工程师"] = 1
        elif dataset_type == "sft":
            team["ML 工程师"] = 1
            team["数据工程师"] = 1

        return team

    def _estimate_duration(
        self,
        target_size: int,
        human_percentage: float,
        complexity_metrics: Optional[Any],
    ) -> int:
        """Estimate project duration in work days."""
        # Base: 1 day per 100 samples for human work
        base_days = (target_size * human_percentage / 100) / 100

        # Minimum 10 days for setup and QA
        base_days = max(base_days, 10)

        # Add buffer for complexity
        if complexity_metrics:
            multiplier = getattr(complexity_metrics, 'time_multiplier', 1.0)
            base_days *= multiplier

        # Add 20% buffer
        base_days *= 1.2

        return int(base_days)

    def to_markdown(self, plan: MilestonePlan) -> str:
        """Generate markdown milestone plan."""
        lines = []

        # Header
        lines.append(f"# {plan.dataset_id} 里程碑计划")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {plan.dataset_type}")
        lines.append(f"> 目标规模: {plan.target_size:,} 条")
        lines.append(f"> 预估工期: {plan.estimated_days} 工作日")
        lines.append("")

        # Overview
        lines.append("---")
        lines.append("")
        lines.append("## 项目概览")
        lines.append("")

        # Gantt-like visualization
        lines.append("```")
        lines.append("阶段进度:")
        cumulative = 0
        for m in plan.milestones:
            bar_len = int(m.effort_percentage / 5)
            bar = "█" * bar_len
            spaces = " " * (20 - bar_len)
            cumulative += m.effort_percentage
            lines.append(f"{m.id} {m.name[:12]:<12} {bar}{spaces} {m.effort_percentage}% (累计 {cumulative}%)")
        lines.append("```")
        lines.append("")

        # Team composition
        lines.append("### 团队配置")
        lines.append("")
        lines.append("| 角色 | 人数 |")
        lines.append("|------|------|")
        for role, count in plan.team_composition.items():
            lines.append(f"| {role} | {count} |")
        lines.append("")

        # Milestones detail
        lines.append("---")
        lines.append("")
        lines.append("## 里程碑详情")
        lines.append("")

        for m in plan.milestones:
            lines.append(f"### {m.id}: {m.name}")
            lines.append("")
            lines.append(f"**描述**: {m.description}")
            lines.append("")

            if m.dependencies:
                lines.append(f"**依赖**: {', '.join(m.dependencies)}")
                lines.append("")

            lines.append(f"**负责团队**: {', '.join(m.team)}")
            lines.append("")

            lines.append("**交付物**:")
            for d in m.deliverables:
                lines.append(f"- [ ] {d}")
            lines.append("")

            lines.append("**验收标准**:")
            for ac in m.acceptance_criteria:
                lines.append(f"- {ac}")
            lines.append("")

        # Acceptance criteria
        lines.append("---")
        lines.append("")
        lines.append("## 验收标准")
        lines.append("")
        lines.append("| 类别 | 标准 | 指标 | 阈值 | 验证方法 |")
        lines.append("|------|------|------|------|----------|")
        for ac in plan.acceptance_criteria:
            lines.append(f"| {ac.category} | {ac.criterion} | {ac.metric} | {ac.threshold} | {ac.verification_method} |")
        lines.append("")

        # Risk management
        lines.append("---")
        lines.append("")
        lines.append("## 风险管理")
        lines.append("")

        for risk in plan.risks:
            prob_icon = "🔴" if risk.probability == "high" else "🟡" if risk.probability == "medium" else "🟢"
            impact_icon = "🔴" if risk.impact == "high" else "🟡" if risk.impact == "medium" else "🟢"

            lines.append(f"### {risk.id}: {risk.description}")
            lines.append("")
            lines.append(f"- **类别**: {risk.category}")
            lines.append(f"- **概率**: {prob_icon} {risk.probability}")
            lines.append(f"- **影响**: {impact_icon} {risk.impact}")
            lines.append(f"- **缓解措施**: {risk.mitigation}")
            lines.append(f"- **应急预案**: {risk.contingency}")
            lines.append("")

        # Checklist
        lines.append("---")
        lines.append("")
        lines.append("## 启动检查清单")
        lines.append("")
        lines.append("### 项目启动前")
        lines.append("- [ ] 需求确认并签字")
        lines.append("- [ ] 团队到位")
        lines.append("- [ ] 工具和环境准备就绪")
        lines.append("- [ ] 培训材料准备完成")
        lines.append("")
        lines.append("### 每个里程碑结束时")
        lines.append("- [ ] 交付物检查完成")
        lines.append("- [ ] 验收标准达成")
        lines.append("- [ ] 问题记录和跟踪")
        lines.append("- [ ] 下阶段计划确认")
        lines.append("")
        lines.append("### 项目结束时")
        lines.append("- [ ] 所有交付物移交")
        lines.append("- [ ] 文档归档")
        lines.append("- [ ] 经验总结")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*本计划由 DataRecipe 自动生成，请根据实际情况调整*")

        return "\n".join(lines)

    def to_dict(self, plan: MilestonePlan) -> dict:
        """Convert plan to dictionary."""
        return {
            "dataset_id": plan.dataset_id,
            "dataset_type": plan.dataset_type,
            "target_size": plan.target_size,
            "estimated_days": plan.estimated_days,
            "team_composition": plan.team_composition,
            "milestones": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "deliverables": m.deliverables,
                    "acceptance_criteria": m.acceptance_criteria,
                    "dependencies": m.dependencies,
                    "team": m.team,
                    "effort_percentage": m.effort_percentage,
                }
                for m in plan.milestones
            ],
            "acceptance_criteria": [
                {
                    "category": ac.category,
                    "criterion": ac.criterion,
                    "metric": ac.metric,
                    "threshold": ac.threshold,
                    "verification_method": ac.verification_method,
                }
                for ac in plan.acceptance_criteria
            ],
            "risks": [
                {
                    "id": r.id,
                    "category": r.category,
                    "description": r.description,
                    "probability": r.probability,
                    "impact": r.impact,
                    "mitigation": r.mitigation,
                    "contingency": r.contingency,
                }
                for r in plan.risks
            ],
        }
