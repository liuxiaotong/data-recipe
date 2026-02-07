"""
Enhanced Production Guide Generator

Generates comprehensive production guides that include:
- Discovered patterns (rubrics, prompts)
- Human-machine task allocation
- Workload estimation
- Quality assurance framework
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..analyzers.context_strategy import ContextStrategy
from ..extractors.prompt_extractor import PromptLibrary
from ..extractors.rubrics_analyzer import RubricsAnalysisResult
from .human_machine_split import HumanMachineAllocation


@dataclass
class WorkloadEstimate:
    """Workload estimate for a production phase."""

    phase_name: str
    person_days: float
    team_size: int
    duration_weeks: float
    cost: float
    critical_path: bool = False
    dependencies: list[str] = field(default_factory=list)


@dataclass
class EnhancedProductionGuide:
    """Complete enhanced production guide."""

    # Basic info
    dataset_name: str
    target_size: int
    generation_date: str = field(default_factory=lambda: datetime.now().isoformat()[:10])

    # Discovered patterns
    rubrics_analysis: Optional[RubricsAnalysisResult] = None
    prompt_library: Optional[PromptLibrary] = None
    context_strategy: Optional[ContextStrategy] = None

    # Allocation
    allocation: Optional[HumanMachineAllocation] = None

    # Workload
    workload_estimates: list[WorkloadEstimate] = field(default_factory=list)
    total_person_days: float = 0.0
    total_weeks: float = 0.0
    total_cost: float = 0.0

    # Team structure
    recommended_team: dict = field(default_factory=dict)

    # Quality framework
    quality_checkpoints: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary statistics."""
        lines = [
            f"Dataset: {self.dataset_name}",
            f"Target Size: {self.target_size:,}",
            f"Total Effort: {self.total_person_days:.0f} person-days",
            f"Timeline: {self.total_weeks:.1f} weeks",
            f"Estimated Cost: ${self.total_cost:,.0f}",
        ]
        return "\n".join(lines)


class EnhancedGuideGenerator:
    """
    Generates enhanced production guides with patterns and allocations.

    Example usage:
        generator = EnhancedGuideGenerator()
        guide = generator.generate(
            dataset_name="my-dataset",
            target_size=10000,
            rubrics_analysis=rubrics_result,
            allocation=allocation_result,
        )
        markdown = generator.to_markdown(guide)
    """

    def generate(
        self,
        dataset_name: str,
        target_size: int,
        rubrics_analysis: Optional[RubricsAnalysisResult] = None,
        prompt_library: Optional[PromptLibrary] = None,
        context_strategy: Optional[ContextStrategy] = None,
        allocation: Optional[HumanMachineAllocation] = None,
        region: str = "china",
    ) -> EnhancedProductionGuide:
        """
        Generate an enhanced production guide.

        Args:
            dataset_name: Name of the dataset
            target_size: Target number of examples
            rubrics_analysis: Rubrics pattern analysis result
            prompt_library: Extracted prompt templates
            context_strategy: Detected context strategy
            allocation: Human-machine allocation
            region: Region for cost calculation

        Returns:
            EnhancedProductionGuide with all information
        """
        guide = EnhancedProductionGuide(
            dataset_name=dataset_name,
            target_size=target_size,
            rubrics_analysis=rubrics_analysis,
            prompt_library=prompt_library,
            context_strategy=context_strategy,
            allocation=allocation,
        )

        # Calculate workload
        guide.workload_estimates = self._estimate_workload(target_size, allocation, region)
        guide.total_person_days = sum(w.person_days for w in guide.workload_estimates)
        guide.total_weeks = max((w.duration_weeks for w in guide.workload_estimates), default=0)
        guide.total_cost = sum(w.cost for w in guide.workload_estimates)

        # Recommend team structure
        guide.recommended_team = self._recommend_team(guide.workload_estimates)

        # Define quality checkpoints
        guide.quality_checkpoints = self._define_quality_checkpoints(
            rubrics_analysis, context_strategy
        )

        return guide

    def _estimate_workload(
        self,
        target_size: int,
        allocation: Optional[HumanMachineAllocation],
        region: str,
    ) -> list[WorkloadEstimate]:
        """Estimate workload for each phase."""
        estimates = []

        # Base estimates (per 1000 examples)
        base_phases = [
            ("Planning & Setup", 5, 2, 1, []),
            ("Context Creation", 100, 4, 6, ["Planning & Setup"]),
            ("Task Design", 40, 3, 4, ["Context Creation"]),
            ("Rubrics Writing", 60, 4, 5, ["Task Design"]),
            ("Quality Review", 20, 2, 4, ["Rubrics Writing"]),
            ("Final Validation", 10, 2, 2, ["Quality Review"]),
        ]

        scale = target_size / 1000

        # Adjust based on allocation if available
        automation_factor = 1.0
        if allocation:
            automation_factor = 1 - (allocation.machine_work_percentage / 100) * 0.5

        # Regional cost multiplier
        cost_multipliers = {"us": 3.0, "china": 1.0, "europe": 2.5, "india": 0.7}
        cost_mult = cost_multipliers.get(region, 1.0)

        for phase_name, base_days, team_size, weeks, deps in base_phases:
            person_days = base_days * scale * automation_factor
            cost = person_days * 8 * 15 * cost_mult  # 8 hours/day, $15 base rate

            estimates.append(
                WorkloadEstimate(
                    phase_name=phase_name,
                    person_days=person_days,
                    team_size=team_size,
                    duration_weeks=weeks * (scale**0.3),  # Sub-linear scaling
                    cost=cost,
                    critical_path=(phase_name in ["Context Creation", "Rubrics Writing"]),
                    dependencies=deps,
                )
            )

        return estimates

    def _recommend_team(self, workload: list[WorkloadEstimate]) -> dict:
        """Recommend team structure based on workload."""
        return {
            "domain_experts": {
                "count": 4,
                "role": "Create and review context content",
                "skills": ["Domain expertise", "Technical writing"],
            },
            "task_designers": {
                "count": 2,
                "role": "Design evaluation tasks and questions",
                "skills": ["Instructional design", "Assessment creation"],
            },
            "annotators": {
                "count": 4,
                "role": "Write rubrics and label data",
                "skills": ["Attention to detail", "Logical thinking"],
            },
            "qa_reviewers": {
                "count": 2,
                "role": "Quality assurance and validation",
                "skills": ["Quality control", "Data analysis"],
            },
            "project_manager": {
                "count": 1,
                "role": "Coordinate team and track progress",
                "skills": ["Project management", "Communication"],
            },
        }

    def _define_quality_checkpoints(
        self,
        rubrics_analysis: Optional[RubricsAnalysisResult],
        context_strategy: Optional[ContextStrategy],
    ) -> list[dict]:
        """Define quality checkpoints based on analysis."""
        checkpoints = [
            {
                "phase": "Context Creation",
                "checks": [
                    "Content is original/not in training data",
                    "Internal consistency verified",
                    "Sufficient complexity for evaluation",
                ],
                "threshold": "100% pass rate",
            },
            {
                "phase": "Task Design",
                "checks": [
                    "Tasks require context to answer",
                    "Clear and unambiguous questions",
                    "Appropriate difficulty distribution",
                ],
                "threshold": "95% pass rate",
            },
            {
                "phase": "Rubrics Writing",
                "checks": [
                    "Follows standard pattern",
                    "Covers all key aspects",
                    "Objectively verifiable",
                ],
                "threshold": "98% pass rate",
            },
        ]

        # Add rubrics-specific checks
        if rubrics_analysis and rubrics_analysis.top_templates:
            checkpoints[2]["checks"].append(
                f"Uses discovered patterns (top: '{rubrics_analysis.top_templates[0][:50]}...')"
            )

        # Add strategy-specific checks
        if context_strategy:
            strategy_checks = {
                "synthetic": "Verify content novelty (not in existing corpora)",
                "modified": "Track modifications from original source",
                "niche": "Verify domain accuracy with experts",
            }
            check = strategy_checks.get(
                context_strategy.primary_strategy.value, "Verify content quality"
            )
            checkpoints[0]["checks"].append(check)

        return checkpoints

    def to_markdown(self, guide: EnhancedProductionGuide) -> str:
        """Convert guide to markdown format."""
        sections = []

        # Header
        sections.append(f"# Production Guide: {guide.dataset_name}")
        sections.append("")
        sections.append(f"> Generated: {guide.generation_date}")
        sections.append(f"> Target Size: {guide.target_size:,} examples")
        sections.append("")

        # Executive Summary
        sections.append("## Executive Summary")
        sections.append("")
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.append(f"| Total Effort | {guide.total_person_days:.0f} person-days |")
        sections.append(f"| Timeline | {guide.total_weeks:.1f} weeks |")
        sections.append(f"| Estimated Cost | ${guide.total_cost:,.0f} |")
        sections.append("")

        # Human-Machine Allocation
        if guide.allocation:
            sections.append("## Human-Machine Allocation")
            sections.append("")
            sections.append("### Summary")
            sections.append("")
            sections.append(f"- Human Work: {guide.allocation.human_work_percentage:.0f}%")
            sections.append(f"- Machine Work: {guide.allocation.machine_work_percentage:.0f}%")
            sections.append(
                f"- Estimated Savings: ${guide.allocation.estimated_savings_vs_all_human:,.0f}"
            )
            sections.append("")
            sections.append("### Task Breakdown")
            sections.append("")
            sections.append("| Task | Allocation | Human % | Human Hours | Cost |")
            sections.append("|------|------------|---------|-------------|------|")
            for task in guide.allocation.tasks:
                sections.append(
                    f"| {task.task_name} | {task.decision.value} | "
                    f"{task.human_percentage:.0f}% | {task.human_hours:.1f}h | "
                    f"${task.human_cost + task.machine_cost:.0f} |"
                )
            sections.append("")

            # Human-only vs Machine sections
            sections.append("### What Humans Must Do")
            sections.append("")
            for task in guide.allocation.human_only_tasks:
                sections.append(f"- **{task.task_name}**: {task.description}")
            sections.append("")
            sections.append("### What Machines Can Do")
            sections.append("")
            for task in guide.allocation.machine_only_tasks:
                sections.append(f"- **{task.task_name}**: {task.machine_method}")
            sections.append("")

        # Discovered Patterns
        if guide.rubrics_analysis:
            sections.append("## Discovered Patterns")
            sections.append("")
            sections.append("### Rubrics Patterns")
            sections.append("")
            sections.append(f"Total analyzed: {guide.rubrics_analysis.total_rubrics}")
            sections.append(f"Unique patterns: {guide.rubrics_analysis.unique_patterns}")
            sections.append("")
            sections.append("**Top Verbs:**")
            sections.append("")
            for verb, count in sorted(
                guide.rubrics_analysis.verb_distribution.items(), key=lambda x: -x[1]
            )[:10]:
                pct = count / guide.rubrics_analysis.total_rubrics * 100
                sections.append(f"- `{verb}`: {count} ({pct:.1f}%)")
            sections.append("")
            sections.append("**Top Templates:**")
            sections.append("")
            for template in guide.rubrics_analysis.top_templates[:5]:
                sections.append(f"- `{template}`")
            sections.append("")

        # Prompt Templates
        if guide.prompt_library:
            sections.append("### System Prompt Templates")
            sections.append("")
            sections.append(f"Extracted: {guide.prompt_library.unique_count} unique templates")
            sections.append("")
            sections.append("**By Category:**")
            sections.append("")
            for cat, count in guide.prompt_library.category_counts.items():
                sections.append(f"- {cat}: {count}")
            sections.append("")

        # Context Strategy
        if guide.context_strategy:
            sections.append("### Context Construction Strategy")
            sections.append("")
            sections.append(
                f"**Primary Strategy:** {guide.context_strategy.primary_strategy.value}"
            )
            sections.append(f"**Confidence:** {guide.context_strategy.confidence:.0%}")
            sections.append("")
            sections.append("**Recommendations:**")
            sections.append("")
            for rec in guide.context_strategy.recommendations[:5]:
                sections.append(f"- {rec}")
            sections.append("")

        # Workload Breakdown
        sections.append("## Workload Breakdown")
        sections.append("")
        sections.append("| Phase | Person-Days | Team Size | Weeks | Cost | Critical |")
        sections.append("|-------|-------------|-----------|-------|------|----------|")
        for w in guide.workload_estimates:
            critical = "Yes" if w.critical_path else "No"
            sections.append(
                f"| {w.phase_name} | {w.person_days:.1f} | {w.team_size} | "
                f"{w.duration_weeks:.1f} | ${w.cost:,.0f} | {critical} |"
            )
        sections.append("")

        # Team Structure
        sections.append("## Recommended Team Structure")
        sections.append("")
        for role, info in guide.recommended_team.items():
            sections.append(f"### {role.replace('_', ' ').title()} ({info['count']})")
            sections.append(f"**Role:** {info['role']}")
            sections.append(f"**Skills:** {', '.join(info['skills'])}")
            sections.append("")

        # Quality Checkpoints
        sections.append("## Quality Checkpoints")
        sections.append("")
        for checkpoint in guide.quality_checkpoints:
            sections.append(f"### {checkpoint['phase']}")
            sections.append(f"**Threshold:** {checkpoint['threshold']}")
            sections.append("")
            for check in checkpoint["checks"]:
                sections.append(f"- [ ] {check}")
            sections.append("")

        # Footer
        sections.append("---")
        sections.append("> Generated by DataRecipe")

        return "\n".join(sections)

    def to_dict(self, guide: EnhancedProductionGuide) -> dict:
        """Convert guide to dictionary for JSON export."""
        return {
            "dataset_name": guide.dataset_name,
            "target_size": guide.target_size,
            "generation_date": guide.generation_date,
            "summary": {
                "total_person_days": guide.total_person_days,
                "total_weeks": guide.total_weeks,
                "total_cost": guide.total_cost,
            },
            "workload": [
                {
                    "phase": w.phase_name,
                    "person_days": w.person_days,
                    "team_size": w.team_size,
                    "weeks": w.duration_weeks,
                    "cost": w.cost,
                    "critical_path": w.critical_path,
                }
                for w in guide.workload_estimates
            ],
            "team": guide.recommended_team,
            "quality_checkpoints": guide.quality_checkpoints,
        }
