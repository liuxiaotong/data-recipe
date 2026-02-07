"""
Human-Machine Task Allocation Generator

Analyzes production workflow and determines optimal allocation
between human workers and automated systems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(Enum):
    """Types of tasks in data production."""

    CONTEXT_CREATION = "context_creation"
    TASK_DESIGN = "task_design"
    RUBRICS_WRITING = "rubrics_writing"
    DATA_COLLECTION = "data_collection"
    DATA_GENERATION = "data_generation"
    DATA_FILTERING = "data_filtering"
    LABELING = "labeling"
    QUALITY_REVIEW = "quality_review"
    FORMAT_CONVERSION = "format_conversion"
    EDGE_CASE_HANDLING = "edge_case_handling"


class AllocationDecision(Enum):
    """Allocation decision for a task."""

    HUMAN_ONLY = "human_only"
    MACHINE_ONLY = "machine_only"
    HUMAN_PRIMARY = "human_primary"  # Human does most, machine assists
    MACHINE_PRIMARY = "machine_primary"  # Machine does most, human reviews


@dataclass
class TaskAllocation:
    """Allocation decision for a single task."""

    task_id: str
    task_name: str
    task_type: TaskType
    description: str = ""

    # Allocation
    decision: AllocationDecision = AllocationDecision.HUMAN_ONLY
    human_percentage: float = 100.0  # % of work done by humans
    machine_percentage: float = 0.0  # % of work done by machines

    # Human effort
    human_role: str = ""  # e.g., "domain expert"
    human_hours: float = 0.0
    human_hourly_rate: float = 25.0
    human_cost: float = 0.0

    # Machine effort
    machine_method: str = ""  # e.g., "LLM generation"
    machine_cost: float = 0.0
    automation_confidence: float = 0.0

    # Reasoning
    rationale: str = ""
    risk_factors: list[str] = field(default_factory=list)
    quality_impact: str = ""

    def __post_init__(self):
        self.human_cost = self.human_hours * self.human_hourly_rate


@dataclass
class HumanMachineAllocation:
    """Complete allocation analysis for a production workflow."""

    tasks: list[TaskAllocation] = field(default_factory=list)

    # Summary stats
    total_human_hours: float = 0.0
    total_human_cost: float = 0.0
    total_machine_cost: float = 0.0
    total_cost: float = 0.0

    # Breakdown
    human_only_tasks: list[TaskAllocation] = field(default_factory=list)
    machine_only_tasks: list[TaskAllocation] = field(default_factory=list)
    hybrid_tasks: list[TaskAllocation] = field(default_factory=list)

    # Percentages
    human_work_percentage: float = 0.0
    machine_work_percentage: float = 0.0

    # Savings estimate
    estimated_savings_vs_all_human: float = 0.0
    timeline_reduction_percentage: float = 0.0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            "HUMAN-MACHINE ALLOCATION SUMMARY",
            "=" * 50,
            "",
            f"Total Tasks: {len(self.tasks)}",
            f"  - Human Only: {len(self.human_only_tasks)}",
            f"  - Machine Only: {len(self.machine_only_tasks)}",
            f"  - Hybrid: {len(self.hybrid_tasks)}",
            "",
            "COSTS:",
            f"  Human Labor: ${self.total_human_cost:,.0f} ({self.total_human_hours:.0f} hours)",
            f"  Machine/API: ${self.total_machine_cost:,.0f}",
            f"  Total: ${self.total_cost:,.0f}",
            "",
            "WORKLOAD SPLIT:",
            f"  Human: {self.human_work_percentage:.0f}%",
            f"  Machine: {self.machine_work_percentage:.0f}%",
            "",
        ]

        if self.estimated_savings_vs_all_human > 0:
            lines.extend(
                [
                    f"Estimated Savings: ${self.estimated_savings_vs_all_human:,.0f}",
                    f"Timeline Reduction: {self.timeline_reduction_percentage:.0f}%",
                ]
            )

        return "\n".join(lines)

    def to_markdown_table(self) -> str:
        """Generate markdown table of allocations."""
        lines = [
            "| Task | Type | Allocation | Human % | Human Hours | Human Cost | Machine Cost |",
            "|------|------|------------|---------|-------------|------------|--------------|",
        ]

        for task in self.tasks:
            lines.append(
                f"| {task.task_name} | {task.task_type.value} | "
                f"{task.decision.value} | {task.human_percentage:.0f}% | "
                f"{task.human_hours:.1f}h | ${task.human_cost:.0f} | "
                f"${task.machine_cost:.0f} |"
            )

        return "\n".join(lines)


class HumanMachineSplitter:
    """
    Analyzes production workflow and allocates tasks between humans and machines.

    Example usage:
        splitter = HumanMachineSplitter()
        allocation = splitter.analyze(
            dataset_size=10000,
            task_types=[TaskType.CONTEXT_CREATION, TaskType.RUBRICS_WRITING],
            region="china"
        )
        print(allocation.summary())
    """

    # Default automation feasibility by task type
    AUTOMATION_FEASIBILITY = {
        TaskType.CONTEXT_CREATION: 0.2,  # Low - requires creativity
        TaskType.TASK_DESIGN: 0.3,  # Low - requires pedagogical skill
        TaskType.RUBRICS_WRITING: 0.4,  # Medium - templates help
        TaskType.DATA_COLLECTION: 0.7,  # High - scraping is easy
        TaskType.DATA_GENERATION: 0.9,  # High - LLMs excel here
        TaskType.DATA_FILTERING: 0.8,  # High - rule-based
        TaskType.LABELING: 0.5,  # Medium - depends on complexity
        TaskType.QUALITY_REVIEW: 0.3,  # Low - needs human judgment
        TaskType.FORMAT_CONVERSION: 1.0,  # Full automation
        TaskType.EDGE_CASE_HANDLING: 0.1,  # Very low - needs expertise
    }

    # Hours per unit by task type (for 1000 examples)
    HOURS_PER_1K = {
        TaskType.CONTEXT_CREATION: 200.0,  # ~12 min per context
        TaskType.TASK_DESIGN: 80.0,  # ~5 min per task
        TaskType.RUBRICS_WRITING: 100.0,  # ~6 min per rubric set
        TaskType.DATA_COLLECTION: 20.0,  # Mostly automated
        TaskType.DATA_GENERATION: 5.0,  # API calls
        TaskType.DATA_FILTERING: 10.0,  # Semi-automated
        TaskType.LABELING: 50.0,  # ~3 min per label
        TaskType.QUALITY_REVIEW: 30.0,  # ~2 min per review
        TaskType.FORMAT_CONVERSION: 2.0,  # Fully automated
        TaskType.EDGE_CASE_HANDLING: 40.0,  # Requires expertise
    }

    # Regional hourly rates
    HOURLY_RATES = {
        "us": {
            "general": 25.0,
            "expert": 75.0,
            "professional": 150.0,
        },
        "china": {
            "general": 8.0,
            "expert": 25.0,
            "professional": 50.0,
        },
        "europe": {
            "general": 20.0,
            "expert": 60.0,
            "professional": 120.0,
        },
        "india": {
            "general": 5.0,
            "expert": 15.0,
            "professional": 35.0,
        },
    }

    # Task type to expertise level mapping
    EXPERTISE_REQUIRED = {
        TaskType.CONTEXT_CREATION: "expert",
        TaskType.TASK_DESIGN: "expert",
        TaskType.RUBRICS_WRITING: "general",
        TaskType.DATA_COLLECTION: "general",
        TaskType.DATA_GENERATION: "general",
        TaskType.DATA_FILTERING: "general",
        TaskType.LABELING: "general",
        TaskType.QUALITY_REVIEW: "expert",
        TaskType.FORMAT_CONVERSION: "general",
        TaskType.EDGE_CASE_HANDLING: "professional",
    }

    def __init__(self, region: str = "china"):
        """
        Initialize the splitter.

        Args:
            region: Region for cost calculation ('us', 'china', 'europe', 'india')
        """
        self.region = region
        self.rates = self.HOURLY_RATES.get(region, self.HOURLY_RATES["us"])

    def analyze(
        self,
        dataset_size: int,
        task_types: Optional[list[TaskType]] = None,
        context_count: Optional[int] = None,
        rubrics_per_task: float = 15.0,
        custom_hours: Optional[dict[TaskType, float]] = None,
    ) -> HumanMachineAllocation:
        """
        Analyze and allocate tasks between humans and machines.

        Args:
            dataset_size: Number of examples in the dataset
            task_types: Types of tasks involved (default: all)
            context_count: Number of unique contexts (default: dataset_size/4)
            rubrics_per_task: Average rubrics per task
            custom_hours: Custom hours per 1K for specific tasks

        Returns:
            HumanMachineAllocation with complete analysis
        """
        if task_types is None:
            task_types = list(TaskType)

        if context_count is None:
            context_count = max(1, dataset_size // 4)

        result = HumanMachineAllocation()
        scale_factor = dataset_size / 1000

        for i, task_type in enumerate(task_types):
            task = self._create_task_allocation(
                task_id=f"T{i + 1:02d}",
                task_type=task_type,
                scale_factor=scale_factor,
                context_count=context_count,
                rubrics_per_task=rubrics_per_task,
                custom_hours=custom_hours,
            )
            result.tasks.append(task)

            # Categorize
            if task.decision == AllocationDecision.HUMAN_ONLY:
                result.human_only_tasks.append(task)
            elif task.decision == AllocationDecision.MACHINE_ONLY:
                result.machine_only_tasks.append(task)
            else:
                result.hybrid_tasks.append(task)

        # Calculate totals
        self._calculate_totals(result)

        return result

    def _create_task_allocation(
        self,
        task_id: str,
        task_type: TaskType,
        scale_factor: float,
        context_count: int,
        rubrics_per_task: float,
        custom_hours: Optional[dict[TaskType, float]],
    ) -> TaskAllocation:
        """Create allocation for a single task."""

        # Get base hours
        if custom_hours and task_type in custom_hours:
            base_hours = custom_hours[task_type]
        else:
            base_hours = self.HOURS_PER_1K.get(task_type, 50.0)

        # Adjust for task type specifics
        if task_type == TaskType.CONTEXT_CREATION:
            total_hours = base_hours * (context_count / 1000)
        elif task_type == TaskType.RUBRICS_WRITING:
            total_hours = base_hours * scale_factor * (rubrics_per_task / 15)
        else:
            total_hours = base_hours * scale_factor

        # Get automation feasibility
        automation = self.AUTOMATION_FEASIBILITY.get(task_type, 0.5)

        # Determine allocation decision
        if automation >= 0.9:
            decision = AllocationDecision.MACHINE_ONLY
            human_pct = 5.0  # Minimal oversight
            machine_pct = 95.0
        elif automation >= 0.7:
            decision = AllocationDecision.MACHINE_PRIMARY
            human_pct = 20.0
            machine_pct = 80.0
        elif automation >= 0.4:
            decision = AllocationDecision.HUMAN_PRIMARY
            human_pct = 70.0
            machine_pct = 30.0
        else:
            decision = AllocationDecision.HUMAN_ONLY
            human_pct = 95.0
            machine_pct = 5.0

        # Calculate costs
        expertise = self.EXPERTISE_REQUIRED.get(task_type, "general")
        hourly_rate = self.rates.get(expertise, self.rates["general"])

        human_hours = total_hours * (human_pct / 100)
        human_cost = human_hours * hourly_rate

        # Machine cost (estimate based on API calls)
        machine_cost = self._estimate_machine_cost(task_type, scale_factor, machine_pct)

        # Build task name and description
        task_name = task_type.value.replace("_", " ").title()
        description = self._get_task_description(task_type)
        rationale = self._get_rationale(task_type, automation)
        risks = self._get_risk_factors(task_type, decision)

        return TaskAllocation(
            task_id=task_id,
            task_name=task_name,
            task_type=task_type,
            description=description,
            decision=decision,
            human_percentage=human_pct,
            machine_percentage=machine_pct,
            human_role=expertise,
            human_hours=human_hours,
            human_hourly_rate=hourly_rate,
            human_cost=human_cost,
            machine_method=self._get_machine_method(task_type),
            machine_cost=machine_cost,
            automation_confidence=automation,
            rationale=rationale,
            risk_factors=risks,
        )

    def _estimate_machine_cost(
        self, task_type: TaskType, scale_factor: float, machine_pct: float
    ) -> float:
        """Estimate machine/API costs."""
        # Base costs per 1K examples
        base_costs = {
            TaskType.DATA_GENERATION: 50.0,  # LLM API calls
            TaskType.DATA_FILTERING: 10.0,  # Compute
            TaskType.FORMAT_CONVERSION: 5.0,  # Compute
            TaskType.LABELING: 30.0,  # Classification API
        }

        base = base_costs.get(task_type, 5.0)
        return base * scale_factor * (machine_pct / 100)

    def _get_task_description(self, task_type: TaskType) -> str:
        """Get description for a task type."""
        descriptions = {
            TaskType.CONTEXT_CREATION: "Create original context documents (rules, scenarios, documents)",
            TaskType.TASK_DESIGN: "Design questions/tasks that test comprehension",
            TaskType.RUBRICS_WRITING: "Write evaluation criteria for responses",
            TaskType.DATA_COLLECTION: "Collect raw data from sources",
            TaskType.DATA_GENERATION: "Generate synthetic data using LLMs",
            TaskType.DATA_FILTERING: "Filter and clean data based on rules",
            TaskType.LABELING: "Label/annotate data examples",
            TaskType.QUALITY_REVIEW: "Review data for quality issues",
            TaskType.FORMAT_CONVERSION: "Convert data between formats",
            TaskType.EDGE_CASE_HANDLING: "Handle ambiguous/difficult cases",
        }
        return descriptions.get(task_type, "")

    def _get_machine_method(self, task_type: TaskType) -> str:
        """Get machine method for a task type."""
        methods = {
            TaskType.DATA_GENERATION: "LLM API (GPT-4, Claude)",
            TaskType.DATA_FILTERING: "Rule-based filtering + ML classifiers",
            TaskType.FORMAT_CONVERSION: "Automated scripts",
            TaskType.LABELING: "Semi-automated classification",
            TaskType.RUBRICS_WRITING: "Template-based generation",
        }
        return methods.get(task_type, "N/A")

    def _get_rationale(self, task_type: TaskType, automation: float) -> str:
        """Get rationale for allocation decision."""
        if automation < 0.3:
            return f"Requires human creativity/judgment (automation feasibility: {automation:.0%})"
        elif automation < 0.6:
            return f"Benefits from human oversight (automation feasibility: {automation:.0%})"
        elif automation < 0.9:
            return (
                f"Mostly automatable with human review (automation feasibility: {automation:.0%})"
            )
        else:
            return f"Fully automatable (automation feasibility: {automation:.0%})"

    def _get_risk_factors(self, task_type: TaskType, decision: AllocationDecision) -> list[str]:
        """Get risk factors for allocation."""
        risks = []

        if decision in [AllocationDecision.MACHINE_ONLY, AllocationDecision.MACHINE_PRIMARY]:
            if task_type in [TaskType.CONTEXT_CREATION, TaskType.TASK_DESIGN]:
                risks.append("Quality may suffer without human creativity")
            if task_type == TaskType.QUALITY_REVIEW:
                risks.append("May miss subtle quality issues")

        if decision == AllocationDecision.HUMAN_ONLY:
            risks.append("Higher cost and longer timeline")
            risks.append("Scaling challenges")

        return risks

    def _calculate_totals(self, result: HumanMachineAllocation) -> None:
        """Calculate total statistics."""
        result.total_human_hours = sum(t.human_hours for t in result.tasks)
        result.total_human_cost = sum(t.human_cost for t in result.tasks)
        result.total_machine_cost = sum(t.machine_cost for t in result.tasks)
        result.total_cost = result.total_human_cost + result.total_machine_cost

        # Calculate work percentages
        total_hours = sum(
            t.human_hours / (t.human_percentage / 100) if t.human_percentage > 0 else 0
            for t in result.tasks
        )
        if total_hours > 0:
            result.human_work_percentage = (result.total_human_hours / total_hours) * 100
            result.machine_work_percentage = 100 - result.human_work_percentage

        # Estimate savings (vs all-human)
        all_human_cost = sum(
            t.human_hours / (t.human_percentage / 100) * t.human_hourly_rate
            if t.human_percentage > 0
            else t.human_cost
            for t in result.tasks
        )
        result.estimated_savings_vs_all_human = max(0, all_human_cost - result.total_cost)

        # Timeline reduction (rough estimate)
        if all_human_cost > 0:
            result.timeline_reduction_percentage = (
                result.estimated_savings_vs_all_human / all_human_cost
            ) * 100

    def to_dict(self, result: HumanMachineAllocation) -> dict:
        """Convert result to dictionary for JSON export."""
        return {
            "summary": {
                "total_tasks": len(result.tasks),
                "human_only_count": len(result.human_only_tasks),
                "machine_only_count": len(result.machine_only_tasks),
                "hybrid_count": len(result.hybrid_tasks),
                "total_human_hours": result.total_human_hours,
                "total_human_cost": result.total_human_cost,
                "total_machine_cost": result.total_machine_cost,
                "total_cost": result.total_cost,
                "human_work_percentage": result.human_work_percentage,
                "machine_work_percentage": result.machine_work_percentage,
                "estimated_savings": result.estimated_savings_vs_all_human,
            },
            "tasks": [
                {
                    "task_id": t.task_id,
                    "task_name": t.task_name,
                    "task_type": t.task_type.value,
                    "decision": t.decision.value,
                    "human_percentage": t.human_percentage,
                    "human_hours": t.human_hours,
                    "human_cost": t.human_cost,
                    "machine_cost": t.machine_cost,
                    "rationale": t.rationale,
                    "risks": t.risk_factors,
                }
                for t in result.tasks
            ],
        }
