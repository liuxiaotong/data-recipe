"""DataRecipe - AI dataset analyzer: reverse-engineer datasets, estimate costs, analyze quality, generate workflows."""

__version__ = "0.3.0"

from datarecipe.analyzer import DatasetAnalyzer
from datarecipe.schema import (
    # v2 additions
    AnnotatorProfile,
    Cost,
    DataRecipe,
    DeploymentProvider,
    EnhancedCost,
    GenerationMethod,
    ProductionConfig,
    Recipe,
    Reproducibility,
)


# Lazy imports for optional modules to avoid requiring all dependencies
def __getattr__(name):
    """Lazy import for optional modules."""
    if name == "CostCalculator":
        from datarecipe.cost_calculator import CostCalculator

        return CostCalculator
    if name == "EnhancedCostCalculator":
        from datarecipe.cost_calculator import EnhancedCostCalculator

        return EnhancedCostCalculator
    if name == "QualityAnalyzer":
        from datarecipe.quality_metrics import QualityAnalyzer

        return QualityAnalyzer
    if name == "BatchAnalyzer":
        from datarecipe.batch_analyzer import BatchAnalyzer

        return BatchAnalyzer
    if name == "DatasetComparator":
        from datarecipe.comparator import DatasetComparator

        return DatasetComparator
    if name == "WorkflowGenerator":
        from datarecipe.workflow import WorkflowGenerator

        return WorkflowGenerator
    if name == "AnnotatorProfiler":
        from datarecipe.profiler import AnnotatorProfiler

        return AnnotatorProfiler
    if name == "ProductionDeployer":
        from datarecipe.deployer import ProductionDeployer

        return ProductionDeployer
    # Task profiles
    if name == "task_profiles":
        from datarecipe import task_profiles

        return task_profiles
    if name == "get_task_profile":
        from datarecipe.task_profiles import get_task_profile

        return get_task_profile
    if name == "TaskTypeProfile":
        from datarecipe.task_profiles import TaskTypeProfile

        return TaskTypeProfile
    # Provider-related
    if name == "get_provider":
        from datarecipe.providers import get_provider

        return get_provider
    if name == "list_providers":
        from datarecipe.providers import list_providers

        return list_providers
    if name == "discover_providers":
        from datarecipe.providers import discover_providers

        return discover_providers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core schema
    "Recipe",
    "Cost",
    "Reproducibility",
    "GenerationMethod",
    "DatasetAnalyzer",
    # v2 schema additions
    "AnnotatorProfile",
    "ProductionConfig",
    "EnhancedCost",
    "DataRecipe",
    "DeploymentProvider",
    # Optional modules (lazy loaded)
    "CostCalculator",
    "EnhancedCostCalculator",
    "QualityAnalyzer",
    "BatchAnalyzer",
    "DatasetComparator",
    "WorkflowGenerator",
    "AnnotatorProfiler",
    "ProductionDeployer",
    # Provider functions (lazy loaded)
    "get_provider",
    "list_providers",
    "discover_providers",
]
