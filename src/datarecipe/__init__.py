"""DataRecipe - AI dataset analyzer: reverse-engineer datasets, estimate costs, analyze quality, generate workflows."""

__version__ = "0.2.0"

from datarecipe.schema import Recipe, Cost, Reproducibility, GenerationMethod
from datarecipe.analyzer import DatasetAnalyzer

# Lazy imports for optional modules to avoid requiring all dependencies
def __getattr__(name):
    """Lazy import for optional modules."""
    if name == "CostCalculator":
        from datarecipe.cost_calculator import CostCalculator
        return CostCalculator
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Recipe",
    "Cost",
    "Reproducibility",
    "GenerationMethod",
    "DatasetAnalyzer",
    # Optional modules (lazy loaded)
    "CostCalculator",
    "QualityAnalyzer",
    "BatchAnalyzer",
    "DatasetComparator",
    "WorkflowGenerator",
]
