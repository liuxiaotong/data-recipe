"""Centralized constants for DataRecipe.

Cross-cutting values that are referenced by multiple modules.
File-local constants (used in only one file) remain in their original location.
"""

# =============================================================================
# LLM Default Models
# =============================================================================

# Default model for API calls (Anthropic)
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Default model for API calls (OpenAI)
DEFAULT_OPENAI_MODEL = "gpt-4o"

# Default fallback model for cost estimation
DEFAULT_COST_ESTIMATION_MODEL = "gpt-4o"


# =============================================================================
# Region Cost Multipliers
# =============================================================================

# Unified region multiplier table - single source of truth
# Used by: cost_calculator.py, phased_model.py, profiler.py, spec_output.py
REGION_COST_MULTIPLIERS: dict[str, float] = {
    "us": 1.0,
    "uk": 0.9,
    "eu": 0.85,
    "europe": 0.85,  # alias
    "cn": 0.4,
    "china": 0.4,  # alias
    "in": 0.25,
    "india": 0.25,  # alias
    "sea": 0.3,
    "latam": 0.35,
}


# =============================================================================
# Project Scale Thresholds
# =============================================================================

# Sample count boundaries for project scale classification
SCALE_SMALL_MAX = 1_000
SCALE_MEDIUM_MAX = 10_000
SCALE_LARGE_MAX = 100_000
# > SCALE_LARGE_MAX → enterprise


# =============================================================================
# Quality & Acceptance Criteria
# =============================================================================

# Inter-annotator agreement (Cohen's Kappa)
ACCEPTANCE_KAPPA_THRESHOLD = 0.7

# Expert review pass rate targets
ACCEPTANCE_EXPERT_REVIEW_PASS_RATE = 0.90
ACCEPTANCE_EXPERT_REVIEW_PASS_RATE_HIGH = 0.95

# Quality sampling rates
QA_SAMPLING_RATE_PILOT = 0.20
QA_SAMPLING_RATE_PRODUCTION = 0.05

# Maximum acceptable rework rate
ACCEPTANCE_MAX_REWORK_RATE = 0.10

# Quality pass rate target
ACCEPTANCE_QUALITY_PASS_RATE = 0.95

# Risk contingency buffer (15%)
CONTINGENCY_BUFFER_RATE = 0.15


# =============================================================================
# Duration & Workload
# =============================================================================

# Human annotation throughput
HUMAN_ANNOTATION_SAMPLES_PER_DAY = 100

# Minimum project duration regardless of scale
MIN_PROJECT_DURATION_DAYS = 10

# Contingency buffer applied to duration
PROJECT_DURATION_BUFFER_FACTOR = 1.2


# =============================================================================
# Team Sizing Defaults
# =============================================================================

BASE_TEAM = {
    "项目经理": 1,
    "领域专家": 2,
    "QA": 1,
}

# Annotator count by project scale (sample count thresholds)
ANNOTATOR_COUNTS = {
    "large": (SCALE_MEDIUM_MAX, 8),    # >= 10,000 samples → 8 annotators
    "medium": (SCALE_SMALL_MAX, 4),    # >= 1,000 samples → 4 annotators
    "small": (0, 2),                   # default → 2 annotators
}
