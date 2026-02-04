"""Data classes for representing dataset recipes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable
import json


class GenerationType(Enum):
    """Type of data generation."""

    SYNTHETIC = "synthetic"
    HUMAN = "human"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SourceType(Enum):
    """Type of data source."""

    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    OPENAI = "openai"
    LOCAL = "local"
    WEB = "web"
    UNKNOWN = "unknown"


@dataclass
class Cost:
    """Estimated cost breakdown for dataset creation."""

    estimated_total_usd: Optional[float] = None
    api_calls_usd: Optional[float] = None
    human_annotation_usd: Optional[float] = None
    compute_usd: Optional[float] = None
    confidence: str = "low"  # low, medium, high
    # New fields for detailed estimation
    low_estimate_usd: Optional[float] = None
    high_estimate_usd: Optional[float] = None
    assumptions: list[str] = field(default_factory=list)
    tokens_estimated: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "estimated_total_usd": self.estimated_total_usd,
            "breakdown": {
                "api_calls": self.api_calls_usd,
                "human_annotation": self.human_annotation_usd,
                "compute": self.compute_usd,
            },
            "confidence": self.confidence,
        }
        if self.low_estimate_usd is not None:
            result["low_estimate_usd"] = self.low_estimate_usd
        if self.high_estimate_usd is not None:
            result["high_estimate_usd"] = self.high_estimate_usd
        if self.assumptions:
            result["assumptions"] = self.assumptions
        if self.tokens_estimated is not None:
            result["tokens_estimated"] = self.tokens_estimated
        return result


@dataclass
class Reproducibility:
    """Reproducibility assessment for a dataset."""

    score: int  # 1-10
    available: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "available": self.available,
            "missing": self.missing,
            "notes": self.notes,
        }


@dataclass
class GenerationMethod:
    """Details about how data was generated."""

    method_type: str  # distillation, human_annotation, web_scrape, etc.
    teacher_model: Optional[str] = None
    prompt_template_available: bool = False
    platform: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {"type": self.method_type}
        if self.teacher_model:
            result["teacher_model"] = self.teacher_model
        if self.prompt_template_available:
            result["prompt_template"] = "available"
        if self.platform:
            result["platform"] = self.platform
        if self.details:
            result.update(self.details)
        return result


@dataclass
class Recipe:
    """Complete recipe for a dataset."""

    name: str
    version: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    source_id: Optional[str] = None

    # Dataset characteristics
    size: Optional[int] = None
    num_examples: Optional[int] = None
    languages: list[str] = field(default_factory=list)
    license: Optional[str] = None
    description: Optional[str] = None

    # Generation details
    generation_type: GenerationType = GenerationType.UNKNOWN
    synthetic_ratio: Optional[float] = None
    human_ratio: Optional[float] = None
    generation_methods: list[GenerationMethod] = field(default_factory=list)
    teacher_models: list[str] = field(default_factory=list)

    # Assessment
    cost: Optional[Cost] = None
    reproducibility: Optional[Reproducibility] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    created_date: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    paper_url: Optional[str] = None
    homepage_url: Optional[str] = None

    # Quality metrics (populated by quality analyzer)
    quality_metrics: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML export."""
        result = {
            "name": self.name,
            "source": {
                "type": self.source_type.value,
                "id": self.source_id,
            },
        }

        if self.version:
            result["version"] = self.version

        # Generation info
        generation = {}
        if self.synthetic_ratio is not None:
            generation["synthetic_ratio"] = self.synthetic_ratio
        if self.human_ratio is not None:
            generation["human_ratio"] = self.human_ratio
        if self.generation_methods:
            generation["methods"] = [m.to_dict() for m in self.generation_methods]
        if self.teacher_models:
            generation["teacher_models"] = self.teacher_models
        if generation:
            result["generation"] = generation

        # Cost
        if self.cost:
            result["cost"] = self.cost.to_dict()

        # Reproducibility
        if self.reproducibility:
            result["reproducibility"] = self.reproducibility.to_dict()

        # Metadata
        metadata = {}
        if self.size:
            metadata["size_bytes"] = self.size
        if self.num_examples:
            metadata["num_examples"] = self.num_examples
        if self.languages:
            metadata["languages"] = self.languages
        if self.license:
            metadata["license"] = self.license
        if self.tags:
            metadata["tags"] = self.tags
        if self.authors:
            metadata["authors"] = self.authors
        if self.paper_url:
            metadata["paper_url"] = self.paper_url
        if metadata:
            result["metadata"] = metadata

        return result

    def to_yaml(self) -> str:
        """Export recipe as YAML string."""
        import yaml

        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        )


# =============================================================================
# V2 Extended Data Models
# =============================================================================


class ExperienceLevel(Enum):
    """Annotator experience level."""
    JUNIOR = "junior"          # 初级：0-1年
    MID = "mid"                # 中级：1-3年
    SENIOR = "senior"          # 高级：3-5年
    EXPERT = "expert"          # 专家：5年+


class EducationLevel(Enum):
    """学历要求"""
    HIGH_SCHOOL = "high_school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    PROFESSIONAL = "professional"  # 持证专业人士


class ReviewWorkflow(Enum):
    """审核流程"""
    SINGLE = "single"          # 单人标注
    DOUBLE = "double"          # 双人交叉验证
    EXPERT = "expert"          # 专家审核


@dataclass
class SkillRequirement:
    """技能要求"""
    skill_type: str            # programming, domain, language, tool, certification
    name: str                  # Python, 医疗, 英语, Excel
    level: str                 # basic, intermediate, advanced, native, required
    required: bool = True      # 必须 vs 加分项
    details: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.skill_type,
            "name": self.name,
            "level": self.level,
            "required": self.required,
            "details": self.details,
        }


@dataclass
class AnnotatorProfile:
    """标注专家画像——开放标准，任何平台都能消费"""

    # 技能要求
    skill_requirements: list[SkillRequirement] = field(default_factory=list)

    # 经验要求
    experience_level: ExperienceLevel = ExperienceLevel.MID
    min_experience_years: int = 1

    # 语言要求
    language_requirements: list[str] = field(default_factory=list)  # ["zh-CN:native", "en:C1"]

    # 领域知识
    domain_knowledge: list[str] = field(default_factory=list)  # ["医疗", "金融"]

    # 学历要求
    education_level: EducationLevel = EducationLevel.BACHELOR

    # 团队规模
    team_size: int = 10
    team_structure: dict = field(default_factory=lambda: {"annotator": 8, "reviewer": 2})

    # 工作量估算
    estimated_person_days: float = 0.0
    estimated_hours_per_example: float = 0.0

    # 费率参考
    hourly_rate_range: dict = field(default_factory=lambda: {
        "min": 15, "max": 45, "currency": "USD"
    })

    # 筛选标准
    screening_criteria: list[str] = field(default_factory=list)

    # 推荐平台
    recommended_platforms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "skill_requirements": [s.to_dict() for s in self.skill_requirements],
            "experience": {
                "level": self.experience_level.value,
                "min_years": self.min_experience_years,
            },
            "language_requirements": self.language_requirements,
            "domain_knowledge": self.domain_knowledge,
            "education_level": self.education_level.value,
            "team": {
                "size": self.team_size,
                "structure": self.team_structure,
            },
            "workload": {
                "estimated_person_days": self.estimated_person_days,
                "hours_per_example": self.estimated_hours_per_example,
            },
            "hourly_rate_range": self.hourly_rate_range,
            "screening_criteria": self.screening_criteria,
            "recommended_platforms": self.recommended_platforms,
        }

    def to_yaml(self) -> str:
        """Export as YAML string."""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class QualityRule:
    """Quality check rule."""
    rule_id: str
    name: str
    description: str
    check_type: str            # format, content, consistency, semantic
    severity: str              # error, warning, info
    auto_check: bool = True    # 是否可自动检查
    check_code: Optional[str] = None  # 检查代码/正则

    def to_dict(self) -> dict:
        return {
            "id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "type": self.check_type,
            "severity": self.severity,
            "auto_check": self.auto_check,
        }


@dataclass
class AcceptanceCriterion:
    """验收标准"""
    criterion_id: str
    name: str
    description: str
    threshold: float           # 达标阈值
    metric_type: str           # accuracy, agreement, completeness, etc.
    priority: str = "required" # required, recommended, optional

    def to_dict(self) -> dict:
        return {
            "id": self.criterion_id,
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "type": self.metric_type,
            "priority": self.priority,
        }


@dataclass
class Milestone:
    """项目里程碑"""
    name: str
    description: str
    deliverables: list[str]
    estimated_days: int
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "deliverables": self.deliverables,
            "estimated_days": self.estimated_days,
            "dependencies": self.dependencies,
        }


@dataclass
class ProductionConfig:
    """投产配置——平台无关的项目定义"""

    # 标注指南
    annotation_guide: str = ""  # Markdown 内容
    annotation_guide_url: Optional[str] = None

    # 质检规则
    quality_rules: list[QualityRule] = field(default_factory=list)

    # 验收标准
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)

    # 审核流程
    review_workflow: ReviewWorkflow = ReviewWorkflow.DOUBLE
    review_sample_rate: float = 0.1  # 抽检比例

    # 时间规划
    estimated_timeline_days: int = 30
    milestones: list[Milestone] = field(default_factory=list)

    # 工具配置
    labeling_tool_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "annotation_guide": self.annotation_guide[:500] + "..." if len(self.annotation_guide) > 500 else self.annotation_guide,
            "annotation_guide_url": self.annotation_guide_url,
            "quality_rules": [r.to_dict() for r in self.quality_rules],
            "acceptance_criteria": [c.to_dict() for c in self.acceptance_criteria],
            "review": {
                "workflow": self.review_workflow.value,
                "sample_rate": self.review_sample_rate,
            },
            "timeline": {
                "estimated_days": self.estimated_timeline_days,
                "milestones": [m.to_dict() for m in self.milestones],
            },
        }


@dataclass
class EnhancedCost:
    """增强的成本估算——包含人力成本"""

    # 现有字段
    api_cost: float = 0.0
    compute_cost: float = 0.0

    # 新增人力成本细分
    human_cost: float = 0.0
    human_cost_breakdown: dict = field(default_factory=lambda: {
        "annotation": 0.0,
        "review": 0.0,
        "expert_consultation": 0.0,
        "project_management": 0.0,
    })

    # 地区系数
    region: str = "us"
    region_multiplier: float = 1.0

    # 总成本
    total_cost: float = 0.0
    total_range: dict = field(default_factory=lambda: {"low": 0.0, "high": 0.0})

    # 置信度
    confidence: str = "medium"
    assumptions: list[str] = field(default_factory=list)

    # ROI 分析
    estimated_dataset_value: Optional[float] = None
    roi_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "api_cost": self.api_cost,
            "compute_cost": self.compute_cost,
            "human_cost": self.human_cost,
            "human_cost_breakdown": self.human_cost_breakdown,
            "region": self.region,
            "region_multiplier": self.region_multiplier,
            "total_cost": self.total_cost,
            "total_range": self.total_range,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
        }


@dataclass
class DataRecipe:
    """Complete data recipe - V2 version with all modules integrated."""

    # === Core fields (from Recipe) ===
    name: str
    version: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    source_id: Optional[str] = None

    size: Optional[int] = None
    num_examples: Optional[int] = None
    languages: list[str] = field(default_factory=list)
    license: Optional[str] = None
    description: Optional[str] = None

    generation_type: GenerationType = GenerationType.UNKNOWN
    synthetic_ratio: Optional[float] = None
    human_ratio: Optional[float] = None
    generation_methods: list[GenerationMethod] = field(default_factory=list)
    teacher_models: list[str] = field(default_factory=list)

    cost: Optional[Cost] = None
    reproducibility: Optional[Reproducibility] = None
    quality_metrics: Optional[dict] = None

    tags: list[str] = field(default_factory=list)
    created_date: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    paper_url: Optional[str] = None
    homepage_url: Optional[str] = None

    # === V2 extended fields ===
    annotator_profile: Optional[AnnotatorProfile] = None
    production_config: Optional[ProductionConfig] = None
    enhanced_cost: Optional[EnhancedCost] = None

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> "DataRecipe":
        """Create DataRecipe from an existing Recipe object."""
        return cls(
            name=recipe.name,
            version=recipe.version,
            source_type=recipe.source_type,
            source_id=recipe.source_id,
            size=recipe.size,
            num_examples=recipe.num_examples,
            languages=recipe.languages.copy() if recipe.languages else [],
            license=recipe.license,
            description=recipe.description,
            generation_type=recipe.generation_type,
            synthetic_ratio=recipe.synthetic_ratio,
            human_ratio=recipe.human_ratio,
            generation_methods=recipe.generation_methods.copy() if recipe.generation_methods else [],
            teacher_models=recipe.teacher_models.copy() if recipe.teacher_models else [],
            cost=recipe.cost,
            reproducibility=recipe.reproducibility,
            quality_metrics=recipe.quality_metrics,
            tags=recipe.tags.copy() if recipe.tags else [],
            created_date=recipe.created_date,
            authors=recipe.authors.copy() if recipe.authors else [],
            paper_url=recipe.paper_url,
            homepage_url=recipe.homepage_url,
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        result = {
            "name": self.name,
            "source": {
                "type": self.source_type.value,
                "id": self.source_id,
            },
        }

        if self.version:
            result["version"] = self.version

        # Generation info
        generation = {}
        if self.synthetic_ratio is not None:
            generation["synthetic_ratio"] = self.synthetic_ratio
        if self.human_ratio is not None:
            generation["human_ratio"] = self.human_ratio
        if self.generation_methods:
            generation["methods"] = [m.to_dict() for m in self.generation_methods]
        if self.teacher_models:
            generation["teacher_models"] = self.teacher_models
        if generation:
            result["generation"] = generation

        # Cost
        if self.cost:
            result["cost"] = self.cost.to_dict()

        # Enhanced cost
        if self.enhanced_cost:
            result["enhanced_cost"] = self.enhanced_cost.to_dict()

        # Reproducibility
        if self.reproducibility:
            result["reproducibility"] = self.reproducibility.to_dict()

        # Annotator profile
        if self.annotator_profile:
            result["annotator_profile"] = self.annotator_profile.to_dict()

        # Production config
        if self.production_config:
            result["production_config"] = self.production_config.to_dict()

        # Metadata
        metadata = {}
        if self.size:
            metadata["size_bytes"] = self.size
        if self.num_examples:
            metadata["num_examples"] = self.num_examples
        if self.languages:
            metadata["languages"] = self.languages
        if self.license:
            metadata["license"] = self.license
        if self.tags:
            metadata["tags"] = self.tags
        if self.authors:
            metadata["authors"] = self.authors
        if self.paper_url:
            metadata["paper_url"] = self.paper_url
        if metadata:
            result["metadata"] = metadata

        return result

    def to_yaml(self) -> str:
        """Export as YAML string."""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, allow_unicode=True)


# =============================================================================
# Provider Protocol Data Classes
# =============================================================================


@dataclass
class ValidationResult:
    """Configuration validation result."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnnotatorMatch:
    """Annotator matching result."""
    annotator_id: str
    name: str
    match_score: float         # 0-1 match score
    skills_matched: list[str] = field(default_factory=list)
    skills_missing: list[str] = field(default_factory=list)
    hourly_rate: float = 0.0
    availability: str = "unknown"  # available, busy, unavailable


@dataclass
class ProjectHandle:
    """项目句柄"""
    project_id: str
    provider: str
    created_at: str
    status: str
    url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """部署结果"""
    success: bool
    project_handle: Optional[ProjectHandle] = None
    error: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class ProjectStatus:
    """项目状态"""
    status: str                # pending, in_progress, completed, failed
    progress: float = 0.0      # 0-100
    completed_count: int = 0
    total_count: int = 0
    quality_score: Optional[float] = None
    estimated_completion: Optional[str] = None


@runtime_checkable
class DeploymentProvider(Protocol):
    """Provider 接口协议——第三方实现此协议即可接入"""

    @property
    def name(self) -> str:
        """Provider 名称"""
        ...

    @property
    def description(self) -> str:
        """Provider 描述"""
        ...

    def validate_config(self, config: ProductionConfig) -> ValidationResult:
        """验证投产配置是否符合平台要求"""
        ...

    def match_annotators(
        self,
        profile: AnnotatorProfile,
        limit: int = 10,
    ) -> list[AnnotatorMatch]:
        """根据画像匹配标注者"""
        ...

    def create_project(
        self,
        recipe: DataRecipe,
        config: Optional[ProductionConfig] = None,
    ) -> ProjectHandle:
        """在平台上创建项目"""
        ...

    def submit(self, project: ProjectHandle) -> DeploymentResult:
        """提交项目开始执行"""
        ...

    def get_status(self, project: ProjectHandle) -> ProjectStatus:
        """获取项目状态"""
        ...

    def cancel(self, project: ProjectHandle) -> bool:
        """取消项目"""
        ...
