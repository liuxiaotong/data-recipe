# DataRecipe v2 架构设计文档

> **版本**: 2.0 Draft
> **作者**: DataRecipe 架构组
> **最后更新**: 2024-12

## 目录

1. [项目定位与设计原则](#1-项目定位与设计原则)
2. [现有功能清单](#2-现有功能清单)
3. [v2 分层架构](#3-v2-分层架构)
4. [核心数据模型](#4-核心数据模型)
5. [新增模块设计](#5-新增模块设计)
6. [CLI 接口设计](#6-cli-接口设计)
7. [Provider 插件系统](#7-provider-插件系统)
8. [目录结构规划](#8-目录结构规划)
9. [开源社区设计](#9-开源社区设计)
10. [实现路线图](#10-实现路线图)

---

## 1. 项目定位与设计原则

### 1.1 双重身份

DataRecipe 同时服务两类用户：

| 身份 | 描述 | 使用方式 |
|------|------|----------|
| **开源工具** | 独立的 AI 数据集逆向工程框架 | `pip install datarecipe` |
| **生产武器** | 集识光年内部的数据生产力工具 | `pip install datarecipe-judgeguild` |

### 1.2 核心设计原则

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 开源核心零耦合：Core Engine 不 import 任何 provider 实现 │
│ 2. Provider 通过 entry_points 注册：pip install 自动发现    │
│ 3. 默认 provider = LocalFiles：不装插件也能完整使用         │
│ 4. 数据模型即协议：标准化 schema，任何 provider 都消费同一套 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 现有功能清单

### 2.1 当前模块架构图

```
src/datarecipe/
├── __init__.py          # 包入口，懒加载可选模块
├── schema.py            # 核心数据模型：Recipe, Cost, Reproducibility
├── analyzer.py          # 主分析器：DatasetAnalyzer
├── cli.py               # CLI 入口：click 命令组
│
├── sources/             # 数据源提取器
│   ├── huggingface.py   # HuggingFace Hub 提取
│   ├── github.py        # GitHub 仓库提取
│   └── web.py           # 通用 Web URL 提取
│
├── cost_calculator.py   # 成本估算器
├── quality_metrics.py   # 质量分析器
├── deep_analyzer.py     # 深度分析（论文/网页）
├── llm_analyzer.py      # LLM 增强分析
├── batch_analyzer.py    # 批量分析
├── comparator.py        # 数据集对比
├── pipeline.py          # 流程模板
└── workflow.py          # 工作流生成
```

### 2.2 现有 CLI 命令

| 命令 | 功能 | 状态 |
|------|------|------|
| `analyze` | 分析数据集，输出配方 | ✅ 完整 |
| `cost` | 估算复制成本 | ✅ 完整 |
| `quality` | 分析数据质量 | ✅ 完整 |
| `guide` | 生成生产指南 | ✅ 完整 |
| `deep-guide` | 深度分析生产指南 | ✅ 完整 |
| `batch` | 批量分析 | ✅ 完整 |
| `compare` | 对比数据集 | ✅ 完整 |
| `workflow` | 生成复制项目 | ✅ 完整 |
| `create` | 交互式创建配方 | ✅ 完整 |
| `show` | 显示 YAML 配方 | ✅ 完整 |
| `list-sources` | 列出支持的来源 | ✅ 完整 |

### 2.3 现有数据模型

```python
# schema.py 现有模型
class GenerationType(Enum):
    SYNTHETIC, HUMAN, MIXED, UNKNOWN

class SourceType(Enum):
    HUGGINGFACE, GITHUB, OPENAI, LOCAL, WEB, UNKNOWN

@dataclass
class Cost:
    estimated_total_usd, api_calls_usd, human_annotation_usd, compute_usd
    confidence, low_estimate_usd, high_estimate_usd, assumptions

@dataclass
class Reproducibility:
    score (1-10), available[], missing[], notes

@dataclass
class GenerationMethod:
    method_type, teacher_model, prompt_template_available, platform, details

@dataclass
class Recipe:
    name, version, source_type, source_id
    size, num_examples, languages, license, description
    generation_type, synthetic_ratio, human_ratio
    generation_methods[], teacher_models[]
    cost, reproducibility, quality_metrics
    tags[], created_date, authors[], paper_url, homepage_url
```

---

## 3. v2 分层架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLI / Python API                                     │
│  datarecipe analyze | cost | quality | profile | deploy | pipeline           │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Core Engine（开源核心）                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐    │
│  │  Analyzer   │  Costing    │  Quality    │  Profiler   │  Deployer   │    │
│  │  数据分析   │  成本估算   │  质量分析   │  画像生成   │  投产部署   │    │
│  │  (现有)     │  (增强)     │  (现有)     │  (新增)     │  (新增)     │    │
│  └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                    Provider Interface（插件接口）                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  class DeploymentProvider(Protocol):                                   │  │
│  │      name: str                                                         │  │
│  │      def validate_config(config: ProductionConfig) -> ValidationResult │  │
│  │      def match_annotators(profile: AnnotatorProfile) -> list[Match]    │  │
│  │      def create_project(recipe: DataRecipe) -> ProjectHandle           │  │
│  │      def submit(project: ProjectHandle) -> DeploymentResult            │  │
│  │      def get_status(project: ProjectHandle) -> ProjectStatus           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                      Built-in Providers（内置实现）                          │
│  ┌────────────────┬─────────────────┬──────────────────────────────────┐   │
│  │   LocalFiles   │   LabelStudio   │        (社区贡献 ...)             │   │
│  │   (默认)       │   (开源示例)    │                                   │   │
│  └────────────────┴─────────────────┴──────────────────────────────────┘   │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│                   External Providers（私有/第三方，不在开源仓库）              │
│  ┌────────────────┬─────────────────┬──────────────────────────────────┐   │
│  │  Judge Guild   │    Scale AI     │     Labelbox / MTurk / ...       │   │
│  │  (集识光年)    │    (示例)       │                                   │   │
│  └────────────────┴─────────────────┴──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   analyze    │────▶│    Recipe    │────▶│   profile    │────▶│ Annotator    │
│   数据集     │     │   数据配方    │     │   画像生成    │     │ Profile      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │                                          │
                            ▼                                          ▼
                     ┌──────────────┐                           ┌──────────────┐
                     │    cost      │                           │   deploy     │
                     │   成本估算    │                           │   投产部署    │
                     └──────────────┘                           └──────────────┘
                            │                                          │
                            ▼                                          ▼
                     ┌──────────────┐                           ┌──────────────┐
                     │  Enhanced    │                           │ Production   │
                     │  Cost        │                           │ Config       │
                     │  (含人力)    │                           └──────────────┘
                     └──────────────┘                                  │
                                                                       ▼
                                                              ┌──────────────────┐
                                                              │     Provider     │
                                                              │  LocalFiles /    │
                                                              │  LabelStudio /   │
                                                              │  JudgeGuild      │
                                                              └──────────────────┘
```

---

## 4. 核心数据模型

### 4.1 新增数据模型

```python
# src/datarecipe/schema.py 新增

from dataclasses import dataclass, field
from typing import Protocol, Optional, Any
from enum import Enum


class ExperienceLevel(Enum):
    """标注者经验等级"""
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
    skill_type: str            # programming, domain, language, tool
    name: str                  # Python, 医疗, 英语, Excel
    level: str                 # basic, intermediate, advanced, native
    required: bool = True      # 必须 vs 加分项
    details: Optional[str] = None


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
            "skill_requirements": [
                {"type": s.skill_type, "name": s.name, "level": s.level, "required": s.required}
                for s in self.skill_requirements
            ],
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
        """导出为 YAML"""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def to_json(self) -> str:
        """导出为 JSON"""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class QualityRule:
    """质检规则"""
    rule_id: str
    name: str
    description: str
    check_type: str            # format, content, consistency, semantic
    severity: str              # error, warning, info
    auto_check: bool = True    # 是否可自动检查
    check_code: Optional[str] = None  # 检查代码/正则


@dataclass
class AcceptanceCriterion:
    """验收标准"""
    criterion_id: str
    name: str
    description: str
    threshold: float           # 达标阈值
    metric_type: str           # accuracy, agreement, completeness, etc.
    priority: str = "required" # required, recommended, optional


@dataclass
class Milestone:
    """项目里程碑"""
    name: str
    description: str
    deliverables: list[str]
    estimated_days: int
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ProductionConfig:
    """投产配置——平台无关的项目定义"""

    # 标注指南
    annotation_guide: str      # Markdown 内容
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
            "quality_rules": [
                {"id": r.rule_id, "name": r.name, "type": r.check_type, "severity": r.severity}
                for r in self.quality_rules
            ],
            "acceptance_criteria": [
                {"id": c.criterion_id, "name": c.name, "threshold": c.threshold, "type": c.metric_type}
                for c in self.acceptance_criteria
            ],
            "review": {
                "workflow": self.review_workflow.value,
                "sample_rate": self.review_sample_rate,
            },
            "timeline": {
                "estimated_days": self.estimated_timeline_days,
                "milestones": [
                    {"name": m.name, "days": m.estimated_days, "deliverables": m.deliverables}
                    for m in self.milestones
                ],
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


@dataclass
class DataRecipe:
    """完整的数据配方——v2 版本，串联所有模块"""

    # === 现有字段（来自 Recipe）===
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

    # === v2 新增字段 ===
    annotator_profile: Optional[AnnotatorProfile] = None
    production_config: Optional[ProductionConfig] = None
    enhanced_cost: Optional[EnhancedCost] = None

    @classmethod
    def from_recipe(cls, recipe: "Recipe") -> "DataRecipe":
        """从现有 Recipe 对象创建 DataRecipe"""
        return cls(
            name=recipe.name,
            version=recipe.version,
            source_type=recipe.source_type,
            source_id=recipe.source_id,
            size=recipe.size,
            num_examples=recipe.num_examples,
            languages=recipe.languages,
            license=recipe.license,
            description=recipe.description,
            generation_type=recipe.generation_type,
            synthetic_ratio=recipe.synthetic_ratio,
            human_ratio=recipe.human_ratio,
            generation_methods=recipe.generation_methods,
            teacher_models=recipe.teacher_models,
            cost=recipe.cost,
            reproducibility=recipe.reproducibility,
            quality_metrics=recipe.quality_metrics,
            tags=recipe.tags,
            created_date=recipe.created_date,
            authors=recipe.authors,
            paper_url=recipe.paper_url,
            homepage_url=recipe.homepage_url,
        )
```

### 4.2 Provider 接口协议

```python
# src/datarecipe/providers/protocol.py

from typing import Protocol, Optional, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """配置验证结果"""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnnotatorMatch:
    """标注者匹配结果"""
    annotator_id: str
    name: str
    match_score: float         # 0-1 匹配度
    skills_matched: list[str]
    skills_missing: list[str]
    hourly_rate: float
    availability: str          # available, busy, unavailable


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
    progress: float            # 0-100
    completed_count: int
    total_count: int
    quality_score: Optional[float] = None
    estimated_completion: Optional[str] = None


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

    def validate_config(self, config: "ProductionConfig") -> ValidationResult:
        """验证投产配置是否符合平台要求"""
        ...

    def match_annotators(
        self,
        profile: "AnnotatorProfile",
        limit: int = 10,
    ) -> list[AnnotatorMatch]:
        """根据画像匹配标注者"""
        ...

    def create_project(
        self,
        recipe: "DataRecipe",
        config: Optional["ProductionConfig"] = None,
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
```

---

## 5. 新增模块设计

### 5.1 Annotator Profiler（标注专家画像生成器）

```python
# src/datarecipe/profiler.py

"""标注专家画像生成器

根据数据集分析结果，自动推导所需的标注专家画像：
- 技能要求
- 经验等级
- 人力规模
- 团队结构
- 费率参考
"""

from dataclasses import dataclass
from typing import Optional

from datarecipe.schema import (
    Recipe,
    DataRecipe,
    AnnotatorProfile,
    SkillRequirement,
    ExperienceLevel,
    EducationLevel,
)


# 数据集类型 → 技能要求映射
DATASET_TYPE_SKILLS = {
    "code": [
        SkillRequirement("programming", "Python", "advanced", True),
        SkillRequirement("programming", "JavaScript", "intermediate", False),
        SkillRequirement("domain", "软件工程", "intermediate", True),
    ],
    "code_review": [
        SkillRequirement("programming", "目标语言", "advanced", True),
        SkillRequirement("domain", "代码审查", "advanced", True),
        SkillRequirement("tool", "Git", "intermediate", True),
    ],
    "rlhf": [
        SkillRequirement("domain", "内容理解", "advanced", True),
        SkillRequirement("language", "目标语言", "native", True),
        SkillRequirement("tool", "标注平台", "basic", True),
    ],
    "medical": [
        SkillRequirement("domain", "医学", "expert", True),
        SkillRequirement("certification", "医师资格", "required", True),
    ],
    "legal": [
        SkillRequirement("domain", "法律", "expert", True),
        SkillRequirement("certification", "法律从业资格", "required", True),
    ],
    "financial": [
        SkillRequirement("domain", "金融", "advanced", True),
        SkillRequirement("certification", "金融从业资格", "required", False),
    ],
    "math": [
        SkillRequirement("domain", "数学", "advanced", True),
        SkillRequirement("education", "理工科研究生", "required", True),
    ],
    "agent": [
        SkillRequirement("domain", "AI工具使用", "advanced", True),
        SkillRequirement("programming", "Python", "intermediate", False),
    ],
    "multilingual": [
        SkillRequirement("language", "目标语言", "native", True),
        SkillRequirement("language", "英语", "C1", False),
    ],
    "general": [
        SkillRequirement("domain", "通用理解", "intermediate", True),
        SkillRequirement("language", "目标语言", "native", True),
    ],
}

# 数据集类型 → 经验等级映射
DATASET_TYPE_EXPERIENCE = {
    "code": (ExperienceLevel.SENIOR, 3),
    "code_review": (ExperienceLevel.SENIOR, 5),
    "rlhf": (ExperienceLevel.MID, 1),
    "medical": (ExperienceLevel.EXPERT, 5),
    "legal": (ExperienceLevel.EXPERT, 5),
    "financial": (ExperienceLevel.SENIOR, 3),
    "math": (ExperienceLevel.SENIOR, 3),
    "agent": (ExperienceLevel.MID, 2),
    "multilingual": (ExperienceLevel.MID, 1),
    "general": (ExperienceLevel.JUNIOR, 0),
}

# 数据集类型 → 学历要求
DATASET_TYPE_EDUCATION = {
    "code": EducationLevel.BACHELOR,
    "medical": EducationLevel.PROFESSIONAL,
    "legal": EducationLevel.PROFESSIONAL,
    "financial": EducationLevel.BACHELOR,
    "math": EducationLevel.MASTER,
    "general": EducationLevel.HIGH_SCHOOL,
}

# 地区费率系数
REGION_MULTIPLIERS = {
    "us": 1.0,
    "uk": 0.9,
    "eu": 0.85,
    "china": 0.4,
    "india": 0.25,
    "latam": 0.35,
    "sea": 0.3,
}

# 基础小时费率（美元）
BASE_HOURLY_RATES = {
    ExperienceLevel.JUNIOR: {"min": 10, "max": 20},
    ExperienceLevel.MID: {"min": 20, "max": 35},
    ExperienceLevel.SENIOR: {"min": 35, "max": 60},
    ExperienceLevel.EXPERT: {"min": 60, "max": 150},
}


class AnnotatorProfiler:
    """标注专家画像生成器"""

    def __init__(self, custom_rules: Optional[dict] = None):
        """初始化

        Args:
            custom_rules: 自定义推导规则，覆盖默认规则
        """
        self.custom_rules = custom_rules or {}

    def generate_profile(
        self,
        recipe: Recipe,
        target_size: Optional[int] = None,
        region: str = "us",
        budget: Optional[float] = None,
    ) -> AnnotatorProfile:
        """根据数据集配方生成标注专家画像

        Args:
            recipe: 数据集配方
            target_size: 目标数据量
            region: 目标地区
            budget: 预算限制

        Returns:
            AnnotatorProfile 标注专家画像
        """
        # 1. 检测数据集类型
        dataset_type = self._detect_dataset_type(recipe)

        # 2. 推导技能要求
        skills = self._derive_skills(dataset_type, recipe)

        # 3. 推导经验等级
        exp_level, min_years = self._derive_experience(dataset_type)

        # 4. 推导学历要求
        education = self._derive_education(dataset_type)

        # 5. 推导语言要求
        languages = self._derive_languages(recipe)

        # 6. 推导领域知识
        domains = self._derive_domains(recipe)

        # 7. 计算团队规模和工作量
        target = target_size or recipe.num_examples or 10000
        team_size, person_days, hours_per_example = self._calculate_workload(
            target, dataset_type, recipe
        )

        # 8. 计算费率
        hourly_rate = self._calculate_hourly_rate(exp_level, region)

        # 9. 生成筛选标准
        screening = self._generate_screening_criteria(skills, exp_level, domains)

        # 10. 推荐平台
        platforms = self._recommend_platforms(dataset_type, region)

        return AnnotatorProfile(
            skill_requirements=skills,
            experience_level=exp_level,
            min_experience_years=min_years,
            language_requirements=languages,
            domain_knowledge=domains,
            education_level=education,
            team_size=team_size,
            team_structure={"annotator": int(team_size * 0.8), "reviewer": int(team_size * 0.2)},
            estimated_person_days=person_days,
            estimated_hours_per_example=hours_per_example,
            hourly_rate_range=hourly_rate,
            screening_criteria=screening,
            recommended_platforms=platforms,
        )

    def _detect_dataset_type(self, recipe: Recipe) -> str:
        """检测数据集类型"""
        # 基于标签检测
        tags = " ".join(recipe.tags).lower() if recipe.tags else ""
        description = (recipe.description or "").lower()
        text = tags + " " + description

        if any(kw in text for kw in ["code", "programming", "github"]):
            if any(kw in text for kw in ["review", "評"]):
                return "code_review"
            return "code"
        if any(kw in text for kw in ["rlhf", "preference", "reward", "alignment"]):
            return "rlhf"
        if any(kw in text for kw in ["medical", "clinical", "health", "医"]):
            return "medical"
        if any(kw in text for kw in ["legal", "law", "法律"]):
            return "legal"
        if any(kw in text for kw in ["financial", "banking", "金融"]):
            return "financial"
        if any(kw in text for kw in ["math", "数学", "reasoning"]):
            return "math"
        if any(kw in text for kw in ["agent", "tool", "function"]):
            return "agent"
        if len(recipe.languages) > 2:
            return "multilingual"

        return "general"

    def _derive_skills(self, dataset_type: str, recipe: Recipe) -> list[SkillRequirement]:
        """推导技能要求"""
        base_skills = DATASET_TYPE_SKILLS.get(dataset_type, DATASET_TYPE_SKILLS["general"])

        # 应用自定义规则
        if dataset_type in self.custom_rules.get("skills", {}):
            return self.custom_rules["skills"][dataset_type]

        return base_skills.copy()

    def _derive_experience(self, dataset_type: str) -> tuple[ExperienceLevel, int]:
        """推导经验要求"""
        return DATASET_TYPE_EXPERIENCE.get(
            dataset_type,
            (ExperienceLevel.MID, 1)
        )

    def _derive_education(self, dataset_type: str) -> EducationLevel:
        """推导学历要求"""
        return DATASET_TYPE_EDUCATION.get(dataset_type, EducationLevel.BACHELOR)

    def _derive_languages(self, recipe: Recipe) -> list[str]:
        """推导语言要求"""
        langs = []
        for lang in recipe.languages or ["en"]:
            if lang in ["zh", "zh-CN", "zh-TW", "chinese"]:
                langs.append("zh-CN:native")
            elif lang in ["en", "english"]:
                langs.append("en:C1")
            else:
                langs.append(f"{lang}:B2")
        return langs

    def _derive_domains(self, recipe: Recipe) -> list[str]:
        """推导领域知识"""
        domains = []
        text = ((recipe.description or "") + " ".join(recipe.tags or [])).lower()

        domain_keywords = {
            "医疗": ["medical", "health", "clinical", "医"],
            "金融": ["financial", "banking", "金融", "投资"],
            "法律": ["legal", "law", "法律", "合同"],
            "技术": ["code", "programming", "software", "技术"],
            "教育": ["education", "learning", "教育"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                domains.append(domain)

        return domains or ["通用"]

    def _calculate_workload(
        self,
        target_size: int,
        dataset_type: str,
        recipe: Recipe,
    ) -> tuple[int, float, float]:
        """计算工作量

        Returns:
            (team_size, person_days, hours_per_example)
        """
        # 每个样本的平均标注时间（分钟）
        time_per_example = {
            "code": 15,
            "code_review": 30,
            "rlhf": 5,
            "medical": 20,
            "legal": 25,
            "financial": 15,
            "math": 20,
            "agent": 10,
            "multilingual": 8,
            "general": 5,
        }

        minutes = time_per_example.get(dataset_type, 5)
        hours_per_example = minutes / 60

        # 总工时
        total_hours = target_size * hours_per_example

        # 人天（8小时/天）
        person_days = total_hours / 8

        # 团队规模（假设项目周期30天）
        team_size = max(2, int(person_days / 30) + 1)

        # 加上审核人员
        team_size = int(team_size * 1.25)

        return team_size, person_days, hours_per_example

    def _calculate_hourly_rate(
        self,
        exp_level: ExperienceLevel,
        region: str,
    ) -> dict:
        """计算小时费率"""
        base = BASE_HOURLY_RATES[exp_level]
        multiplier = REGION_MULTIPLIERS.get(region, 1.0)

        return {
            "min": round(base["min"] * multiplier, 2),
            "max": round(base["max"] * multiplier, 2),
            "currency": "USD",
            "region": region,
        }

    def _generate_screening_criteria(
        self,
        skills: list[SkillRequirement],
        exp_level: ExperienceLevel,
        domains: list[str],
    ) -> list[str]:
        """生成筛选标准"""
        criteria = []

        for skill in skills:
            if skill.required:
                criteria.append(f"具备{skill.name}能力（{skill.level}级）")

        if exp_level in [ExperienceLevel.SENIOR, ExperienceLevel.EXPERT]:
            criteria.append(f"至少 {exp_level.value} 级别工作经验")

        for domain in domains:
            if domain != "通用":
                criteria.append(f"具有{domain}领域背景")

        criteria.append("通过平台资质审核")
        criteria.append("完成样例任务测试")

        return criteria

    def _recommend_platforms(self, dataset_type: str, region: str) -> list[str]:
        """推荐标注平台"""
        platforms = {
            "code": ["Scale AI", "Surge AI", "集识光年"],
            "medical": ["集识光年", "Scale AI"],
            "legal": ["集识光年", "Scale AI"],
            "general": ["Amazon MTurk", "Prolific", "集识光年"],
            "multilingual": ["Appen", "集识光年"],
        }

        return platforms.get(dataset_type, ["集识光年", "Scale AI", "Amazon MTurk"])
```

### 5.2 Production Deployer（投产部署生成器）

```python
# src/datarecipe/deployer.py

"""投产部署生成器

根据数据配方和标注画像，生成投产配置：
- 标注指南
- 质检规则
- 验收标准
- 项目结构

支持输出到本地文件或调用 Provider API 部署到平台。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import importlib.metadata

from datarecipe.schema import (
    DataRecipe,
    AnnotatorProfile,
    ProductionConfig,
    QualityRule,
    AcceptanceCriterion,
    Milestone,
    ReviewWorkflow,
)
from datarecipe.providers.protocol import DeploymentProvider, ProjectHandle, DeploymentResult


class ProductionDeployer:
    """投产部署生成器"""

    def __init__(self):
        """初始化"""
        self._providers: dict[str, DeploymentProvider] = {}
        self._load_providers()

    def _load_providers(self):
        """通过 entry_points 加载已安装的 providers"""
        try:
            eps = importlib.metadata.entry_points(group="datarecipe.providers")
            for ep in eps:
                try:
                    provider_class = ep.load()
                    provider = provider_class()
                    self._providers[ep.name] = provider
                except Exception as e:
                    print(f"Warning: Failed to load provider {ep.name}: {e}")
        except Exception:
            pass

        # 确保 LocalFiles provider 始终可用
        if "local" not in self._providers:
            from datarecipe.providers.local import LocalFilesProvider
            self._providers["local"] = LocalFilesProvider()

    def list_providers(self) -> list[dict]:
        """列出可用的 providers"""
        return [
            {"name": name, "description": p.description}
            for name, p in self._providers.items()
        ]

    def get_provider(self, name: str) -> DeploymentProvider:
        """获取指定 provider"""
        if name not in self._providers:
            raise ValueError(
                f"Provider '{name}' not found. "
                f"Available: {list(self._providers.keys())}"
            )
        return self._providers[name]

    def generate_config(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile] = None,
    ) -> ProductionConfig:
        """生成投产配置

        Args:
            recipe: 数据配方
            profile: 标注专家画像（可选）

        Returns:
            ProductionConfig 投产配置
        """
        # 1. 生成标注指南
        guide = self._generate_annotation_guide(recipe, profile)

        # 2. 生成质检规则
        quality_rules = self._generate_quality_rules(recipe)

        # 3. 生成验收标准
        acceptance = self._generate_acceptance_criteria(recipe)

        # 4. 确定审核流程
        workflow = self._determine_review_workflow(recipe)

        # 5. 生成里程碑
        milestones = self._generate_milestones(recipe, profile)

        # 6. 估算时间
        days = self._estimate_timeline(recipe, profile)

        return ProductionConfig(
            annotation_guide=guide,
            quality_rules=quality_rules,
            acceptance_criteria=acceptance,
            review_workflow=workflow,
            estimated_timeline_days=days,
            milestones=milestones,
        )

    def deploy(
        self,
        recipe: DataRecipe,
        output: str,
        provider: str = "local",
        config: Optional[ProductionConfig] = None,
        profile: Optional[AnnotatorProfile] = None,
    ) -> DeploymentResult:
        """部署投产项目

        Args:
            recipe: 数据配方
            output: 输出路径（本地）或项目名称（平台）
            provider: Provider 名称
            config: 投产配置（可选，自动生成）
            profile: 标注画像（可选，自动生成）

        Returns:
            DeploymentResult 部署结果
        """
        # 自动生成缺失的配置
        if profile is None:
            from datarecipe.profiler import AnnotatorProfiler
            profiler = AnnotatorProfiler()
            # 从 DataRecipe 转换回 Recipe 用于 profiler
            profile = profiler.generate_profile(recipe)

        if config is None:
            config = self.generate_config(recipe, profile)

        # 获取 provider
        p = self.get_provider(provider)

        # 验证配置
        validation = p.validate_config(config)
        if not validation.valid:
            return DeploymentResult(
                success=False,
                error=f"Config validation failed: {validation.errors}",
            )

        # 创建项目
        try:
            handle = p.create_project(recipe, config)

            # 对于 local provider，handle 已包含所有信息
            if provider == "local":
                return DeploymentResult(
                    success=True,
                    project_handle=handle,
                    details={"output_path": output},
                )

            # 对于平台 provider，提交项目
            result = p.submit(handle)
            return result

        except Exception as e:
            return DeploymentResult(
                success=False,
                error=str(e),
            )

    def _generate_annotation_guide(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile],
    ) -> str:
        """生成标注指南"""
        lines = []

        lines.append(f"# 标注指南：{recipe.name}")
        lines.append("")
        lines.append("## 1. 任务概述")
        lines.append("")
        if recipe.description:
            lines.append(recipe.description)
        lines.append("")

        lines.append("## 2. 标注目标")
        lines.append("")
        lines.append("请根据以下标准对每条数据进行标注：")
        lines.append("")
        # 根据数据集类型生成具体目标
        if recipe.generation_type.value == "synthetic":
            lines.append("- 验证生成内容的准确性")
            lines.append("- 检查格式是否符合要求")
            lines.append("- 评估内容质量")
        else:
            lines.append("- 按照预定义类别进行分类")
            lines.append("- 标注关键信息")
            lines.append("- 评估数据质量")
        lines.append("")

        lines.append("## 3. 标注标准")
        lines.append("")
        lines.append("### 3.1 质量要求")
        lines.append("")
        lines.append("- 准确性：标注必须符合实际内容")
        lines.append("- 完整性：所有必填字段都需填写")
        lines.append("- 一致性：相似内容应有相似的标注")
        lines.append("")

        lines.append("### 3.2 格式要求")
        lines.append("")
        lines.append("- 遵循字段定义中的格式说明")
        lines.append("- 使用规定的标签/类别")
        lines.append("- 避免拼写和语法错误")
        lines.append("")

        lines.append("## 4. 示例")
        lines.append("")
        lines.append("### 正确示例")
        lines.append("")
        lines.append("```")
        lines.append("待补充具体示例")
        lines.append("```")
        lines.append("")

        lines.append("### 错误示例")
        lines.append("")
        lines.append("```")
        lines.append("待补充错误示例及原因")
        lines.append("```")
        lines.append("")

        lines.append("## 5. 常见问题")
        lines.append("")
        lines.append("Q: 遇到模糊情况怎么办？")
        lines.append("A: 请联系审核员或在备注中说明。")
        lines.append("")

        lines.append("---")
        lines.append("*由 DataRecipe 自动生成，请根据实际情况补充完善*")

        return "\n".join(lines)

    def _generate_quality_rules(self, recipe: DataRecipe) -> list[QualityRule]:
        """生成质检规则"""
        rules = []

        # 通用规则
        rules.append(QualityRule(
            rule_id="QR001",
            name="非空检查",
            description="必填字段不能为空",
            check_type="format",
            severity="error",
            auto_check=True,
        ))

        rules.append(QualityRule(
            rule_id="QR002",
            name="长度检查",
            description="文本长度在合理范围内",
            check_type="format",
            severity="warning",
            auto_check=True,
        ))

        rules.append(QualityRule(
            rule_id="QR003",
            name="重复检查",
            description="不能与已有数据重复",
            check_type="consistency",
            severity="error",
            auto_check=True,
        ))

        # 根据数据集类型添加特定规则
        if recipe.generation_type.value in ["synthetic", "mixed"]:
            rules.append(QualityRule(
                rule_id="QR004",
                name="事实性检查",
                description="生成内容不能包含明显的事实错误",
                check_type="content",
                severity="error",
                auto_check=False,
            ))

        return rules

    def _generate_acceptance_criteria(self, recipe: DataRecipe) -> list[AcceptanceCriterion]:
        """生成验收标准"""
        criteria = []

        criteria.append(AcceptanceCriterion(
            criterion_id="AC001",
            name="完成率",
            description="标注任务完成比例",
            threshold=0.98,
            metric_type="completeness",
            priority="required",
        ))

        criteria.append(AcceptanceCriterion(
            criterion_id="AC002",
            name="准确率",
            description="抽检准确率",
            threshold=0.95,
            metric_type="accuracy",
            priority="required",
        ))

        criteria.append(AcceptanceCriterion(
            criterion_id="AC003",
            name="一致性",
            description="标注者间一致性（Kappa系数）",
            threshold=0.7,
            metric_type="agreement",
            priority="required",
        ))

        return criteria

    def _determine_review_workflow(self, recipe: DataRecipe) -> ReviewWorkflow:
        """确定审核流程"""
        # 高质量要求的数据集使用双人审核
        if recipe.human_ratio and recipe.human_ratio > 0.5:
            return ReviewWorkflow.DOUBLE
        # 专业领域使用专家审核
        tags = " ".join(recipe.tags or []).lower()
        if any(kw in tags for kw in ["medical", "legal", "financial"]):
            return ReviewWorkflow.EXPERT
        return ReviewWorkflow.SINGLE

    def _generate_milestones(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile],
    ) -> list[Milestone]:
        """生成项目里程碑"""
        milestones = []

        milestones.append(Milestone(
            name="项目启动",
            description="完成项目设置和团队组建",
            deliverables=["标注指南定稿", "团队培训完成", "工具配置完成"],
            estimated_days=3,
        ))

        milestones.append(Milestone(
            name="试标注",
            description="小规模试标注验证流程",
            deliverables=["100条试标注完成", "问题清单", "指南修订"],
            estimated_days=5,
            dependencies=["项目启动"],
        ))

        milestones.append(Milestone(
            name="正式标注",
            description="大规模标注执行",
            deliverables=["全量数据标注完成", "日报"],
            estimated_days=20,
            dependencies=["试标注"],
        ))

        milestones.append(Milestone(
            name="质检验收",
            description="质量检查和验收",
            deliverables=["质检报告", "最终数据集", "项目总结"],
            estimated_days=5,
            dependencies=["正式标注"],
        ))

        return milestones

    def _estimate_timeline(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile],
    ) -> int:
        """估算项目时间"""
        base_days = 30

        if profile and profile.estimated_person_days:
            # 假设团队并行工作
            team_size = profile.team_size or 10
            base_days = int(profile.estimated_person_days / team_size) + 10

        return max(14, base_days)  # 最少2周
```

### 5.3 增强成本计算器

```python
# src/datarecipe/cost_calculator.py 增强部分

# 新增人力成本基准数据
LABOR_COST_RATES = {
    # 经验等级 → 基础小时费率（美元）
    "junior": {"base": 12, "annotation": 10, "review": 15},
    "mid": {"base": 25, "annotation": 20, "review": 30},
    "senior": {"base": 45, "annotation": 35, "review": 55},
    "expert": {"base": 80, "annotation": 60, "review": 100},
}

# 地区费率系数
REGION_COST_MULTIPLIERS = {
    "us": 1.0,
    "uk": 0.9,
    "eu": 0.85,
    "cn": 0.4,      # 中国
    "in": 0.25,     # 印度
    "sea": 0.3,     # 东南亚
    "latam": 0.35,  # 拉美
}

# 标注类型 → 每条平均时间（分钟）
ANNOTATION_TIME_ESTIMATES = {
    "simple_label": 1,
    "text_classification": 2,
    "preference_ranking": 5,
    "text_generation": 15,
    "complex_annotation": 20,
    "expert_annotation": 30,
    "code_review": 20,
}


class EnhancedCostCalculator(CostCalculator):
    """增强的成本计算器——包含人力成本"""

    def calculate_labor_cost(
        self,
        num_examples: int,
        annotation_type: str,
        experience_level: str = "mid",
        region: str = "us",
        review_rate: float = 0.2,  # 审核比例
    ) -> dict:
        """计算人力成本

        Args:
            num_examples: 数据量
            annotation_type: 标注类型
            experience_level: 经验等级
            region: 地区
            review_rate: 审核比例

        Returns:
            人力成本详情
        """
        # 获取基础费率
        rates = LABOR_COST_RATES.get(experience_level, LABOR_COST_RATES["mid"])
        region_mult = REGION_COST_MULTIPLIERS.get(region, 1.0)

        # 计算标注时间
        minutes_per_item = ANNOTATION_TIME_ESTIMATES.get(
            annotation_type,
            ANNOTATION_TIME_ESTIMATES["text_classification"]
        )

        # 标注成本
        annotation_hours = (num_examples * minutes_per_item) / 60
        annotation_cost = annotation_hours * rates["annotation"] * region_mult

        # 审核成本
        review_items = int(num_examples * review_rate)
        review_hours = (review_items * minutes_per_item * 1.5) / 60  # 审核慢1.5倍
        review_cost = review_hours * rates["review"] * region_mult

        # 项目管理成本（估算为总成本的10%）
        pm_cost = (annotation_cost + review_cost) * 0.1

        total = annotation_cost + review_cost + pm_cost

        return {
            "annotation_cost": round(annotation_cost, 2),
            "review_cost": round(review_cost, 2),
            "pm_cost": round(pm_cost, 2),
            "total": round(total, 2),
            "details": {
                "annotation_hours": round(annotation_hours, 1),
                "review_hours": round(review_hours, 1),
                "effective_hourly_rate": round(rates["annotation"] * region_mult, 2),
                "region": region,
                "region_multiplier": region_mult,
            },
        }

    def calculate_total_cost(
        self,
        recipe: Recipe,
        target_size: int,
        model: str = "gpt-4o",
        region: str = "us",
    ) -> EnhancedCost:
        """计算总成本（API + 算力 + 人力）"""
        # 现有 API 成本
        api_breakdown = self.estimate_from_recipe(recipe, target_size, model)

        # 人力成本
        annotation_type = self._infer_annotation_type(recipe)
        labor = self.calculate_labor_cost(
            num_examples=target_size,
            annotation_type=annotation_type or "text_classification",
            experience_level="mid",
            region=region,
        )

        # 合并
        total = api_breakdown.total.expected + labor["total"]

        return EnhancedCost(
            api_cost=api_breakdown.total.expected,
            compute_cost=api_breakdown.compute_cost.expected,
            human_cost=labor["total"],
            human_cost_breakdown={
                "annotation": labor["annotation_cost"],
                "review": labor["review_cost"],
                "project_management": labor["pm_cost"],
            },
            region=region,
            region_multiplier=REGION_COST_MULTIPLIERS.get(region, 1.0),
            total_cost=total,
            total_range={
                "low": total * 0.7,
                "high": total * 1.5,
            },
            confidence="medium",
            assumptions=api_breakdown.assumptions + [
                f"人力成本基于 {region} 地区费率",
                f"标注类型: {annotation_type}",
            ],
        )
```

---

## 6. CLI 接口设计

### 6.1 新增命令

```bash
# 标注专家画像
datarecipe profile <dataset>
datarecipe profile <dataset> --region china --budget 50000
datarecipe profile <dataset> --format json
datarecipe profile <dataset> -o profile.yaml

# 投产部署（默认输出到本地文件）
datarecipe deploy <dataset> -o ./project
datarecipe deploy <dataset> -o ./project --provider local
datarecipe deploy <dataset> --provider labelstudio --url http://localhost:8080
datarecipe deploy <dataset> --provider judgeguild

# 全流程 pipeline
datarecipe pipeline <dataset> -o ./project
datarecipe pipeline <dataset> --provider judgeguild

# 列出已安装的 providers
datarecipe providers list
datarecipe providers info <provider_name>

# 增强的成本命令
datarecipe cost <dataset> --include-labor --region china
```

### 6.2 CLI 实现

```python
# src/datarecipe/cli.py 新增命令

@main.command()
@click.argument("dataset_id")
@click.option("--region", "-r", default="us", help="Target region for cost estimation")
@click.option("--budget", "-b", type=float, help="Budget constraint (USD)")
@click.option("--format", "fmt", type=click.Choice(["yaml", "json", "markdown"]), default="yaml")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def profile(dataset_id: str, region: str, budget: float, fmt: str, output: str):
    """Generate annotator profile for a dataset.

    Analyzes a dataset and outputs the required annotator profile:
    skill requirements, experience level, team size, and cost estimates.

    DATASET_ID is the identifier of the dataset to analyze.
    """
    from datarecipe.profiler import AnnotatorProfiler

    analyzer = DatasetAnalyzer()
    profiler = AnnotatorProfiler()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        recipe = analyzer.analyze(dataset_id)

    with console.status("[cyan]Generating annotator profile...[/cyan]"):
        profile = profiler.generate_profile(
            recipe,
            target_size=recipe.num_examples,
            region=region,
            budget=budget,
        )

    # 输出格式
    if fmt == "json":
        content = profile.to_json()
    elif fmt == "markdown":
        content = profile_to_markdown(profile, recipe.name)
    else:
        content = profile.to_yaml()

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]✓ Profile saved to:[/green] {output}")
    else:
        print(content)

    # 显示摘要
    console.print(f"\n[bold cyan]Annotator Profile Summary:[/bold cyan]")
    console.print(f"  Experience: {profile.experience_level.value}")
    console.print(f"  Team Size: {profile.team_size}")
    console.print(f"  Estimated Days: {profile.estimated_person_days:.0f}")
    console.print(f"  Hourly Rate: ${profile.hourly_rate_range['min']}-${profile.hourly_rate_range['max']}")


@main.command()
@click.argument("dataset_id")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--provider", "-p", default="local", help="Deployment provider")
@click.option("--url", type=str, help="Provider URL (for remote providers)")
def deploy(dataset_id: str, output: str, provider: str, url: str):
    """Deploy a dataset production project.

    Generates a complete production project including:
    - Annotation guidelines
    - Quality rules
    - Acceptance criteria
    - Project structure

    Default provider is 'local' which outputs to local files.
    """
    from datarecipe.deployer import ProductionDeployer
    from datarecipe.schema import DataRecipe

    analyzer = DatasetAnalyzer()
    deployer = ProductionDeployer()

    with console.status(f"[cyan]Analyzing {dataset_id}...[/cyan]"):
        recipe = analyzer.analyze(dataset_id)
        data_recipe = DataRecipe.from_recipe(recipe)

    with console.status(f"[cyan]Deploying with provider '{provider}'...[/cyan]"):
        result = deployer.deploy(
            recipe=data_recipe,
            output=output,
            provider=provider,
        )

    if result.success:
        console.print(f"[green]✓ Deployment successful![/green]")
        if result.project_handle:
            console.print(f"  Project ID: {result.project_handle.project_id}")
            if result.project_handle.url:
                console.print(f"  URL: {result.project_handle.url}")
        console.print(f"  Output: {output}")
    else:
        console.print(f"[red]✗ Deployment failed:[/red] {result.error}")
        sys.exit(1)


@main.command("providers")
@click.argument("action", type=click.Choice(["list", "info"]))
@click.argument("name", required=False)
def providers(action: str, name: str):
    """Manage deployment providers.

    Commands:
        list  - List all available providers
        info  - Show details about a specific provider
    """
    from datarecipe.deployer import ProductionDeployer

    deployer = ProductionDeployer()

    if action == "list":
        providers = deployer.list_providers()

        table = Table(title="Available Providers")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Status", style="green")

        for p in providers:
            table.add_row(p["name"], p["description"], "✓ Installed")

        console.print(table)

    elif action == "info":
        if not name:
            console.print("[red]Error:[/red] Please specify a provider name")
            sys.exit(1)

        try:
            p = deployer.get_provider(name)
            console.print(f"[bold]{name}[/bold]")
            console.print(f"Description: {p.description}")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
```

---

## 7. Provider 插件系统

### 7.1 entry_points 注册机制

```toml
# datarecipe 核心包的 pyproject.toml
[project.entry-points."datarecipe.providers"]
local = "datarecipe.providers.local:LocalFilesProvider"

# datarecipe-labelstudio 包的 pyproject.toml
[project.entry-points."datarecipe.providers"]
labelstudio = "datarecipe_labelstudio:LabelStudioProvider"

# datarecipe-judgeguild 包的 pyproject.toml（私有）
[project.entry-points."datarecipe.providers"]
judgeguild = "datarecipe_judgeguild:JudgeGuildProvider"
```

### 7.2 Provider 发现逻辑

```python
# src/datarecipe/providers/__init__.py

import importlib.metadata
from typing import Optional
from datarecipe.providers.protocol import DeploymentProvider


class ProviderNotFoundError(Exception):
    """Provider 未找到"""
    pass


def discover_providers() -> dict[str, type[DeploymentProvider]]:
    """发现所有已安装的 providers"""
    providers = {}

    try:
        eps = importlib.metadata.entry_points(group="datarecipe.providers")
        for ep in eps:
            try:
                provider_class = ep.load()
                providers[ep.name] = provider_class
            except Exception as e:
                print(f"Warning: Failed to load provider {ep.name}: {e}")
    except Exception:
        pass

    return providers


def get_provider(name: str) -> DeploymentProvider:
    """获取指定名称的 provider 实例"""
    providers = discover_providers()

    if name not in providers:
        available = list(providers.keys())
        raise ProviderNotFoundError(
            f"Provider '{name}' not found. "
            f"Available providers: {available}. "
            f"Install with: pip install datarecipe-{name}"
        )

    return providers[name]()


def list_providers() -> list[str]:
    """列出所有可用的 provider 名称"""
    return list(discover_providers().keys())
```

### 7.3 LocalFiles Provider 实现

```python
# src/datarecipe/providers/local.py

"""本地文件 Provider——默认实现

将投产配置输出到本地文件系统，生成完整的项目结构。
"""

from pathlib import Path
from datetime import datetime
import yaml
import json

from datarecipe.providers.protocol import (
    DeploymentProvider,
    ValidationResult,
    AnnotatorMatch,
    ProjectHandle,
    DeploymentResult,
    ProjectStatus,
)
from datarecipe.schema import DataRecipe, ProductionConfig, AnnotatorProfile


class LocalFilesProvider:
    """本地文件 Provider"""

    @property
    def name(self) -> str:
        return "local"

    @property
    def description(self) -> str:
        return "Output to local files (default provider)"

    def validate_config(self, config: ProductionConfig) -> ValidationResult:
        """验证配置"""
        errors = []
        warnings = []

        if not config.annotation_guide:
            warnings.append("Annotation guide is empty")

        if not config.quality_rules:
            warnings.append("No quality rules defined")

        if not config.acceptance_criteria:
            warnings.append("No acceptance criteria defined")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def match_annotators(
        self,
        profile: AnnotatorProfile,
        limit: int = 10,
    ) -> list[AnnotatorMatch]:
        """本地模式不支持匹配标注者"""
        return []

    def create_project(
        self,
        recipe: DataRecipe,
        config: ProductionConfig = None,
        output_dir: str = "./project",
    ) -> ProjectHandle:
        """创建本地项目结构"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成项目文件
        files_created = []

        # 1. recipe.yaml
        recipe_path = output_path / "recipe.yaml"
        recipe_path.write_text(
            yaml.dump(recipe.to_dict() if hasattr(recipe, 'to_dict') else {},
                     default_flow_style=False, allow_unicode=True),
            encoding="utf-8"
        )
        files_created.append(str(recipe_path))

        # 2. annotator_profile.yaml
        if recipe.annotator_profile:
            profile_path = output_path / "annotator_profile.yaml"
            profile_path.write_text(recipe.annotator_profile.to_yaml(), encoding="utf-8")
            files_created.append(str(profile_path))

        # 3. annotation_guide.md
        if config and config.annotation_guide:
            guide_path = output_path / "annotation_guide.md"
            guide_path.write_text(config.annotation_guide, encoding="utf-8")
            files_created.append(str(guide_path))

        # 4. quality_rules.yaml
        if config and config.quality_rules:
            rules_path = output_path / "quality_rules.yaml"
            rules_data = [
                {"id": r.rule_id, "name": r.name, "type": r.check_type,
                 "severity": r.severity, "description": r.description}
                for r in config.quality_rules
            ]
            rules_path.write_text(
                yaml.dump(rules_data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8"
            )
            files_created.append(str(rules_path))

        # 5. acceptance_criteria.yaml
        if config and config.acceptance_criteria:
            criteria_path = output_path / "acceptance_criteria.yaml"
            criteria_data = [
                {"id": c.criterion_id, "name": c.name, "threshold": c.threshold,
                 "type": c.metric_type, "priority": c.priority}
                for c in config.acceptance_criteria
            ]
            criteria_path.write_text(
                yaml.dump(criteria_data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8"
            )
            files_created.append(str(criteria_path))

        # 6. timeline.md
        if config and config.milestones:
            timeline_path = output_path / "timeline.md"
            timeline_content = self._generate_timeline_md(config)
            timeline_path.write_text(timeline_content, encoding="utf-8")
            files_created.append(str(timeline_path))

        # 7. README.md
        readme_path = output_path / "README.md"
        readme_content = self._generate_readme(recipe, config)
        readme_path.write_text(readme_content, encoding="utf-8")
        files_created.append(str(readme_path))

        # 8. scripts/ 目录
        scripts_dir = output_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        return ProjectHandle(
            project_id=f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            provider="local",
            created_at=datetime.now().isoformat(),
            status="created",
            metadata={
                "output_dir": str(output_path),
                "files_created": files_created,
            },
        )

    def submit(self, project: ProjectHandle) -> DeploymentResult:
        """本地模式不需要提交"""
        return DeploymentResult(
            success=True,
            project_handle=project,
            details={"message": "Local project created successfully"},
        )

    def get_status(self, project: ProjectHandle) -> ProjectStatus:
        """获取项目状态"""
        return ProjectStatus(
            status="completed",
            progress=100.0,
            completed_count=0,
            total_count=0,
        )

    def cancel(self, project: ProjectHandle) -> bool:
        """取消项目"""
        return True

    def _generate_readme(self, recipe: DataRecipe, config: ProductionConfig) -> str:
        """生成 README"""
        lines = []
        lines.append(f"# {recipe.name} 投产项目")
        lines.append("")
        lines.append("## 项目概述")
        lines.append("")
        if recipe.description:
            lines.append(recipe.description)
        lines.append("")
        lines.append("## 项目结构")
        lines.append("")
        lines.append("```")
        lines.append("./")
        lines.append("├── README.md              # 本文件")
        lines.append("├── recipe.yaml            # 数据配方")
        lines.append("├── annotator_profile.yaml # 标注专家画像")
        lines.append("├── annotation_guide.md    # 标注指南")
        lines.append("├── quality_rules.yaml     # 质检规则")
        lines.append("├── acceptance_criteria.yaml # 验收标准")
        lines.append("├── timeline.md            # 项目时间线")
        lines.append("└── scripts/               # 脚本目录")
        lines.append("```")
        lines.append("")
        lines.append("## 快速开始")
        lines.append("")
        lines.append("1. 阅读 `annotation_guide.md` 了解标注要求")
        lines.append("2. 根据 `annotator_profile.yaml` 组建团队")
        lines.append("3. 按照 `timeline.md` 执行项目")
        lines.append("4. 使用 `quality_rules.yaml` 进行质检")
        lines.append("")
        lines.append("---")
        lines.append("*由 DataRecipe 生成*")
        return "\n".join(lines)

    def _generate_timeline_md(self, config: ProductionConfig) -> str:
        """生成时间线文档"""
        lines = []
        lines.append("# 项目时间线")
        lines.append("")
        lines.append(f"预计总工期：{config.estimated_timeline_days} 天")
        lines.append("")
        lines.append("## 里程碑")
        lines.append("")

        for i, m in enumerate(config.milestones, 1):
            lines.append(f"### M{i}: {m.name}")
            lines.append("")
            lines.append(f"**描述**: {m.description}")
            lines.append(f"**预计天数**: {m.estimated_days}")
            lines.append("")
            lines.append("**交付物**:")
            for d in m.deliverables:
                lines.append(f"- [ ] {d}")
            lines.append("")

        return "\n".join(lines)
```

---

## 8. 目录结构规划

### 8.1 v2 目录结构

```
src/datarecipe/
├── __init__.py              # 包入口
├── __main__.py              # python -m datarecipe
├── cli.py                   # CLI 命令（增强）
├── schema.py                # 数据模型（增强）
│
├── sources/                 # 数据源提取器（现有）
│   ├── __init__.py
│   ├── huggingface.py
│   ├── github.py
│   └── web.py
│
├── analyzer.py              # 主分析器（现有）
├── deep_analyzer.py         # 深度分析（现有）
├── llm_analyzer.py          # LLM 分析（现有）
│
├── cost_calculator.py       # 成本计算（增强）
├── quality_metrics.py       # 质量分析（现有）
│
├── profiler.py              # 🆕 标注专家画像生成器
├── deployer.py              # 🆕 投产部署生成器
│
├── providers/               # 🆕 Provider 插件系统
│   ├── __init__.py          # Provider 发现和注册
│   ├── protocol.py          # Provider 接口协议
│   └── local.py             # LocalFiles Provider（默认）
│
├── templates/               # 🆕 模板目录
│   ├── annotation_guide.md.jinja
│   ├── quality_rules.yaml.jinja
│   └── project_readme.md.jinja
│
├── pipeline.py              # 流程模板（现有）
├── workflow.py              # 工作流生成（现有）
├── batch_analyzer.py        # 批量分析（现有）
└── comparator.py            # 数据集对比（现有）
```

---

## 9. 开源社区设计

### 9.1 CONTRIBUTING.md 模板

```markdown
# Contributing to DataRecipe

感谢您对 DataRecipe 的关注！

## 贡献方式

### 1. 添加新的 Profiling Rule

标注画像推导规则位于 `src/datarecipe/profiler.py`：

```python
# 在 DATASET_TYPE_SKILLS 中添加新的数据集类型
DATASET_TYPE_SKILLS["your_type"] = [
    SkillRequirement("domain", "你的领域", "advanced", True),
]
```

### 2. 添加新的 Provider

1. 创建独立包：`datarecipe-yourprovider`
2. 实现 `DeploymentProvider` 协议
3. 在 `pyproject.toml` 中注册 entry_point

```toml
[project.entry-points."datarecipe.providers"]
yourprovider = "datarecipe_yourprovider:YourProvider"
```

### 3. 改进现有功能

- Fork 仓库
- 创建 feature 分支
- 提交 PR

## 代码规范

- 使用 `ruff` 进行代码检查
- 使用 `pytest` 编写测试
- 文档使用中文，代码使用英文
```

### 9.2 开放数据标准

DataRecipe 定义的数据模型作为开放标准发布：

```yaml
# AnnotatorProfile Schema (v1.0)
$schema: "https://datarecipe.dev/schemas/annotator-profile-v1.yaml"

skill_requirements:
  type: array
  items:
    type: object
    properties:
      type: {type: string, enum: [programming, domain, language, tool, certification]}
      name: {type: string}
      level: {type: string, enum: [basic, intermediate, advanced, native, required]}
      required: {type: boolean}

experience:
  type: object
  properties:
    level: {type: string, enum: [junior, mid, senior, expert]}
    min_years: {type: integer, minimum: 0}

# ... 完整 schema
```

### 9.3 README 改写指引

```markdown
# DataRecipe 🧬

> AI 数据集的「营养成分表」—— 5 分钟搞懂任何数据集是怎么造的

## 一行命令，看穿数据集

```bash
datarecipe analyze Anthropic/hh-rlhf
```

输出：
- 🧬 生成方式：40% 合成 + 60% 人工
- 🎓 教师模型：GPT-4, Claude
- 💰 复制成本：$25,000 - $75,000
- 🔄 可复现性：8/10

## 完整复制一份？

```bash
# 生成标注专家画像
datarecipe profile Anthropic/hh-rlhf

# 一键生成投产项目
datarecipe deploy Anthropic/hh-rlhf -o ./my-project
```

## 架构

```
┌─────────────────────────────────────────┐
│            DataRecipe Core              │  ← pip install datarecipe
├─────────────────────────────────────────┤
│         Provider Interface              │  ← 插件接口
├────────────┬────────────┬───────────────┤
│  LocalFiles│ LabelStudio│  Your Provider│  ← pip install datarecipe-xxx
└────────────┴────────────┴───────────────┘
```

## Built with DataRecipe

- **集识光年** - 万人标注专家平台，首个企业级 Provider
- *期待你的项目...*

## 安装

```bash
pip install datarecipe           # 核心工具
pip install datarecipe[all]      # 含所有可选依赖
pip install datarecipe-labelstudio  # Label Studio 集成
```
```

---

## 10. 实现路线图

### Phase 1（1-2 周）：Provider 基础设施

**目标**: 建立插件系统基础

- [ ] `src/datarecipe/providers/protocol.py` - Provider 接口协议
- [ ] `src/datarecipe/providers/__init__.py` - entry_points 发现机制
- [ ] `src/datarecipe/providers/local.py` - LocalFiles Provider
- [ ] `src/datarecipe/schema.py` - AnnotatorProfile 数据模型
- [ ] `src/datarecipe/profiler.py` - `profile` 命令基础实现
- [ ] `src/datarecipe/cli.py` - 新增 `profile` 和 `providers` 命令

**交付物**:
```bash
datarecipe profile <dataset>
datarecipe providers list
```

### Phase 2（1-2 周）：投产部署

**目标**: 完成本地项目生成

- [ ] `src/datarecipe/schema.py` - ProductionConfig, QualityRule 等模型
- [ ] `src/datarecipe/deployer.py` - ProductionDeployer
- [ ] `src/datarecipe/templates/` - 模板文件
- [ ] `src/datarecipe/cli.py` - `deploy` 命令
- [ ] 标注指南、质检规则、验收标准模板化生成

**交付物**:
```bash
datarecipe deploy <dataset> -o ./project
# 生成完整项目结构
```

### Phase 3（1 周）：成本增强

**目标**: 完善成本估算

- [ ] `src/datarecipe/cost_calculator.py` - 人力成本估算
- [ ] 地区系数支持
- [ ] ROI 计算
- [ ] `pipeline` 命令串联全流程

**交付物**:
```bash
datarecipe cost <dataset> --include-labor --region china
datarecipe pipeline <dataset> -o ./project
```

### Phase 4（1 周）：开源准备

**目标**: 发布准备

- [ ] `datarecipe-labelstudio` 包作为开源示例 Provider
- [ ] `CONTRIBUTING.md` 完善
- [ ] README 改写
- [ ] PyPI 发布准备
- [ ] 文档网站

**交付物**:
```bash
pip install datarecipe
pip install datarecipe-labelstudio
```

### Phase 5（独立仓库）：Judge Guild Provider

**目标**: 私有 Provider 实现（不在开源仓库）

- [ ] `datarecipe-judgeguild` 包
- [ ] Judge Guild 平台 API 对接
- [ ] 标注者匹配功能
- [ ] 项目状态同步

**交付物**:
```bash
pip install datarecipe-judgeguild  # 私有 PyPI
datarecipe deploy <dataset> --provider judgeguild
```

---

## 附录

### A. 数据模型关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              DataRecipe                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │     Recipe      │  │AnnotatorProfile │  │ProductionConfig │         │
│  │  (数据配方)      │  │  (标注画像)     │  │  (投产配置)      │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           ▼                    ▼                    ▼                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │      Cost       │  │SkillRequirement │  │  QualityRule    │         │
│  │ Reproducibility │  │ ExperienceLevel │  │AcceptCriterion  │         │
│  │GenerationMethod │  │     ...         │  │   Milestone     │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DeploymentProvider                                │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  LocalFiles    │  │  LabelStudio    │  │   JudgeGuild    │          │
│  │   (内置)       │  │    (开源)       │  │    (私有)       │          │
│  └────────────────┘  └─────────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### B. 常用命令速查

```bash
# 分析
datarecipe analyze <dataset>
datarecipe analyze <dataset> -o report.md

# 成本
datarecipe cost <dataset>
datarecipe cost <dataset> --include-labor --region china

# 质量
datarecipe quality <dataset> --detect-ai

# 画像
datarecipe profile <dataset>
datarecipe profile <dataset> --region china --format json

# 部署
datarecipe deploy <dataset> -o ./project
datarecipe deploy <dataset> --provider labelstudio

# 全流程
datarecipe pipeline <dataset> -o ./project

# Provider
datarecipe providers list
datarecipe providers info labelstudio
```

---

*文档版本: v2.0-draft*
*最后更新: 2024-12*
