"""Unified task type profiles — central registry for dataset type configurations.

Replaces scattered TASK_TYPE_INFO / DEFAULT_QUALITY_CONSTRAINTS with a single
queryable registry so every module uses consistent defaults.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskTypeProfile:
    """Complete profile for a dataset task type."""

    type_id: str  # e.g. "preference", "evaluation", "sft"
    name: str
    description: str
    cognitive_requirements: list[str] = field(default_factory=list)
    reasoning_chain: str = ""
    default_quality_constraints: list[str] = field(default_factory=list)
    default_field_constraints: list[dict[str, Any]] = field(default_factory=list)
    default_fields: list[dict[str, Any]] = field(default_factory=list)
    cost_multiplier: float = 1.0
    default_human_percentage: float = 95.0
    preferred_pipeline: str = "hybrid"
    default_scoring_dimensions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type_id": self.type_id,
            "name": self.name,
            "description": self.description,
            "cognitive_requirements": self.cognitive_requirements,
            "reasoning_chain": self.reasoning_chain,
            "default_quality_constraints": self.default_quality_constraints,
            "default_field_constraints": self.default_field_constraints,
            "default_fields": self.default_fields,
            "cost_multiplier": self.cost_multiplier,
            "default_human_percentage": self.default_human_percentage,
            "preferred_pipeline": self.preferred_pipeline,
            "default_scoring_dimensions": self.default_scoring_dimensions,
            "extra": self.extra,
        }


# ---- Registry ----

_PROFILE_REGISTRY: dict[str, TaskTypeProfile] = {}


def register_task_profile(profile: TaskTypeProfile) -> None:
    """Register (or overwrite) a task type profile."""
    _PROFILE_REGISTRY[profile.type_id] = profile


def get_task_profile(type_id: str) -> TaskTypeProfile:
    """Return profile for *type_id*, falling back to ``unknown``."""
    return _PROFILE_REGISTRY.get(type_id, _PROFILE_REGISTRY["unknown"])


def list_task_profiles() -> list[TaskTypeProfile]:
    """Return all registered profiles."""
    return list(_PROFILE_REGISTRY.values())


# ---- Built-in profiles ----

register_task_profile(
    TaskTypeProfile(
        type_id="preference",
        name="偏好对比数据",
        description="人类偏好对比数据集，用于训练奖励模型或直接偏好优化。标注员需要对比两个回答，选择更符合人类偏好的一个。",
        cognitive_requirements=[
            "理解问题上下文",
            "评估回答质量（准确性、有用性、安全性）",
            "进行相对比较判断",
        ],
        reasoning_chain="理解问题 → 阅读两个回答 → 多维度评估 → 选择更优回答",
        default_quality_constraints=[
            "两个回答必须有明显的质量差异",
            "选择理由需要有明确依据",
            "避免基于主观喜好的判断",
        ],
        default_fields=[
            {"name": "prompt", "type": "string", "required": True, "description": "用户问题"},
            {"name": "chosen", "type": "string", "required": True, "description": "更优回答"},
            {"name": "rejected", "type": "string", "required": True, "description": "较差回答"},
            {"name": "reason", "type": "string", "required": False, "description": "选择理由"},
        ],
        cost_multiplier=1.0,
        default_human_percentage=90.0,
        preferred_pipeline="human_annotation",
        default_scoring_dimensions=[
            "准确性",
            "有用性",
            "安全性",
            "流畅性",
        ],
    )
)

register_task_profile(
    TaskTypeProfile(
        type_id="evaluation",
        name="评测基准数据",
        description="用于评估模型能力的测试数据集。每道题目需要有明确的评分标准和参考答案。",
        cognitive_requirements=[
            "理解任务目标",
            "设计有区分度的题目",
            "制定清晰的评分标准",
            "确保答案唯一性或评分客观性",
        ],
        reasoning_chain="定义考察点 → 设计题目 → 制定评分标准 → 验证难度",
        default_quality_constraints=[
            "题目答案必须唯一或有明确的评分标准",
            "题目难度需要经过模型验证",
            "禁止使用 AI 生成的内容作为题目或答案",
        ],
        default_fields=[
            {"name": "question", "type": "string", "required": True, "description": "题目"},
            {
                "name": "answer",
                "type": "object",
                "required": True,
                "description": "答案",
                "properties": [
                    {
                        "name": "value",
                        "type": "string",
                        "required": True,
                        "description": "标准答案",
                    },
                    {
                        "name": "is_unique",
                        "type": "boolean",
                        "required": True,
                        "description": "答案是否唯一",
                    },
                    {
                        "name": "alternatives",
                        "type": "array",
                        "items": {"name": "alt", "type": "string"},
                        "description": "其他可接受答案",
                    },
                ],
            },
            {"name": "explanation", "type": "string", "required": True, "description": "解题过程"},
            {
                "name": "scoring_rubric",
                "type": "object",
                "required": True,
                "description": "评分标准",
                "properties": [
                    {
                        "name": "full_score",
                        "type": "string",
                        "required": True,
                        "description": "满分条件",
                    },
                    {"name": "partial_score", "type": "string", "description": "部分得分条件"},
                    {
                        "name": "zero_score",
                        "type": "string",
                        "required": True,
                        "description": "零分条件",
                    },
                ],
            },
        ],
        cost_multiplier=1.5,
        default_human_percentage=95.0,
        preferred_pipeline="benchmark",
        default_scoring_dimensions=[
            "准确性: 回答是否正确",
            "完整性: 回答是否覆盖所有要点",
            "清晰性: 回答是否易于理解",
            "相关性: 回答是否切题",
        ],
    )
)

register_task_profile(
    TaskTypeProfile(
        type_id="sft",
        name="监督微调数据",
        description="用于模型监督微调的指令-回答对数据。需要高质量的指令和对应的标准回答。",
        cognitive_requirements=[
            "理解指令意图",
            "生成准确、有帮助的回答",
            "保持一致的风格和格式",
        ],
        reasoning_chain="理解指令 → 规划回答结构 → 生成内容 → 质量检查",
        default_quality_constraints=[
            "回答必须准确、有帮助",
            "格式和风格保持一致",
            "避免有害或不当内容",
        ],
        default_fields=[
            {"name": "instruction", "type": "string", "required": True, "description": "指令"},
            {"name": "input", "type": "string", "required": False, "description": "附加输入"},
            {"name": "output", "type": "string", "required": True, "description": "标准回答"},
        ],
        cost_multiplier=0.8,
        default_human_percentage=60.0,
        preferred_pipeline="hybrid",
        default_scoring_dimensions=[
            "准确性",
            "有用性",
            "格式一致性",
        ],
    )
)

register_task_profile(
    TaskTypeProfile(
        type_id="swe_bench",
        name="软件工程评测数据",
        description="用于评估模型代码理解和修复能力的数据集。包含真实的代码问题和对应的修复补丁。",
        cognitive_requirements=[
            "理解代码结构和逻辑",
            "定位问题根因",
            "设计正确的修复方案",
            "验证修复的正确性",
        ],
        reasoning_chain="理解问题描述 → 分析代码 → 定位bug → 设计修复 → 验证补丁",
        default_quality_constraints=[
            "补丁必须能够正确修复问题",
            "代码风格与原项目保持一致",
            "包含必要的测试验证",
        ],
        default_fields=[
            {"name": "repo", "type": "string", "required": True, "description": "仓库名称"},
            {"name": "issue", "type": "string", "required": True, "description": "问题描述"},
            {"name": "patch", "type": "string", "required": True, "description": "修复补丁"},
            {"name": "test_patch", "type": "string", "required": False, "description": "测试补丁"},
        ],
        cost_multiplier=2.0,
        default_human_percentage=80.0,
        preferred_pipeline="programmatic",
        default_scoring_dimensions=[
            "正确性: 补丁是否修复问题",
            "代码质量: 风格是否一致",
            "测试覆盖: 是否有充分测试",
        ],
    )
)

register_task_profile(
    TaskTypeProfile(
        type_id="unknown",
        name="通用数据标注",
        description="通用数据标注任务，请根据具体数据特点制定标注规范。",
        cognitive_requirements=[
            "理解数据结构",
            "掌握标注标准",
            "保持标注一致性",
        ],
        reasoning_chain="理解任务 → 分析样本 → 执行标注 → 质量检查",
        default_quality_constraints=[
            "数据格式必须符合 Schema 规范",
            "每条数据需经过质量检查",
        ],
        default_fields=[
            {"name": "question", "type": "string", "required": True, "description": "题目"},
            {"name": "answer", "type": "string", "required": True, "description": "答案"},
            {"name": "explanation", "type": "string", "required": False, "description": "解析"},
        ],
        cost_multiplier=1.0,
        default_human_percentage=90.0,
        preferred_pipeline="hybrid",
        default_scoring_dimensions=[
            "准确性: 回答是否正确",
            "完整性: 回答是否覆盖所有要点",
        ],
    )
)
