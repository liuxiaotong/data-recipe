"""Pipeline extraction and production guide generation."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class PipelineStepType(Enum):
    """Types of pipeline steps."""
    DATA_COLLECTION = "data_collection"
    SEED_DATA = "seed_data"
    PROMPT_DESIGN = "prompt_design"
    LLM_GENERATION = "llm_generation"
    HUMAN_ANNOTATION = "human_annotation"
    QUALITY_FILTER = "quality_filter"
    DEDUPLICATION = "deduplication"
    FORMAT_CONVERSION = "format_conversion"
    VALIDATION = "validation"
    POST_PROCESSING = "post_processing"


@dataclass
class PipelineStep:
    """A single step in the data production pipeline."""
    step_number: int
    step_type: PipelineStepType
    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    estimated_cost: Optional[float] = None
    estimated_time: Optional[str] = None
    code_snippet: Optional[str] = None
    tips: list[str] = field(default_factory=list)


@dataclass
class ProductionPipeline:
    """Complete data production pipeline."""
    name: str
    description: str
    target_size: Optional[int] = None
    estimated_total_cost: Optional[float] = None
    estimated_total_time: Optional[str] = None
    prerequisites: list[str] = field(default_factory=list)
    steps: list[PipelineStep] = field(default_factory=list)
    quality_criteria: list[str] = field(default_factory=list)
    common_pitfalls: list[str] = field(default_factory=list)


# Pre-defined pipeline templates for common dataset types
PIPELINE_TEMPLATES = {
    "distillation": ProductionPipeline(
        name="LLM 蒸馏数据集生产流程",
        description="通过大型语言模型生成高质量训练数据",
        prerequisites=[
            "OpenAI/Anthropic/其他 LLM API 访问权限",
            "种子数据或提示词模板",
            "Python 环境 + requests/openai 库",
            "数据存储空间",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.SEED_DATA,
                name="准备种子数据",
                description="收集或创建初始数据样本，作为生成的基础",
                inputs=["领域知识", "示例数据"],
                outputs=["seed_data.jsonl"],
                tips=[
                    "种子数据质量直接影响生成质量",
                    "建议人工审核种子数据",
                    "多样性比数量更重要",
                ],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.PROMPT_DESIGN,
                name="设计提示词模板",
                description="创建用于调用 LLM 的提示词模板",
                inputs=["任务需求", "输出格式要求"],
                outputs=["prompt_templates.yaml"],
                code_snippet='''PROMPT_TEMPLATE = """
你是一个专业的数据生成助手。请根据以下要求生成数据：

任务：{task_description}
输入：{input_data}
要求：
1. 输出格式为 JSON
2. 确保数据多样性
3. 保持逻辑一致性

请生成：
"""''',
                tips=[
                    "使用 few-shot 示例提高质量",
                    "明确输出格式要求",
                    "添加质量约束条件",
                ],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.LLM_GENERATION,
                name="调用 LLM 生成数据",
                description="批量调用 LLM API 生成数据",
                inputs=["seed_data.jsonl", "prompt_templates.yaml"],
                outputs=["raw_generated.jsonl"],
                tools=["OpenAI API", "Anthropic API", "本地模型"],
                estimated_cost=0.01,  # per 1k tokens
                code_snippet='''import openai
from tqdm import tqdm

def generate_batch(prompts, model="gpt-4"):
    results = []
    for prompt in tqdm(prompts):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        results.append(response.choices[0].message.content)
    return results''',
                tips=[
                    "使用批量 API 降低成本",
                    "设置合理的 temperature",
                    "添加重试机制处理 API 错误",
                    "记录所有请求用于复现",
                ],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.QUALITY_FILTER,
                name="质量过滤",
                description="过滤低质量或不符合要求的生成结果",
                inputs=["raw_generated.jsonl"],
                outputs=["filtered_data.jsonl"],
                code_snippet='''def quality_filter(data):
    filtered = []
    for item in data:
        # 长度检查
        if len(item["text"]) < 50:
            continue
        # 格式检查
        if not is_valid_json(item):
            continue
        # 去重检查
        if is_duplicate(item, filtered):
            continue
        filtered.append(item)
    return filtered''',
                tips=[
                    "定义明确的质量标准",
                    "保留过滤日志用于分析",
                    "考虑使用 LLM 进行质量评分",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.VALIDATION,
                name="人工抽样验证",
                description="人工检查部分数据确保质量",
                inputs=["filtered_data.jsonl"],
                outputs=["validated_data.jsonl", "quality_report.md"],
                tips=[
                    "抽样比例建议 5-10%",
                    "建立评分标准",
                    "记录常见问题用于迭代改进",
                ],
            ),
            PipelineStep(
                step_number=6,
                step_type=PipelineStepType.FORMAT_CONVERSION,
                name="格式转换与发布",
                description="转换为目标格式并准备发布",
                inputs=["validated_data.jsonl"],
                outputs=["final_dataset/"],
                tools=["HuggingFace datasets", "pandas"],
                code_snippet='''from datasets import Dataset

dataset = Dataset.from_json("validated_data.jsonl")
dataset.push_to_hub("your-org/dataset-name")''',
            ),
        ],
        quality_criteria=[
            "生成内容与种子数据主题一致",
            "无事实性错误",
            "格式符合要求",
            "无重复或高度相似内容",
            "语言流畅自然",
        ],
        common_pitfalls=[
            "种子数据质量差导致生成质量差",
            "提示词不够明确导致输出不一致",
            "未设置 temperature 导致生成单一",
            "跳过人工验证导致质量问题",
            "未记录生成参数导致无法复现",
        ],
    ),

    "human_annotation": ProductionPipeline(
        name="人工标注数据集生产流程",
        description="通过众包或专家标注创建高质量数据",
        prerequisites=[
            "标注平台账号（Scale AI/Labelbox/Amazon MTurk）",
            "标注指南文档",
            "原始数据",
            "质量控制机制",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.DATA_COLLECTION,
                name="收集原始数据",
                description="收集需要标注的原始数据",
                inputs=["数据来源"],
                outputs=["raw_data.jsonl"],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.PROMPT_DESIGN,
                name="编写标注指南",
                description="创建详细的标注指南和示例",
                inputs=["任务需求", "质量标准"],
                outputs=["annotation_guidelines.md", "examples.json"],
                tips=[
                    "包含正面和负面示例",
                    "定义边界情况处理方式",
                    "使用截图和可视化",
                ],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.HUMAN_ANNOTATION,
                name="执行标注",
                description="通过标注平台分发任务并收集结果",
                inputs=["raw_data.jsonl", "annotation_guidelines.md"],
                outputs=["annotated_data.jsonl"],
                tools=["Scale AI", "Labelbox", "Amazon MTurk", "Prolific"],
                estimated_cost=0.10,  # per annotation
                tips=[
                    "先进行小规模试标注",
                    "设置标注者资格要求",
                    "使用多人标注提高质量",
                ],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.QUALITY_FILTER,
                name="质量控制",
                description="检查标注一致性和质量",
                inputs=["annotated_data.jsonl"],
                outputs=["quality_checked.jsonl"],
                code_snippet='''def check_agreement(annotations):
    """计算标注者一致性"""
    from sklearn.metrics import cohen_kappa_score
    # 计算 Cohen's Kappa
    kappa = cohen_kappa_score(annotator1, annotator2)
    return kappa > 0.7  # 阈值''',
                tips=[
                    "计算标注者间一致性（Inter-annotator agreement）",
                    "对低一致性样本进行仲裁",
                    "定期校准标注者",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.VALIDATION,
                name="专家审核",
                description="专家审核最终数据集",
                inputs=["quality_checked.jsonl"],
                outputs=["final_dataset.jsonl"],
            ),
        ],
        quality_criteria=[
            "标注者间一致性 > 0.7",
            "无遗漏标注",
            "标注符合指南要求",
            "边界情况处理一致",
        ],
        common_pitfalls=[
            "标注指南不够清晰",
            "未进行试标注直接大规模标注",
            "忽略标注者反馈",
            "未计算标注一致性",
            "标注者疲劳导致质量下降",
        ],
    ),

    "hybrid": ProductionPipeline(
        name="混合数据集生产流程（LLM + 人工）",
        description="结合 LLM 生成和人工验证/修正",
        prerequisites=[
            "LLM API 访问权限",
            "标注平台或内部标注团队",
            "明确的质量标准",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.SEED_DATA,
                name="准备种子数据",
                description="收集高质量种子样本",
                inputs=["领域数据"],
                outputs=["seed_data.jsonl"],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.LLM_GENERATION,
                name="LLM 批量生成",
                description="使用 LLM 生成初始数据",
                inputs=["seed_data.jsonl"],
                outputs=["llm_generated.jsonl"],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.QUALITY_FILTER,
                name="自动质量过滤",
                description="使用规则或模型过滤明显低质量数据",
                inputs=["llm_generated.jsonl"],
                outputs=["auto_filtered.jsonl"],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.HUMAN_ANNOTATION,
                name="人工验证与修正",
                description="人工验证 LLM 生成的数据并修正错误",
                inputs=["auto_filtered.jsonl"],
                outputs=["human_verified.jsonl"],
                tips=[
                    "重点关注事实性验证",
                    "允许标注者修改而非仅判断",
                    "收集修改原因用于改进提示词",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.POST_PROCESSING,
                name="最终处理",
                description="去重、格式化、准备发布",
                inputs=["human_verified.jsonl"],
                outputs=["final_dataset/"],
            ),
        ],
        quality_criteria=[
            "LLM 生成通过率 > 70%",
            "人工修正率 < 30%",
            "最终数据无事实错误",
        ],
        common_pitfalls=[
            "LLM 生成质量差导致人工成本过高",
            "人工修正标准不一致",
            "未利用修正反馈改进 LLM 提示词",
        ],
    ),
}


def get_pipeline_template(generation_type: str, synthetic_ratio: float = None) -> ProductionPipeline:
    """Get appropriate pipeline template based on generation type."""
    if synthetic_ratio is not None:
        if synthetic_ratio >= 0.9:
            return PIPELINE_TEMPLATES["distillation"]
        elif synthetic_ratio <= 0.1:
            return PIPELINE_TEMPLATES["human_annotation"]
        else:
            return PIPELINE_TEMPLATES["hybrid"]

    if generation_type in ["synthetic", "distillation"]:
        return PIPELINE_TEMPLATES["distillation"]
    elif generation_type in ["human", "human_annotation"]:
        return PIPELINE_TEMPLATES["human_annotation"]
    else:
        return PIPELINE_TEMPLATES["hybrid"]


def pipeline_to_markdown(pipeline: ProductionPipeline, dataset_name: str = None) -> str:
    """Convert pipeline to production guide markdown."""
    lines = []

    # Title
    title = f"数据生产指南：{dataset_name}" if dataset_name else f"数据生产指南：{pipeline.name}"
    lines.append(f"# {title}")
    lines.append("")

    # Overview
    lines.append("## 概述")
    lines.append("")
    lines.append(f"> {pipeline.description}")
    lines.append("")

    if pipeline.estimated_total_cost:
        lines.append(f"**预估总成本**: ${pipeline.estimated_total_cost:,.0f}")
    if pipeline.estimated_total_time:
        lines.append(f"**预估时间**: {pipeline.estimated_total_time}")
    if pipeline.target_size:
        lines.append(f"**目标数据量**: {pipeline.target_size:,} 条")
    lines.append("")

    # Prerequisites
    lines.append("## 前置准备")
    lines.append("")
    for prereq in pipeline.prerequisites:
        lines.append(f"- [ ] {prereq}")
    lines.append("")

    # Pipeline steps
    lines.append("## 生产流程")
    lines.append("")

    # Flow diagram
    lines.append("### 流程图")
    lines.append("")
    lines.append("```")
    step_names = [f"[{s.step_number}. {s.name}]" for s in pipeline.steps]
    lines.append(" → ".join(step_names))
    lines.append("```")
    lines.append("")

    # Detailed steps
    lines.append("### 详细步骤")
    lines.append("")

    for step in pipeline.steps:
        lines.append(f"#### 步骤 {step.step_number}: {step.name}")
        lines.append("")
        lines.append(f"**描述**: {step.description}")
        lines.append("")

        if step.inputs:
            lines.append(f"**输入**: {', '.join(step.inputs)}")
        if step.outputs:
            lines.append(f"**输出**: {', '.join(step.outputs)}")
        if step.tools:
            lines.append(f"**工具**: {', '.join(step.tools)}")
        if step.estimated_cost:
            lines.append(f"**成本**: ${step.estimated_cost} per item")
        lines.append("")

        if step.code_snippet:
            lines.append("**代码示例**:")
            lines.append("")
            lines.append("```python")
            lines.append(step.code_snippet.strip())
            lines.append("```")
            lines.append("")

        if step.tips:
            lines.append("**提示**:")
            lines.append("")
            for tip in step.tips:
                lines.append(f"- {tip}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Quality criteria
    lines.append("## 质量标准")
    lines.append("")
    for criterion in pipeline.quality_criteria:
        lines.append(f"- [ ] {criterion}")
    lines.append("")

    # Common pitfalls
    lines.append("## 常见问题与避坑指南")
    lines.append("")
    for i, pitfall in enumerate(pipeline.common_pitfalls, 1):
        lines.append(f"{i}. ⚠️ {pitfall}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*由 DataRecipe 生成 - 数据生产指南*")

    return "\n".join(lines)
