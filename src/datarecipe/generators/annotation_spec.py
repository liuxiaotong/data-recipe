"""Annotation Specification Generator.

Generates forward-looking annotation specification documents
that can be used to guide annotators in producing new data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from datarecipe.task_profiles import get_task_profile


@dataclass
class ScoringCriterion:
    """A single scoring criterion."""

    score: str  # e.g., "1分", "0分"
    description: str
    examples: List[str] = field(default_factory=list)


@dataclass
class ExampleItem:
    """An example item for the annotation spec."""

    id: int
    question_text: str
    answer: str
    scoring_criteria: List[ScoringCriterion] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationSpec:
    """Complete annotation specification."""

    # Basic info
    dataset_id: str
    dataset_type: str

    # Task description
    task_name: str = ""
    task_description: str = ""
    cognitive_requirements: List[str] = field(default_factory=list)
    reasoning_chain: str = ""

    # Data requirements
    data_requirements: List[str] = field(default_factory=list)
    quality_constraints: List[str] = field(default_factory=list)
    difficulty_calibration: str = ""
    format_requirements: List[str] = field(default_factory=list)

    # Examples
    examples: List[ExampleItem] = field(default_factory=list)

    # Scoring
    scoring_dimensions: List[str] = field(default_factory=list)
    scoring_rubrics: List[Dict[str, Any]] = field(default_factory=list)
    partial_credit_rules: List[str] = field(default_factory=list)

    # Metadata
    generated_at: str = ""
    source_samples: int = 0


class AnnotationSpecGenerator:
    """Generates annotation specifications from dataset analysis."""

    # Task type descriptions
    TASK_TYPE_INFO = {
        "preference": {
            "name": "偏好对比数据",
            "description": "人类偏好对比数据集，用于训练奖励模型或直接偏好优化。标注员需要对比两个回答，选择更符合人类偏好的一个。",
            "cognitive_requirements": [
                "理解问题上下文",
                "评估回答质量（准确性、有用性、安全性）",
                "进行相对比较判断",
            ],
            "reasoning_chain": "理解问题 → 阅读两个回答 → 多维度评估 → 选择更优回答",
        },
        "evaluation": {
            "name": "评测基准数据",
            "description": "用于评估模型能力的测试数据集。每道题目需要有明确的评分标准和参考答案。",
            "cognitive_requirements": [
                "理解任务目标",
                "设计有区分度的题目",
                "制定清晰的评分标准",
                "确保答案唯一性或评分客观性",
            ],
            "reasoning_chain": "定义考察点 → 设计题目 → 制定评分标准 → 验证难度",
        },
        "sft": {
            "name": "监督微调数据",
            "description": "用于模型监督微调的指令-回答对数据。需要高质量的指令和对应的标准回答。",
            "cognitive_requirements": [
                "理解指令意图",
                "生成准确、有帮助的回答",
                "保持一致的风格和格式",
            ],
            "reasoning_chain": "理解指令 → 规划回答结构 → 生成内容 → 质量检查",
        },
        "swe_bench": {
            "name": "软件工程评测数据",
            "description": "用于评估模型代码理解和修复能力的数据集。包含真实的代码问题和对应的修复补丁。",
            "cognitive_requirements": [
                "理解代码结构和逻辑",
                "定位问题根因",
                "设计正确的修复方案",
                "验证修复的正确性",
            ],
            "reasoning_chain": "理解问题描述 → 分析代码 → 定位bug → 设计修复 → 验证补丁",
        },
        "unknown": {
            "name": "通用数据标注",
            "description": "通用数据标注任务，请根据具体数据特点制定标注规范。",
            "cognitive_requirements": [
                "理解数据结构",
                "掌握标注标准",
                "保持标注一致性",
            ],
            "reasoning_chain": "理解任务 → 分析样本 → 执行标注 → 质量检查",
        },
    }

    # Default quality constraints by type
    DEFAULT_QUALITY_CONSTRAINTS = {
        "preference": [
            "两个回答必须有明显的质量差异",
            "选择理由需要有明确依据",
            "避免基于主观喜好的判断",
        ],
        "evaluation": [
            "题目答案必须唯一或有明确的评分标准",
            "题目难度需要经过模型验证",
            "禁止使用 AI 生成的内容作为题目或答案",
        ],
        "sft": [
            "回答必须准确、有帮助",
            "格式和风格保持一致",
            "避免有害或不当内容",
        ],
        "swe_bench": [
            "补丁必须能够正确修复问题",
            "代码风格与原项目保持一致",
            "包含必要的测试验证",
        ],
    }

    def __init__(self):
        pass

    def generate(
        self,
        dataset_id: str,
        dataset_type: str,
        schema_info: Dict[str, Any],
        sample_items: List[Dict[str, Any]],
        rubrics_result: Optional[Any] = None,
        llm_analysis: Optional[Any] = None,
        complexity_metrics: Optional[Any] = None,
        enhanced_context: Optional[Any] = None,
    ) -> AnnotationSpec:
        """Generate annotation specification.

        Args:
            dataset_id: Dataset identifier
            dataset_type: Type of dataset
            schema_info: Schema information
            sample_items: Sample data items
            rubrics_result: Rubrics analysis result (optional)
            llm_analysis: LLM analysis result (optional)
            complexity_metrics: Complexity metrics (optional)

        Returns:
            AnnotationSpec with complete specification
        """
        spec = AnnotationSpec(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            source_samples=len(sample_items),
        )

        # Get base task info from unified profile registry
        profile = get_task_profile(dataset_type)

        spec.task_name = profile.name
        spec.task_description = profile.description
        spec.cognitive_requirements = list(profile.cognitive_requirements)
        spec.reasoning_chain = profile.reasoning_chain

        # Override with LLM analysis if available
        if llm_analysis:
            if hasattr(llm_analysis, "purpose") and llm_analysis.purpose:
                spec.task_description = llm_analysis.purpose
            if hasattr(llm_analysis, "production_steps") and llm_analysis.production_steps:
                spec.reasoning_chain = " → ".join(llm_analysis.production_steps[:5])

        # Generate data requirements
        self._generate_data_requirements(spec, schema_info, complexity_metrics)

        # Generate quality constraints
        self._generate_quality_constraints(spec, dataset_type, rubrics_result)

        # Generate format requirements
        self._generate_format_requirements(spec, schema_info)

        # Generate examples
        self._generate_examples(spec, sample_items, rubrics_result)

        # Generate scoring rubrics
        self._generate_scoring_rubrics(spec, rubrics_result, llm_analysis)

        # Store enhanced_context for to_markdown to use
        spec._enhanced_context = enhanced_context

        return spec

    def _generate_data_requirements(
        self,
        spec: AnnotationSpec,
        schema_info: Dict[str, Any],
        complexity_metrics: Optional[Any],
    ) -> None:
        """Generate data requirements section."""
        requirements = []

        # Schema-based requirements
        field_count = len(schema_info)
        requirements.append(f"每条数据需包含 {field_count} 个字段")

        # Check for required fields
        required_fields = []
        for field_name, info in schema_info.items():
            if info.get("type") in ["str", "list"]:
                required_fields.append(field_name)

        if required_fields:
            requirements.append(f"必填字段: {', '.join(required_fields[:5])}")

        # Complexity-based requirements
        if complexity_metrics:
            if hasattr(complexity_metrics, "length_category"):
                length_desc = {
                    "short": "简短（< 500 字符）",
                    "medium": "中等（500-2000 字符）",
                    "long": "较长（2000-5000 字符）",
                    "very_long": "长文本（> 5000 字符）",
                }.get(complexity_metrics.length_category, "")
                if length_desc:
                    requirements.append(f"文本长度要求: {length_desc}")

            if hasattr(complexity_metrics, "primary_domain"):
                domain = complexity_metrics.primary_domain.value
                if domain != "general":
                    requirements.append(f"领域要求: 需要 {domain} 领域专业知识")

        spec.data_requirements = requirements

    def _generate_quality_constraints(
        self,
        spec: AnnotationSpec,
        dataset_type: str,
        rubrics_result: Optional[Any],
    ) -> None:
        """Generate quality constraints section."""
        constraints = []

        # Default constraints by type (from unified profile registry)
        profile = get_task_profile(dataset_type)
        constraints.extend(profile.default_quality_constraints)

        # Add universal constraints
        constraints.extend(
            [
                "交付数据中不能含有任何 AI 产出的内容",
                "数据格式必须符合 Schema 规范",
                "每条数据需经过质量检查",
            ]
        )

        # Rubric-based constraints
        if rubrics_result and hasattr(rubrics_result, "verb_distribution"):
            top_verbs = list(rubrics_result.verb_distribution.keys())[:3]
            if top_verbs:
                constraints.append(f"评分标准应包含: {', '.join(top_verbs)} 等动词")

        spec.quality_constraints = constraints

        # Difficulty calibration
        spec.difficulty_calibration = (
            "建议使用主流模型（如 GPT-4、Claude）进行难度校准，确保题目具有适当的区分度"
        )

    def _generate_format_requirements(
        self,
        spec: AnnotationSpec,
        schema_info: Dict[str, Any],
    ) -> None:
        """Generate format requirements section."""
        requirements = []

        for field_name, info in schema_info.items():
            field_type = info.get("type", "unknown")
            nested = info.get("nested_type")

            if field_type == "str":
                requirements.append(f"`{field_name}`: 字符串类型")
            elif field_type == "list":
                if nested:
                    requirements.append(f"`{field_name}`: 列表类型，元素为 {nested}")
                else:
                    requirements.append(f"`{field_name}`: 列表类型")
            elif field_type == "dict":
                requirements.append(f"`{field_name}`: 对象类型")
            elif field_type == "int":
                requirements.append(f"`{field_name}`: 整数类型")
            elif field_type == "float":
                requirements.append(f"`{field_name}`: 浮点数类型")
            elif field_type == "bool":
                requirements.append(f"`{field_name}`: 布尔类型")

        spec.format_requirements = requirements

    def _score_example_quality(self, item: Dict[str, Any], dataset_type: str) -> float:
        """Score the quality of an example for selection.

        Returns a score from 0-10, higher is better.
        """
        score = 5.0  # Base score

        # Check for preference dataset quality
        if dataset_type == "preference":
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            if isinstance(chosen, str) and isinstance(rejected, str):
                # Penalize if both are very similar
                if chosen == rejected:
                    return 0  # Duplicate, skip

                # Check length difference (some difference is good)
                len_diff = abs(len(chosen) - len(rejected))
                if len_diff > 50:
                    score += 1  # Good differentiation

                # Penalize very short responses
                if len(chosen) < 20 or len(rejected) < 20:
                    score -= 2

                # Check for multi-turn conversations (better examples)
                if "Human:" in chosen and "Assistant:" in chosen:
                    # Count turns
                    turns = chosen.count("Human:")
                    if turns >= 2:
                        score += 1  # Multi-turn is more interesting

                # Penalize if chosen seems worse than rejected (quality issue)
                # Simple heuristic: chosen should not be much shorter if it's supposed to be better
                if len(chosen) < len(rejected) * 0.3 and len(rejected) > 100:
                    score -= 2  # Suspicious - chosen much shorter

                # Check for harmful content markers (prefer safe examples)
                harmful_markers = ["fuck", "shit", "kill", "steal", "hack"]
                if any(m in chosen.lower() for m in harmful_markers):
                    score -= 1  # Less suitable as example

                # Prefer examples with clear topic
                clear_topics = ["explain", "what is", "how to", "help me", "can you"]
                human_query = ""
                if "Human:" in chosen:
                    parts = chosen.split("Human:")
                    if len(parts) > 1:
                        human_query = parts[1].split("Assistant:")[0].lower()

                if any(t in human_query for t in clear_topics):
                    score += 1  # Clear, instructive topic

        else:
            # For other dataset types
            # Prefer items with more complete data
            non_empty_fields = sum(
                1 for v in item.values() if v and (not isinstance(v, str) or len(v) > 10)
            )
            score += non_empty_fields * 0.5

            # Prefer items with reasonable text length
            for v in item.values():
                if isinstance(v, str):
                    if 100 <= len(v) <= 2000:
                        score += 0.5  # Good length

        return max(0, min(10, score))

    def _select_best_examples(
        self,
        sample_items: List[Dict[str, Any]],
        dataset_type: str,
        count: int = 3,
    ) -> List[Dict[str, Any]]:
        """Select the best examples based on quality scoring."""
        if not sample_items:
            return []

        # Score all items
        scored = [(item, self._score_example_quality(item, dataset_type)) for item in sample_items]

        # Filter out very low quality
        scored = [(item, score) for item, score in scored if score >= 3]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return [item for item, score in scored[:count]]

    def _generate_examples(
        self,
        spec: AnnotationSpec,
        sample_items: List[Dict[str, Any]],
        rubrics_result: Optional[Any],
    ) -> None:
        """Generate example items section."""
        examples = []

        # Select best examples based on quality
        best_items = self._select_best_examples(sample_items, spec.dataset_type, count=3)

        for i, item in enumerate(best_items):
            example = ExampleItem(
                id=i + 1,
                question_text="",
                answer="",
            )

            # Extract question/input
            for field in ["question", "input", "prompt", "instruction", "problem_statement"]:
                if field in item and item[field]:
                    val = item[field]
                    if isinstance(val, str):
                        example.question_text = val[:500] + ("..." if len(val) > 500 else "")
                    break

            # Try messages format
            if not example.question_text and "messages" in item:
                messages = item["messages"]
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                example.question_text = content[:500] + (
                                    "..." if len(content) > 500 else ""
                                )
                                break

            # Fallback to first string field
            if not example.question_text:
                for field, val in item.items():
                    if isinstance(val, str) and len(val) > 50:
                        example.question_text = val[:500] + ("..." if len(val) > 500 else "")
                        break

            # Extract answer/output
            for field in ["answer", "output", "response", "completion", "solution"]:
                if field in item and item[field]:
                    val = item[field]
                    if isinstance(val, str):
                        example.answer = val[:200] + ("..." if len(val) > 200 else "")
                    break

            # Extract rubrics as scoring criteria
            for field in ["rubrics", "rubric", "criteria"]:
                if field in item and item[field]:
                    rubrics = item[field]
                    if isinstance(rubrics, list):
                        for r in rubrics[:3]:
                            if isinstance(r, str):
                                example.scoring_criteria.append(
                                    ScoringCriterion(score="✓", description=r)
                                )
                    elif isinstance(rubrics, str):
                        example.scoring_criteria.append(
                            ScoringCriterion(score="✓", description=rubrics)
                        )
                    break

            # Add default scoring if none found
            if not example.scoring_criteria:
                example.scoring_criteria = [
                    ScoringCriterion(
                        score="1分",
                        description="回答正确且完整",
                    ),
                    ScoringCriterion(
                        score="0分",
                        description="回答错误或不完整",
                    ),
                ]

            examples.append(example)

        spec.examples = examples

    def _generate_scoring_rubrics(
        self,
        spec: AnnotationSpec,
        rubrics_result: Optional[Any],
        llm_analysis: Optional[Any],
    ) -> None:
        """Generate scoring rubrics section."""
        dimensions = []
        rubrics = []
        partial_credit = []

        # From LLM analysis
        if llm_analysis and hasattr(llm_analysis, "quality_criteria"):
            for criterion in llm_analysis.quality_criteria[:5]:
                dimensions.append(criterion)

        # From rubrics analysis
        if rubrics_result:
            if hasattr(rubrics_result, "verb_distribution"):
                top_verbs = list(rubrics_result.verb_distribution.items())[:5]
                for verb, count in top_verbs:
                    rubrics.append(
                        {
                            "dimension": verb,
                            "frequency": count,
                            "description": f"回答应 {verb} 相关内容",
                        }
                    )

            if hasattr(rubrics_result, "structured_patterns"):
                for pattern in rubrics_result.structured_patterns[:3]:
                    if isinstance(pattern, dict):
                        rubrics.append(pattern)

        # Default dimensions if none found
        if not dimensions:
            dimensions = [
                "准确性: 回答是否正确",
                "完整性: 回答是否覆盖所有要点",
                "清晰性: 回答是否易于理解",
                "相关性: 回答是否切题",
            ]

        # Default partial credit rules
        partial_credit = [
            "部分正确的回答可获得部分分数",
            "格式错误但内容正确可酌情扣分",
            "有多个要点时按要点给分",
        ]

        spec.scoring_dimensions = dimensions
        spec.scoring_rubrics = rubrics
        spec.partial_credit_rules = partial_credit

    def to_markdown(self, spec: AnnotationSpec) -> str:
        """Convert specification to Markdown format."""
        lines = []

        # Header
        lines.append(f"# {spec.dataset_id} 标注规范")
        lines.append("")
        lines.append(f"> 生成时间: {spec.generated_at}")
        lines.append(f"> 数据类型: {spec.task_name}")
        lines.append(f"> 样本来源: {spec.source_samples} 条")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Task Description
        lines.append("## 一、题目类型描述")
        lines.append("")
        lines.append(f"**任务名称**: {spec.task_name}")
        lines.append("")
        lines.append(f"**任务说明**: {spec.task_description}")
        lines.append("")

        if spec.cognitive_requirements:
            lines.append("**认知要求**:")
            lines.append("")
            for req in spec.cognitive_requirements:
                lines.append(f"- {req}")
            lines.append("")

        if spec.reasoning_chain:
            lines.append(f"**推理链**: {spec.reasoning_chain}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Data Requirements
        lines.append("## 二、数据要求")
        lines.append("")

        if spec.data_requirements:
            for i, req in enumerate(spec.data_requirements, 1):
                lines.append(f"{i}. {req}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Quality Constraints
        lines.append("## 三、质量约束")
        lines.append("")

        if spec.quality_constraints:
            for i, constraint in enumerate(spec.quality_constraints, 1):
                # Highlight important constraints
                if "禁止" in constraint or "不能" in constraint or "必须" in constraint:
                    lines.append(f"{i}. **{constraint}**")
                else:
                    lines.append(f"{i}. {constraint}")
            lines.append("")

        if spec.difficulty_calibration:
            lines.append(f"**难度校准**: {spec.difficulty_calibration}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Format Requirements
        lines.append("## 四、格式要求")
        lines.append("")
        lines.append("### Schema 定义")
        lines.append("")
        lines.append("| 字段 | 要求 |")
        lines.append("|------|------|")
        for req in spec.format_requirements:
            lines.append(f"| {req} |")
        lines.append("")

        lines.append("---")
        lines.append("")

        # Examples
        lines.append("## 五、例题")
        lines.append("")

        if spec.examples:
            lines.append("| 序号 | 题目 | 答案 | 打分标准 |")
            lines.append("|------|------|------|----------|")

            for ex in spec.examples:
                question = ex.question_text.replace("\n", " ")[:100]
                answer = ex.answer.replace("\n", " ")[:50] if ex.answer else "-"
                scoring = "; ".join(
                    [f"{c.score}: {c.description[:30]}" for c in ex.scoring_criteria[:2]]
                )
                lines.append(f"| {ex.id} | {question}... | {answer} | {scoring} |")

            lines.append("")

            # Detailed examples
            lines.append("### 详细示例")
            lines.append("")

            for ex in spec.examples:
                lines.append(f"#### 例题 {ex.id}")
                lines.append("")
                lines.append("**题目**:")
                lines.append("")
                lines.append(f"> {ex.question_text}")
                lines.append("")

                if ex.answer:
                    lines.append("**参考答案**:")
                    lines.append("")
                    lines.append(f"> {ex.answer}")
                    lines.append("")

                if ex.scoring_criteria:
                    lines.append("**打分标准**:")
                    lines.append("")
                    for criterion in ex.scoring_criteria:
                        lines.append(f"- {criterion.score}: {criterion.description}")
                    lines.append("")

        lines.append("---")
        lines.append("")

        # Scoring Rubrics
        lines.append("## 六、打分标准")
        lines.append("")

        if spec.scoring_dimensions:
            lines.append("### 评分维度")
            lines.append("")
            for dim in spec.scoring_dimensions:
                lines.append(f"- {dim}")
            lines.append("")

        if spec.scoring_rubrics:
            lines.append("### 评分细则")
            lines.append("")
            lines.append("| 维度 | 说明 |")
            lines.append("|------|------|")
            for rubric in spec.scoring_rubrics:
                if isinstance(rubric, dict):
                    dim = rubric.get("dimension", "-")
                    desc = rubric.get("description", "-")
                    lines.append(f"| {dim} | {desc} |")
            lines.append("")

        if spec.partial_credit_rules:
            lines.append("### 部分得分规则")
            lines.append("")
            for rule in spec.partial_credit_rules:
                lines.append(f"- {rule}")
            lines.append("")

        # LLM-enhanced: Domain-specific guidelines
        ec = getattr(spec, "_enhanced_context", None)
        if ec and getattr(ec, "generated", False):
            if ec.domain_specific_guidelines:
                lines.append("## 七、领域标注指导")
                lines.append("")
                lines.append(ec.domain_specific_guidelines)
                lines.append("")

            if ec.quality_pitfalls:
                lines.append("### 常见错误")
                lines.append("")
                for pitfall in ec.quality_pitfalls:
                    lines.append(f"- ⚠️ {pitfall}")
                lines.append("")

            if ec.example_analysis:
                lines.append("### 样本点评")
                lines.append("")
                for analysis in ec.example_analysis:
                    if isinstance(analysis, dict):
                        idx = analysis.get("sample_index", "?")
                        lines.append(f"**样本 {idx}**:")
                        if analysis.get("strengths"):
                            lines.append(f"- 优点: {analysis['strengths']}")
                        if analysis.get("weaknesses"):
                            lines.append(f"- 改进: {analysis['weaknesses']}")
                        if analysis.get("annotation_tips"):
                            lines.append(f"- 建议: {analysis['annotation_tips']}")
                        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本规范由 DataRecipe 自动生成，请根据实际需求调整")

        return "\n".join(lines)

    def to_dict(self, spec: AnnotationSpec) -> dict:
        """Convert specification to dictionary."""
        return {
            "dataset_id": spec.dataset_id,
            "dataset_type": spec.dataset_type,
            "task": {
                "name": spec.task_name,
                "description": spec.task_description,
                "cognitive_requirements": spec.cognitive_requirements,
                "reasoning_chain": spec.reasoning_chain,
            },
            "requirements": {
                "data": spec.data_requirements,
                "quality": spec.quality_constraints,
                "difficulty": spec.difficulty_calibration,
                "format": spec.format_requirements,
            },
            "examples": [
                {
                    "id": ex.id,
                    "question": ex.question_text,
                    "answer": ex.answer,
                    "scoring": [
                        {"score": c.score, "description": c.description}
                        for c in ex.scoring_criteria
                    ],
                }
                for ex in spec.examples
            ],
            "scoring": {
                "dimensions": spec.scoring_dimensions,
                "rubrics": spec.scoring_rubrics,
                "partial_credit": spec.partial_credit_rules,
            },
            "metadata": {
                "generated_at": spec.generated_at,
                "source_samples": spec.source_samples,
            },
        }
