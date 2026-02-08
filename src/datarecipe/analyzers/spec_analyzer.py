"""Specification document analyzer using LLM."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from datarecipe.constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL
from datarecipe.parsers import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)

# --- Type mapping utility ---


def _map_type(type_str: str) -> str:
    """Map common type names to JSON Schema types."""
    mapping = {
        "string": "string",
        "text": "string",
        "code": "string",
        "image": "string",
        "number": "number",
        "float": "number",
        "double": "number",
        "integer": "integer",
        "int": "integer",
        "boolean": "boolean",
        "bool": "boolean",
        "array": "array",
        "list": "array",
        "object": "object",
        "dict": "object",
        "map": "object",
    }
    return mapping.get(type_str.lower().strip(), "string")


# --- FieldDefinition: nested schema modeling ---


@dataclass
class FieldDefinition:
    """Rich field definition supporting nested/complex types (JSON Schema compatible)."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False

    # Nested structure (for array items / object properties)
    items: Optional["FieldDefinition"] = None  # array element type
    properties: Optional[list["FieldDefinition"]] = None  # object sub-fields

    # Enum / union
    enum: Optional[list[Any]] = None
    any_of: Optional[list["FieldDefinition"]] = None

    # Constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None

    # --- serialization ---

    def to_dict(self) -> dict:
        """Serialize to plain dict (round-trippable with from_dict)."""
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.description:
            d["description"] = self.description
        if self.required:
            d["required"] = True
        if self.items is not None:
            d["items"] = self.items.to_dict()
        if self.properties is not None:
            d["properties"] = [p.to_dict() for p in self.properties]
        if self.enum is not None:
            d["enum"] = self.enum
        if self.any_of is not None:
            d["any_of"] = [a.to_dict() for a in self.any_of]
        for attr in ("min_length", "max_length", "minimum", "maximum", "pattern"):
            val = getattr(self, attr)
            if val is not None:
                d[attr] = val
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldDefinition":
        """Deserialize from dict. Backwards-compatible with old flat format
        ``{"name": "x", "type": "string", "required": true, "description": "..."}``."""
        items = None
        if "items" in data and isinstance(data["items"], dict):
            items = cls.from_dict(data["items"])

        properties = None
        if "properties" in data and isinstance(data["properties"], list):
            properties = [cls.from_dict(p) for p in data["properties"]]

        any_of = None
        if "any_of" in data and isinstance(data["any_of"], list):
            any_of = [cls.from_dict(a) for a in data["any_of"]]

        # Handle old-style "required" which may be bool or truthy string
        raw_req = data.get("required", False)
        req = raw_req if isinstance(raw_req, bool) else str(raw_req).lower() in ("true", "1", "yes")

        return cls(
            name=data.get("name", ""),
            type=data.get("type", "string"),
            description=data.get("description", ""),
            required=req,
            items=items,
            properties=properties,
            enum=data.get("enum"),
            any_of=any_of,
            min_length=data.get("min_length") or data.get("minLength"),
            max_length=data.get("max_length") or data.get("maxLength"),
            minimum=data.get("minimum"),
            maximum=data.get("maximum"),
            pattern=data.get("pattern"),
        )

    def to_json_schema(self) -> dict:
        """Convert to a JSON Schema property definition."""
        json_type = _map_type(self.type)
        schema: dict[str, Any] = {"type": json_type}

        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.pattern is not None:
            schema["pattern"] = self.pattern

        # String constraints
        if json_type == "string":
            if self.min_length is not None:
                schema["minLength"] = self.min_length
            if self.max_length is not None:
                schema["maxLength"] = self.max_length

        # Numeric constraints
        if json_type in ("number", "integer"):
            if self.minimum is not None:
                schema["minimum"] = self.minimum
            if self.maximum is not None:
                schema["maximum"] = self.maximum

        # Array items
        if json_type == "array" and self.items is not None:
            schema["items"] = self.items.to_json_schema()

        # Object properties
        if json_type == "object" and self.properties is not None:
            props = {}
            req = []
            for p in self.properties:
                props[p.name] = p.to_json_schema()
                if p.required:
                    req.append(p.name)
            schema["properties"] = props
            if req:
                schema["required"] = req

        # Union (anyOf)
        if self.any_of is not None:
            schema.pop("type", None)
            schema["anyOf"] = [a.to_json_schema() for a in self.any_of]

        return schema


# --- FieldConstraint: structured constraints ---


@dataclass
class FieldConstraint:
    """Structured constraint for a field."""

    field_name: str
    constraint_type: str = "general"  # format, range, content, uniqueness, custom
    rule: str = ""
    severity: str = "error"  # error, warning, info
    auto_checkable: bool = False

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "constraint_type": self.constraint_type,
            "rule": self.rule,
            "severity": self.severity,
            "auto_checkable": self.auto_checkable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldConstraint":
        return cls(
            field_name=data.get("field_name", ""),
            constraint_type=data.get("constraint_type", "general"),
            rule=data.get("rule", ""),
            severity=data.get("severity", "error"),
            auto_checkable=data.get("auto_checkable", False),
        )


# --- ValidationStrategy ---


@dataclass
class ValidationStrategy:
    """A pluggable validation strategy."""

    strategy_type: str  # model_test, human_review, format_check, cross_validation, auto_scoring
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy_type": self.strategy_type,
            "enabled": self.enabled,
            "config": self.config,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationStrategy":
        return cls(
            strategy_type=data.get("strategy_type", ""),
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
            description=data.get("description", ""),
        )

    @classmethod
    def from_difficulty_validation(cls, diff_val: dict[str, Any]) -> "ValidationStrategy":
        """Convert legacy difficulty_validation dict to a ValidationStrategy."""
        return cls(
            strategy_type="model_test",
            enabled=True,
            config={
                "model": diff_val.get("model", ""),
                "settings": diff_val.get("settings", ""),
                "test_count": diff_val.get("test_count", 3),
                "max_correct": diff_val.get("max_correct", 1),
                "pass_criteria": diff_val.get("pass_criteria", ""),
                "requires_record": diff_val.get("requires_record", True),
            },
            description=f"使用 {diff_val.get('model', '模型')} 进行难度验证",
        )


# --- SpecificationAnalysis ---


@dataclass
class SpecificationAnalysis:
    """Structured analysis of a specification document."""

    # Basic info
    project_name: str = ""
    dataset_type: str = ""  # evaluation, preference, sft, etc.
    description: str = ""

    # Task definition
    task_type: str = ""  # 题目类型
    task_description: str = ""  # 题目类型描述
    cognitive_requirements: list[str] = field(default_factory=list)  # 认知要求
    reasoning_chain: list[str] = field(default_factory=list)  # 推理链

    # Data requirements
    data_requirements: list[str] = field(default_factory=list)  # 数据要求
    quality_constraints: list[str] = field(default_factory=list)  # 质量约束
    forbidden_items: list[str] = field(default_factory=list)  # 禁止项
    difficulty_criteria: str = ""  # 难度标准

    # Difficulty validation config (extracted from spec, None if not specified)
    difficulty_validation: Optional[dict[str, Any]] = None  # 难度验证配置
    # Example: {"model": "doubao1.8", "settings": "高思考深度", "test_count": 3, "max_correct": 1, "requires_record": True}

    # Data structure
    fields: list[dict[str, str]] = field(default_factory=list)  # 字段定义
    field_requirements: dict[str, str] = field(default_factory=dict)  # 字段要求

    # Structured field constraints (Upgrade 6)
    field_constraints: list[dict] = field(default_factory=list)

    # Validation strategies (Upgrade 3)
    validation_strategies: list[dict] = field(default_factory=list)

    # Quality gates (Upgrade 4)
    quality_gates: list[dict] = field(default_factory=list)

    # Examples
    examples: list[dict[str, Any]] = field(default_factory=list)  # 示例

    # Scoring
    scoring_rubric: list[dict[str, str]] = field(default_factory=list)  # 打分标准

    # Estimates
    estimated_difficulty: str = ""  # easy/medium/hard/expert
    estimated_domain: str = ""  # 领域
    estimated_human_percentage: float = 95.0  # 人工比例估计
    similar_datasets: list[str] = field(default_factory=list)

    # Raw
    raw_text: str = ""
    has_images: bool = False
    image_count: int = 0

    # --- Computed properties ---

    @property
    def field_definitions(self) -> list[FieldDefinition]:
        """Parse ``fields`` into rich FieldDefinition objects (cached on instance)."""
        cache_attr = "_cached_field_definitions"
        if not hasattr(self, cache_attr) or getattr(self, cache_attr) is None:
            defs = [FieldDefinition.from_dict(f) for f in self.fields] if self.fields else []
            object.__setattr__(self, cache_attr, defs)
        return getattr(self, cache_attr)

    @property
    def parsed_constraints(self) -> list[FieldConstraint]:
        """Merge ``field_constraints`` (new) + ``field_requirements`` + ``quality_constraints`` (legacy)."""
        constraints: list[FieldConstraint] = []
        # New-format field_constraints
        for c in self.field_constraints:
            constraints.append(FieldConstraint.from_dict(c))
        # Legacy field_requirements
        for fname, rule_text in self.field_requirements.items():
            if not any(c.field_name == fname and c.rule == rule_text for c in constraints):
                constraints.append(
                    FieldConstraint(
                        field_name=fname,
                        constraint_type="general",
                        rule=rule_text,
                        severity="error",
                        auto_checkable=False,
                    )
                )
        # Legacy quality_constraints (global, not per-field)
        for qc in self.quality_constraints:
            if not any(c.field_name == "_global" and c.rule == qc for c in constraints):
                constraints.append(
                    FieldConstraint(
                        field_name="_global",
                        constraint_type="content",
                        rule=qc,
                        severity="error",
                        auto_checkable=False,
                    )
                )
        return constraints

    def constraints_for_field(self, field_name: str) -> list[FieldConstraint]:
        """Return constraints applicable to a specific field (including _global)."""
        return [c for c in self.parsed_constraints if c.field_name in (field_name, "_global")]

    @property
    def parsed_validation_strategies(self) -> list[ValidationStrategy]:
        """Merge ``validation_strategies`` (new) + legacy ``difficulty_validation``."""
        strategies: list[ValidationStrategy] = []
        for vs in self.validation_strategies:
            strategies.append(ValidationStrategy.from_dict(vs))
        # Legacy difficulty_validation → model_test strategy
        if self.difficulty_validation is not None:
            if not any(s.strategy_type == "model_test" for s in strategies):
                strategies.append(
                    ValidationStrategy.from_difficulty_validation(self.difficulty_validation)
                )
        return strategies

    def get_strategy(self, strategy_type: str) -> Optional[ValidationStrategy]:
        """Get a specific validation strategy by type."""
        for s in self.parsed_validation_strategies:
            if s.strategy_type == strategy_type and s.enabled:
                return s
        return None

    def has_strategy(self, strategy_type: str) -> bool:
        """Check if a specific validation strategy is present and enabled."""
        return self.get_strategy(strategy_type) is not None

    def to_dict(self) -> dict:
        d = {
            "project_name": self.project_name,
            "dataset_type": self.dataset_type,
            "description": self.description,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "cognitive_requirements": self.cognitive_requirements,
            "reasoning_chain": self.reasoning_chain,
            "data_requirements": self.data_requirements,
            "quality_constraints": self.quality_constraints,
            "forbidden_items": self.forbidden_items,
            "difficulty_criteria": self.difficulty_criteria,
            "difficulty_validation": self.difficulty_validation,
            "fields": self.fields,
            "field_requirements": self.field_requirements,
            "field_constraints": self.field_constraints,
            "validation_strategies": self.validation_strategies,
            "quality_gates": self.quality_gates,
            "examples": self.examples,
            "scoring_rubric": self.scoring_rubric,
            "estimated_difficulty": self.estimated_difficulty,
            "estimated_domain": self.estimated_domain,
            "estimated_human_percentage": self.estimated_human_percentage,
            "similar_datasets": self.similar_datasets,
            "has_images": self.has_images,
            "image_count": self.image_count,
        }
        # Include rich field_definitions for downstream consumers
        if self.fields:
            d["field_definitions"] = [fd.to_dict() for fd in self.field_definitions]
        return d

    def has_difficulty_validation(self) -> bool:
        """Check if difficulty validation is required (legacy or new strategies)."""
        if self.difficulty_validation is not None:
            return True
        return self.has_strategy("model_test")


class SpecAnalyzer:
    """Analyze specification documents using LLM."""

    EXTRACTION_PROMPT = """你是一个数据标注项目分析专家。请分析以下需求文档，提取结构化信息。

## 需求文档内容

{document_content}

## 请提取以下信息，以 JSON 格式返回：

```json
{{
  "project_name": "项目名称（从文档标题或内容推断）",
  "dataset_type": "数据集类型（evaluation/preference/sft/multimodal/reasoning 等）",
  "description": "项目简短描述（1-2句话）",

  "task_type": "题目类型名称",
  "task_description": "题目类型的详细描述",
  "cognitive_requirements": ["认知要求1", "认知要求2"],
  "reasoning_chain": ["推理步骤1", "推理步骤2", "推理步骤3"],

  "data_requirements": ["数据要求1", "数据要求2"],
  "quality_constraints": ["质量约束1", "质量约束2"],
  "forbidden_items": ["禁止项1（如禁止AI内容）", "禁止项2"],
  "difficulty_criteria": "难度验证标准描述（如有）",

  "difficulty_validation": {{
    "enabled": true,
    "model": "用于验证难度的模型名称（如doubao1.8、gpt-4等）",
    "settings": "模型设置（如高思考深度）",
    "test_count": 3,
    "max_correct": 1,
    "pass_criteria": "通过标准描述（如：跑3次最多1次正确）",
    "requires_record": true
  }},

  "fields": [
    {{"name": "字段名", "type": "类型", "required": true, "description": "说明"}},
    {{"name": "对话历史", "type": "array", "required": true, "description": "多轮对话",
      "items": {{"name": "turn", "type": "object", "properties": [
        {{"name": "role", "type": "string", "enum": ["user", "assistant"]}},
        {{"name": "content", "type": "string"}}
      ]}}
    }},
    {{"name": "答案", "type": "string", "enum": ["A", "B", "C", "D"], "description": "选项"}}
  ],
  "field_requirements": {{
    "字段名": "具体要求"
  }},

  "field_constraints": [
    {{
      "field_name": "字段名",
      "constraint_type": "format|range|content|uniqueness|custom",
      "rule": "约束规则描述",
      "severity": "error|warning|info",
      "auto_checkable": true
    }}
  ],

  "validation_strategies": [
    {{
      "strategy_type": "model_test|human_review|format_check|cross_validation|auto_scoring",
      "enabled": true,
      "config": {{}},
      "description": "策略描述"
    }}
  ],

  "quality_gates": [
    {{
      "gate_id": "min_overall_score",
      "name": "最低综合分",
      "metric": "overall_score",
      "operator": ">=",
      "threshold": 60,
      "severity": "blocker"
    }}
  ],

  "examples": [
    {{
      "id": 1,
      "question": "题目文字",
      "answer": "答案",
      "has_image": true,
      "scoring_rubric": "打分标准（如有）"
    }}
  ],

  "scoring_rubric": [
    {{"score": "1分", "criteria": "得分标准"}}
  ],

  "estimated_difficulty": "hard",
  "estimated_domain": "multimodal_reasoning",
  "estimated_human_percentage": 95,
  "similar_datasets": ["类似数据集1", "类似数据集2"]
}}
```

注意：
1. 如果某项信息在文档中没有，请合理推断或留空
2. examples 数组最多包含 3 个示例
3. 确保返回有效的 JSON 格式
4. difficulty_validation: 如果文档中没有提到需要用模型验证难度，则设置 enabled 为 false 或整个字段设为 null
5. fields 支持嵌套结构：用 items 表示数组元素类型，用 properties 表示对象子字段，用 enum 表示枚举值
6. field_constraints: 如文档未提及具体约束，可留空数组
7. validation_strategies: 如文档未提及验证策略，可留空数组
8. quality_gates: 如文档未提及质量门禁，可留空数组
"""

    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.parser = DocumentParser()
        self._last_doc: Optional[ParsedDocument] = None

    def parse_document(self, file_path: str) -> ParsedDocument:
        """Parse document without LLM analysis.

        Args:
            file_path: Path to the specification file

        Returns:
            ParsedDocument with text and images
        """
        self._last_doc = self.parser.parse(file_path)
        return self._last_doc

    def get_extraction_prompt(self, doc: Optional[ParsedDocument] = None) -> str:
        """Get the LLM extraction prompt for a parsed document.

        Args:
            doc: ParsedDocument to analyze (uses last parsed if None)

        Returns:
            Formatted prompt string for LLM
        """
        if doc is None:
            doc = self._last_doc
        if doc is None:
            raise ValueError("No document parsed. Call parse_document first.")

        return self.EXTRACTION_PROMPT.format(document_content=doc.text_content[:15000])

    def create_analysis_from_json(
        self, extracted: dict, doc: Optional[ParsedDocument] = None
    ) -> SpecificationAnalysis:
        """Create SpecificationAnalysis from extracted JSON data.

        Args:
            extracted: Dictionary with extracted information
            doc: ParsedDocument for metadata (uses last parsed if None)

        Returns:
            SpecificationAnalysis populated with data
        """
        if doc is None:
            doc = self._last_doc

        analysis = SpecificationAnalysis(
            raw_text=doc.text_content if doc else "",
            has_images=doc.has_images() if doc else False,
            image_count=len(doc.images) if doc else 0,
        )

        # Populate from extracted data
        analysis.project_name = extracted.get("project_name", "")
        analysis.dataset_type = extracted.get("dataset_type", "")
        analysis.description = extracted.get("description", "")
        analysis.task_type = extracted.get("task_type", "")
        analysis.task_description = extracted.get("task_description", "")
        analysis.cognitive_requirements = extracted.get("cognitive_requirements", [])
        analysis.reasoning_chain = extracted.get("reasoning_chain", [])
        analysis.data_requirements = extracted.get("data_requirements", [])
        analysis.quality_constraints = extracted.get("quality_constraints", [])
        analysis.forbidden_items = extracted.get("forbidden_items", [])
        analysis.difficulty_criteria = extracted.get("difficulty_criteria", "")
        analysis.fields = extracted.get("fields", [])
        analysis.field_requirements = extracted.get("field_requirements", {})
        analysis.field_constraints = extracted.get("field_constraints", [])
        analysis.validation_strategies = extracted.get("validation_strategies", [])
        analysis.quality_gates = extracted.get("quality_gates", [])
        analysis.examples = extracted.get("examples", [])
        analysis.scoring_rubric = extracted.get("scoring_rubric", [])
        analysis.estimated_difficulty = extracted.get("estimated_difficulty", "hard")
        analysis.estimated_domain = extracted.get("estimated_domain", "")
        analysis.estimated_human_percentage = extracted.get("estimated_human_percentage", 95)
        analysis.similar_datasets = extracted.get("similar_datasets", [])

        # Handle difficulty validation config
        diff_val = extracted.get("difficulty_validation")
        if diff_val and isinstance(diff_val, dict):
            # Check if validation is enabled (explicit false means disabled)
            if diff_val.get("enabled", True) and diff_val.get("model"):
                analysis.difficulty_validation = {
                    "model": diff_val.get("model", ""),
                    "settings": diff_val.get("settings", ""),
                    "test_count": diff_val.get("test_count", 3),
                    "max_correct": diff_val.get("max_correct", 1),
                    "pass_criteria": diff_val.get("pass_criteria", ""),
                    "requires_record": diff_val.get("requires_record", True),
                }

        return analysis

    def analyze(self, file_path: str) -> SpecificationAnalysis:
        """Analyze a specification document using LLM.

        Args:
            file_path: Path to the specification file

        Returns:
            SpecificationAnalysis with extracted information
        """
        # Parse document
        doc = self.parse_document(file_path)

        # Use LLM to extract structured information
        extracted = self._extract_with_llm(doc)

        # Create analysis from extracted data (or empty if LLM failed)
        if extracted:
            return self.create_analysis_from_json(extracted, doc)
        else:
            # Return minimal analysis with just document info
            return SpecificationAnalysis(
                raw_text=doc.text_content,
                has_images=doc.has_images(),
                image_count=len(doc.images),
            )

    def _extract_with_llm(self, doc: ParsedDocument) -> Optional[dict]:
        """Extract structured information using LLM."""
        prompt = self.EXTRACTION_PROMPT.format(
            document_content=doc.text_content[:15000]  # Limit content length
        )

        if self.provider == "anthropic":
            return self._call_anthropic(prompt, doc.images)
        elif self.provider == "openai":
            return self._call_openai(prompt, doc.images)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_anthropic(self, prompt: str, images: list[dict]) -> Optional[dict]:
        """Call Anthropic Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # Build message content
            content = []

            # Add images if available (for vision)
            for img in images[:5]:  # Limit to 5 images
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img["type"],
                            "data": img["data"],
                        },
                    }
                )

            # Add text prompt
            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            response = client.messages.create(
                model=DEFAULT_ANTHROPIC_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )

            # Extract JSON from response
            response_text = response.content[0].text
            return self._parse_json_response(response_text)

        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        except anthropic.AuthenticationError:
            logger.error(
                "LLM call failed: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=your_key"
            )
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _call_openai(self, prompt: str, images: list[dict]) -> Optional[dict]:
        """Call OpenAI API."""
        try:
            import openai

            client = openai.OpenAI()

            # Build message content
            content = []

            # Add images if available
            for img in images[:5]:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['type']};base64,{img['data']}"},
                    }
                )

            # Add text prompt
            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            response = client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )

            response_text = response.choices[0].message.content
            return self._parse_json_response(response_text)

        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        # Try to find JSON block
        import re

        # Look for ```json ... ``` block
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
