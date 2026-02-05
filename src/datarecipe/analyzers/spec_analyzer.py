"""Specification document analyzer using LLM."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datarecipe.parsers import DocumentParser, ParsedDocument


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
    cognitive_requirements: List[str] = field(default_factory=list)  # 认知要求
    reasoning_chain: List[str] = field(default_factory=list)  # 推理链

    # Data requirements
    data_requirements: List[str] = field(default_factory=list)  # 数据要求
    quality_constraints: List[str] = field(default_factory=list)  # 质量约束
    forbidden_items: List[str] = field(default_factory=list)  # 禁止项
    difficulty_criteria: str = ""  # 难度标准

    # Data structure
    fields: List[Dict[str, str]] = field(default_factory=list)  # 字段定义
    field_requirements: Dict[str, str] = field(default_factory=dict)  # 字段要求

    # Examples
    examples: List[Dict[str, Any]] = field(default_factory=list)  # 示例

    # Scoring
    scoring_rubric: List[Dict[str, str]] = field(default_factory=list)  # 打分标准

    # Estimates
    estimated_difficulty: str = ""  # easy/medium/hard/expert
    estimated_domain: str = ""  # 领域
    estimated_human_percentage: float = 95.0  # 人工比例估计
    similar_datasets: List[str] = field(default_factory=list)

    # Raw
    raw_text: str = ""
    has_images: bool = False
    image_count: int = 0

    def to_dict(self) -> dict:
        return {
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
            "fields": self.fields,
            "field_requirements": self.field_requirements,
            "examples": self.examples,
            "scoring_rubric": self.scoring_rubric,
            "estimated_difficulty": self.estimated_difficulty,
            "estimated_domain": self.estimated_domain,
            "estimated_human_percentage": self.estimated_human_percentage,
            "similar_datasets": self.similar_datasets,
            "has_images": self.has_images,
            "image_count": self.image_count,
        }


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
  "difficulty_criteria": "难度验证标准描述",

  "fields": [
    {{"name": "字段名", "type": "类型", "required": true, "description": "说明"}}
  ],
  "field_requirements": {{
    "字段名": "具体要求"
  }},

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

        return self.EXTRACTION_PROMPT.format(
            document_content=doc.text_content[:15000]
        )

    def create_analysis_from_json(
        self,
        extracted: Dict,
        doc: Optional[ParsedDocument] = None
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
        analysis.examples = extracted.get("examples", [])
        analysis.scoring_rubric = extracted.get("scoring_rubric", [])
        analysis.estimated_difficulty = extracted.get("estimated_difficulty", "hard")
        analysis.estimated_domain = extracted.get("estimated_domain", "")
        analysis.estimated_human_percentage = extracted.get("estimated_human_percentage", 95)
        analysis.similar_datasets = extracted.get("similar_datasets", [])

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

    def _extract_with_llm(self, doc: ParsedDocument) -> Optional[Dict]:
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

    def _call_anthropic(self, prompt: str, images: List[dict]) -> Optional[Dict]:
        """Call Anthropic Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # Build message content
            content = []

            # Add images if available (for vision)
            for img in images[:5]:  # Limit to 5 images
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["type"],
                        "data": img["data"],
                    }
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt,
            })

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )

            # Extract JSON from response
            response_text = response.content[0].text
            return self._parse_json_response(response_text)

        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        except anthropic.AuthenticationError:
            print("LLM call failed: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=your_key")
            return None
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None

    def _call_openai(self, prompt: str, images: List[dict]) -> Optional[Dict]:
        """Call OpenAI API."""
        try:
            import openai

            client = openai.OpenAI()

            # Build message content
            content = []

            # Add images if available
            for img in images[:5]:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['type']};base64,{img['data']}"
                    }
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt,
            })

            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )

            response_text = response.choices[0].message.content
            return self._parse_json_response(response_text)

        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None

    def _parse_json_response(self, text: str) -> Optional[Dict]:
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
