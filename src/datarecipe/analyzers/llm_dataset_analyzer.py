"""LLM-based analyzer for unknown dataset types."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMDatasetAnalysis:
    """Result of LLM analysis on dataset samples."""

    dataset_type: str = ""  # e.g., "instruction_tuning", "qa", "classification", etc.
    purpose: str = ""  # What is this dataset for?
    structure_description: str = ""  # Description of data structure
    key_fields: list = field(default_factory=list)  # Important fields and their roles
    production_steps: list = field(default_factory=list)  # How to produce similar data
    quality_criteria: list = field(default_factory=list)  # Quality standards
    annotation_guidelines: str = ""  # How to annotate/label data
    example_analysis: list = field(default_factory=list)  # Analysis of example items
    recommended_team: dict = field(default_factory=dict)  # Team composition
    estimated_difficulty: str = ""  # easy/medium/hard
    similar_datasets: list = field(default_factory=list)  # Known similar datasets
    raw_response: str = ""  # Raw LLM response for debugging


DATASET_ANALYSIS_PROMPT = '''你是一个数据集工程专家。请分析以下数据集样本，帮助用户理解如何复刻类似的数据集。

## 数据集信息
- 名称: {dataset_id}
- 样本数: {sample_count}

## 数据结构 (Schema)
```json
{schema}
```

## 样本数据 (前 {num_examples} 条)
```json
{examples}
```

## 请分析并输出以下内容 (JSON 格式)

```json
{{
  "dataset_type": "数据集类型，如: instruction_tuning, preference_ranking, qa, classification, code_generation, summarization, translation, dialogue, benchmark 等",

  "purpose": "这个数据集的用途是什么？用一句话描述",

  "structure_description": "数据结构的详细描述，包括各字段的含义和关系",

  "key_fields": [
    {{"field": "字段名", "role": "字段作用", "format": "数据格式", "example_pattern": "典型内容模式"}}
  ],

  "production_steps": [
    {{"step": 1, "name": "步骤名称", "description": "详细说明", "who": "谁来做 (人工/机器/混合)", "tools": ["可能用到的工具"]}}
  ],

  "quality_criteria": [
    {{"criterion": "质量标准名称", "description": "具体要求", "check_method": "如何检查"}}
  ],

  "annotation_guidelines": "如果需要人工标注，给出详细的标注指南",

  "example_analysis": [
    {{"example_index": 0, "analysis": "对这条样本的分析，包括好的地方和可改进的地方"}}
  ],

  "recommended_team": {{
    "roles": [{{"role": "角色名称", "count": "人数", "skills": ["所需技能"]}}],
    "total_people": "总人数范围"
  }},

  "estimated_difficulty": "easy/medium/hard，并说明原因",

  "similar_datasets": ["类似的知名数据集名称"]
}}
```

请确保输出有效的 JSON 格式。基于样本数据给出具体、可操作的建议。'''


class LLMDatasetAnalyzer:
    """Analyze unknown dataset types using LLM."""

    def __init__(self, provider: str = "anthropic"):
        """Initialize analyzer.

        Args:
            provider: LLM provider ("anthropic" or "openai")
        """
        self.provider = provider
        self._client = None

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            try:
                import anthropic

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Please install: pip install anthropic")
        elif self.provider == "openai":
            try:
                import openai

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Please install: pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def analyze(
        self,
        dataset_id: str,
        schema_info: dict,
        sample_items: list,
        sample_count: int,
    ) -> LLMDatasetAnalysis:
        """Analyze dataset samples using LLM.

        Args:
            dataset_id: Dataset identifier
            schema_info: Schema information dict
            sample_items: List of sample data items
            sample_count: Total number of samples analyzed

        Returns:
            LLMDatasetAnalysis with extracted insights
        """
        # Prepare schema for prompt
        schema_str = json.dumps(
            {k: {"type": v["type"], "nested": v.get("nested_type")} for k, v in schema_info.items()},
            indent=2,
            ensure_ascii=False,
        )

        # Prepare examples (truncate long values)
        def truncate_item(item: dict, max_len: int = 500) -> dict:
            result = {}
            for k, v in item.items():
                if isinstance(v, str) and len(v) > max_len:
                    result[k] = v[:max_len] + "..."
                elif isinstance(v, list) and len(v) > 5:
                    result[k] = v[:5] + ["..."]
                else:
                    result[k] = v
            return result

        examples = [truncate_item(item) for item in sample_items[:5]]
        examples_str = json.dumps(examples, indent=2, ensure_ascii=False)

        # Build prompt
        prompt = DATASET_ANALYSIS_PROMPT.format(
            dataset_id=dataset_id,
            sample_count=sample_count,
            schema=schema_str,
            examples=examples_str,
            num_examples=len(examples),
        )

        # Call LLM
        try:
            client = self._get_client()

            if self.provider == "anthropic":
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content

            # Parse JSON from response
            result = self._parse_response(response_text)
            result.raw_response = response_text
            return result

        except Exception as e:
            # Return empty result with error info
            return LLMDatasetAnalysis(
                dataset_type="unknown",
                purpose=f"LLM analysis failed: {e}",
                raw_response=str(e),
            )

    def _parse_response(self, response_text: str) -> LLMDatasetAnalysis:
        """Parse LLM response into structured result."""
        import re

        # Try to extract JSON block
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try parsing entire response
            json_str = response_text

        try:
            data = json.loads(json_str)
            return LLMDatasetAnalysis(
                dataset_type=data.get("dataset_type", "unknown"),
                purpose=data.get("purpose", ""),
                structure_description=data.get("structure_description", ""),
                key_fields=data.get("key_fields", []),
                production_steps=data.get("production_steps", []),
                quality_criteria=data.get("quality_criteria", []),
                annotation_guidelines=data.get("annotation_guidelines", ""),
                example_analysis=data.get("example_analysis", []),
                recommended_team=data.get("recommended_team", {}),
                estimated_difficulty=data.get("estimated_difficulty", ""),
                similar_datasets=data.get("similar_datasets", []),
            )
        except json.JSONDecodeError:
            return LLMDatasetAnalysis(
                dataset_type="unknown",
                purpose="Failed to parse LLM response",
                raw_response=response_text,
            )


def generate_llm_guide_section(analysis: LLMDatasetAnalysis) -> str:
    """Generate reproduction guide section from LLM analysis.

    Args:
        analysis: LLM analysis result

    Returns:
        Markdown string for the guide
    """
    lines = []

    lines.append("## 🤖 LLM 智能分析结果")
    lines.append("")
    lines.append(f"> 数据集类型: **{analysis.dataset_type}**")
    lines.append(f">")
    lines.append(f"> {analysis.purpose}")
    lines.append("")

    # Structure description
    if analysis.structure_description:
        lines.append("### 数据结构解读")
        lines.append("")
        lines.append(analysis.structure_description)
        lines.append("")

    # Key fields
    if analysis.key_fields:
        lines.append("### 关键字段说明")
        lines.append("")
        lines.append("| 字段 | 作用 | 格式 | 内容模式 |")
        lines.append("|------|------|------|----------|")
        for f in analysis.key_fields:
            if isinstance(f, dict):
                lines.append(
                    f"| `{f.get('field', '')}` | {f.get('role', '')} | {f.get('format', '')} | {f.get('example_pattern', '')} |"
                )
        lines.append("")

    # Production steps
    if analysis.production_steps:
        lines.append("### 生产流程 (SOP)")
        lines.append("")
        lines.append("```")
        for step in analysis.production_steps:
            if isinstance(step, dict):
                step_num = step.get("step", "?")
                name = step.get("name", "")
                desc = step.get("description", "")
                who = step.get("who", "")
                tools = step.get("tools", [])
                lines.append(f"Step {step_num}: {name}")
                lines.append(f"  描述: {desc}")
                lines.append(f"  执行者: {who}")
                if tools:
                    lines.append(f"  工具: {', '.join(tools)}")
                lines.append("")
        lines.append("```")
        lines.append("")

    # Quality criteria
    if analysis.quality_criteria:
        lines.append("### 质量标准")
        lines.append("")
        lines.append("| 标准 | 要求 | 检查方法 |")
        lines.append("|------|------|----------|")
        for c in analysis.quality_criteria:
            if isinstance(c, dict):
                lines.append(
                    f"| {c.get('criterion', '')} | {c.get('description', '')} | {c.get('check_method', '')} |"
                )
        lines.append("")

    # Annotation guidelines
    if analysis.annotation_guidelines:
        lines.append("### 标注指南")
        lines.append("")
        lines.append(analysis.annotation_guidelines)
        lines.append("")

    # Team recommendation
    if analysis.recommended_team and analysis.recommended_team.get("roles"):
        lines.append("### 团队配置建议")
        lines.append("")
        lines.append("| 角色 | 人数 | 所需技能 |")
        lines.append("|------|------|----------|")
        for role in analysis.recommended_team.get("roles", []):
            if isinstance(role, dict):
                skills = ", ".join(role.get("skills", []))
                lines.append(f"| {role.get('role', '')} | {role.get('count', '')} | {skills} |")
        total = analysis.recommended_team.get("total_people", "")
        if total:
            lines.append(f"\n**总人数**: {total}")
        lines.append("")

    # Difficulty
    if analysis.estimated_difficulty:
        lines.append("### 复刻难度评估")
        lines.append("")
        lines.append(f"**难度**: {analysis.estimated_difficulty}")
        lines.append("")

    # Similar datasets
    if analysis.similar_datasets:
        lines.append("### 相似数据集参考")
        lines.append("")
        for ds in analysis.similar_datasets:
            lines.append(f"- {ds}")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)
