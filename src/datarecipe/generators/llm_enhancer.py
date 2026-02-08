"""LLM Enhancement Layer for generating dataset-specific insights.

Sits between the analysis phase and the generation phase.
Supports three modes:
  1. interactive: Outputs prompt to stdout, reads JSON from stdin (Claude Code/App)
  2. from-json:   Loads from a pre-computed JSON file
  3. api:         Calls Anthropic/OpenAI API directly (standalone use)
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from datarecipe.constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL


@dataclass
class EnhancedContext:
    """LLM-generated enhancements that augment template outputs."""

    # Dataset-specific insights (REPRODUCTION_GUIDE, ANALYSIS_REPORT)
    dataset_purpose_summary: str = ""
    key_methodology_insights: list[str] = field(default_factory=list)
    reproduction_strategy: str = ""
    domain_specific_tips: list[str] = field(default_factory=list)

    # EXECUTIVE_SUMMARY
    tailored_use_cases: list[str] = field(default_factory=list)
    tailored_roi_scenarios: list[str] = field(default_factory=list)
    tailored_risks: list[dict[str, str]] = field(default_factory=list)
    competitive_positioning: str = ""

    # ANNOTATION_SPEC / TRAINING_GUIDE
    domain_specific_guidelines: str = ""
    quality_pitfalls: list[str] = field(default_factory=list)
    example_analysis: list[dict[str, str]] = field(default_factory=list)

    # MILESTONE_PLAN
    phase_specific_risks: list[dict[str, str]] = field(default_factory=list)
    team_recommendations: str = ""

    # Sample generation seeds
    realistic_sample_seeds: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    llm_provider: str = ""
    generated: bool = False
    raw_response: str = ""


ENHANCEMENT_PROMPT = """你是一个数据集工程和项目管理专家。请基于以下数据集分析结果，生成针对性的增强内容。

## 数据集信息
- 名称: {dataset_id}
- 类型: {dataset_type}
- 领域: {domain}
- 样本数: {sample_count}
- 复杂度: {difficulty}
- 人工比例: {human_percentage}%
- 总成本预估: ${total_cost}

## 数据结构 (Schema)
```json
{schema_summary}
```

## 样本数据摘要 (前3条)
```json
{sample_summaries}
```

## 已有分析摘要
{existing_analysis_summary}

## 请生成以下内容 (JSON格式)

```json
{{
  "dataset_purpose_summary": "这个数据集的具体用途和价值（基于实际数据内容，2-3句话，不要泛泛而谈）",

  "key_methodology_insights": [
    "这个数据集的构建方法学洞察1（基于数据结构和内容推断）",
    "洞察2",
    "洞察3"
  ],

  "reproduction_strategy": "复刻这个数据集的具体策略（200-300字，包含数据来源、标注流程、质控方法等具体步骤建议）",

  "domain_specific_tips": [
    "针对{domain}领域的具体建议1",
    "建议2",
    "建议3"
  ],

  "tailored_use_cases": [
    "基于数据内容的具体用途1（写具体场景，不要写'场景A'之类的占位符）",
    "具体用途2",
    "具体用途3"
  ],

  "tailored_roi_scenarios": [
    "基于${total_cost}成本的具体回报场景1（要有数字和时间线）",
    "具体回报场景2",
    "具体回报场景3"
  ],

  "tailored_risks": [
    {{"level": "高/中/低", "description": "针对此数据集的具体风险", "mitigation": "具体缓解措施"}}
  ],

  "competitive_positioning": "相对于类似数据集的竞争定位分析（100-150字）",

  "domain_specific_guidelines": "针对{domain}领域的标注指导（100-200字，包含标注时需要注意的领域知识和常见陷阱）",

  "quality_pitfalls": [
    "标注时容易犯的领域特定错误1",
    "错误2",
    "错误3"
  ],

  "example_analysis": [
    {{"sample_index": 0, "strengths": "这条数据的优点", "weaknesses": "可改进的地方", "annotation_tips": "标注此类数据的建议"}}
  ],

  "phase_specific_risks": [
    {{"phase": "试点阶段", "risk": "具体风险", "mitigation": "缓解措施"}},
    {{"phase": "主体生产", "risk": "具体风险", "mitigation": "缓解措施"}},
    {{"phase": "质检阶段", "risk": "具体风险", "mitigation": "缓解措施"}}
  ],

  "team_recommendations": "团队配置建议（50-100字，考虑领域专业知识需求）",

  "realistic_sample_seeds": [
    {{"instruction": "一个符合此数据集风格的真实指令/问题", "expected_output_description": "期望输出的描述", "difficulty": "easy/medium/hard"}}
  ]
}}
```

请确保输出有效的 JSON 格式。所有内容必须具体、可操作，避免空泛的模板化表述。"""


class LLMEnhancer:
    """Generate dataset-specific enhancements using LLM.

    Supports three modes:
      - interactive: Output prompt to stdout, read JSON from stdin
                     (for use inside Claude Code / Claude App)
      - api:         Call Anthropic/OpenAI API directly
                     (for standalone use with API keys)
      - from-json:   Load from pre-computed JSON file
    """

    def __init__(self, mode: str = "auto", provider: str = "anthropic"):
        """Initialize enhancer.

        Args:
            mode: "interactive", "api", "from-json", or "auto"
                  "auto" will try interactive first (if stdin is a pipe),
                  then api (if API key is set), then skip.
            provider: LLM provider for API mode ("anthropic" or "openai")
        """
        self.mode = mode
        self.provider = provider
        self._client = None

    def _build_prompt(
        self,
        dataset_id: str,
        dataset_type: str = "unknown",
        schema_info: Optional[dict] = None,
        sample_items: Optional[list[dict]] = None,
        sample_count: int = 0,
        domain: str = "通用",
        difficulty: str = "medium",
        human_percentage: float = 0,
        total_cost: float = 0,
        complexity_metrics: Optional[Any] = None,
        allocation: Optional[Any] = None,
        rubrics_result: Optional[Any] = None,
        llm_analysis: Optional[Any] = None,
    ) -> str:
        """Build the enhancement prompt from analysis data."""
        # Build schema summary
        schema_summary = "{}"
        if schema_info:
            schema_brief = {}
            for k, v in schema_info.items():
                schema_brief[k] = {
                    "type": v.get("type", "unknown"),
                    "nested": v.get("nested_type"),
                }
            schema_summary = json.dumps(schema_brief, indent=2, ensure_ascii=False)

        # Build sample summaries (truncate long values)
        sample_summaries = "[]"
        if sample_items:
            truncated = []
            for item in sample_items[:3]:
                t = {}
                for k, v in item.items():
                    if isinstance(v, str) and len(v) > 300:
                        t[k] = v[:300] + "..."
                    elif isinstance(v, list) and len(v) > 3:
                        t[k] = v[:3] + ["..."]
                    else:
                        t[k] = v
                truncated.append(t)
            sample_summaries = json.dumps(truncated, indent=2, ensure_ascii=False)

        # Extract metrics from analysis objects
        if complexity_metrics:
            if hasattr(complexity_metrics, "primary_domain"):
                domain = complexity_metrics.primary_domain.value
            if hasattr(complexity_metrics, "difficulty_score"):
                difficulty = f"{complexity_metrics.difficulty_score}/10"
        if allocation:
            human_percentage = allocation.human_work_percentage
            total_cost = allocation.total_cost

        # Build existing analysis summary
        analysis_parts = []
        if rubrics_result:
            analysis_parts.append(
                f"- 评分标准: {rubrics_result.total_rubrics} 条, "
                f"{rubrics_result.unique_patterns} 种独特模式"
            )
        if llm_analysis and hasattr(llm_analysis, "purpose") and llm_analysis.purpose:
            analysis_parts.append(f"- LLM 分析: {llm_analysis.purpose}")
        existing_analysis = "\n".join(analysis_parts) if analysis_parts else "无额外分析"

        return ENHANCEMENT_PROMPT.format(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            domain=domain,
            sample_count=sample_count,
            difficulty=difficulty,
            human_percentage=round(human_percentage, 1),
            total_cost=round(total_cost, 2),
            schema_summary=schema_summary,
            sample_summaries=sample_summaries,
            existing_analysis_summary=existing_analysis,
        )

    def get_prompt(self, **kwargs) -> str:
        """Get the enhancement prompt (for external LLM processing).

        This is useful when running inside Claude Code or other LLM environments
        where the caller can process the prompt directly.

        Returns:
            The formatted prompt string
        """
        return self._build_prompt(**kwargs)

    def enhance(self, **kwargs) -> EnhancedContext:
        """Generate enhanced context.

        Automatically selects the best available mode:
          1. If mode="interactive": stdout/stdin exchange
          2. If mode="api": direct API call
          3. If mode="auto": try API first (if key available), else interactive
        """
        mode = self._resolve_mode()

        if mode == "interactive":
            return self._enhance_interactive(**kwargs)
        elif mode == "api":
            return self._enhance_api(**kwargs)
        else:
            return EnhancedContext(generated=False, raw_response="No LLM mode available")

    def enhance_from_json(self, json_path: str) -> EnhancedContext:
        """Load enhanced context from a pre-computed JSON file.

        Args:
            json_path: Path to enhanced_context.json

        Returns:
            EnhancedContext loaded from file
        """
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            return self._dict_to_context(data)
        except Exception as e:
            return EnhancedContext(
                generated=False,
                raw_response=f"Failed to load from {json_path}: {e}",
            )

    def enhance_from_response(self, response_text: str) -> EnhancedContext:
        """Parse an LLM response text into EnhancedContext.

        Use this when the caller has already obtained the LLM response
        (e.g., Claude Code processed the prompt from get_prompt()).

        Args:
            response_text: Raw LLM response containing JSON

        Returns:
            EnhancedContext parsed from the response
        """
        return self._parse_response(response_text)

    def _resolve_mode(self) -> str:
        """Determine the actual mode to use."""
        if self.mode != "auto":
            return self.mode

        # Auto mode: try API first (if key set), then interactive
        if self.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            return "api"
        if self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            return "api"

        # Fall back to interactive if stdin is a pipe (not a terminal)
        if not sys.stdin.isatty():
            return "interactive"

        return "none"

    def _enhance_interactive(self, **kwargs) -> EnhancedContext:
        """Interactive mode: output prompt to stdout, read JSON from stdin."""
        prompt = self._build_prompt(**kwargs)
        stderr = sys.stderr

        # Output prompt to stdout (for the LLM environment to process)
        stderr.write("\n" + "=" * 60 + "\n")
        stderr.write("DataRecipe LLM Enhancement - 请将以下 prompt 交给 LLM 分析\n")
        stderr.write("=" * 60 + "\n\n")

        print(prompt)

        stderr.write("\n" + "=" * 60 + "\n")
        stderr.write("请输入 LLM 返回的 JSON (以空行结束):\n")
        stderr.write("=" * 60 + "\n\n")

        # Read response from stdin
        json_lines = []
        try:
            for line in sys.stdin:
                if line.strip() == "":
                    break
                json_lines.append(line)
        except EOFError:
            pass

        json_text = "".join(json_lines)
        if not json_text.strip():
            return EnhancedContext(
                generated=False,
                raw_response="No JSON received from stdin",
            )

        return self._parse_response(json_text)

    def _enhance_api(self, **kwargs) -> EnhancedContext:
        """API mode: call Anthropic/OpenAI API directly."""
        prompt = self._build_prompt(**kwargs)

        try:
            client = self._get_client()

            if self.provider == "anthropic":
                response = client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content

            return self._parse_response(response_text)

        except Exception as e:
            return EnhancedContext(
                generated=False,
                raw_response=f"API call failed: {e}",
            )

    def _get_client(self):
        """Get or create LLM client for API mode."""
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

    def _parse_response(self, response_text: str) -> EnhancedContext:
        """Parse LLM response text into EnhancedContext."""
        # Extract JSON block from markdown code fence
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_obj_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_obj_match:
                json_str = json_obj_match.group(0)
            else:
                json_str = response_text

        try:
            data = json.loads(json_str)
            return self._dict_to_context(data)
        except json.JSONDecodeError:
            return EnhancedContext(
                generated=False,
                raw_response=response_text,
            )

    def _dict_to_context(self, data: dict) -> EnhancedContext:
        """Convert a dict to EnhancedContext."""
        return EnhancedContext(
            # Dataset insights
            dataset_purpose_summary=data.get("dataset_purpose_summary", ""),
            key_methodology_insights=data.get("key_methodology_insights", []),
            reproduction_strategy=data.get("reproduction_strategy", ""),
            domain_specific_tips=data.get("domain_specific_tips", []),
            # Executive summary
            tailored_use_cases=data.get("tailored_use_cases", []),
            tailored_roi_scenarios=data.get("tailored_roi_scenarios", []),
            tailored_risks=data.get("tailored_risks", []),
            competitive_positioning=data.get("competitive_positioning", ""),
            # Annotation
            domain_specific_guidelines=data.get("domain_specific_guidelines", ""),
            quality_pitfalls=data.get("quality_pitfalls", []),
            example_analysis=data.get("example_analysis", []),
            # Milestone
            phase_specific_risks=data.get("phase_specific_risks", []),
            team_recommendations=data.get("team_recommendations", ""),
            # Samples
            realistic_sample_seeds=data.get("realistic_sample_seeds", []),
            # Metadata
            llm_provider=data.get("llm_provider", self.provider),
            generated=data.get("generated", True),
            raw_response=data.get("raw_response", ""),
        )
