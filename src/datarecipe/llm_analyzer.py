"""LLM-enhanced analyzer for extracting detailed dataset information.

.. deprecated::
    This module will be removed in v0.4.0.
"""

import logging
import warnings

warnings.warn(
    "datarecipe.llm_analyzer is deprecated. This module will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2,
)

import os
import re
import tempfile
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

from datarecipe.constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL
from datarecipe.deep_analyzer import (
    DatasetCategory,
    DeepAnalysisResult,
    DeepAnalyzer,
)

# LLM analysis prompt template
ANALYSIS_PROMPT = """你是一个专业的数据集分析专家。请分析以下数据集的相关内容，提取关键信息。

数据集名称: {name}
来源URL: {url}

内容:
{content}

请以JSON格式输出以下信息（如果无法确定某项，设为null）：

```json
{{
  "category": "数据集类别: llm_distillation/human_annotation/programmatic/simulation/benchmark/hybrid",
  "methodology": "数据生成方法论的简要描述（中文，50字以内）",
  "domain": "领域: 电信/医疗/金融/教育/通用等",
  "key_innovations": ["创新点1", "创新点2"],
  "generation_steps": [
    {{"step": 1, "name": "步骤名称", "description": "步骤描述"}},
    {{"step": 2, "name": "步骤名称", "description": "步骤描述"}}
  ],
  "quality_methods": ["质量控制方法1", "质量控制方法2"],
  "data_format": "数据格式描述",
  "size_info": {{"total": 数量, "train": 训练集数量, "test": 测试集数量}},
  "evaluation_metrics": ["评估指标1", "评估指标2"],
  "modeling_approach": "建模方法（如Dec-POMDP、MDP等，如果有的话）",
  "limitations": ["局限性1", "局限性2"],
  "use_cases": ["使用场景1", "使用场景2"]
}}
```

请确保输出有效的JSON格式。只输出JSON，不要有其他内容。"""


@dataclass
class MultiSourceContent:
    """Content aggregated from multiple sources."""

    website_content: str = ""
    paper_content: str = ""
    github_content: str = ""
    paper_url: str | None = None
    github_url: str | None = None


class LLMAnalyzer(DeepAnalyzer):
    """Enhanced analyzer using LLM for better extraction."""

    def __init__(
        self,
        auto_search_paper: bool = True,
        use_llm: bool = True,
        llm_provider: str = "anthropic",  # "anthropic" or "openai"
        parse_pdf: bool = True,
    ):
        """Initialize the LLM analyzer.

        Args:
            auto_search_paper: If True, automatically search for related papers
            use_llm: If True, use LLM for analysis
            llm_provider: LLM provider to use ("anthropic" or "openai")
            parse_pdf: If True, parse PDF content directly
        """
        super().__init__(auto_search_paper=auto_search_paper)
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.parse_pdf = parse_pdf
        self._llm_client = None

    def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is not None:
            return self._llm_client

        if self.llm_provider == "anthropic":
            try:
                import anthropic

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                self._llm_client = anthropic.Anthropic(api_key=api_key)
                return self._llm_client
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        elif self.llm_provider == "openai":
            try:
                import openai

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self._llm_client = openai.OpenAI(api_key=api_key)
                return self._llm_client
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def _fetch_pdf_content(self, url: str) -> str | None:
        """Fetch and extract text from PDF.

        Args:
            url: URL of the PDF file

        Returns:
            Extracted text content or None if failed
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        try:
            # Convert arXiv abstract URL to PDF URL
            if "arxiv.org/abs/" in url:
                url = url.replace("/abs/", "/pdf/") + ".pdf"

            # Download PDF to temp file
            headers = {"User-Agent": "Mozilla/5.0 (compatible; DataRecipe/1.0)"}
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                return None

            # Save to temp file and extract text
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            try:
                doc = fitz.open(temp_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                return "\n".join(text_parts)
            finally:
                os.unlink(temp_path)

        except Exception:
            return None

    def _fetch_github_readme(self, repo_url: str) -> str | None:
        """Fetch README content from GitHub repository.

        Args:
            repo_url: GitHub repository URL

        Returns:
            README content or None if failed
        """
        try:
            # Extract owner/repo from URL
            match = re.search(r"github\.com/([^/]+/[^/]+)", repo_url)
            if not match:
                return None

            repo_path = match.group(1).rstrip("/")

            # Try GitHub API for README
            api_url = f"https://api.github.com/repos/{repo_path}/readme"
            headers = {
                "Accept": "application/vnd.github.v3.raw",
                "User-Agent": "DataRecipe/1.0",
            }

            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.text

            return None

        except Exception:
            return None

    def _aggregate_sources(self, url: str, name: str) -> MultiSourceContent:
        """Aggregate content from multiple sources.

        Args:
            url: Primary URL
            name: Dataset name for searching

        Returns:
            MultiSourceContent with all gathered content
        """
        result = MultiSourceContent()

        # 1. Fetch website content
        website_content = self._fetch_content(url)
        if website_content:
            result.website_content = website_content

        # 2. Search and fetch paper
        if self.auto_search_paper:
            paper_url = self.search_related_paper(name)
            if paper_url:
                result.paper_url = paper_url

                # Try to fetch PDF content
                if self.parse_pdf:
                    pdf_content = self._fetch_pdf_content(paper_url)
                    if pdf_content:
                        result.paper_content = pdf_content
                    else:
                        # Fallback to abstract page
                        result.paper_content = self._fetch_content(paper_url) or ""
                else:
                    result.paper_content = self._fetch_content(paper_url) or ""

        # 3. Extract and fetch GitHub repo
        all_content = result.website_content + result.paper_content
        github_match = re.search(r"github\.com/([^/\s\"'<>]+/[^/\s\"'<>]+)", all_content)
        if github_match:
            result.github_url = f"https://github.com/{github_match.group(1)}"
            result.github_content = self._fetch_github_readme(result.github_url) or ""

        return result

    def _analyze_with_llm(
        self,
        name: str,
        url: str,
        content: str,
    ) -> dict:
        """Use LLM to analyze content and extract information.

        Args:
            name: Dataset name
            url: Source URL
            content: Content to analyze

        Returns:
            Dict with extracted information
        """
        # Truncate content if too long
        max_content_length = 15000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[内容已截断...]"

        prompt = ANALYSIS_PROMPT.format(name=name, url=url, content=content)

        try:
            client = self._get_llm_client()

            if self.llm_provider == "anthropic":
                response = client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            else:  # openai
                response = client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content

            # Extract JSON from response
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                import json

                return json.loads(json_match.group(1))

            # Try parsing entire response as JSON
            import json

            return json.loads(response_text)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {}

    def analyze(self, url: str, search_paper_if_needed: bool = None) -> DeepAnalysisResult:
        """Perform enhanced analysis using multiple sources and LLM.

        Args:
            url: URL of the paper or dataset page
            search_paper_if_needed: If True, search for related papers

        Returns:
            DeepAnalysisResult with extracted information
        """
        if search_paper_if_needed is None:
            search_paper_if_needed = self.auto_search_paper

        # First, do basic analysis
        content = self._fetch_content(url)
        if not content:
            raise ValueError(f"Could not fetch content from: {url}")

        name = self._extract_name(content, url)
        description = self._extract_description(content)

        # Aggregate from multiple sources
        sources = self._aggregate_sources(url, name)

        # Combine all content
        combined_content = "\n\n---\n\n".join(
            filter(
                None,
                [
                    f"=== 网站内容 ===\n{sources.website_content[:5000]}"
                    if sources.website_content
                    else "",
                    f"=== 论文内容 ===\n{sources.paper_content[:10000]}"
                    if sources.paper_content
                    else "",
                    f"=== GitHub README ===\n{sources.github_content[:3000]}"
                    if sources.github_content
                    else "",
                ],
            )
        )

        # Use LLM if enabled and content is available
        llm_result = {}
        if self.use_llm and combined_content:
            try:
                llm_result = self._analyze_with_llm(name, url, combined_content)
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")

        # Create result with basic extraction using combined content (including PDF)
        category = self._detect_category(combined_content)
        if llm_result.get("category"):
            category_map = {
                "llm_distillation": DatasetCategory.LLM_DISTILLATION,
                "human_annotation": DatasetCategory.HUMAN_ANNOTATION,
                "programmatic": DatasetCategory.PROGRAMMATIC,
                "simulation": DatasetCategory.SIMULATION,
                "benchmark": DatasetCategory.BENCHMARK,
                "hybrid": DatasetCategory.HYBRID,
            }
            category = category_map.get(llm_result["category"], category)

        result = DeepAnalysisResult(
            name=name,
            category=category,
            description=description,
        )

        # Fill in LLM results
        if llm_result:
            result.methodology = llm_result.get("methodology") or self._extract_methodology(
                combined_content
            )
            result.domain = llm_result.get("domain") or self._extract_domain(combined_content)
            result.key_innovations = llm_result.get("key_innovations") or self._extract_innovations(
                combined_content
            )
            result.generation_steps = llm_result.get(
                "generation_steps"
            ) or self._extract_generation_steps(combined_content, category)
            result.quality_methods = llm_result.get(
                "quality_methods"
            ) or self._extract_quality_methods(combined_content)
            result.data_format = llm_result.get("data_format")
            result.evaluation_metrics = llm_result.get(
                "evaluation_metrics"
            ) or self._extract_metrics(combined_content)
            result.modeling_approach = llm_result.get(
                "modeling_approach"
            ) or self._extract_modeling_approach(combined_content)
            result.limitations = llm_result.get("limitations") or self._extract_limitations(
                combined_content
            )
            result.use_cases = llm_result.get("use_cases") or self._extract_use_cases(
                combined_content
            )

            if llm_result.get("size_info"):
                result.size_info = llm_result["size_info"]
            else:
                result.size_info = self._extract_size_info(combined_content)
        else:
            # Fallback to pattern-based extraction
            result.methodology = self._extract_methodology(combined_content)
            result.domain = self._extract_domain(combined_content)
            result.key_innovations = self._extract_innovations(combined_content)
            result.generation_steps = self._extract_generation_steps(combined_content, category)
            result.quality_methods = self._extract_quality_methods(combined_content)
            result.evaluation_metrics = self._extract_metrics(combined_content)
            result.modeling_approach = self._extract_modeling_approach(combined_content)
            result.limitations = self._extract_limitations(combined_content)
            result.use_cases = self._extract_use_cases(combined_content)
            result.size_info = self._extract_size_info(combined_content)

        # Extract code/data info from combined content
        result.code_available, result.code_url = self._extract_code_info(combined_content)
        result.data_available, result.data_url = self._extract_data_info(combined_content)

        # Add discovered URLs
        if sources.paper_url:
            result.paper_url = sources.paper_url
        if sources.github_url and not result.code_url:
            result.code_available = True
            result.code_url = sources.github_url

        result.human_verification = self._check_human_verification(combined_content)

        return result
