"""Deep analyzer for extracting detailed dataset information from papers and web pages.

.. deprecated::
    This module handles URL/paper analysis. For HuggingFace dataset analysis,
    use :class:`datarecipe.core.deep_analyzer.DeepAnalyzerCore` instead.
    This module is retained because :mod:`datarecipe.llm_analyzer` depends on
    :class:`DeepAnalyzer` for web content extraction. A future release will
    migrate URL analysis into ``datarecipe.core`` and remove this file.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests


class DatasetCategory(Enum):
    """Categories of datasets based on their construction method."""

    LLM_DISTILLATION = "llm_distillation"
    HUMAN_ANNOTATION = "human_annotation"
    PROGRAMMATIC = "programmatic"
    SIMULATION = "simulation"
    WEB_SCRAPE = "web_scrape"
    HYBRID = "hybrid"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"


@dataclass
class DeepAnalysisResult:
    """Result of deep analysis."""

    name: str
    category: DatasetCategory
    description: str

    # Core methodology
    methodology: str = ""
    key_innovations: list[str] = field(default_factory=list)

    # Data composition
    data_sources: list[str] = field(default_factory=list)
    data_format: Optional[str] = None
    size_info: dict = field(default_factory=dict)

    # Generation details
    generation_method: str = ""
    generation_tools: list[str] = field(default_factory=list)
    generation_steps: list[dict] = field(default_factory=list)

    # Quality control
    quality_methods: list[str] = field(default_factory=list)
    human_verification: bool = False
    verification_details: str = ""

    # Technical details
    domain: str = ""
    modeling_approach: str = ""
    evaluation_metrics: list[str] = field(default_factory=list)

    # Reproducibility
    code_available: bool = False
    code_url: Optional[str] = None
    data_available: bool = False
    data_url: Optional[str] = None
    paper_url: Optional[str] = None  # Auto-discovered paper URL

    # Cost and resources
    estimated_cost: Optional[float] = None
    resource_requirements: list[str] = field(default_factory=list)

    # Additional insights
    limitations: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    related_datasets: list[str] = field(default_factory=list)


class DeepAnalyzer:
    """Analyzer for extracting detailed information from dataset papers and pages."""

    # Patterns for detecting dataset categories
    CATEGORY_PATTERNS = {
        DatasetCategory.LLM_DISTILLATION: [
            r"distill",
            r"teacher.?model",
            r"gpt-?\d",
            r"claude",
            r"llama",
            r"synthetic.?data",
            r"generated.?by",
            r"api.?call",
        ],
        DatasetCategory.HUMAN_ANNOTATION: [
            r"human.?annotat",
            r"crowdsourc",
            r"mturk",
            r"mechanical.?turk",
            r"expert.?label",
            r"manual.?annotation",
            r"annotator",
        ],
        DatasetCategory.PROGRAMMATIC: [
            r"programmat",
            r"procedural",
            r"compositional.?generat",
            r"rule.?based",
            r"template",
            r"automatic.?generat",
        ],
        DatasetCategory.SIMULATION: [
            r"simulat",
            r"environment",
            r"agent",
            r"interact",
            r"pomdp",
            r"mdp",
            r"reinforcement",
        ],
        DatasetCategory.BENCHMARK: [
            r"benchmark",
            r"evaluat",
            r"test.?set",
            r"leaderboard",
        ],
    }

    # Patterns for extracting specific information
    SIZE_PATTERNS = [
        (r"(\d{1,3}(?:,\d{3})+)\s*(?:examples?|samples?|instances?|tasks?|items?)", "count"),
        (r"(\d+(?:\.\d+)?)\s*[kK]\s*(?:examples?|samples?|instances?|tasks?)", "count_k"),
        (r"(\d+(?:\.\d+)?)\s*[mM]\s*(?:examples?|samples?|instances?|tasks?)", "count_m"),
        (r"training.{0,20}(\d{1,3}(?:,\d{3})*)", "train"),
        (r"test.{0,20}(\d{1,3}(?:,\d{3})*)", "test"),
        (r"validation.{0,20}(\d{1,3}(?:,\d{3})*)", "val"),
    ]

    METRIC_PATTERNS = [
        r"accuracy",
        r"f1.?score",
        r"precision",
        r"recall",
        r"bleu",
        r"rouge",
        r"perplexity",
        r"success.?rate",
        r"completion.?rate",
        r"efficiency",
        r"human.?eval",
    ]

    def __init__(self, auto_search_paper: bool = True):
        """Initialize the deep analyzer.

        Args:
            auto_search_paper: If True, automatically search for related papers
                              when the initial analysis lacks methodology details.
        """
        self.auto_search_paper = auto_search_paper

    def search_related_paper(self, dataset_name: str) -> Optional[str]:
        """Search for related papers on arXiv.

        Args:
            dataset_name: Name of the dataset to search for

        Returns:
            URL of the most relevant paper, or None if not found
        """
        try:
            # Clean up dataset name for search - use quotes for exact phrase
            clean_name = dataset_name.replace("-", " ").replace("_", " ").strip()

            # Try multiple search strategies
            search_queries = [
                f'ti:"{clean_name}"',  # Title search with quotes
                f'all:"{clean_name}" AND (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)',
                f"all:{clean_name.replace(' ', '+')}+dataset",  # With dataset keyword
            ]

            arxiv_api = "http://export.arxiv.org/api/query"

            for query in search_queries:
                params = {
                    "search_query": query,
                    "start": 0,
                    "max_results": 5,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                }

                response = requests.get(arxiv_api, params=params, timeout=10)
                if response.status_code != 200:
                    continue

                content = response.text

                # Extract arXiv IDs from the response (format: 2312.11456v4)
                arxiv_ids = re.findall(
                    r"<id>http://arxiv.org/abs/(\d+\.\d+(?:v\d+)?)</id>", content
                )

                if arxiv_ids:
                    # Return the first (most relevant) result, strip version
                    arxiv_id = arxiv_ids[0].split("v")[0]  # Remove version suffix
                    return f"https://arxiv.org/abs/{arxiv_id}"

            return None

        except (ImportError, OSError, ValueError, AttributeError):
            return None

    def analyze(self, url: str, search_paper_if_needed: bool = None) -> DeepAnalysisResult:
        """Perform deep analysis on a URL.

        Args:
            url: URL of the paper or dataset page
            search_paper_if_needed: If True, search for related papers when
                                   methodology details are lacking. Defaults to
                                   self.auto_search_paper.

        Returns:
            DeepAnalysisResult with extracted information
        """
        if search_paper_if_needed is None:
            search_paper_if_needed = self.auto_search_paper

        content = self._fetch_content(url)
        if not content:
            raise ValueError(f"Could not fetch content from: {url}")

        # Extract basic info
        name = self._extract_name(content, url)
        description = self._extract_description(content)

        # Detect category
        category = self._detect_category(content)

        # Create result
        result = DeepAnalysisResult(
            name=name,
            category=category,
            description=description,
        )

        # Extract detailed information
        result.methodology = self._extract_methodology(content)
        result.key_innovations = self._extract_innovations(content)
        result.data_sources = self._extract_data_sources(content)
        result.size_info = self._extract_size_info(content)
        result.generation_method = self._extract_generation_method(content, category)
        result.generation_steps = self._extract_generation_steps(content, category)
        result.quality_methods = self._extract_quality_methods(content)
        result.human_verification = self._check_human_verification(content)
        result.domain = self._extract_domain(content)
        result.modeling_approach = self._extract_modeling_approach(content)
        result.evaluation_metrics = self._extract_metrics(content)
        result.code_available, result.code_url = self._extract_code_info(content)
        result.data_available, result.data_url = self._extract_data_info(content)
        result.limitations = self._extract_limitations(content)
        result.use_cases = self._extract_use_cases(content)

        # Check if methodology details are lacking and search for paper if needed
        methodology_lacking = result.methodology in ["", "未知"] and not result.generation_steps

        if methodology_lacking and search_paper_if_needed and name:
            paper_url = self.search_related_paper(name)
            if paper_url:
                # Store the found paper URL
                result.paper_url = paper_url

                # Fetch and analyze the paper
                paper_content = self._fetch_content(paper_url)
                if paper_content:
                    # Merge paper information into result
                    paper_category = self._detect_category(paper_content)
                    if paper_category != DatasetCategory.UNKNOWN:
                        result.category = paper_category

                    paper_methodology = self._extract_methodology(paper_content)
                    if paper_methodology and paper_methodology != "未知":
                        result.methodology = paper_methodology

                    paper_innovations = self._extract_innovations(paper_content)
                    if paper_innovations:
                        result.key_innovations = list(
                            set(result.key_innovations + paper_innovations)
                        )

                    paper_steps = self._extract_generation_steps(paper_content, result.category)
                    if paper_steps:
                        result.generation_steps = paper_steps

                    paper_quality = self._extract_quality_methods(paper_content)
                    if paper_quality:
                        result.quality_methods = list(set(result.quality_methods + paper_quality))

                    if not result.modeling_approach:
                        result.modeling_approach = self._extract_modeling_approach(paper_content)

                    if not result.domain or result.domain == "通用":
                        result.domain = self._extract_domain(paper_content)

                    paper_metrics = self._extract_metrics(paper_content)
                    if paper_metrics:
                        result.evaluation_metrics = list(
                            set(result.evaluation_metrics + paper_metrics)
                        )

                    # Check for code/data in paper
                    if not result.code_available:
                        result.code_available, result.code_url = self._extract_code_info(
                            paper_content
                        )
                    if not result.data_available:
                        result.data_available, result.data_url = self._extract_data_info(
                            paper_content
                        )

        return result

    def _fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL."""
        try:
            # Convert PDF URL to abstract page for arXiv
            if "arxiv.org/pdf" in url:
                url = url.replace("/pdf/", "/abs/").replace(".pdf", "")

            headers = {"User-Agent": "Mozilla/5.0 (compatible; DataRecipe/1.0)"}
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.text
        except (OSError, requests.RequestException):
            pass
        return None

    def _extract_name(self, content: str, url: str) -> str:
        """Extract dataset/paper name."""
        # Try title tag
        match = re.search(r"<title>([^<]+)</title>", content, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean up
            for suffix in [" - arXiv", " | Papers With Code", " - GitHub"]:
                title = title.replace(suffix, "")
            return title
        return url.split("/")[-1]

    def _extract_description(self, content: str) -> str:
        """Extract description."""
        # Try meta description
        match = re.search(
            r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
            content,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try abstract
        match = re.search(r"abstract[^>]*>([^<]{100,500})", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return ""

    def _detect_category(self, content: str) -> DatasetCategory:
        """Detect the dataset category based on content."""
        content_lower = content.lower()
        scores = {}

        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, content_lower))
            scores[category] = score

        if max(scores.values()) == 0:
            return DatasetCategory.UNKNOWN

        return max(scores, key=scores.get)

    def _extract_methodology(self, content: str) -> str:
        """Extract methodology description."""
        content_lower = content.lower()

        # More comprehensive methodology patterns
        methodology_patterns = [
            # Programmatic/Procedural
            (r"compositional.{0,20}generat", "组合式任务生成器，从原子组件程序化创建多样化任务"),
            (r"procedural.{0,20}generat", "程序化生成，通过算法自动创建任务"),
            (r"rule.?based.{0,20}generat", "基于规则的生成方法"),
            (r"programmat.{0,20}creat", "程序化创建数据集"),
            # Human design
            (r"human.?design", "人工设计的任务和谜题"),
            (r"manually.{0,20}creat", "人工创建的数据集"),
            (r"expert.{0,20}design", "专家设计的任务"),
            (r"hand.?craft", "手工构建的数据"),
            # Abstraction/Reasoning specific
            (r"abstract.{0,20}reason", "抽象推理任务，测试归纳和泛化能力"),
            (r"visual.{0,20}puzzle", "视觉谜题，需要模式识别和推理"),
            (r"grid.?based.{0,20}task", "基于网格的任务"),
            (r"few.?shot.{0,20}learn", "少样本学习任务"),
            # Distillation
            (r"distill", "知识蒸馏，使用大模型生成训练数据"),
            (r"teacher.{0,20}model", "教师模型生成数据"),
            (r"synthetic.{0,20}data.{0,20}generat", "合成数据生成"),
            # Human annotation
            (r"crowdsourc", "众包标注"),
            (r"human.{0,20}annot", "人工标注"),
            (r"mturk|mechanical.?turk", "Amazon MTurk 众包标注"),
            # Simulation
            (r"simulat.{0,20}environ", "模拟环境驱动的数据生成"),
            (r"interactive.{0,20}environ", "交互式环境"),
            # Collection
            (r"web.?scrap|crawl", "网络爬取收集"),
            (r"curated.{0,20}from", "从现有资源筛选整理"),
        ]

        for pattern, description in methodology_patterns:
            if re.search(pattern, content_lower):
                return description

        return "未知"

    def _extract_innovations(self, content: str) -> list[str]:
        """Extract key innovations."""
        innovations = []
        content_lower = content.lower()

        innovation_indicators = [
            # Environment/Control
            ("dual.?control", "双控制环境：用户和代理都能使用工具"),
            ("dec.?pomdp", "Dec-POMDP 建模：分布式部分可观测马尔可夫决策过程"),
            ("user.?simulator", "用户模拟器：模拟真实用户行为"),
            # Generation
            ("compositional", "组合式生成：从原子组件构建复杂任务"),
            ("procedural.{0,10}generat", "程序化生成：算法自动创建任务"),
            # Evaluation
            ("efficiency.?scor", "效率评分：同时考虑成本和能力"),
            ("fine.?grain", "细粒度分析：区分推理和协调能力"),
            # Reasoning
            ("abstract.{0,10}reason", "抽象推理：测试泛化和归纳能力"),
            ("few.?shot", "少样本学习：用极少示例推断规则"),
            ("novel.{0,10}task", "新颖任务：评估时使用训练中未见的任务"),
            ("out.?of.?distribution", "分布外泛化：测试对新情况的适应"),
            # Task design
            ("grid.?based", "网格任务：基于二维网格的视觉推理"),
            ("input.?output.{0,10}pair", "输入输出对：从示例中学习转换规则"),
            ("transformation.{0,10}rule", "转换规则：发现并应用抽象规则"),
            # Difficulty
            ("difficult.{0,10}for.{0,10}(ai|llm|model)", "AI难题：对当前模型具有挑战性"),
            ("human.{0,10}baseline", "人类基准：与人类表现对比"),
            ("unsolved", "未解决：目前没有系统能完全解决"),
            # Scale
            ("private.{0,10}test", "私有测试集：防止过拟合"),
            ("held.?out", "保留集：用于公平评估"),
            # Prize/Competition
            ("prize|competition|challenge", "竞赛/挑战：设有奖金激励研究"),
        ]

        for pattern, description in innovation_indicators:
            if re.search(pattern, content_lower):
                innovations.append(description)

        return innovations

    def _extract_data_sources(self, content: str) -> list[str]:
        """Extract data sources."""
        sources = []
        content_lower = content.lower()

        source_patterns = [
            (r"telecom", "电信领域数据"),
            (r"medical|health", "医疗健康数据"),
            (r"wikipedia", "Wikipedia"),
            (r"common.?crawl", "Common Crawl"),
            (r"github", "GitHub 代码"),
            (r"reddit", "Reddit"),
            (r"stack.?overflow", "Stack Overflow"),
        ]

        for pattern, name in source_patterns:
            if re.search(pattern, content_lower):
                sources.append(name)

        return sources if sources else ["自定义/专有数据"]

    def _extract_size_info(self, content: str) -> dict:
        """Extract dataset size information."""
        size_info = {}

        for pattern, key in self.SIZE_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                num_str = match.group(1).replace(",", "")
                try:
                    if key == "count_k":
                        size_info["total"] = int(float(num_str) * 1000)
                    elif key == "count_m":
                        size_info["total"] = int(float(num_str) * 1000000)
                    else:
                        size_info[key] = int(num_str)
                except ValueError:
                    pass

        return size_info

    def _extract_generation_method(self, content: str, category: DatasetCategory) -> str:
        """Extract generation method based on category."""
        methods = {
            DatasetCategory.LLM_DISTILLATION: "LLM API 调用生成",
            DatasetCategory.HUMAN_ANNOTATION: "人工标注",
            DatasetCategory.PROGRAMMATIC: "程序化/规则生成",
            DatasetCategory.SIMULATION: "环境模拟生成",
            DatasetCategory.BENCHMARK: "多来源整合",
            DatasetCategory.HYBRID: "混合方法",
        }
        return methods.get(category, "未知")

    def _extract_generation_steps(self, content: str, category: DatasetCategory) -> list[dict]:
        """Extract generation steps based on category."""
        if category == DatasetCategory.PROGRAMMATIC or category == DatasetCategory.SIMULATION:
            return [
                {"step": 1, "name": "设计领域环境", "description": "定义实体、状态、可观测性"},
                {"step": 2, "name": "定义原子操作", "description": "用户操作和代理操作"},
                {"step": 3, "name": "组合任务生成", "description": "程序化生成多样任务"},
                {"step": 4, "name": "构建模拟器", "description": "用户/环境模拟器"},
                {"step": 5, "name": "人工验证", "description": "确保任务可解"},
                {"step": 6, "name": "校准发布", "description": "数据集划分和发布"},
            ]
        elif category == DatasetCategory.LLM_DISTILLATION:
            return [
                {"step": 1, "name": "准备种子数据", "description": "收集高质量种子样本"},
                {"step": 2, "name": "设计提示词", "description": "创建生成模板"},
                {"step": 3, "name": "LLM 生成", "description": "批量调用 API"},
                {"step": 4, "name": "质量过滤", "description": "自动过滤低质量数据"},
                {"step": 5, "name": "人工验证", "description": "抽样检查"},
                {"step": 6, "name": "格式化发布", "description": "转换格式并发布"},
            ]
        elif category == DatasetCategory.HUMAN_ANNOTATION:
            return [
                {"step": 1, "name": "收集原始数据", "description": "获取需标注的数据"},
                {"step": 2, "name": "编写标注指南", "description": "定义标注标准"},
                {"step": 3, "name": "试标注", "description": "小规模测试"},
                {"step": 4, "name": "大规模标注", "description": "众包或专家标注"},
                {"step": 5, "name": "质量控制", "description": "一致性检查"},
                {"step": 6, "name": "仲裁发布", "description": "处理争议并发布"},
            ]
        elif category == DatasetCategory.BENCHMARK:
            return [
                {"step": 1, "name": "定义评估目标", "description": "明确要测试的能力维度"},
                {"step": 2, "name": "设计任务格式", "description": "确定输入输出格式和难度级别"},
                {"step": 3, "name": "创建任务实例", "description": "人工设计或程序生成任务"},
                {"step": 4, "name": "人类基准测试", "description": "收集人类表现数据"},
                {"step": 5, "name": "数据集划分", "description": "划分公开/私有测试集"},
                {"step": 6, "name": "发布与维护", "description": "建立排行榜和评估协议"},
            ]
        else:
            return []

    def _extract_quality_methods(self, content: str) -> list[str]:
        """Extract quality control methods."""
        methods = []
        content_lower = content.lower()

        quality_patterns = [
            (r"human.?verif", "人工验证"),
            (r"inter.?annotator", "标注者间一致性检查"),
            (r"automat.?filter", "自动过滤"),
            (r"dedup", "去重"),
            (r"quality.?score", "质量评分"),
            (r"calibrat", "校准测试"),
        ]

        for pattern, name in quality_patterns:
            if re.search(pattern, content_lower):
                methods.append(name)

        return methods

    def _check_human_verification(self, content: str) -> bool:
        """Check if human verification was used."""
        content_lower = content.lower()
        patterns = [r"human.?verif", r"human.?test", r"participant", r"annotator.?agreement"]
        return any(re.search(p, content_lower) for p in patterns)

    def _extract_domain(self, content: str) -> str:
        """Extract domain."""
        content_lower = content.lower()
        domains = {
            "telecom": "电信",
            "medical|health|clinical": "医疗",
            "legal|law": "法律",
            "financial|banking": "金融",
            "education": "教育",
            "e-commerce|retail": "电商",
            "customer.?service": "客服",
        }

        for pattern, name in domains.items():
            if re.search(pattern, content_lower):
                return name
        return "通用"

    def _extract_modeling_approach(self, content: str) -> str:
        """Extract modeling approach."""
        content_lower = content.lower()

        if "dec-pomdp" in content_lower or "decentralized" in content_lower:
            return "Dec-POMDP（分布式部分可观测马尔可夫决策过程）"
        elif "pomdp" in content_lower:
            return "POMDP（部分可观测马尔可夫决策过程）"
        elif "mdp" in content_lower:
            return "MDP（马尔可夫决策过程）"
        elif "multi-agent" in content_lower:
            return "多代理系统"

        return ""

    def _extract_metrics(self, content: str) -> list[str]:
        """Extract evaluation metrics."""
        metrics = []
        content_lower = content.lower()

        for pattern in self.METRIC_PATTERNS:
            if re.search(pattern, content_lower):
                metrics.append(pattern.replace(".?", " ").replace("\\", ""))

        return metrics

    def _extract_code_info(self, content: str) -> tuple[bool, Optional[str]]:
        """Extract code availability info."""
        # Look for GitHub links - clean up any trailing punctuation
        match = re.search(r"github\.com/([^/\s\"'<>]+/[a-zA-Z0-9_.-]+)", content)
        if match:
            repo_path = match.group(1).rstrip(".,;:!?)]}")
            return True, f"https://github.com/{repo_path}"

        if "code" in content.lower() and "available" in content.lower():
            return True, None

        return False, None

    def _extract_data_info(self, content: str) -> tuple[bool, Optional[str]]:
        """Extract data availability info."""
        # Look for HuggingFace links
        match = re.search(r"huggingface\.co/datasets/([^/\s\"'<>]+/[^/\s\"'<>]+)", content)
        if match:
            return True, f"https://huggingface.co/datasets/{match.group(1)}"

        if "data" in content.lower() and "available" in content.lower():
            return True, None

        return False, None

    def _extract_limitations(self, content: str) -> list[str]:
        """Extract limitations."""
        limitations = []
        content_lower = content.lower()

        limitation_patterns = [
            (r"limited to", "领域限制"),
            (r"does not", "功能限制"),
            (r"future work", "待改进"),
            (r"challenging", "技术挑战"),
        ]

        for pattern, category in limitation_patterns:
            if re.search(pattern, content_lower):
                limitations.append(category)

        return limitations

    def _extract_use_cases(self, content: str) -> list[str]:
        """Extract use cases."""
        use_cases = []
        content_lower = content.lower()

        use_case_patterns = [
            (r"customer.?service|support", "客户服务"),
            (r"technical.?support", "技术支持"),
            (r"chatbot|conversational", "对话系统"),
            (r"evaluation|benchmark", "模型评估"),
            (r"training", "模型训练"),
        ]

        for pattern, name in use_case_patterns:
            if re.search(pattern, content_lower):
                use_cases.append(name)

        return use_cases


def deep_analysis_to_markdown(result: DeepAnalysisResult) -> str:
    """Generate a customized production guide from deep analysis result.

    This creates a detailed, domain-specific guide based on the analyzed
    dataset's methodology and characteristics.
    """
    lines = []

    # Title
    lines.append(f"# 数据生产指南：{result.name}")
    lines.append("")

    # Reference info
    lines.append("## 参考数据集")
    lines.append("")
    lines.append("| 属性 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| **名称** | {result.name} |")
    if hasattr(result, "paper_url") and result.paper_url:
        lines.append(f"| **论文** | {result.paper_url} |")
    if result.domain:
        lines.append(f"| **领域** | {result.domain} |")
    lines.append(f"| **分类** | {result.category.value.replace('_', ' ').title()} |")
    if result.modeling_approach:
        lines.append(f"| **建模方式** | {result.modeling_approach} |")
    lines.append("")

    # Key innovations
    if result.key_innovations:
        lines.append("### 核心特点")
        lines.append("")
        for innovation in result.key_innovations:
            lines.append(f"- {innovation}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Core concept
    lines.append("## 核心理念")
    lines.append("")
    if result.description:
        lines.append(f"> {result.description}")
        lines.append("")
    if result.methodology:
        lines.append(f"**方法论**: {result.methodology}")
        lines.append("")
    lines.append("")

    # Production flow
    lines.append("---")
    lines.append("")
    lines.append("## 生产流程")
    lines.append("")

    # Flow diagram
    if result.generation_steps:
        lines.append("### 流程图")
        lines.append("")
        lines.append("```")
        step_names = [f"[{s['step']}. {s['name']}]" for s in result.generation_steps]
        lines.append(" → ".join(step_names))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Detailed steps
        for step in result.generation_steps:
            lines.append(f"### 步骤 {step['step']}: {step['name']}")
            lines.append("")
            lines.append(f"**目标**: {step['description']}")
            lines.append("")

            # Add category-specific code examples
            if (
                result.category == DatasetCategory.PROGRAMMATIC
                or result.category == DatasetCategory.SIMULATION
            ):
                if step["step"] == 1:
                    lines.append("**输出**: `environment_spec.yaml`")
                    lines.append("")
                    lines.append("**示例（{}领域）**:".format(result.domain or "通用"))
                    lines.append("")
                    lines.append("```yaml")
                    lines.append(f"domain: {result.domain.lower() if result.domain else 'general'}")
                    lines.append("entities:")
                    lines.append("  - primary_entity:")
                    lines.append("      attributes: [attr1, attr2, status]")
                    lines.append("")
                    lines.append("world_state:")
                    lines.append("  observable_by_user: [public_info]")
                    lines.append("  observable_by_agent: [full_info, internal_state]")
                    lines.append("  shared: [conversation_history]")
                    lines.append("```")
                    lines.append("")
                elif step["step"] == 2:
                    lines.append("**输出**: `actions.yaml`")
                    lines.append("")
                    lines.append("```yaml")
                    lines.append("user_actions:")
                    lines.append("  - check_status:")
                    lines.append("      preconditions: [logged_in]")
                    lines.append("      effects: [display_status]")
                    lines.append("  - execute_action:")
                    lines.append("      params: [action_type]")
                    lines.append("      effects: [update_state]")
                    lines.append("")
                    lines.append("agent_actions:")
                    lines.append("  - lookup_info:")
                    lines.append("      params: [entity_id]")
                    lines.append("      effects: [retrieve_info]")
                    lines.append("  - guide_user:")
                    lines.append("      params: [instruction]")
                    lines.append("      effects: [user_receives_guidance]")
                    lines.append("```")
                    lines.append("")
                elif step["step"] == 3:
                    lines.append("**输出**: `tasks.jsonl`")
                    lines.append("")
                    lines.append("**生成策略**:")
                    lines.append("")
                    lines.append("```python")
                    lines.append("class TaskGenerator:")
                    lines.append("    def generate_task(self, complexity_level):")
                    lines.append("        # 1. 随机初始化世界状态")
                    lines.append("        initial_state = self.random_initial_state()")
                    lines.append("")
                    lines.append("        # 2. 定义目标状态")
                    lines.append(
                        "        goal_state = self.define_goal(initial_state, complexity_level)"
                    )
                    lines.append("")
                    lines.append("        # 3. 计算最优解（用于验证）")
                    lines.append(
                        "        optimal_solution = self.compute_solution(initial_state, goal_state)"
                    )
                    lines.append("")
                    lines.append("        return {")
                    lines.append("            'initial_state': initial_state,")
                    lines.append("            'goal_state': goal_state,")
                    lines.append("            'optimal_solution': optimal_solution,")
                    lines.append("        }")
                    lines.append("```")
                    lines.append("")
                    lines.append("**复杂度控制**:")
                    lines.append("")
                    lines.append("| 等级 | 描述 |")
                    lines.append("|------|------|")
                    lines.append("| 简单 | 单步解决，无需用户操作 |")
                    lines.append("| 中等 | 需要用户提供信息或执行简单操作 |")
                    lines.append("| 困难 | 多步协作，用户需执行关键操作 |")
                    lines.append("")
                elif step["step"] == 4:
                    lines.append("**输出**: `simulator.py`")
                    lines.append("")
                    lines.append("```python")
                    lines.append("class Simulator:")
                    lines.append("    def __init__(self, task, behavior_params):")
                    lines.append("        self.task = task")
                    lines.append(
                        "        self.cooperation_level = behavior_params.get('cooperation', 0.9)"
                    )
                    lines.append(
                        "        self.comprehension_level = behavior_params.get('comprehension', 0.8)"
                    )
                    lines.append("")
                    lines.append("    def respond(self, message, available_actions):")
                    lines.append("        # 根据理解能力和配合度生成响应")
                    lines.append("        pass")
                    lines.append("```")
                    lines.append("")
                    lines.append("**行为参数**:")
                    lines.append("- `cooperation_level`: 配合度 (0-1)")
                    lines.append("- `comprehension_level`: 理解能力 (0-1)")
                    lines.append("- `patience`: 耐心程度")
                    lines.append("")
                elif step["step"] == 5:
                    lines.append("**方法**: 人工测试验证")
                    lines.append("")
                    lines.append("```")
                    lines.append("验证标准：")
                    lines.append("- 每个任务至少 2 名人类测试者")
                    lines.append("- 在 2 次尝试内解决")
                    lines.append("- 记录人类平均表现")
                    lines.append("```")
                    lines.append("")
                    lines.append("**输出**: `human_baseline.json`")
                    lines.append("")

            lines.append("---")
            lines.append("")

    # Size info
    if result.size_info:
        lines.append("## 数据集规模")
        lines.append("")
        lines.append("| 划分 | 数量 |")
        lines.append("|------|------|")
        for key, value in result.size_info.items():
            lines.append(f"| {key} | {value:,} |")
        lines.append("")

    # Quality standards
    lines.append("## 质量标准")
    lines.append("")
    if result.quality_methods:
        for method in result.quality_methods:
            lines.append(f"- [ ] {method}")
    else:
        # Default quality criteria based on category
        if (
            result.category == DatasetCategory.PROGRAMMATIC
            or result.category == DatasetCategory.SIMULATION
        ):
            lines.append("- [ ] 每个任务有明确的最优解")
            lines.append("- [ ] 需要双方协作的任务占比 > 50%")
            lines.append("- [ ] 人类测试通过率 > 90%")
            lines.append("- [ ] 任务多样性：覆盖所有操作类型")
        elif result.category == DatasetCategory.LLM_DISTILLATION:
            lines.append("- [ ] 生成内容与种子数据主题一致")
            lines.append("- [ ] 无事实性错误")
            lines.append("- [ ] 格式符合要求")
            lines.append("- [ ] 无重复或高度相似内容")
        elif result.category == DatasetCategory.HUMAN_ANNOTATION:
            lines.append("- [ ] 标注者间一致性 > 0.7")
            lines.append("- [ ] 无遗漏标注")
            lines.append("- [ ] 标注符合指南要求")
    lines.append("")

    # Evaluation metrics
    if result.evaluation_metrics:
        lines.append("---")
        lines.append("")
        lines.append("## 评估指标")
        lines.append("")
        for metric in result.evaluation_metrics:
            lines.append(f"- **{metric}**")
        lines.append("")

    # Pitfalls
    lines.append("---")
    lines.append("")
    lines.append("## 常见问题与避坑指南")
    lines.append("")
    if (
        result.category == DatasetCategory.PROGRAMMATIC
        or result.category == DatasetCategory.SIMULATION
    ):
        pitfalls = [
            "环境设计过于简单 → 无法产生有意义的交互",
            "模拟器太完美 → 不能反映真实用户的错误和误解",
            "任务无解或有多解 → 导致评估不公平",
            "跳过人工验证 → 可能包含无法解决的任务",
            "忽略效率指标 → 只看完成率会鼓励低效的暴力尝试",
        ]
    elif result.category == DatasetCategory.LLM_DISTILLATION:
        pitfalls = [
            "种子数据质量差导致生成质量差",
            "提示词不够明确导致输出不一致",
            "未设置 temperature 导致生成单一",
            "跳过人工验证导致质量问题",
            "未记录生成参数导致无法复现",
        ]
    else:
        pitfalls = [
            "数据质量不一致",
            "缺乏明确的质量标准",
            "文档不完整",
        ]

    for i, pitfall in enumerate(pitfalls, 1):
        lines.append(f"{i}. ⚠️ {pitfall}")
    lines.append("")

    # Domain migration
    if (
        result.category == DatasetCategory.PROGRAMMATIC
        or result.category == DatasetCategory.SIMULATION
    ):
        lines.append("---")
        lines.append("")
        lines.append("## 迁移到其他领域")
        lines.append("")
        lines.append("该框架可以迁移到：")
        lines.append("")
        lines.append("| 领域 | 用户操作示例 | 代理操作示例 |")
        lines.append("|------|------------|------------|")
        lines.append("| 医疗问诊 | 测量体温、描述症状 | 查看病历、开具处方 |")
        lines.append("| IT 支持 | 重启电脑、检查网络 | 远程诊断、推送更新 |")
        lines.append("| 金融服务 | 验证身份、确认交易 | 查询账户、处理申请 |")
        lines.append("| 智能家居 | 检查设备、执行操作 | 分析数据、调整设置 |")
        lines.append("")

    # Code and data availability
    if result.code_available or result.data_available:
        lines.append("---")
        lines.append("")
        lines.append("## 资源链接")
        lines.append("")
        if result.code_available:
            if result.code_url:
                lines.append(f"- 代码: {result.code_url}")
            else:
                lines.append("- 代码: 可用（具体链接请参考原论文）")
        if result.data_available:
            if result.data_url:
                lines.append(f"- 数据: {result.data_url}")
            else:
                lines.append("- 数据: 可用（具体链接请参考原论文）")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("> 由 DataRecipe 生成 — 专项数据生产指南")

    return "\n".join(lines)
