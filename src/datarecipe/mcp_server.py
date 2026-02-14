"""MCP Server for DataRecipe.

This server exposes DataRecipe functionality as MCP tools,
allowing Claude Desktop/Claude Code to directly use datarecipe
without requiring external API calls.

Usage:
    # Add to Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
        "mcpServers": {
            "datarecipe": {
                "command": "python",
                "args": ["-m", "datarecipe.mcp_server"]
            }
        }
    }

    # Or run directly:
    python -m datarecipe.mcp_server
"""

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def create_server() -> "Server":
    """Create and configure the MCP server."""
    server = Server("datarecipe")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="parse_spec_document",
                description="Parse a specification document (PDF, Word, image, text) and extract text content. Returns the document text and a prompt for LLM analysis.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file (PDF, docx, png, jpg, txt, md)",
                        }
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="generate_spec_output",
                description="Generate project artifacts (annotation spec, executive summary, milestone plan, cost breakdown) from analysis JSON.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_json": {
                            "type": "object",
                            "description": "Analysis result with project_name, dataset_type, task_type, fields, examples, etc.",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Output directory path",
                            "default": "./projects",
                        },
                        "target_size": {
                            "type": "integer",
                            "description": "Target dataset size for cost estimation",
                            "default": 100,
                        },
                        "region": {
                            "type": "string",
                            "description": "Region for cost calculation (china/us)",
                            "default": "china",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: original document path for metadata",
                        },
                    },
                    "required": ["analysis_json"],
                },
            ),
            Tool(
                name="analyze_huggingface_dataset",
                description="Run deep analysis on a HuggingFace dataset and generate reproduction guide.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Output directory path",
                            "default": "./projects",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of samples to analyze",
                            "default": 500,
                        },
                        "target_size": {
                            "type": "integer",
                            "description": "Target dataset size for cost estimation",
                        },
                        "region": {
                            "type": "string",
                            "description": "Region for cost calculation",
                            "default": "china",
                        },
                    },
                    "required": ["dataset_id"],
                },
            ),
            Tool(
                name="get_extraction_prompt",
                description="Get the LLM extraction prompt template for analyzing a specification document. Use this when you want to analyze a document yourself instead of using an external API.",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="extract_rubrics",
                description="Extract scoring rubrics and evaluation patterns from a HuggingFace dataset. Returns structured templates for annotation guidelines.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of samples to analyze",
                            "default": 500,
                        },
                    },
                    "required": ["dataset_id"],
                },
            ),
            Tool(
                name="extract_prompts",
                description="Extract system prompt templates from a HuggingFace dataset. Returns unique prompts categorized by domain.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of samples to analyze",
                            "default": 500,
                        },
                    },
                    "required": ["dataset_id"],
                },
            ),
            Tool(
                name="compare_datasets",
                description="Compare multiple HuggingFace datasets side by side. Returns comparison metrics and recommendations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of HuggingFace dataset IDs to compare (minimum 2)",
                            "minItems": 2,
                        },
                        "include_quality": {
                            "type": "boolean",
                            "description": "Include quality metrics in comparison",
                            "default": False,
                        },
                    },
                    "required": ["dataset_ids"],
                },
            ),
            Tool(
                name="profile_dataset",
                description="Generate annotator profile and cost estimation for a dataset. Returns required skills, team size, and budget.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')",
                        },
                        "region": {
                            "type": "string",
                            "description": "Region for cost calculation (china/us/europe/india/sea)",
                            "default": "china",
                        },
                    },
                    "required": ["dataset_id"],
                },
            ),
            Tool(
                name="get_agent_context",
                description="Get the AI Agent context file from a previous analysis. Returns structured data for AI Agent consumption.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Path to the analysis output directory",
                        }
                    },
                    "required": ["output_dir"],
                },
            ),
            Tool(
                name="enhance_analysis_reports",
                description=(
                    "Apply LLM-enhanced context to regenerate analysis reports with rich, "
                    "dataset-specific content. Use after analyze_huggingface_dataset returns "
                    "an enhancement_prompt. Process the prompt yourself and pass the resulting "
                    "JSON here to replace template placeholders with tailored analysis."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Output directory from previous analyze_huggingface_dataset call",
                        },
                        "enhanced_context": {
                            "type": "object",
                            "description": "LLM-generated enhanced context JSON (from processing the enhancement_prompt)",
                        },
                    },
                    "required": ["output_dir", "enhanced_context"],
                },
            ),
            Tool(
                name="recipe_template",
                description="从分析结果生成标注模板（接 data-label）。读取 DATA_SCHEMA.json 和 ANNOTATION_SPEC.md，生成 data-label 兼容的 HTML 标注模板。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_dir": {
                            "type": "string",
                            "description": "DataRecipe 分析输出目录",
                        },
                        "template_type": {
                            "type": "string",
                            "enum": ["classification", "ranking", "qa", "preference", "auto"],
                            "description": "标注模板类型，auto 自动推断（默认 auto）",
                            "default": "auto",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "保存模板文件的路径（可选）",
                        },
                    },
                    "required": ["analysis_dir"],
                },
            ),
            Tool(
                name="recipe_diff",
                description="对比两次分析结果的差异。比较 schema 字段、统计数据、评分规范等。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_dir_a": {
                            "type": "string",
                            "description": "第一个分析输出目录",
                        },
                        "analysis_dir_b": {
                            "type": "string",
                            "description": "第二个分析输出目录",
                        },
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["schema", "stats", "rubrics", "cost", "all"],
                            },
                            "description": "要比较的板块（默认 all）",
                            "default": ["all"],
                        },
                    },
                    "required": ["analysis_dir_a", "analysis_dir_b"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""

        if name == "parse_spec_document":
            return await _parse_spec_document(arguments)
        elif name == "generate_spec_output":
            return await _generate_spec_output(arguments)
        elif name == "analyze_huggingface_dataset":
            return await _analyze_huggingface_dataset(arguments)
        elif name == "get_extraction_prompt":
            return await _get_extraction_prompt(arguments)
        elif name == "extract_rubrics":
            return await _extract_rubrics(arguments)
        elif name == "extract_prompts":
            return await _extract_prompts(arguments)
        elif name == "compare_datasets":
            return await _compare_datasets(arguments)
        elif name == "profile_dataset":
            return await _profile_dataset(arguments)
        elif name == "get_agent_context":
            return await _get_agent_context(arguments)
        elif name == "enhance_analysis_reports":
            return await _enhance_analysis_reports(arguments)
        elif name == "recipe_template":
            return await _recipe_template(arguments)
        elif name == "recipe_diff":
            return await _recipe_diff(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def _parse_spec_document(arguments: dict[str, Any]) -> list[TextContent]:
    """Parse a specification document."""
    from datarecipe.analyzers.spec_analyzer import SpecAnalyzer

    file_path = arguments.get("file_path")
    if not file_path:
        return [TextContent(type="text", text="Error: file_path is required")]

    try:
        analyzer = SpecAnalyzer()
        doc = analyzer.parse_document(file_path)
        prompt = analyzer.get_extraction_prompt(doc)

        result = {
            "success": True,
            "file_path": file_path,
            "file_type": doc.file_type,
            "text_length": len(doc.text_content),
            "has_images": doc.has_images(),
            "image_count": len(doc.images),
            "pages": doc.pages,
            "extraction_prompt": prompt,
            "instructions": "Please analyze the document content in the extraction_prompt and return a JSON object with the extracted information. The JSON should include: project_name, dataset_type, task_type, task_description, cognitive_requirements, reasoning_chain, data_requirements, quality_constraints, forbidden_items, difficulty_criteria, fields, field_requirements, examples, scoring_rubric, estimated_difficulty, estimated_domain, estimated_human_percentage, similar_datasets.",
        }

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    except FileNotFoundError:
        return [TextContent(type="text", text=f"Error: File not found: {file_path}")]
    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing document: {e}")]


async def _generate_spec_output(arguments: dict[str, Any]) -> list[TextContent]:
    """Generate project artifacts from analysis JSON."""
    from datarecipe.analyzers.spec_analyzer import SpecAnalyzer
    from datarecipe.generators.spec_output import SpecOutputGenerator

    analysis_json = arguments.get("analysis_json")
    if not analysis_json:
        return [TextContent(type="text", text="Error: analysis_json is required")]

    output_dir = arguments.get("output_dir", "./projects")
    target_size = arguments.get("target_size", 100)
    region = arguments.get("region", "china")
    file_path = arguments.get("file_path")

    try:
        analyzer = SpecAnalyzer()

        # Parse original document if provided
        doc = None
        if file_path:
            try:
                doc = analyzer.parse_document(file_path)
            except Exception:
                pass

        # Create analysis from JSON
        analysis = analyzer.create_analysis_from_json(analysis_json, doc)

        # Generate outputs
        generator = SpecOutputGenerator(output_dir=output_dir)
        result = generator.generate(
            analysis=analysis,
            target_size=target_size,
            region=region,
        )

        if not result.success:
            return [TextContent(type="text", text=f"Error: {result.error}")]

        output = {
            "success": True,
            "output_dir": result.output_dir,
            "files_generated": result.files_generated,
            "project_name": analysis.project_name,
            "key_files": {
                "executive_summary": f"{result.output_dir}/01_决策参考/EXECUTIVE_SUMMARY.md",
                "milestone_plan": f"{result.output_dir}/02_项目管理/MILESTONE_PLAN.md",
                "annotation_spec": f"{result.output_dir}/03_标注规范/ANNOTATION_SPEC.md",
                "cost_breakdown": f"{result.output_dir}/05_成本分析/COST_BREAKDOWN.md",
            },
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating output: {e}")]


async def _analyze_huggingface_dataset(arguments: dict[str, Any]) -> list[TextContent]:
    """Run deep analysis on a HuggingFace dataset."""
    from datarecipe.core.deep_analyzer import DeepAnalyzerCore

    dataset_id = arguments.get("dataset_id")
    if not dataset_id:
        return [TextContent(type="text", text="Error: dataset_id is required")]

    output_dir = arguments.get("output_dir", "./projects")
    sample_size = arguments.get("sample_size", 500)
    target_size = arguments.get("target_size")
    region = arguments.get("region", "china")

    try:
        analyzer = DeepAnalyzerCore(
            output_dir=output_dir,
            region=region,
            use_llm=False,  # Don't use external LLM
        )

        result = analyzer.analyze(
            dataset_id=dataset_id,
            sample_size=sample_size,
            target_size=target_size,
        )

        if not result.success:
            return [TextContent(type="text", text=f"Error: {result.error}")]

        output = {
            "success": True,
            "dataset_id": dataset_id,
            "output_dir": result.output_dir,
            "dataset_type": result.dataset_type,
            "sample_count": result.sample_count,
            "rubric_patterns": result.rubric_patterns,
            "prompt_templates": result.prompt_templates,
            "human_percentage": result.human_percentage,
            "files_generated": result.files_generated,
            "key_files": {
                "executive_summary": f"{result.output_dir}/01_决策参考/EXECUTIVE_SUMMARY.md",
                "reproduction_guide": f"{result.output_dir}/04_复刻指南/REPRODUCTION_GUIDE.md",
                "annotation_spec": f"{result.output_dir}/03_标注规范/ANNOTATION_SPEC.md",
                "agent_context": f"{result.output_dir}/08_AI_Agent/agent_context.json",
            },
        }

        # Include enhancement prompt for MCP two-step workflow
        if result.enhancement_prompt:
            output["enhancement_prompt"] = result.enhancement_prompt
            output["enhancement_instructions"] = (
                "报告已生成，但部分内容为模板化占位符。"
                "上面的 enhancement_prompt 包含一个数据集增强分析提示。"
                "请根据此提示生成 JSON 格式的增强内容，"
                "然后调用 enhance_analysis_reports 工具将增强内容应用到报告中，"
                "使报告包含针对性的、具体的分析内容。"
            )

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        import traceback

        return [
            TextContent(type="text", text=f"Error analyzing dataset: {e}\n{traceback.format_exc()}")
        ]


async def _get_extraction_prompt(arguments: dict[str, Any]) -> list[TextContent]:
    """Get the extraction prompt template."""
    from datarecipe.analyzers.spec_analyzer import SpecAnalyzer

    prompt_template = SpecAnalyzer.EXTRACTION_PROMPT

    result = {
        "prompt_template": prompt_template,
        "usage": "Replace {document_content} with the actual document text, then analyze and return JSON.",
        "output_fields": [
            "project_name",
            "dataset_type",
            "description",
            "task_type",
            "task_description",
            "cognitive_requirements",
            "reasoning_chain",
            "data_requirements",
            "quality_constraints",
            "forbidden_items",
            "difficulty_criteria",
            "fields",
            "field_requirements",
            "examples",
            "scoring_rubric",
            "estimated_difficulty",
            "estimated_domain",
            "estimated_human_percentage",
            "similar_datasets",
        ],
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _extract_rubrics(arguments: dict[str, Any]) -> list[TextContent]:
    """Extract scoring rubrics from a dataset."""
    from datarecipe.extractors import RubricsAnalyzer

    dataset_id = arguments.get("dataset_id")
    if not dataset_id:
        return [TextContent(type="text", text="Error: dataset_id is required")]

    sample_size = arguments.get("sample_size", 500)

    try:
        from datasets import load_dataset

        # Load dataset
        ds = load_dataset(dataset_id, split="train", streaming=True)

        # Collect rubrics
        rubrics = []
        task_count = 0
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            task_count = i + 1
            for field in ["rubrics", "rubric", "criteria", "evaluation"]:
                if field in item:
                    value = item[field]
                    if isinstance(value, list):
                        rubrics.extend(value)
                    elif isinstance(value, str):
                        rubrics.append(value)

        if not rubrics:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": "No rubrics found in dataset",
                            "tried_fields": ["rubrics", "rubric", "criteria", "evaluation"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            ]

        # Analyze
        analyzer = RubricsAnalyzer()
        result = analyzer.analyze(rubrics, task_count=task_count)

        output = {
            "success": True,
            "dataset_id": dataset_id,
            "total_rubrics": result.total_rubrics,
            "unique_patterns": result.unique_patterns,
            "avg_rubrics_per_task": result.avg_rubrics_per_task,
            "top_verbs": dict(list(result.verb_distribution.items())[:10]),
            "category_distribution": dict(result.category_distribution),
            "structured_templates": result.structured_templates[:10],
            "summary": result.summary(),
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting rubrics: {e}")]


async def _extract_prompts(arguments: dict[str, Any]) -> list[TextContent]:
    """Extract prompt templates from a dataset."""
    from datarecipe.extractors import PromptExtractor

    dataset_id = arguments.get("dataset_id")
    if not dataset_id:
        return [TextContent(type="text", text="Error: dataset_id is required")]

    sample_size = arguments.get("sample_size", 500)

    try:
        from datasets import load_dataset

        # Load dataset
        ds = load_dataset(dataset_id, split="train", streaming=True)

        # Collect messages
        messages = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            if "messages" in item and isinstance(item["messages"], list):
                messages.extend(item["messages"])

        if not messages:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": "No messages found in dataset",
                            "hint": "Dataset may not contain 'messages' field",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            ]

        # Extract prompts
        extractor = PromptExtractor()
        library = extractor.extract(messages)

        output = {
            "success": True,
            "dataset_id": dataset_id,
            "unique_count": library.unique_count,
            "total_extracted": library.total_extracted,
            "deduplication_ratio": library.deduplication_ratio,
            "category_counts": dict(library.category_counts),
            "domain_counts": dict(library.domain_counts),
            "sample_prompts": [
                {
                    "category": p.category,
                    "domain": p.domain,
                    "preview": p.content[:200] + "..." if len(p.content) > 200 else p.content,
                }
                for p in library.templates[:5]
            ],
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting prompts: {e}")]


async def _compare_datasets(arguments: dict[str, Any]) -> list[TextContent]:
    """Compare multiple datasets."""
    from datarecipe.comparator import DatasetComparator

    dataset_ids = arguments.get("dataset_ids")
    if not dataset_ids or len(dataset_ids) < 2:
        return [TextContent(type="text", text="Error: At least 2 dataset_ids are required")]

    include_quality = arguments.get("include_quality", False)

    try:
        comparator = DatasetComparator(include_quality=include_quality)
        report = comparator.compare_by_ids(list(dataset_ids))

        output = {
            "success": True,
            "datasets_compared": dataset_ids,
            "comparison_table": report.to_markdown(),
            "recommendations": report.recommendations,
            "metrics": {
                ds_id: {
                    "sample_count": metrics.get("sample_count", 0),
                    "field_count": metrics.get("field_count", 0),
                    "estimated_cost": metrics.get("estimated_cost", 0),
                }
                for ds_id, metrics in report.dataset_metrics.items()
            }
            if hasattr(report, "dataset_metrics")
            else {},
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error comparing datasets: {e}")]


async def _profile_dataset(arguments: dict[str, Any]) -> list[TextContent]:
    """Generate annotator profile for a dataset."""
    from datarecipe.analyzer import DatasetAnalyzer
    from datarecipe.profiler import AnnotatorProfiler

    dataset_id = arguments.get("dataset_id")
    if not dataset_id:
        return [TextContent(type="text", text="Error: dataset_id is required")]

    region = arguments.get("region", "china")

    try:
        analyzer = DatasetAnalyzer()
        recipe = analyzer.analyze(dataset_id)

        profiler = AnnotatorProfiler()
        profile = profiler.generate_profile(recipe, region=region)

        profile_dict = profile.to_dict()
        output = {
            "success": True,
            "dataset_id": dataset_id,
            "region": region,
            "profile": profile_dict,
            "summary": {
                "skill_requirements": [
                    s.get("name", "") for s in profile_dict.get("skill_requirements", [])[:5]
                ],
                "education_level": profile_dict.get("education_level", ""),
                "experience_level": profile_dict.get("experience", {}).get("level", ""),
                "min_experience_years": profile_dict.get("experience", {}).get("min_years", 0),
                "team_size": profile_dict.get("team", {}).get("size", 0),
                "estimated_person_days": profile_dict.get("workload", {}).get(
                    "estimated_person_days", 0
                ),
                "hourly_rate_range": profile_dict.get("hourly_rate_range", {}),
            },
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error profiling dataset: {e}")]


async def _get_agent_context(arguments: dict[str, Any]) -> list[TextContent]:
    """Get AI Agent context from analysis output."""
    import os

    output_dir = arguments.get("output_dir")
    if not output_dir:
        return [TextContent(type="text", text="Error: output_dir is required")]

    agent_context_path = os.path.join(output_dir, "08_AI_Agent", "agent_context.json")

    if not os.path.exists(agent_context_path):
        # Try without 08_AI_Agent subdirectory
        alt_path = os.path.join(output_dir, "agent_context.json")
        if os.path.exists(alt_path):
            agent_context_path = alt_path
        else:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": "agent_context.json not found",
                            "searched_paths": [agent_context_path, alt_path],
                            "hint": "Run analyze_huggingface_dataset or generate_spec_output first",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            ]

    try:
        with open(agent_context_path, encoding="utf-8") as f:
            context = json.load(f)

        # Also try to load workflow state
        workflow_path = os.path.join(os.path.dirname(agent_context_path), "workflow_state.json")
        workflow_state = None
        if os.path.exists(workflow_path):
            with open(workflow_path, encoding="utf-8") as f:
                workflow_state = json.load(f)

        output = {
            "success": True,
            "output_dir": output_dir,
            "agent_context": context,
            "workflow_state": workflow_state,
            "available_files": os.listdir(os.path.dirname(agent_context_path)),
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error reading agent context: {e}")]


async def _enhance_analysis_reports(arguments: dict[str, Any]) -> list[TextContent]:
    """Apply LLM-enhanced context to regenerate analysis reports."""
    import os

    from datarecipe.core.deep_analyzer import DeepAnalyzerCore

    output_dir = arguments.get("output_dir")
    enhanced_context_data = arguments.get("enhanced_context")

    if not output_dir:
        return [TextContent(type="text", text="Error: output_dir is required")]
    if not enhanced_context_data:
        return [TextContent(type="text", text="Error: enhanced_context is required")]

    # Read _enhancement_state.json for re-run parameters
    state_path = os.path.join(output_dir, "_enhancement_state.json")
    if not os.path.exists(state_path):
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "error": "_enhancement_state.json not found",
                        "hint": "Run analyze_huggingface_dataset first to generate the analysis state.",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        ]

    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)

        # Convert JSON dict to EnhancedContext
        from datarecipe.generators.llm_enhancer import LLMEnhancer

        enhancer = LLMEnhancer()
        enhanced_context = enhancer._dict_to_context(enhanced_context_data)

        # Re-run analysis with pre-loaded enhanced context
        base_dir = os.path.dirname(output_dir)
        analyzer = DeepAnalyzerCore(
            output_dir=base_dir,
            region=state.get("region", "china"),
            pre_enhanced_context=enhanced_context,
        )

        result = analyzer.analyze(
            dataset_id=state["dataset_id"],
            sample_size=state.get("sample_size", 500),
            target_size=state.get("target_size"),
            split=state.get("split"),
        )

        if not result.success:
            return [TextContent(type="text", text=f"Error: {result.error}")]

        output = {
            "success": True,
            "output_dir": result.output_dir,
            "files_regenerated": result.files_generated,
            "message": "报告已使用 LLM 增强内容重新生成",
            "key_files": {
                "executive_summary": f"{result.output_dir}/01_决策参考/EXECUTIVE_SUMMARY.md",
                "reproduction_guide": f"{result.output_dir}/04_复刻指南/REPRODUCTION_GUIDE.md",
                "annotation_spec": f"{result.output_dir}/03_标注规范/ANNOTATION_SPEC.md",
                "agent_context": f"{result.output_dir}/08_AI_Agent/agent_context.json",
            },
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        import traceback

        return [
            TextContent(
                type="text",
                text=f"Error enhancing reports: {e}\n{traceback.format_exc()}",
            )
        ]


async def _recipe_template(arguments: dict[str, Any]) -> list[TextContent]:
    """Generate annotation template from analysis results."""
    import os

    analysis_dir = arguments.get("analysis_dir", "")
    template_type = arguments.get("template_type", "auto")
    output_path = arguments.get("output_path")

    schema_path = os.path.join(analysis_dir, "02_数据结构", "DATA_SCHEMA.json")
    if not os.path.exists(schema_path):
        # Try flat layout
        schema_path = os.path.join(analysis_dir, "DATA_SCHEMA.json")
    if not os.path.exists(schema_path):
        return [TextContent(type="text", text=f"Error: DATA_SCHEMA.json not found in {analysis_dir}")]

    try:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        fields = schema.get("fields", {})
        if not fields and "schema" in schema:
            fields = schema["schema"].get("fields", {})

        # Auto-detect template type
        if template_type == "auto":
            field_names = " ".join(fields.keys()).lower()
            if "chosen" in field_names or "rejected" in field_names or "preference" in field_names:
                template_type = "preference"
            elif "rank" in field_names or "score" in field_names:
                template_type = "ranking"
            elif "question" in field_names and "answer" in field_names:
                template_type = "qa"
            else:
                template_type = "classification"

        # Generate template
        lines = [
            f"## 标注模板 (类型: {template_type})",
            "",
            f"从 `{os.path.basename(analysis_dir)}` 的 schema 自动生成",
            "",
            "### 字段定义",
            "",
            "| 字段 | 类型 | 必填 | 说明 |",
            "|------|------|------|------|",
        ]
        for fname, fdef in fields.items():
            ftype = fdef.get("type", "-") if isinstance(fdef, dict) else str(fdef)
            req = "是" if (isinstance(fdef, dict) and fdef.get("required")) else "否"
            desc = fdef.get("description", "-") if isinstance(fdef, dict) else "-"
            lines.append(f"| {fname} | {ftype} | {req} | {desc} |")

        lines.extend(["", f"### 推荐模板类型: `{template_type}`", ""])

        # Generate a minimal data-label compatible config
        label_config = {
            "template_type": template_type,
            "fields": {k: {"type": v.get("type", "text") if isinstance(v, dict) else "text"} for k, v in fields.items()},
        }
        lines.extend(["### data-label 配置", "", "```json", json.dumps(label_config, ensure_ascii=False, indent=2), "```"])

        text = "\n".join(lines)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            return [TextContent(type="text", text=f"模板已保存到 {output_path}\n\n{text}")]
        return [TextContent(type="text", text=text)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating template: {e}")]


async def _recipe_diff(arguments: dict[str, Any]) -> list[TextContent]:
    """Compare two analysis outputs."""
    import os

    dir_a = arguments["analysis_dir_a"]
    dir_b = arguments["analysis_dir_b"]
    sections = arguments.get("sections", ["all"])
    if "all" in sections:
        sections = ["schema", "stats", "rubrics", "cost"]

    lines = [f"## 分析对比", "", f"- A: `{os.path.basename(dir_a)}`", f"- B: `{os.path.basename(dir_b)}`", ""]

    def _load_json(base_dir: str, *paths: str) -> dict | None:
        for p in paths:
            fp = os.path.join(base_dir, p)
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as f:
                    return json.load(f)
        return None

    if "schema" in sections:
        schema_a = _load_json(dir_a, "02_数据结构/DATA_SCHEMA.json", "DATA_SCHEMA.json")
        schema_b = _load_json(dir_b, "02_数据结构/DATA_SCHEMA.json", "DATA_SCHEMA.json")
        lines.append("### Schema 对比")
        if schema_a and schema_b:
            fields_a = set((schema_a.get("fields") or schema_a.get("schema", {}).get("fields", {})).keys())
            fields_b = set((schema_b.get("fields") or schema_b.get("schema", {}).get("fields", {})).keys())
            added = fields_b - fields_a
            removed = fields_a - fields_b
            shared = fields_a & fields_b
            lines.append(f"- 共有字段: {len(shared)}")
            if added:
                lines.append(f"- B 新增: {', '.join(sorted(added))}")
            if removed:
                lines.append(f"- B 移除: {', '.join(sorted(removed))}")
            if not added and not removed:
                lines.append("- 字段完全一致")
        else:
            lines.append("- 无法加载 schema（至少一个目录缺少 DATA_SCHEMA.json）")
        lines.append("")

    if "stats" in sections:
        ctx_a = _load_json(dir_a, "08_AI_Agent/agent_context.json", "agent_context.json")
        ctx_b = _load_json(dir_b, "08_AI_Agent/agent_context.json", "agent_context.json")
        lines.append("### 统计对比")
        if ctx_a and ctx_b:
            for key in ["sample_count", "estimated_difficulty", "estimated_domain"]:
                va = ctx_a.get(key, ctx_a.get("dataset", {}).get(key, "-"))
                vb = ctx_b.get(key, ctx_b.get("dataset", {}).get(key, "-"))
                lines.append(f"- {key}: {va} → {vb}")
        else:
            lines.append("- 无法加载 agent_context.json")
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        logger.error("MCP SDK not installed. Run: pip install mcp")
        logger.error("Alternatively, use the CLI with --interactive mode:")
        logger.error("  datarecipe analyze-spec document.pdf --interactive")
        sys.exit(1)

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


def run():
    """Entry point for datarecipe-mcp command."""
    import asyncio

    asyncio.run(main())
