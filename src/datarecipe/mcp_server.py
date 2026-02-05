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
import sys
from typing import Any

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
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
                            "description": "Path to the document file (PDF, docx, png, jpg, txt, md)"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="generate_spec_output",
                description="Generate project artifacts (annotation spec, executive summary, milestone plan, cost breakdown) from analysis JSON.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_json": {
                            "type": "object",
                            "description": "Analysis result with project_name, dataset_type, task_type, fields, examples, etc."
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Output directory path",
                            "default": "./spec_output"
                        },
                        "target_size": {
                            "type": "integer",
                            "description": "Target dataset size for cost estimation",
                            "default": 100
                        },
                        "region": {
                            "type": "string",
                            "description": "Region for cost calculation (china/us)",
                            "default": "china"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: original document path for metadata"
                        }
                    },
                    "required": ["analysis_json"]
                }
            ),
            Tool(
                name="analyze_huggingface_dataset",
                description="Run deep analysis on a HuggingFace dataset and generate reproduction guide.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')"
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Output directory path",
                            "default": "./analysis_output"
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of samples to analyze",
                            "default": 500
                        },
                        "target_size": {
                            "type": "integer",
                            "description": "Target dataset size for cost estimation"
                        },
                        "region": {
                            "type": "string",
                            "description": "Region for cost calculation",
                            "default": "china"
                        }
                    },
                    "required": ["dataset_id"]
                }
            ),
            Tool(
                name="get_extraction_prompt",
                description="Get the LLM extraction prompt template for analyzing a specification document. Use this when you want to analyze a document yourself instead of using an external API.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="extract_rubrics",
                description="Extract scoring rubrics and evaluation patterns from a HuggingFace dataset. Returns structured templates for annotation guidelines.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')"
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of samples to analyze",
                            "default": 500
                        }
                    },
                    "required": ["dataset_id"]
                }
            ),
            Tool(
                name="extract_prompts",
                description="Extract system prompt templates from a HuggingFace dataset. Returns unique prompts categorized by domain.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')"
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of samples to analyze",
                            "default": 500
                        }
                    },
                    "required": ["dataset_id"]
                }
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
                            "minItems": 2
                        },
                        "include_quality": {
                            "type": "boolean",
                            "description": "Include quality metrics in comparison",
                            "default": False
                        }
                    },
                    "required": ["dataset_ids"]
                }
            ),
            Tool(
                name="profile_dataset",
                description="Generate annotator profile and cost estimation for a dataset. Returns required skills, team size, and budget.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "HuggingFace dataset ID (e.g., 'tencent/CL-bench')"
                        },
                        "region": {
                            "type": "string",
                            "description": "Region for cost calculation (china/us/europe/india/sea)",
                            "default": "china"
                        }
                    },
                    "required": ["dataset_id"]
                }
            ),
            Tool(
                name="get_agent_context",
                description="Get the AI Agent context file from a previous analysis. Returns structured data for AI Agent consumption.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Path to the analysis output directory"
                        }
                    },
                    "required": ["output_dir"]
                }
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
            "instructions": "Please analyze the document content in the extraction_prompt and return a JSON object with the extracted information. The JSON should include: project_name, dataset_type, task_type, task_description, cognitive_requirements, reasoning_chain, data_requirements, quality_constraints, forbidden_items, difficulty_criteria, fields, field_requirements, examples, scoring_rubric, estimated_difficulty, estimated_domain, estimated_human_percentage, similar_datasets."
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

    output_dir = arguments.get("output_dir", "./spec_output")
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
            }
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

    output_dir = arguments.get("output_dir", "./analysis_output")
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
            }
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error analyzing dataset: {e}\n{traceback.format_exc()}")]


async def _get_extraction_prompt(arguments: dict[str, Any]) -> list[TextContent]:
    """Get the extraction prompt template."""
    from datarecipe.analyzers.spec_analyzer import SpecAnalyzer

    prompt_template = SpecAnalyzer.EXTRACTION_PROMPT

    result = {
        "prompt_template": prompt_template,
        "usage": "Replace {document_content} with the actual document text, then analyze and return JSON.",
        "output_fields": [
            "project_name", "dataset_type", "description",
            "task_type", "task_description", "cognitive_requirements", "reasoning_chain",
            "data_requirements", "quality_constraints", "forbidden_items", "difficulty_criteria",
            "fields", "field_requirements", "examples", "scoring_rubric",
            "estimated_difficulty", "estimated_domain", "estimated_human_percentage", "similar_datasets"
        ]
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
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "No rubrics found in dataset",
                "tried_fields": ["rubrics", "rubric", "criteria", "evaluation"]
            }, ensure_ascii=False, indent=2))]

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
            "summary": result.summary()
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
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "No messages found in dataset",
                "hint": "Dataset may not contain 'messages' field"
            }, ensure_ascii=False, indent=2))]

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
                    "preview": p.content[:200] + "..." if len(p.content) > 200 else p.content
                }
                for p in library.templates[:5]
            ]
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
            } if hasattr(report, 'dataset_metrics') else {}
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
                "skill_requirements": [s.get("name", "") for s in profile_dict.get("skill_requirements", [])[:5]],
                "education_level": profile_dict.get("education_level", ""),
                "experience_level": profile_dict.get("experience", {}).get("level", ""),
                "min_experience_years": profile_dict.get("experience", {}).get("min_years", 0),
                "team_size": profile_dict.get("team", {}).get("size", 0),
                "estimated_person_days": profile_dict.get("workload", {}).get("estimated_person_days", 0),
                "hourly_rate_range": profile_dict.get("hourly_rate_range", {}),
            }
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
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "agent_context.json not found",
                "searched_paths": [agent_context_path, alt_path],
                "hint": "Run analyze_huggingface_dataset or generate_spec_output first"
            }, ensure_ascii=False, indent=2))]

    try:
        with open(agent_context_path, "r", encoding="utf-8") as f:
            context = json.load(f)

        # Also try to load workflow state
        workflow_path = os.path.join(os.path.dirname(agent_context_path), "workflow_state.json")
        workflow_state = None
        if os.path.exists(workflow_path):
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow_state = json.load(f)

        output = {
            "success": True,
            "output_dir": output_dir,
            "agent_context": context,
            "workflow_state": workflow_state,
            "available_files": os.listdir(os.path.dirname(agent_context_path))
        }

        return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error reading agent context: {e}")]


async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
        print("\nAlternatively, use the CLI with --interactive mode:", file=sys.stderr)
        print("  datarecipe analyze-spec document.pdf --interactive", file=sys.stderr)
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
