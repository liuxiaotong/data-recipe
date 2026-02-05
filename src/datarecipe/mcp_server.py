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
                "analysis_report": f"{result.output_dir}/01_决策参考/ANALYSIS_REPORT.md",
                "reproduction_guide": f"{result.output_dir}/04_复刻指南/REPRODUCTION_GUIDE.md",
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
