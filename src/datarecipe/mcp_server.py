#!/usr/bin/env python3
"""
DataRecipe MCP Server

让 Claude App 可以调用 DataRecipe 功能进行数据集分析。

Usage:
    # 启动 server
    python -m datarecipe.mcp_server

    # 或者通过入口点
    datarecipe-mcp
"""

import json
import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建 MCP Server
server = Server("datarecipe")


def analyze_dataset(dataset_id: str) -> dict:
    """分析数据集"""
    from datarecipe.analyzer import DatasetAnalyzer

    analyzer = DatasetAnalyzer()
    recipe = analyzer.analyze(dataset_id)

    return {
        "name": recipe.name,
        "source_type": recipe.source_type.value,
        "generation": {
            "synthetic_ratio": recipe.synthetic_ratio,
            "human_ratio": recipe.human_ratio,
            "type": recipe.generation_type.value if recipe.generation_type else "unknown",
        },
        "teacher_models": recipe.teacher_models or [],
        "reproducibility": {
            "score": recipe.reproducibility.score if recipe.reproducibility else None,
            "available": recipe.reproducibility.available if recipe.reproducibility else [],
            "missing": recipe.reproducibility.missing if recipe.reproducibility else [],
        },
        "cost": {
            "estimated_total_usd": recipe.cost.estimated_total_usd if recipe.cost else None,
            "confidence": recipe.cost.confidence if recipe.cost else None,
        },
        "metadata": {
            "num_examples": recipe.num_examples,
            "languages": recipe.languages,
            "license": recipe.license,
        }
    }


def profile_annotators(dataset_id: str, region: str = "china") -> dict:
    """生成标注专家画像"""
    from datarecipe.analyzer import DatasetAnalyzer
    from datarecipe.profiler import AnnotatorProfiler

    analyzer = DatasetAnalyzer()
    profiler = AnnotatorProfiler()

    recipe = analyzer.analyze(dataset_id)
    profile = profiler.generate_profile(recipe, region=region)

    # 计算成本
    hourly_rate = (profile.hourly_rate_range.get("min", 15) + profile.hourly_rate_range.get("max", 45)) / 2
    estimated_labor_cost = profile.estimated_person_days * 8 * hourly_rate

    return {
        "dataset": dataset_id,
        "region": region,
        "skills": [
            {
                "name": s.name,
                "level": s.level,
                "required": s.required,
            }
            for s in profile.skill_requirements
        ],
        "requirements": {
            "experience_level": profile.experience_level.value,
            "education_level": profile.education_level.value,
            "min_experience_years": profile.min_experience_years,
            "languages": profile.language_requirements,
            "domain_knowledge": profile.domain_knowledge,
        },
        "workload": {
            "team_size": profile.team_size,
            "team_structure": profile.team_structure,
            "person_days": profile.estimated_person_days,
            "hours_per_example": profile.estimated_hours_per_example,
        },
        "cost": {
            "hourly_rate_range": profile.hourly_rate_range,
            "hourly_rate_avg": hourly_rate,
            "estimated_labor_cost": estimated_labor_cost,
        },
        "screening_criteria": profile.screening_criteria,
        "recommended_platforms": profile.recommended_platforms,
    }


def deploy_project(dataset_id: str, output: str = None, region: str = "china") -> dict:
    """生成投产部署项目"""
    from datarecipe.analyzer import DatasetAnalyzer
    from datarecipe.deployer import ProductionDeployer
    from datarecipe.profiler import AnnotatorProfiler
    from datarecipe.schema import DataRecipe

    # 默认输出目录
    if not output:
        safe_name = dataset_id.replace("/", "_").replace(" ", "_").lower()
        output = f"./projects/{safe_name}"

    analyzer = DatasetAnalyzer()
    deployer = ProductionDeployer()
    profiler = AnnotatorProfiler()

    recipe = analyzer.analyze(dataset_id)
    profile = profiler.generate_profile(recipe, region=region)

    # 转换为 DataRecipe
    data_recipe = DataRecipe(
        name=recipe.name,
        version=recipe.version,
        source_type=recipe.source_type,
        source_id=recipe.source_id,
        num_examples=recipe.num_examples,
        languages=recipe.languages or [],
        license=recipe.license,
        description=recipe.description,
        generation_type=recipe.generation_type,
        synthetic_ratio=recipe.synthetic_ratio,
        human_ratio=recipe.human_ratio,
        generation_methods=recipe.generation_methods or [],
        teacher_models=recipe.teacher_models or [],
        tags=recipe.tags or [],
    )

    config = deployer.generate_config(data_recipe, profile=profile)
    result = deployer.deploy(data_recipe, output, provider="local", config=config, profile=profile)

    return {
        "success": result.success,
        "project_id": result.project_handle.project_id if result.project_handle else None,
        "output_path": output,
        "files_created": result.details.get("files_created", []),
        "error": result.error,
    }


def list_providers() -> dict:
    """列出可用的 Provider"""
    from datarecipe.providers import list_providers

    providers = list_providers()
    return {
        "providers": providers,
        "count": len(providers),
    }


def estimate_cost(dataset_id: str, target_size: int = None, model: str = "gpt-4o") -> dict:
    """估算数据集生产成本"""
    from datarecipe.analyzer import DatasetAnalyzer
    from datarecipe.cost_calculator import CostCalculator

    analyzer = DatasetAnalyzer()
    calculator = CostCalculator()

    recipe = analyzer.analyze(dataset_id)
    target = target_size or recipe.num_examples or 10000

    breakdown = calculator.estimate_from_recipe(recipe, target, model)

    return {
        "dataset": dataset_id,
        "target_size": target,
        "model": model,
        "cost": {
            "api": {
                "low": breakdown.api_cost.low,
                "expected": breakdown.api_cost.expected,
                "high": breakdown.api_cost.high,
            },
            "human_annotation": {
                "low": breakdown.human_annotation_cost.low,
                "expected": breakdown.human_annotation_cost.expected,
                "high": breakdown.human_annotation_cost.high,
            },
            "compute": {
                "low": breakdown.compute_cost.low,
                "expected": breakdown.compute_cost.expected,
                "high": breakdown.compute_cost.high,
            },
            "total": {
                "low": breakdown.total.low,
                "expected": breakdown.total.expected,
                "high": breakdown.total.high,
            },
        },
        "assumptions": breakdown.assumptions,
    }


# 注册工具
@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的工具"""
    return [
        Tool(
            name="analyze_dataset",
            description="分析 AI 数据集，提取元数据、检测生成方法、教师模型和可复现性评分。支持 HuggingFace 数据集。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集 ID，如 'Anthropic/hh-rlhf' 或 'AI-MO/NuminaMath-CoT'",
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="profile_annotators",
            description="生成标注专家画像，包括技能要求、学历要求、团队规模和人力成本估算。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集 ID",
                    },
                    "region": {
                        "type": "string",
                        "description": "地区，用于成本估算。可选: china, us, europe, india, sea",
                        "default": "china",
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="deploy_project",
            description="生成完整的标注投产项目，包括标注指南、质量规则、验收标准、时间线等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集 ID",
                    },
                    "output": {
                        "type": "string",
                        "description": "输出目录路径，默认为 ./projects/<dataset_name>/",
                    },
                    "region": {
                        "type": "string",
                        "description": "地区，用于成本估算",
                        "default": "china",
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="estimate_cost",
            description="估算数据集生产成本，包括 API 调用、人工标注和计算资源成本。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集 ID",
                    },
                    "target_size": {
                        "type": "integer",
                        "description": "目标数据量",
                    },
                    "model": {
                        "type": "string",
                        "description": "用于成本估算的模型，如 gpt-4o, claude-3-sonnet",
                        "default": "gpt-4o",
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="list_providers",
            description="列出可用的部署 Provider。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """处理工具调用"""
    try:
        if name == "analyze_dataset":
            result = analyze_dataset(arguments["dataset_id"])
        elif name == "profile_annotators":
            result = profile_annotators(
                arguments["dataset_id"],
                arguments.get("region", "china"),
            )
        elif name == "deploy_project":
            result = deploy_project(
                arguments["dataset_id"],
                arguments.get("output"),
                arguments.get("region", "china"),
            )
        elif name == "estimate_cost":
            result = estimate_cost(
                arguments["dataset_id"],
                arguments.get("target_size"),
                arguments.get("model", "gpt-4o"),
            )
        elif name == "list_providers":
            result = list_providers()
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """运行 MCP Server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    """入口点"""
    asyncio.run(main())


if __name__ == "__main__":
    run()
