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


def deep_analyze_dataset(dataset_id: str, sample_size: int = 200, use_llm: bool = False) -> dict:
    """深度分析数据集，生成复刻指南"""
    import os
    from datarecipe.integrations.radar import RadarIntegration

    try:
        from datasets import load_dataset
        from datarecipe.extractors import RubricsAnalyzer, PromptExtractor
        from datarecipe.analyzers import ContextStrategyDetector
        from datarecipe.generators import HumanMachineSplitter, TaskType

        # Create output directory
        safe_name = dataset_id.replace("/", "_").replace("\\", "_")
        output_dir = f"./analysis_output/{safe_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        try:
            ds = load_dataset(dataset_id, split="train", streaming=True)
        except ValueError:
            ds = load_dataset(dataset_id, split="test", streaming=True)

        # Collect samples
        schema_info = {}
        sample_items = []
        rubrics = []
        messages = []
        sample_count = 0

        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            sample_count = i + 1

            if i < 5:
                for field, value in item.items():
                    if field not in schema_info:
                        schema_info[field] = {"type": type(value).__name__}
                sample_items.append(item)

            for field in ["rubrics", "rubric", "criteria"]:
                if field in item:
                    v = item[field]
                    if isinstance(v, list):
                        rubrics.extend(v)
                    elif isinstance(v, str):
                        rubrics.append(v)

            if "messages" in item:
                messages.extend(item.get("messages", []))

        # Detect dataset type
        is_preference = "chosen" in schema_info and "rejected" in schema_info
        is_swe = "repo" in schema_info and "patch" in schema_info

        dataset_type = ""
        if is_preference:
            dataset_type = "preference"
        elif is_swe:
            dataset_type = "swe_bench"
        elif rubrics:
            dataset_type = "evaluation"

        # Analyze
        rubrics_result = None
        if rubrics:
            analyzer = RubricsAnalyzer()
            rubrics_result = analyzer.analyze(rubrics, task_count=sample_count)

        prompt_library = None
        if messages:
            extractor = PromptExtractor()
            prompt_library = extractor.extract(messages)

        # Allocation
        splitter = HumanMachineSplitter(region="china")
        allocation = splitter.analyze(
            dataset_size=sample_count,
            task_types=[
                TaskType.CONTEXT_CREATION,
                TaskType.TASK_DESIGN,
                TaskType.RUBRICS_WRITING,
                TaskType.DATA_GENERATION,
                TaskType.QUALITY_REVIEW,
            ]
        )

        # LLM analysis
        llm_analysis = None
        if use_llm and not dataset_type:
            try:
                from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer
                llm_analyzer = LLMDatasetAnalyzer()
                llm_analysis = llm_analyzer.analyze(
                    dataset_id=dataset_id,
                    schema_info=schema_info,
                    sample_items=sample_items,
                    sample_count=sample_count,
                )
                dataset_type = llm_analysis.dataset_type
            except Exception:
                pass

        # Create summary
        summary = RadarIntegration.create_summary(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            allocation=allocation,
            rubrics_result=rubrics_result,
            prompt_library=prompt_library,
            schema_info=schema_info,
            sample_count=sample_count,
            llm_analysis=llm_analysis,
            output_dir=output_dir,
        )
        RadarIntegration.save_summary(summary, output_dir)

        return {
            "dataset_id": dataset_id,
            "dataset_type": dataset_type or "unknown",
            "sample_count": sample_count,
            "fields": list(schema_info.keys()),
            "reproduction_cost": summary.reproduction_cost,
            "human_percentage": summary.human_percentage,
            "rubric_patterns": summary.rubric_patterns,
            "prompt_templates": summary.prompt_templates,
            "output_dir": output_dir,
            "files": ["recipe_summary.json"],
        }

    except Exception as e:
        return {"error": str(e), "dataset_id": dataset_id}


def compare_datasets(dataset_ids: list[str]) -> dict:
    """对比多个数据集的构建方式"""
    import os
    from datarecipe.integrations.radar import RadarIntegration, RecipeSummary

    results = []
    for dataset_id in dataset_ids:
        safe_name = dataset_id.replace("/", "_").replace("\\", "_")
        summary_path = f"./analysis_output/{safe_name}/recipe_summary.json"

        if os.path.exists(summary_path):
            summary = RadarIntegration.load_summary(summary_path)
            results.append(summary)
        else:
            # Run quick analysis
            result = deep_analyze_dataset(dataset_id, sample_size=100)
            if "error" not in result:
                summary = RadarIntegration.load_summary(f"./analysis_output/{safe_name}/recipe_summary.json")
                results.append(summary)

    if not results:
        return {"error": "No datasets could be analyzed", "dataset_ids": dataset_ids}

    # Build comparison
    comparison = {
        "datasets": [],
        "summary": {
            "total_datasets": len(results),
            "type_distribution": {},
            "avg_human_percentage": 0,
            "total_reproduction_cost": 0,
        }
    }

    for s in results:
        comparison["datasets"].append({
            "id": s.dataset_id,
            "type": s.dataset_type,
            "cost": s.reproduction_cost.get("total", 0),
            "human_pct": s.human_percentage,
            "fields": s.fields,
            "rubric_patterns": s.rubric_patterns,
            "difficulty": s.difficulty,
        })

        # Aggregate
        if s.dataset_type:
            comparison["summary"]["type_distribution"][s.dataset_type] = \
                comparison["summary"]["type_distribution"].get(s.dataset_type, 0) + 1
        comparison["summary"]["total_reproduction_cost"] += s.reproduction_cost.get("total", 0)

    comparison["summary"]["avg_human_percentage"] = \
        sum(d["human_pct"] for d in comparison["datasets"]) / len(comparison["datasets"])

    return comparison


def get_reproduction_guide(dataset_id: str) -> dict:
    """获取已分析数据集的复刻指南"""
    import os

    safe_name = dataset_id.replace("/", "_").replace("\\", "_")
    guide_path = f"./analysis_output/{safe_name}/REPRODUCTION_GUIDE.md"
    summary_path = f"./analysis_output/{safe_name}/recipe_summary.json"

    if not os.path.exists(summary_path):
        # Run analysis first
        result = deep_analyze_dataset(dataset_id, sample_size=200)
        if "error" in result:
            return result

    # Load summary
    from datarecipe.integrations.radar import RadarIntegration
    summary = RadarIntegration.load_summary(summary_path)

    # Load guide if exists
    guide_content = ""
    if os.path.exists(guide_path):
        with open(guide_path, "r", encoding="utf-8") as f:
            guide_content = f.read()

    return {
        "dataset_id": dataset_id,
        "summary": summary.to_dict(),
        "guide_path": guide_path,
        "guide_available": os.path.exists(guide_path),
        "guide_preview": guide_content[:2000] + "..." if len(guide_content) > 2000 else guide_content,
    }


def batch_analyze_from_radar(radar_report_path: str, limit: int = 5, orgs: list[str] = None) -> dict:
    """从 Radar 报告批量分析数据集"""
    from datarecipe.integrations.radar import RadarIntegration

    try:
        integration = RadarIntegration()
        datasets = integration.load_radar_report(radar_report_path)

        # Filter
        if orgs:
            datasets = [d for d in datasets if d.org.lower() in [o.lower() for o in orgs]]

        # Limit
        datasets = datasets[:limit]

        # Analyze each
        results = []
        for ds in datasets:
            result = deep_analyze_dataset(ds.id, sample_size=100)
            results.append({
                "dataset_id": ds.id,
                "category": ds.category,
                "downloads": ds.downloads,
                "analysis": result,
            })

        # Aggregate
        successful = [r for r in results if "error" not in r["analysis"]]

        return {
            "total_datasets": len(datasets),
            "analyzed": len(successful),
            "failed": len(results) - len(successful),
            "results": results,
            "total_cost": sum(
                r["analysis"].get("reproduction_cost", {}).get("total", 0)
                for r in successful
            ),
        }

    except Exception as e:
        return {"error": str(e)}


def find_similar_datasets(dataset_id: str) -> dict:
    """基于已分析的数据集找相似的数据集"""
    import os
    from datarecipe.integrations.radar import RadarIntegration

    # First ensure the target is analyzed
    safe_name = dataset_id.replace("/", "_").replace("\\", "_")
    summary_path = f"./analysis_output/{safe_name}/recipe_summary.json"

    if not os.path.exists(summary_path):
        result = deep_analyze_dataset(dataset_id, sample_size=100, use_llm=True)
        if "error" in result:
            return result

    target_summary = RadarIntegration.load_summary(summary_path)

    # Scan all analyzed datasets
    similar = []
    analysis_dir = "./analysis_output"

    if os.path.exists(analysis_dir):
        for name in os.listdir(analysis_dir):
            if name == safe_name:
                continue
            other_path = os.path.join(analysis_dir, name, "recipe_summary.json")
            if os.path.exists(other_path):
                try:
                    other = RadarIntegration.load_summary(other_path)

                    # Calculate similarity score
                    score = 0
                    if other.dataset_type == target_summary.dataset_type:
                        score += 50
                    if set(other.fields) & set(target_summary.fields):
                        score += 20
                    if abs(other.human_percentage - target_summary.human_percentage) < 10:
                        score += 15
                    if other.difficulty == target_summary.difficulty:
                        score += 15

                    if score > 30:
                        similar.append({
                            "dataset_id": other.dataset_id,
                            "type": other.dataset_type,
                            "similarity_score": score,
                            "cost": other.reproduction_cost.get("total", 0),
                            "fields": other.fields,
                        })
                except Exception:
                    continue

    # Sort by similarity
    similar.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Also include LLM suggestions if available
    llm_suggestions = target_summary.similar_datasets or []

    return {
        "dataset_id": dataset_id,
        "dataset_type": target_summary.dataset_type,
        "analyzed_similar": similar[:10],
        "llm_suggested": llm_suggestions,
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
        Tool(
            name="deep_analyze",
            description="深度分析数据集，提取评分标准、Prompt模板、生成复刻指南和成本估算。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集 ID，如 'tencent/CL-bench'",
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "分析的样本数量，默认 200",
                        "default": 200,
                    },
                    "use_llm": {
                        "type": "boolean",
                        "description": "是否使用 LLM 分析未知类型数据集",
                        "default": False,
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="compare_datasets",
            description="对比多个数据集的构建方式、成本和复杂度。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的数据集 ID 列表",
                    },
                },
                "required": ["dataset_ids"],
            },
        ),
        Tool(
            name="get_reproduction_guide",
            description="获取数据集的复刻指南，包括 Schema、SOP、成本估算等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集 ID",
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="batch_analyze_from_radar",
            description="从 ai-dataset-radar 报告批量分析数据集。",
            inputSchema={
                "type": "object",
                "properties": {
                    "radar_report_path": {
                        "type": "string",
                        "description": "Radar 报告 JSON 文件路径",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多分析的数据集数量",
                        "default": 5,
                    },
                    "orgs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "按组织筛选",
                    },
                },
                "required": ["radar_report_path"],
            },
        ),
        Tool(
            name="find_similar_datasets",
            description="找与指定数据集相似的数据集。",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "数据集 ID",
                    },
                },
                "required": ["dataset_id"],
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
        elif name == "deep_analyze":
            result = deep_analyze_dataset(
                arguments["dataset_id"],
                arguments.get("sample_size", 200),
                arguments.get("use_llm", False),
            )
        elif name == "compare_datasets":
            result = compare_datasets(arguments["dataset_ids"])
        elif name == "get_reproduction_guide":
            result = get_reproduction_guide(arguments["dataset_id"])
        elif name == "batch_analyze_from_radar":
            result = batch_analyze_from_radar(
                arguments["radar_report_path"],
                arguments.get("limit", 5),
                arguments.get("orgs"),
            )
        elif name == "find_similar_datasets":
            result = find_similar_datasets(arguments["dataset_id"])
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
