"""Core deep analysis functionality shared between CLI and MCP."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Output directory structure
OUTPUT_SUBDIRS = {
    "decision": "01_决策参考",  # Executive summary
    "project": "02_项目管理",  # Milestone plan, industry benchmark
    "annotation": "03_标注规范",  # Annotation spec, rubric templates
    "guide": "04_复刻指南",  # Reproduction guide, analysis report
    "cost": "05_成本分析",  # Cost breakdown, allocation, token analysis
    "data": "06_原始数据",  # Raw analysis data
    "ai_agent": "08_AI_Agent",  # AI Agent layer
}


class OutputManager:
    """Manage organized output directory structure."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.subdirs = {}
        self._create_structure()

    def _create_structure(self):
        """Create subdirectory structure."""
        os.makedirs(self.base_dir, exist_ok=True)
        for key, subdir in OUTPUT_SUBDIRS.items():
            path = os.path.join(self.base_dir, subdir)
            os.makedirs(path, exist_ok=True)
            self.subdirs[key] = path

    def get_path(self, category: str, filename: str) -> str:
        """Get full path for a file in a category."""
        if category in self.subdirs:
            return os.path.join(self.subdirs[category], filename)
        return os.path.join(self.base_dir, filename)

    def get_relative_path(self, category: str, filename: str) -> str:
        """Get relative path for display."""
        if category in self.subdirs:
            return f"{OUTPUT_SUBDIRS[category]}/{filename}"
        return filename

    def generate_readme(self, dataset_id: str, dataset_type: str) -> str:
        """Generate README.md explaining directory structure."""
        content = f"""# {dataset_id} 分析产出

> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}
> 数据类型: {dataset_type}

## 目录结构

```
{os.path.basename(self.base_dir)}/
├── README.md                    # 本文件
├── recipe_summary.json          # 核心摘要 (Radar 兼容)
│
├── {OUTPUT_SUBDIRS["decision"]}/           # 👔 决策层
│   ├── EXECUTIVE_SUMMARY.md     # 执行摘要 (价值评分、ROI)
│   └── executive_summary.json
│
├── {OUTPUT_SUBDIRS["project"]}/           # 📋 项目管理
│   ├── MILESTONE_PLAN.md        # 里程碑计划 (验收标准)
│   ├── milestone_plan.json
│   ├── INDUSTRY_BENCHMARK.md    # 行业基准对比
│   └── industry_benchmark.json
│
├── {OUTPUT_SUBDIRS["annotation"]}/           # 📝 标注团队
│   ├── ANNOTATION_SPEC.md       # 标注规范 (外包交付用)
│   ├── annotation_spec.json
│   ├── rubric_template.md       # 评分标准模板
│   └── rubric_template.json
│
├── {OUTPUT_SUBDIRS["guide"]}/           # 🔧 技术团队
│   ├── REPRODUCTION_GUIDE.md    # 复刻指南
│   └── ANALYSIS_REPORT.md       # 分析报告
│
├── {OUTPUT_SUBDIRS["cost"]}/           # 💰 成本分析
│   ├── COST_BREAKDOWN.md        # 成本明细
│   ├── allocation.json          # 人机分配
│   ├── phased_cost.json         # 分阶段成本
│   ├── cost_comparison.json     # 模型成本对比
│   ├── cost_calibration.json    # 成本校准
│   └── token_analysis.json      # Token 分析
│
├── {OUTPUT_SUBDIRS["data"]}/           # 📊 原始数据
│   ├── complexity_analysis.json # 复杂度分析
│   ├── prompt_templates.json    # Prompt 模板
│   └── ...                      # 其他分析数据
│
└── {OUTPUT_SUBDIRS["ai_agent"]}/          # 🤖 AI Agent
    ├── agent_context.json       # 聚合入口
    ├── workflow_state.json      # 工作流状态
    ├── reasoning_traces.json    # 推理链
    ├── pipeline.yaml            # 可执行流水线
    └── README.md                # Agent 说明
```

## 快速导航

| 目标 | 查看文件 |
|------|----------|
| **快速决策** | `{OUTPUT_SUBDIRS["decision"]}/EXECUTIVE_SUMMARY.md` |
| **项目规划** | `{OUTPUT_SUBDIRS["project"]}/MILESTONE_PLAN.md` |
| **外包标注** | `{OUTPUT_SUBDIRS["annotation"]}/ANNOTATION_SPEC.md` |
| **技术复刻** | `{OUTPUT_SUBDIRS["guide"]}/REPRODUCTION_GUIDE.md` |
| **成本预算** | `{OUTPUT_SUBDIRS["cost"]}/COST_BREAKDOWN.md` |
| **AI Agent** | `{OUTPUT_SUBDIRS["ai_agent"]}/agent_context.json` |

---

> 由 DataRecipe 自动生成
"""
        return content


@dataclass
class AnalysisResult:
    """Result of deep analysis."""

    dataset_id: str
    success: bool = True
    error: str = ""

    # Dataset info
    dataset_type: str = ""
    sample_count: int = 0
    fields: list[str] = field(default_factory=list)

    # Cost info
    reproduction_cost: dict[str, float] = field(default_factory=dict)
    human_percentage: float = 0.0

    # Analysis stats
    rubric_patterns: int = 0
    prompt_templates: int = 0

    # Output paths
    output_dir: str = ""
    files_generated: list[str] = field(default_factory=list)

    # Warnings collected during analysis (non-fatal issues)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "success": self.success,
            "error": self.error,
            "dataset_type": self.dataset_type,
            "sample_count": self.sample_count,
            "fields": self.fields,
            "reproduction_cost": self.reproduction_cost,
            "human_percentage": self.human_percentage,
            "rubric_patterns": self.rubric_patterns,
            "prompt_templates": self.prompt_templates,
            "output_dir": self.output_dir,
            "files_generated": self.files_generated,
            "warnings": self.warnings,
        }


class DeepAnalyzerCore:
    """Core deep analysis engine shared between CLI and MCP."""

    def __init__(
        self,
        output_dir: str = "./analysis_output",
        region: str = "china",
        use_llm: bool = False,
        llm_provider: str = "anthropic",
        enhance_mode: str = "auto",
    ):
        self.output_dir = output_dir
        self.region = region
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.enhance_mode = enhance_mode

    def analyze(
        self,
        dataset_id: str,
        sample_size: int = 500,
        split: str = None,
        target_size: int = None,
    ) -> AnalysisResult:
        """Run full deep analysis on a dataset.

        Args:
            dataset_id: Dataset identifier (e.g., "Anthropic/hh-rlhf")
            sample_size: Number of samples to analyze
            split: Dataset split (auto-detect if None)
            target_size: Target size for cost estimation

        Returns:
            AnalysisResult with all analysis data
        """
        result = AnalysisResult(dataset_id=dataset_id)

        try:
            from datasets import load_dataset

            from datarecipe.analyzers import ContextStrategyDetector
            from datarecipe.extractors import PromptExtractor, RubricsAnalyzer
            from datarecipe.generators import HumanMachineSplitter, TaskType
            from datarecipe.integrations.radar import RadarIntegration

            # Create output directory with organized structure
            safe_name = dataset_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            dataset_output_dir = os.path.join(self.output_dir, safe_name)
            output_mgr = OutputManager(dataset_output_dir)
            result.output_dir = dataset_output_dir

            # Auto-detect split
            if split is None:
                try:
                    ds = load_dataset(dataset_id, split="train", streaming=True)
                    split = "train"
                except ValueError:
                    for try_split in ["test", "validation", "dev"]:
                        try:
                            ds = load_dataset(dataset_id, split=try_split, streaming=True)
                            split = try_split
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError("Cannot find available split")
            else:
                ds = load_dataset(dataset_id, split=split, streaming=True)

            # Initialize collectors
            schema_info = {}
            category_set = set()
            sub_category_set = set()
            system_prompts_by_domain = {}
            rubrics_examples = []
            sample_items = []
            rubrics = []
            messages = []
            contexts = []

            # RLHF preference dataset support
            is_preference_dataset = False
            preference_pairs = []
            preference_topics = {}
            preference_patterns = {
                "chosen_longer": 0,
                "rejected_longer": 0,
                "same_length": 0,
                "chosen_more_detailed": 0,
                "chosen_more_helpful": 0,
                "chosen_safer": 0,
            }

            # SWE-bench support
            is_swe_dataset = False
            swe_stats = {
                "repos": {},
                "languages": {},
                "issue_types": {},
                "issue_categories": {},
                "patch_lines": [],
                "examples": [],
            }

            # Collect samples
            sample_count = 0
            for i, item in enumerate(ds):
                if i >= sample_size:
                    break
                sample_count = i + 1

                # Schema info (first 10 items)
                if i < 10:
                    for fld, value in item.items():
                        if fld not in schema_info:
                            schema_info[fld] = {
                                "type": type(value).__name__,
                                "examples": [],
                                "nested_type": None,
                            }
                            if isinstance(value, list) and value:
                                schema_info[fld]["nested_type"] = type(value[0]).__name__
                            elif isinstance(value, dict) and value:
                                schema_info[fld]["nested_type"] = list(value.keys())
                        if len(schema_info[fld]["examples"]) < 3:
                            if isinstance(value, str) and len(value) > 500:
                                schema_info[fld]["examples"].append(value[:500] + "...")
                            elif not isinstance(value, (list, dict)):
                                schema_info[fld]["examples"].append(value)

                # Sample items (first 5)
                if i < 5:
                    sample_items.append(item)

                # Categories from metadata
                if "metadata" in item and isinstance(item["metadata"], dict):
                    meta = item["metadata"]
                    if "context_category" in meta:
                        category_set.add(meta["context_category"])
                    if "sub_category" in meta:
                        sub_category_set.add(meta["sub_category"])
                    if "category" in meta:
                        category_set.add(meta["category"])

                # Rubrics
                item_rubrics = []
                for fld in ["rubrics", "rubric", "criteria"]:
                    if fld in item:
                        value = item[fld]
                        if isinstance(value, list):
                            rubrics.extend(value)
                            item_rubrics.extend(value)
                        elif isinstance(value, str):
                            rubrics.append(value)
                            item_rubrics.append(value)

                if item_rubrics and len(rubrics_examples) < 10:
                    rubrics_examples.append(
                        {
                            "rubrics": item_rubrics,
                            "metadata": item.get("metadata", {}),
                            "messages": item.get("messages", []),
                        }
                    )

                # Messages and system prompts
                if "messages" in item and isinstance(item["messages"], list):
                    messages.extend(item["messages"])
                    for msg in item["messages"]:
                        if isinstance(msg, dict) and msg.get("role") == "system":
                            content = msg.get("content", "")
                            if content and len(content) > 50:
                                domain = "general"
                                if "metadata" in item and isinstance(item["metadata"], dict):
                                    domain = item["metadata"].get(
                                        "context_category",
                                        item["metadata"].get("category", "general"),
                                    )
                                if domain not in system_prompts_by_domain:
                                    system_prompts_by_domain[domain] = []
                                if len(system_prompts_by_domain[domain]) < 3:
                                    system_prompts_by_domain[domain].append(
                                        {"content": content, "metadata": item.get("metadata", {})}
                                    )

                # Contexts
                context_found = False
                for fld in ["context", "input", "text", "document", "passage", "content"]:
                    if fld in item and isinstance(item[fld], str) and len(item[fld]) > 50:
                        contexts.append(item[fld])
                        context_found = True
                        break
                if not context_found and "messages" in item:
                    for msg in item.get("messages", []):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str) and len(content) > 100:
                                contexts.append(content)
                                break

                # RLHF preference detection
                if "chosen" in item and "rejected" in item:
                    is_preference_dataset = True
                    self._analyze_preference_pair(
                        item, preference_pairs, preference_topics, preference_patterns
                    )

                # SWE-bench detection
                if "repo" in item and "patch" in item and "problem_statement" in item:
                    is_swe_dataset = True
                    self._analyze_swe_item(item, swe_stats)

            result.sample_count = sample_count
            result.fields = list(schema_info.keys())
            actual_size = target_size or sample_count

            # Detect dataset type
            detected_type = ""
            if is_swe_dataset:
                detected_type = "swe_bench"
            elif is_preference_dataset:
                detected_type = "preference"
            elif rubrics:
                detected_type = "evaluation"

            # Run analyzers
            rubrics_result = None
            if rubrics:
                analyzer = RubricsAnalyzer()
                rubrics_result = analyzer.analyze(rubrics, task_count=sample_count)
                result.rubric_patterns = rubrics_result.unique_patterns

                # Save rubric analysis to data/
                with open(
                    output_mgr.get_path("data", "rubrics_analysis.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(analyzer.to_dict(rubrics_result), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("data", "rubrics_analysis.json")
                )

                # Save rubric templates to annotation/
                with open(
                    output_mgr.get_path("annotation", "rubric_template.yaml"), "w", encoding="utf-8"
                ) as f:
                    f.write(analyzer.to_yaml_templates(rubrics_result))
                result.files_generated.append(
                    output_mgr.get_relative_path("annotation", "rubric_template.yaml")
                )

                with open(
                    output_mgr.get_path("annotation", "rubric_template.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(analyzer.to_markdown_templates(rubrics_result))
                result.files_generated.append(
                    output_mgr.get_relative_path("annotation", "rubric_template.md")
                )

            prompt_library = None
            if messages:
                extractor = PromptExtractor()
                prompt_library = extractor.extract(messages)
                result.prompt_templates = prompt_library.unique_count

                with open(
                    output_mgr.get_path("data", "prompt_templates.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(extractor.to_dict(prompt_library), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("data", "prompt_templates.json")
                )

            strategy_result = None
            if contexts:
                detector = ContextStrategyDetector()
                strategy_result = detector.analyze(contexts[:100])
                with open(
                    output_mgr.get_path("data", "context_strategy.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(detector.to_dict(strategy_result), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("data", "context_strategy.json")
                )

            # Preference analysis
            if is_preference_dataset and preference_pairs:
                preference_analysis = {
                    "is_preference_dataset": True,
                    "total_pairs": sample_count,
                    "topic_distribution": preference_topics,
                    "patterns": preference_patterns,
                    "examples": preference_pairs[:10],
                }
                with open(
                    output_mgr.get_path("data", "preference_analysis.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(preference_analysis, f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("data", "preference_analysis.json")
                )

            # SWE analysis
            if is_swe_dataset and swe_stats["repos"]:
                avg_patch = (
                    sum(swe_stats["patch_lines"]) / len(swe_stats["patch_lines"])
                    if swe_stats["patch_lines"]
                    else 0
                )
                swe_analysis = {
                    "is_swe_dataset": True,
                    "total_tasks": sample_count,
                    "repos_count": len(swe_stats["repos"]),
                    "repo_distribution": dict(
                        sorted(swe_stats["repos"].items(), key=lambda x: -x[1])[:20]
                    ),
                    "language_distribution": swe_stats["languages"],
                    "avg_patch_lines": avg_patch,
                    "examples": swe_stats["examples"],
                }
                with open(
                    output_mgr.get_path("data", "swe_analysis.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(swe_analysis, f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("data", "swe_analysis.json")
                )

            # LLM analysis
            llm_analysis = None
            is_known_type = is_preference_dataset or is_swe_dataset or rubrics or messages
            if self.use_llm and not is_known_type:
                try:
                    from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

                    llm_analyzer = LLMDatasetAnalyzer(provider=self.llm_provider)
                    llm_analysis = llm_analyzer.analyze(
                        dataset_id=dataset_id,
                        schema_info=schema_info,
                        sample_items=sample_items,
                        sample_count=sample_count,
                    )
                    detected_type = llm_analysis.dataset_type

                    llm_result_dict = {
                        "dataset_type": llm_analysis.dataset_type,
                        "purpose": llm_analysis.purpose,
                        "structure_description": llm_analysis.structure_description,
                        "key_fields": llm_analysis.key_fields,
                        "production_steps": llm_analysis.production_steps,
                        "quality_criteria": llm_analysis.quality_criteria,
                        "estimated_difficulty": llm_analysis.estimated_difficulty,
                        "similar_datasets": llm_analysis.similar_datasets,
                    }
                    with open(
                        output_mgr.get_path("data", "llm_analysis.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(llm_result_dict, f, indent=2, ensure_ascii=False)
                    result.files_generated.append(
                        output_mgr.get_relative_path("data", "llm_analysis.json")
                    )
                except Exception as e:
                    result.warnings.append(f"LLM 数据集分析跳过: {e}")

            result.dataset_type = detected_type

            # Precise token-based API cost calculation
            precise_api_cost = None
            token_stats = None
            try:
                from datarecipe.cost import PreciseCostCalculator

                cost_calc = PreciseCostCalculator()
                precise_estimate = cost_calc.calculate(
                    samples=sample_items,
                    target_size=actual_size,
                    model="gpt-4o",
                    iteration_factor=1.2,
                )
                precise_api_cost = precise_estimate.adjusted_cost
                token_stats = precise_estimate.token_stats

                # Save token analysis to cost/
                with open(
                    output_mgr.get_path("cost", "token_analysis.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(precise_estimate.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("cost", "token_analysis.json")
                )

                # Model comparison
                comparisons = cost_calc.compare_models(
                    samples=sample_items,
                    target_size=actual_size,
                    models=["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "deepseek-v3"],
                )
                comparison_data = {m: e.to_dict() for m, e in comparisons.items()}
                with open(
                    output_mgr.get_path("cost", "cost_comparison.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(comparison_data, f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("cost", "cost_comparison.json")
                )

            except Exception as e:
                result.warnings.append(f"Token 成本计算跳过: {e}")

            # Complexity analysis for dynamic cost adjustment
            complexity_metrics = None
            try:
                from datarecipe.cost import ComplexityAnalyzer

                complexity_analyzer = ComplexityAnalyzer()
                complexity_metrics = complexity_analyzer.analyze(
                    samples=sample_items,
                    schema_info=schema_info,
                    rubrics=rubrics if rubrics else None,
                )

                # Save complexity analysis to data/
                with open(
                    output_mgr.get_path("data", "complexity_analysis.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(complexity_metrics.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("data", "complexity_analysis.json")
                )

            except Exception as e:
                result.warnings.append(f"复杂度分析跳过: {e}")

            # Human-machine allocation
            splitter = HumanMachineSplitter(region=self.region)
            allocation = splitter.analyze(
                dataset_size=actual_size,
                task_types=[
                    TaskType.CONTEXT_CREATION,
                    TaskType.TASK_DESIGN,
                    TaskType.RUBRICS_WRITING,
                    TaskType.DATA_GENERATION,
                    TaskType.QUALITY_REVIEW,
                ],
            )

            # Apply complexity multipliers to human cost
            human_cost = allocation.total_human_cost
            if complexity_metrics:
                human_cost = human_cost * complexity_metrics.cost_multiplier

            # Use precise API cost if available, otherwise use allocation estimate
            api_cost = precise_api_cost if precise_api_cost else allocation.total_machine_cost

            # Calibrate using historical data
            calibration_result = None
            try:
                from datarecipe.cost import CostCalibrator

                calibrator = CostCalibrator()
                calibration_result = calibrator.calibrate(
                    dataset_type=detected_type or "unknown",
                    human_cost=human_cost,
                    api_cost=api_cost,
                    complexity_metrics=complexity_metrics,
                    sample_count=sample_count,
                )

                # Use calibrated costs
                human_cost = calibration_result.calibrated_human_cost
                api_cost = calibration_result.calibrated_api_cost

                # Save calibration analysis to cost/
                with open(
                    output_mgr.get_path("cost", "cost_calibration.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(calibration_result.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("cost", "cost_calibration.json")
                )

            except Exception as e:
                result.warnings.append(f"成本校准跳过: {e}")

            total_cost = human_cost + api_cost

            # Phased cost breakdown
            phased_breakdown = None
            try:
                from datarecipe.cost import PhasedCostModel

                phased_model = PhasedCostModel(region=self.region)

                # Calculate API cost per sample for phased model
                api_per_sample = api_cost / actual_size if actual_size > 0 else 0.01
                complexity_mult = complexity_metrics.cost_multiplier if complexity_metrics else 1.0
                quality_req = (
                    complexity_metrics.quality_requirement if complexity_metrics else "standard"
                )

                phased_breakdown = phased_model.calculate(
                    target_size=actual_size,
                    dataset_type=detected_type or "unknown",
                    human_percentage=allocation.human_work_percentage,
                    api_cost_per_sample=api_per_sample,
                    complexity_multiplier=complexity_mult,
                    quality_requirement=quality_req,
                )

                # Save phased cost analysis to cost/
                with open(
                    output_mgr.get_path("cost", "phased_cost.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(phased_breakdown.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("cost", "phased_cost.json")
                )

                # Save phased cost report to cost/
                phased_report = phased_model.format_report(phased_breakdown)
                with open(
                    output_mgr.get_path("cost", "COST_BREAKDOWN.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(phased_report)
                result.files_generated.append(
                    output_mgr.get_relative_path("cost", "COST_BREAKDOWN.md")
                )

            except Exception as e:
                result.warnings.append(f"分阶段成本分析跳过: {e}")

            result.reproduction_cost = {
                "human": round(human_cost, 2),
                "api": round(api_cost, 2),
                "total": round(total_cost, 2),
            }

            # Add phased total if available (includes contingency)
            if phased_breakdown:
                result.reproduction_cost["phased_total"] = round(phased_breakdown.grand_total, 2)

            result.human_percentage = round(
                human_cost / total_cost * 100 if total_cost > 0 else 0, 1
            )

            # Add analysis details to allocation output
            allocation_dict = splitter.to_dict(allocation)
            if token_stats:
                allocation_dict["token_analysis"] = token_stats.to_dict()
            if precise_api_cost:
                allocation_dict["precise_api_cost"] = round(precise_api_cost, 2)
            if complexity_metrics:
                allocation_dict["complexity"] = {
                    "domain": complexity_metrics.primary_domain.value,
                    "difficulty_score": complexity_metrics.difficulty_score,
                    "time_multiplier": complexity_metrics.time_multiplier,
                    "cost_multiplier": complexity_metrics.cost_multiplier,
                }
            if calibration_result:
                allocation_dict["calibration"] = {
                    "method": calibration_result.calibration_method,
                    "confidence": calibration_result.confidence,
                    "based_on": calibration_result.based_on_datasets,
                    "range": {
                        "low": round(calibration_result.cost_range_low, 2),
                        "high": round(calibration_result.cost_range_high, 2),
                    },
                }

            # Final adjusted costs
            allocation_dict["final_costs"] = {
                "human": round(human_cost, 2),
                "api": round(api_cost, 2),
                "total": round(total_cost, 2),
            }

            # Save allocation to cost/
            with open(output_mgr.get_path("cost", "allocation.json"), "w", encoding="utf-8") as f:
                json.dump(allocation_dict, f, indent=2, ensure_ascii=False)
            result.files_generated.append(output_mgr.get_relative_path("cost", "allocation.json"))

            # LLM Enhancement Layer (optional, generates rich context for all reports)
            enhanced_context = None
            if self.use_llm:
                try:
                    from datarecipe.generators.llm_enhancer import LLMEnhancer

                    enhancer = LLMEnhancer(mode=self.enhance_mode, provider=self.llm_provider)
                    enhanced_context = enhancer.enhance(
                        dataset_id=dataset_id,
                        dataset_type=detected_type or "unknown",
                        schema_info=schema_info,
                        sample_items=sample_items,
                        sample_count=sample_count,
                        complexity_metrics=complexity_metrics,
                        allocation=allocation,
                        rubrics_result=rubrics_result,
                        llm_analysis=llm_analysis,
                    )
                    if enhanced_context and enhanced_context.generated:
                        enhanced_dict = {
                            k: v
                            for k, v in enhanced_context.__dict__.items()
                            if k not in ("raw_response",)
                        }
                        with open(
                            output_mgr.get_path("data", "enhanced_context.json"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(enhanced_dict, f, indent=2, ensure_ascii=False, default=str)
                        result.files_generated.append(
                            output_mgr.get_relative_path("data", "enhanced_context.json")
                        )
                except Exception as e:
                    result.warnings.append(f"LLM 增强跳过: {e}")

            # Generate reports to guide/
            report = self._generate_analysis_report(
                dataset_id,
                sample_count,
                actual_size,
                rubrics_result,
                prompt_library,
                strategy_result,
                allocation,
                self.region,
                enhanced_context=enhanced_context,
            )
            with open(
                output_mgr.get_path("guide", "ANALYSIS_REPORT.md"), "w", encoding="utf-8"
            ) as f:
                f.write(report)
            result.files_generated.append(
                output_mgr.get_relative_path("guide", "ANALYSIS_REPORT.md")
            )

            guide = self._generate_reproduction_guide(
                dataset_id,
                schema_info,
                category_set,
                sub_category_set,
                system_prompts_by_domain,
                rubrics_examples,
                sample_items,
                rubrics_result,
                prompt_library,
                allocation,
                is_preference_dataset,
                preference_pairs,
                preference_topics,
                preference_patterns,
                is_swe_dataset,
                swe_stats,
                llm_analysis,
                enhanced_context=enhanced_context,
            )
            with open(
                output_mgr.get_path("guide", "REPRODUCTION_GUIDE.md"), "w", encoding="utf-8"
            ) as f:
                f.write(guide)
            result.files_generated.append(
                output_mgr.get_relative_path("guide", "REPRODUCTION_GUIDE.md")
            )

            # Annotation specification (forward-looking production guide)
            try:
                from datarecipe.generators.annotation_spec import AnnotationSpecGenerator

                spec_generator = AnnotationSpecGenerator()
                annotation_spec = spec_generator.generate(
                    dataset_id=dataset_id,
                    dataset_type=detected_type or "unknown",
                    schema_info=schema_info,
                    sample_items=sample_items,
                    rubrics_result=rubrics_result,
                    llm_analysis=llm_analysis,
                    complexity_metrics=complexity_metrics,
                    enhanced_context=enhanced_context,
                )

                # Save as Markdown to annotation/
                spec_md = spec_generator.to_markdown(annotation_spec)
                with open(
                    output_mgr.get_path("annotation", "ANNOTATION_SPEC.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(spec_md)
                result.files_generated.append(
                    output_mgr.get_relative_path("annotation", "ANNOTATION_SPEC.md")
                )

                # Save as JSON to annotation/
                spec_dict = spec_generator.to_dict(annotation_spec)
                with open(
                    output_mgr.get_path("annotation", "annotation_spec.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(spec_dict, f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("annotation", "annotation_spec.json")
                )

            except Exception as e:
                result.warnings.append(f"标注规范生成失败: {e}")

            # Milestone plan (for project management)
            try:
                from datarecipe.generators.milestone_plan import MilestonePlanGenerator

                milestone_generator = MilestonePlanGenerator()
                milestone_plan = milestone_generator.generate(
                    dataset_id=dataset_id,
                    dataset_type=detected_type or "unknown",
                    target_size=actual_size,
                    reproduction_cost=result.reproduction_cost,
                    human_percentage=result.human_percentage,
                    complexity_metrics=complexity_metrics,
                    phased_breakdown=phased_breakdown,
                    enhanced_context=enhanced_context,
                )

                # Save as Markdown to project/
                milestone_md = milestone_generator.to_markdown(milestone_plan)
                with open(
                    output_mgr.get_path("project", "MILESTONE_PLAN.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(milestone_md)
                result.files_generated.append(
                    output_mgr.get_relative_path("project", "MILESTONE_PLAN.md")
                )

                # Save as JSON to project/
                milestone_dict = milestone_generator.to_dict(milestone_plan)
                with open(
                    output_mgr.get_path("project", "milestone_plan.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(milestone_dict, f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("project", "milestone_plan.json")
                )

            except Exception as e:
                result.warnings.append(f"里程碑计划生成失败: {e}")

            # Executive summary (for decision makers)
            try:
                from datarecipe.generators.executive_summary import ExecutiveSummaryGenerator

                exec_generator = ExecutiveSummaryGenerator()
                exec_assessment = exec_generator.generate(
                    dataset_id=dataset_id,
                    dataset_type=detected_type or "unknown",
                    sample_count=sample_count,
                    reproduction_cost=result.reproduction_cost,
                    human_percentage=result.human_percentage,
                    complexity_metrics=complexity_metrics,
                    phased_breakdown=phased_breakdown,
                    llm_analysis=llm_analysis,
                    enhanced_context=enhanced_context,
                )

                # Save as Markdown to decision/
                exec_md = exec_generator.to_markdown(
                    assessment=exec_assessment,
                    dataset_id=dataset_id,
                    dataset_type=detected_type or "unknown",
                    reproduction_cost=result.reproduction_cost,
                    phased_breakdown=phased_breakdown,
                )
                with open(
                    output_mgr.get_path("decision", "EXECUTIVE_SUMMARY.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(exec_md)
                result.files_generated.append(
                    output_mgr.get_relative_path("decision", "EXECUTIVE_SUMMARY.md")
                )

                # Save as JSON to decision/
                exec_dict = exec_generator.to_dict(exec_assessment)
                with open(
                    output_mgr.get_path("decision", "executive_summary.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(exec_dict, f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("decision", "executive_summary.json")
                )

            except Exception as e:
                result.warnings.append(f"执行摘要生成失败: {e}")

            # Industry benchmark comparison
            try:
                from datarecipe.generators.industry_benchmark import IndustryBenchmarkGenerator

                benchmark_generator = IndustryBenchmarkGenerator()
                benchmark_comparison = benchmark_generator.generate(
                    dataset_id=dataset_id,
                    dataset_type=detected_type or "unknown",
                    sample_count=actual_size,
                    reproduction_cost=result.reproduction_cost,
                    human_percentage=result.human_percentage,
                )

                # Save as Markdown to project/
                benchmark_md = benchmark_generator.to_markdown(benchmark_comparison)
                with open(
                    output_mgr.get_path("project", "INDUSTRY_BENCHMARK.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(benchmark_md)
                result.files_generated.append(
                    output_mgr.get_relative_path("project", "INDUSTRY_BENCHMARK.md")
                )

                # Save as JSON to project/
                benchmark_dict = benchmark_generator.to_dict(benchmark_comparison)
                with open(
                    output_mgr.get_path("project", "industry_benchmark.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(benchmark_dict, f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("project", "industry_benchmark.json")
                )

            except Exception as e:
                result.warnings.append(f"行业基准对比生成失败: {e}")

            # Recipe summary (stays in root)
            summary = RadarIntegration.create_summary(
                dataset_id=dataset_id,
                dataset_type=detected_type,
                purpose=llm_analysis.purpose if llm_analysis else "",
                allocation=allocation,
                rubrics_result=rubrics_result,
                prompt_library=prompt_library,
                schema_info=schema_info,
                sample_count=sample_count,
                llm_analysis=llm_analysis,
                output_dir=dataset_output_dir,
                complexity_metrics=complexity_metrics,
            )
            RadarIntegration.save_summary(summary, dataset_output_dir)
            result.files_generated.append("recipe_summary.json")

            # Generate README.md for directory navigation
            readme_content = output_mgr.generate_readme(dataset_id, detected_type or "unknown")
            with open(os.path.join(dataset_output_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme_content)
            result.files_generated.append("README.md")

            # Generate AI Agent layer
            try:
                self._generate_ai_agent_layer(
                    output_mgr=output_mgr,
                    result=result,
                    dataset_id=dataset_id,
                    dataset_type=detected_type or "unknown",
                    sample_count=sample_count,
                    actual_size=actual_size,
                    allocation=allocation,
                    complexity_metrics=complexity_metrics,
                    rubrics_result=rubrics_result,
                    prompt_library=prompt_library,
                    llm_analysis=llm_analysis,
                    is_preference_dataset=is_preference_dataset,
                    is_swe_dataset=is_swe_dataset,
                )
            except Exception as e:
                result.warnings.append(f"AI Agent 层生成失败: {e}")

            # Update knowledge base
            try:
                from datarecipe.knowledge import KnowledgeBase

                kb = KnowledgeBase()
                kb.ingest_analysis(
                    dataset_id=dataset_id,
                    summary=summary,
                    rubrics_result=rubrics_result,
                    prompt_library=prompt_library,
                )
            except Exception:
                pass  # Knowledge base is optional

            # Update cache
            try:
                from datarecipe.cache import AnalysisCache

                cache = AnalysisCache()
                cache.put(
                    dataset_id=dataset_id,
                    output_dir=dataset_output_dir,
                    dataset_type=detected_type,
                    sample_count=sample_count,
                )
            except Exception:
                pass  # Cache is optional

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _analyze_preference_pair(self, item, pairs, topics, patterns):
        """Analyze a preference pair item."""
        import re

        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        if not isinstance(chosen, str) or not isinstance(rejected, str):
            return

        # Parse conversation
        def parse_conv(text):
            turns = []
            for h_pat, a_pat in [
                (r"\n\nHuman:", r"\n\nAssistant:"),
                (r"\nHuman:", r"\nAssistant:"),
            ]:
                if h_pat.replace(r"\n", "\n") in text:
                    parts = re.split(r"(" + h_pat + "|" + a_pat + ")", text)
                    role, content = None, ""
                    for part in parts:
                        if re.match(h_pat, part):
                            if role and content:
                                turns.append({"role": role, "content": content.strip()})
                            role, content = "human", ""
                        elif re.match(a_pat, part):
                            if role and content:
                                turns.append({"role": role, "content": content.strip()})
                            role, content = "assistant", ""
                        else:
                            content += part
                    if role and content:
                        turns.append({"role": role, "content": content.strip()})
                    break
            return turns

        chosen_turns = parse_conv(chosen)

        # Topic classification
        topic = "general"
        for turn in chosen_turns:
            if turn.get("role") == "human":
                t = turn.get("content", "")[:100].lower()
                if any(w in t for w in ["code", "program", "python", "function"]):
                    topic = "coding"
                elif any(w in t for w in ["write", "story", "poem", "essay"]):
                    topic = "creative_writing"
                elif any(w in t for w in ["explain", "what is", "how does"]):
                    topic = "explanation"
                elif any(w in t for w in ["help", "advice", "suggest"]):
                    topic = "advice"
                elif any(w in t for w in ["translate", "chinese", "spanish"]):
                    topic = "translation"
                break

        topics[topic] = topics.get(topic, 0) + 1

        # Length patterns
        if len(chosen) > len(rejected) * 1.2:
            patterns["chosen_longer"] += 1
        elif len(rejected) > len(chosen) * 1.2:
            patterns["rejected_longer"] += 1
        else:
            patterns["same_length"] += 1

        # Safety patterns
        safety_words = ["sorry", "can't", "cannot", "won't", "inappropriate"]
        if any(w in rejected.lower() for w in safety_words) and not any(
            w in chosen.lower() for w in safety_words
        ):
            patterns["chosen_safer"] += 1

        # Save example
        if len(pairs) < 20:
            pairs.append(
                {
                    "topic": topic,
                    "turn_count": len(chosen_turns),
                    "human_query": chosen_turns[0].get("content", "")[:300] if chosen_turns else "",
                }
            )

    def _analyze_swe_item(self, item, stats):
        """Analyze a SWE-bench style item."""
        import ast

        repo = item.get("repo", "unknown")
        stats["repos"][repo] = stats["repos"].get(repo, 0) + 1

        lang = item.get("repo_language", "unknown")
        stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

        # Issue types
        issue_spec = item.get("issue_specificity", "")
        if isinstance(issue_spec, str) and issue_spec.startswith("["):
            try:
                types = ast.literal_eval(issue_spec)
                for t in types:
                    stats["issue_types"][t] = stats["issue_types"].get(t, 0) + 1
            except Exception:
                pass

        # Patch lines
        patch = item.get("patch", "")
        if isinstance(patch, str):
            lines = len([l for l in patch.split("\n") if l.startswith("+") or l.startswith("-")])
            stats["patch_lines"].append(lines)

        # Examples
        if len(stats["examples"]) < 5:
            stats["examples"].append(
                {
                    "repo": repo,
                    "language": lang,
                    "problem_statement": item.get("problem_statement", "")[:800],
                }
            )

    def _generate_analysis_report(
        self,
        dataset_id,
        sample_count,
        actual_size,
        rubrics_result,
        prompt_library,
        strategy_result,
        allocation,
        region,
        enhanced_context=None,
    ) -> str:
        """Generate analysis report markdown."""
        lines = []
        lines.append(f"# 🔬 {dataset_id} 深度逆向分析报告")
        lines.append("")
        lines.append(f"> **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> **数据集**: {dataset_id}")
        lines.append(f"> **分析样本**: {sample_count} 条")
        lines.append(f"> **目标规模**: {actual_size:,} 条")
        lines.append("")

        # LLM-enhanced purpose summary
        if (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.dataset_purpose_summary
        ):
            lines.append(f"> {enhanced_context.dataset_purpose_summary}")
            lines.append("")

        lines.append("---")
        lines.append("")

        lines.append("## 📊 执行摘要")
        lines.append("")
        lines.append("| 维度 | 发现 |")
        lines.append("|------|------|")

        if rubrics_result:
            lines.append(
                f"| **评分标准** | {rubrics_result.total_rubrics:,} 条，{rubrics_result.unique_patterns:,} 种独特模式 |"
            )
        if prompt_library:
            lines.append(
                f"| **Prompt模板** | {prompt_library.unique_count} 个去重后的系统提示模板 |"
            )
        if strategy_result:
            lines.append(
                f"| **数据来源** | 混合策略（合成 {strategy_result.synthetic_score * 100:.0f}% + 改编 {strategy_result.modified_score * 100:.0f}%） |"
            )

        lines.append(
            f"| **复现成本** | 约 ${allocation.total_cost:,.0f}（人工 ${allocation.total_human_cost:,.0f} + API ${allocation.total_machine_cost:,.0f}） |"
        )
        lines.append(
            f"| **人机分配** | 人工 {allocation.human_work_percentage:.0f}%，机器 {allocation.machine_work_percentage:.0f}% |"
        )
        lines.append("")

        # LLM-enhanced methodology insights
        if (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.key_methodology_insights
        ):
            lines.append("## 🔍 方法学洞察")
            lines.append("")
            for insight in enhanced_context.key_methodology_insights:
                lines.append(f"- {insight}")
            lines.append("")

        # LLM-enhanced competitive positioning
        if (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.competitive_positioning
        ):
            lines.append("## 🏆 竞争定位")
            lines.append("")
            lines.append(enhanced_context.competitive_positioning)
            lines.append("")

        # LLM-enhanced domain tips
        if (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.domain_specific_tips
        ):
            lines.append("## 💡 领域建议")
            lines.append("")
            for tip in enhanced_context.domain_specific_tips:
                lines.append(f"- {tip}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 报告由 DataRecipe 自动生成")

        return "\n".join(lines)

    def _generate_reproduction_guide(
        self,
        dataset_id,
        schema_info,
        category_set,
        sub_category_set,
        system_prompts_by_domain,
        rubrics_examples,
        sample_items,
        rubrics_result,
        prompt_library,
        allocation,
        is_preference_dataset,
        preference_pairs,
        preference_topics,
        preference_patterns,
        is_swe_dataset,
        swe_stats,
        llm_analysis,
        enhanced_context=None,
    ) -> str:
        """Generate reproduction guide markdown."""
        lines = []
        lines.append(f"# 📋 {dataset_id} 复刻指南")
        lines.append("")

        if is_swe_dataset:
            lines.append("> **这是一个软件工程评测数据集 (SWE-bench 风格)。**")
        elif is_preference_dataset:
            lines.append("> **这是一个 RLHF 偏好数据集。**")
        elif (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.dataset_purpose_summary
        ):
            lines.append(f"> {enhanced_context.dataset_purpose_summary}")
        elif llm_analysis and llm_analysis.dataset_type != "unknown":
            lines.append(f"> **数据集类型: {llm_analysis.dataset_type}。{llm_analysis.purpose}**")
        else:
            lines.append("> **本指南提供可直接操作的模板和规范。**")
        lines.append("")
        lines.append("---")
        lines.append("")

        # LLM-enhanced reproduction strategy
        if (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.reproduction_strategy
        ):
            lines.append("## 🎯 复刻策略")
            lines.append("")
            lines.append(enhanced_context.reproduction_strategy)
            lines.append("")

        # LLM-enhanced methodology insights
        if (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.key_methodology_insights
        ):
            lines.append("## 🔍 方法学洞察")
            lines.append("")
            for insight in enhanced_context.key_methodology_insights:
                lines.append(f"- {insight}")
            lines.append("")

        # LLM analysis section (for unknown types analyzed by LLM)
        if llm_analysis and llm_analysis.dataset_type != "unknown":
            from datarecipe.analyzers.llm_dataset_analyzer import generate_llm_guide_section

            lines.append(generate_llm_guide_section(llm_analysis))
            lines.append("")

        # Schema section
        lines.append("## 📐 数据结构规范 (Schema)")
        lines.append("")
        lines.append("| 字段名 | 类型 | 说明 |")
        lines.append("|--------|------|------|")
        for fld, info in schema_info.items():
            lines.append(f"| `{fld}` | `{info['type']}` | — |")
        lines.append("")

        # Cost section
        if allocation:
            lines.append("## 💰 成本估算")
            lines.append("")
            lines.append(f"- **人工成本**: ${allocation.total_human_cost:,.0f}")
            lines.append(f"- **API 成本**: ${allocation.total_machine_cost:,.0f}")
            lines.append(f"- **总计**: ${allocation.total_cost:,.0f}")
            lines.append(f"- **人工占比**: {allocation.human_work_percentage:.0f}%")
            lines.append("")

        # LLM-enhanced domain tips
        if (
            enhanced_context
            and enhanced_context.generated
            and enhanced_context.domain_specific_tips
        ):
            lines.append("## 💡 领域建议")
            lines.append("")
            for tip in enhanced_context.domain_specific_tips:
                lines.append(f"- {tip}")
            lines.append("")

        # LLM-enhanced risks
        if enhanced_context and enhanced_context.generated and enhanced_context.tailored_risks:
            lines.append("## ⚠️ 风险提示")
            lines.append("")
            lines.append("| 等级 | 风险 | 缓解措施 |")
            lines.append("|------|------|----------|")
            for risk in enhanced_context.tailored_risks:
                if isinstance(risk, dict):
                    lines.append(
                        f"| {risk.get('level', '')} | {risk.get('description', '')} | {risk.get('mitigation', '')} |"
                    )
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 指南由 DataRecipe 自动生成")

        return "\n".join(lines)

    def _generate_ai_agent_layer(
        self,
        output_mgr: "OutputManager",
        result: AnalysisResult,
        dataset_id: str,
        dataset_type: str,
        sample_count: int,
        actual_size: int,
        allocation: Any,
        complexity_metrics: Any,
        rubrics_result: Any,
        prompt_library: Any,
        llm_analysis: Any,
        is_preference_dataset: bool,
        is_swe_dataset: bool,
    ):
        """Generate AI Agent layer files."""
        subdirs = OUTPUT_SUBDIRS

        # Generate agent_context.json
        self._generate_ai_agent_context(
            output_mgr,
            result,
            dataset_id,
            dataset_type,
            sample_count,
            actual_size,
            allocation,
            complexity_metrics,
            subdirs,
        )

        # Generate workflow_state.json
        self._generate_ai_workflow_state(output_mgr, result, dataset_id, dataset_type, subdirs)

        # Generate reasoning_traces.json
        self._generate_ai_reasoning_traces(
            output_mgr,
            result,
            dataset_id,
            dataset_type,
            actual_size,
            allocation,
            complexity_metrics,
            rubrics_result,
            prompt_library,
            subdirs,
        )

        # Generate pipeline.yaml
        self._generate_ai_pipeline(
            output_mgr,
            result,
            dataset_id,
            dataset_type,
            is_preference_dataset,
            is_swe_dataset,
            subdirs,
        )

        # Generate README.md for AI Agent directory
        self._generate_ai_agent_readme(output_mgr, result, dataset_id, dataset_type, subdirs)

    def _generate_ai_agent_context(
        self,
        output_mgr: "OutputManager",
        result: AnalysisResult,
        dataset_id: str,
        dataset_type: str,
        sample_count: int,
        actual_size: int,
        allocation: Any,
        complexity_metrics: Any,
        subdirs: dict,
    ):
        """Generate agent_context.json - aggregated entry point for AI agents."""
        context = {
            "_meta": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "generator": "DataRecipe",
                "purpose": "AI Agent 聚合入口，引用其他文件而非复制",
            },
            "project": {
                "name": dataset_id,
                "type": dataset_type or "unknown",
                "source": "huggingface",
                "sample_count": sample_count,
                "target_size": actual_size,
            },
            "summary": {
                "total_cost": result.reproduction_cost.get("total", 0),
                "human_cost": result.reproduction_cost.get("human", 0),
                "api_cost": result.reproduction_cost.get("api", 0),
                "human_percentage": result.human_percentage,
                "rubric_patterns": result.rubric_patterns,
                "prompt_templates": result.prompt_templates,
                "field_count": len(result.fields),
            },
            "key_decisions": [
                {
                    "decision": "dataset_type",
                    "value": dataset_type or "unknown",
                    "reasoning_ref": "#/reasoning/dataset_type",
                },
                {
                    "decision": "human_percentage",
                    "value": result.human_percentage,
                    "reasoning_ref": "#/reasoning/human_percentage",
                },
                {
                    "decision": "cost_estimate",
                    "value": result.reproduction_cost.get("total", 0),
                    "reasoning_ref": "#/reasoning/cost",
                },
            ],
            "complexity": None,
            "file_references": {
                "executive_summary": f"../{subdirs['decision']}/EXECUTIVE_SUMMARY.md",
                "milestone_plan": f"../{subdirs['project']}/MILESTONE_PLAN.md",
                "annotation_spec": f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                "reproduction_guide": f"../{subdirs['guide']}/REPRODUCTION_GUIDE.md",
                "cost_breakdown": f"../{subdirs['cost']}/COST_BREAKDOWN.md",
                "allocation": f"../{subdirs['cost']}/allocation.json",
                "recipe_summary": "../recipe_summary.json",
            },
            "quick_actions": [
                {
                    "action": "review_spec",
                    "description": "审核标注规范",
                    "file": f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                    "assignee": "human",
                },
                {
                    "action": "review_cost",
                    "description": "审核成本估算",
                    "file": f"../{subdirs['cost']}/COST_BREAKDOWN.md",
                    "assignee": "human",
                },
                {
                    "action": "start_reproduction",
                    "description": "开始复刻生产",
                    "file": f"../{subdirs['guide']}/REPRODUCTION_GUIDE.md",
                    "assignee": "human",
                },
            ],
        }

        # Add complexity info if available
        if complexity_metrics:
            context["complexity"] = {
                "domain": complexity_metrics.primary_domain.value
                if hasattr(complexity_metrics.primary_domain, "value")
                else str(complexity_metrics.primary_domain),
                "difficulty_score": complexity_metrics.difficulty_score,
                "time_multiplier": complexity_metrics.time_multiplier,
                "cost_multiplier": complexity_metrics.cost_multiplier,
            }

        path = output_mgr.get_path("ai_agent", "agent_context.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)
        result.files_generated.append(
            output_mgr.get_relative_path("ai_agent", "agent_context.json")
        )

    def _generate_ai_workflow_state(
        self,
        output_mgr: "OutputManager",
        result: AnalysisResult,
        dataset_id: str,
        dataset_type: str,
        subdirs: dict,
    ):
        """Generate workflow_state.json - workflow state tracking."""
        state = {
            "_meta": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "purpose": "工作流状态追踪，供 AI Agent 了解当前进度和下一步",
            },
            "current_phase": "analysis_complete",
            "phases": {
                "data_loading": {
                    "status": "completed",
                    "description": "数据集加载",
                },
                "analysis": {
                    "status": "completed",
                    "description": "深度逆向分析",
                    "outputs": [
                        f"../{subdirs['data']}/complexity_analysis.json",
                        f"../{subdirs['cost']}/allocation.json",
                    ],
                },
                "report_generation": {
                    "status": "completed",
                    "description": "报告生成",
                    "outputs": [
                        f"../{subdirs['decision']}/EXECUTIVE_SUMMARY.md",
                        f"../{subdirs['project']}/MILESTONE_PLAN.md",
                        f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                        f"../{subdirs['guide']}/REPRODUCTION_GUIDE.md",
                    ],
                },
                "review": {
                    "status": "pending",
                    "description": "人工审核分析结果",
                    "blocked_by": [],
                    "assignee": "human",
                },
                "reproduction_planning": {
                    "status": "pending",
                    "description": "制定复刻计划",
                    "blocked_by": ["review"],
                    "assignee": "human",
                },
                "production": {
                    "status": "pending",
                    "description": "开始数据生产",
                    "blocked_by": ["reproduction_planning"],
                    "assignee": "human",
                },
            },
            "next_actions": [
                {
                    "action": "review_executive_summary",
                    "description": "审核执行摘要，确认分析结论",
                    "file": f"../{subdirs['decision']}/EXECUTIVE_SUMMARY.md",
                    "assignee": "human",
                    "priority": "high",
                },
                {
                    "action": "review_cost_estimate",
                    "description": "审核成本估算，确认预算",
                    "file": f"../{subdirs['cost']}/COST_BREAKDOWN.md",
                    "assignee": "human",
                    "priority": "high",
                },
                {
                    "action": "review_annotation_spec",
                    "description": "审核标注规范，准备生产",
                    "file": f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                    "assignee": "human",
                    "priority": "medium",
                },
            ],
            "blockers": [],
            "decisions_needed": [
                {
                    "question": "是否采用此数据集的方法论？",
                    "options": ["approved", "needs_modification", "rejected"],
                    "impact": "影响后续复刻策略",
                },
                {
                    "question": "成本预算是否可接受？",
                    "options": ["approved", "needs_adjustment"],
                    "impact": "影响项目规模和时间线",
                },
            ],
        }

        path = output_mgr.get_path("ai_agent", "workflow_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        result.files_generated.append(
            output_mgr.get_relative_path("ai_agent", "workflow_state.json")
        )

    def _generate_ai_reasoning_traces(
        self,
        output_mgr: "OutputManager",
        result: AnalysisResult,
        dataset_id: str,
        dataset_type: str,
        actual_size: int,
        allocation: Any,
        complexity_metrics: Any,
        rubrics_result: Any,
        prompt_library: Any,
        subdirs: dict,
    ):
        """Generate reasoning_traces.json - reasoning chains for all conclusions."""
        total_cost = result.reproduction_cost.get("total", 0)
        human_cost = result.reproduction_cost.get("human", 0)
        api_cost = result.reproduction_cost.get("api", 0)

        traces = {
            "_meta": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "purpose": "所有结论的推理链，供人类理解和 AI 验证",
            },
            "reasoning": {
                "dataset_type": {
                    "conclusion": {
                        "value": dataset_type or "unknown",
                        "display": f"数据集类型: {dataset_type or 'unknown'}",
                    },
                    "chain": [],
                    "confidence": 0.0,
                    "human_explanation": "",
                },
                "human_percentage": {
                    "conclusion": {
                        "value": result.human_percentage,
                        "display": f"人工比例: {result.human_percentage}%",
                    },
                    "chain": [],
                    "confidence": 0.0,
                    "human_explanation": "",
                },
                "cost": {
                    "conclusion": {"value": total_cost, "display": f"总成本: ${total_cost:,.0f}"},
                    "chain": [],
                    "confidence": 0.0,
                    "range": {
                        "low": round(total_cost * 0.7, 2),
                        "high": round(total_cost * 1.4, 2),
                    },
                    "human_explanation": "",
                },
            },
        }

        # Build dataset type reasoning chain
        type_chain = []
        type_confidence = 0.5

        if dataset_type == "preference":
            type_chain.append(
                {
                    "step": "检测偏好数据结构",
                    "evidence": "发现 chosen/rejected 字段对",
                    "impact": "判定为 RLHF 偏好数据集",
                }
            )
            type_confidence = 0.95
        elif dataset_type == "evaluation":
            type_chain.append(
                {
                    "step": "检测评分标准",
                    "evidence": f"发现 {result.rubric_patterns} 种评分模式",
                    "impact": "判定为评测数据集",
                }
            )
            type_confidence = 0.9
        elif dataset_type == "swe_bench":
            type_chain.append(
                {
                    "step": "检测 SWE 结构",
                    "evidence": "发现 repo/patch/problem_statement 字段",
                    "impact": "判定为软件工程评测数据集",
                }
            )
            type_confidence = 0.95

        traces["reasoning"]["dataset_type"]["chain"] = type_chain
        traces["reasoning"]["dataset_type"]["confidence"] = type_confidence
        traces["reasoning"]["dataset_type"]["human_explanation"] = (
            f"通过分析数据结构和字段，判定为 {dataset_type or 'unknown'} 类型数据集。"
        )

        # Build human percentage reasoning chain
        human_chain = []
        human_confidence = 0.7

        if allocation:
            human_chain.append(
                {
                    "step": "分析任务类型",
                    "evidence": f"包含 {len(allocation.tasks)} 种任务类型",
                    "impact": f"人工占比 {result.human_percentage}%",
                }
            )
            human_confidence = 0.8

        if complexity_metrics:
            domain = (
                complexity_metrics.primary_domain.value
                if hasattr(complexity_metrics.primary_domain, "value")
                else str(complexity_metrics.primary_domain)
            )
            human_chain.append(
                {
                    "step": "评估复杂度",
                    "evidence": f"领域: {domain}, 难度分数: {complexity_metrics.difficulty_score:.2f}",
                    "impact": f"成本乘数: {complexity_metrics.cost_multiplier:.2f}",
                }
            )
            human_confidence += 0.1

        traces["reasoning"]["human_percentage"]["chain"] = human_chain
        traces["reasoning"]["human_percentage"]["confidence"] = min(human_confidence, 0.95)
        traces["reasoning"]["human_percentage"]["human_explanation"] = (
            f"基于任务分析，预估人工比例为 {result.human_percentage}%。"
        )

        # Build cost reasoning chain
        cost_chain = [
            {
                "step": "计算人工成本",
                "evidence": f"人工任务成本 ${human_cost:,.0f}",
                "value": human_cost,
            },
            {
                "step": "计算 API 成本",
                "evidence": f"API 调用成本 ${api_cost:,.0f}",
                "value": api_cost,
            },
        ]

        if complexity_metrics:
            cost_chain.append(
                {
                    "step": "应用复杂度乘数",
                    "evidence": f"复杂度乘数 {complexity_metrics.cost_multiplier:.2f}",
                    "multiplier": complexity_metrics.cost_multiplier,
                }
            )

        cost_chain.append(
            {
                "step": "计算总成本",
                "evidence": f"人工 ${human_cost:,.0f} + API ${api_cost:,.0f}",
                "result": total_cost,
            }
        )

        traces["reasoning"]["cost"]["chain"] = cost_chain
        traces["reasoning"]["cost"]["confidence"] = 0.75
        traces["reasoning"]["cost"]["human_explanation"] = (
            f"基于任务分解和 Token 分析，预估总成本 ${total_cost:,.0f}，"
            f"置信区间 ${total_cost * 0.7:,.0f} - ${total_cost * 1.4:,.0f}。"
        )

        path = output_mgr.get_path("ai_agent", "reasoning_traces.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(traces, f, indent=2, ensure_ascii=False)
        result.files_generated.append(
            output_mgr.get_relative_path("ai_agent", "reasoning_traces.json")
        )

    def _generate_ai_pipeline(
        self,
        output_mgr: "OutputManager",
        result: AnalysisResult,
        dataset_id: str,
        dataset_type: str,
        is_preference_dataset: bool,
        is_swe_dataset: bool,
        subdirs: dict,
    ):
        """Generate pipeline.yaml - executable pipeline for AI agents."""
        lines = []
        lines.append("# 数据复刻流水线")
        lines.append("# 供 AI Agent 执行的可操作步骤")
        lines.append("")
        lines.append("name: 数据复刻流水线")
        lines.append("version: '1.0'")
        lines.append(f"source_dataset: {dataset_id}")
        lines.append(f"dataset_type: {dataset_type or 'unknown'}")
        lines.append(f"generated_at: {datetime.now().isoformat()}")
        lines.append("")

        # Variables section
        lines.append("variables:")
        lines.append(f'  source_dataset: "{dataset_id}"')
        lines.append("  target_size: 1000  # 可调整")
        lines.append(f"  human_percentage: {result.human_percentage}")
        lines.append(f"  estimated_cost: {result.reproduction_cost.get('total', 0)}")
        lines.append("")

        # Phases
        lines.append("phases:")
        lines.append("")

        # Phase 1: Analysis Review
        lines.append("  - name: analysis_review")
        lines.append("    description: 审核分析结果")
        lines.append("    steps:")
        lines.append("      - action: review_executive_summary")
        lines.append("        description: 审核执行摘要")
        lines.append(f"        input: ../{subdirs['decision']}/EXECUTIVE_SUMMARY.md")
        lines.append("        assignee: human")
        lines.append("        required: true")
        lines.append("")
        lines.append("      - action: review_cost_estimate")
        lines.append("        description: 审核成本估算")
        lines.append(f"        input: ../{subdirs['cost']}/COST_BREAKDOWN.md")
        lines.append("        assignee: human")
        lines.append("")
        lines.append("      - action: approve_methodology")
        lines.append("        description: 确认复刻方法论")
        lines.append(f"        input: ../{subdirs['guide']}/REPRODUCTION_GUIDE.md")
        lines.append("        assignee: human")
        lines.append("        required: true")
        lines.append("")

        # Phase 2: Setup
        lines.append("  - name: setup")
        lines.append("    description: 环境准备")
        lines.append("    depends_on: [analysis_review]")
        lines.append("    steps:")
        lines.append("      - action: setup_annotation_tool")
        lines.append("        description: 配置标注工具")
        lines.append(f"        spec: ../{subdirs['annotation']}/ANNOTATION_SPEC.md")
        lines.append("        assignee: agent")
        lines.append("")
        lines.append("      - action: prepare_rubric_templates")
        lines.append("        description: 准备评分模板")
        lines.append(f"        input: ../{subdirs['annotation']}/rubric_template.yaml")
        lines.append("        assignee: agent")
        lines.append("")

        # Phase 3: Pilot
        lines.append("  - name: pilot")
        lines.append("    description: 试点生产")
        lines.append("    depends_on: [setup]")
        lines.append("    steps:")
        lines.append("      - action: create_pilot_batch")
        lines.append("        description: 创建试点批次 (50 条)")
        lines.append("        count: 50")
        lines.append("        assignee: human")
        lines.append("")
        lines.append("      - action: quality_review_pilot")
        lines.append("        description: 试点质量审核")
        lines.append("        assignee: human")
        lines.append("")

        # Phase 4: Production
        lines.append("  - name: production")
        lines.append("    description: 主体生产")
        lines.append("    depends_on: [pilot]")
        lines.append("    steps:")
        lines.append("      - action: batch_production")
        lines.append("        description: 批量生产")
        lines.append('        count: "{{ target_size }}"')
        lines.append("        assignee: human")
        lines.append("")
        lines.append("      - action: incremental_qa")
        lines.append("        description: 增量质检")
        lines.append("        sample_rate: 0.2")
        lines.append("        assignee: human")
        lines.append("")

        # Phase 5: Final QA
        lines.append("  - name: final_qa")
        lines.append("    description: 最终质量审核")
        lines.append("    depends_on: [production]")
        lines.append("    steps:")
        lines.append("      - action: full_qa_review")
        lines.append("        description: 全量质检")
        lines.append("        assignee: human")
        lines.append("")
        lines.append("      - action: generate_qa_report")
        lines.append("        description: 生成质检报告")
        lines.append("        assignee: agent")
        lines.append("")
        lines.append("      - action: final_approval")
        lines.append("        description: 最终审批")
        lines.append("        assignee: human")
        lines.append("        required: true")
        lines.append("")

        # Error handling
        lines.append("error_handling:")
        lines.append("  on_qa_failure:")
        lines.append("    action: flag_for_revision")
        lines.append("    notify: human")
        lines.append("  on_budget_exceeded:")
        lines.append("    action: pause_and_review")
        lines.append("    notify: human")

        path = output_mgr.get_path("ai_agent", "pipeline.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(output_mgr.get_relative_path("ai_agent", "pipeline.yaml"))

    def _generate_ai_agent_readme(
        self,
        output_mgr: "OutputManager",
        result: AnalysisResult,
        dataset_id: str,
        dataset_type: str,
        subdirs: dict,
    ):
        """Generate README.md for AI Agent directory."""
        lines = []
        lines.append(f"# {dataset_id} - AI Agent 入口")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {dataset_type or 'unknown'}")
        lines.append("")
        lines.append("本目录包含供 AI Agent 消费的结构化数据，与人类可读的 Markdown 文档互补。")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## 文件说明")
        lines.append("")
        lines.append("| 文件 | 用途 | 消费者 |")
        lines.append("|------|------|--------|")
        lines.append("| `agent_context.json` | 聚合入口，引用其他文件 | AI Agent |")
        lines.append("| `workflow_state.json` | 工作流状态，当前阶段和下一步 | AI Agent |")
        lines.append("| `reasoning_traces.json` | 推理链，解释每个结论的原因 | AI Agent + 人类 |")
        lines.append("| `pipeline.yaml` | 可执行流水线，定义标准操作步骤 | AI Agent |")
        lines.append("")

        lines.append("## 快速开始")
        lines.append("")
        lines.append("### 1. 获取项目上下文")
        lines.append("")
        lines.append("```python")
        lines.append("import json")
        lines.append("")
        lines.append("with open('agent_context.json') as f:")
        lines.append("    context = json.load(f)")
        lines.append("")
        lines.append("print(f\"数据集: {context['project']['name']}\")")
        lines.append("print(f\"类型: {context['project']['type']}\")")
        lines.append("print(f\"总成本: ${context['summary']['total_cost']}\")")
        lines.append("```")
        lines.append("")

        lines.append("### 2. 检查工作流状态")
        lines.append("")
        lines.append("```python")
        lines.append("with open('workflow_state.json') as f:")
        lines.append("    state = json.load(f)")
        lines.append("")
        lines.append("print(f\"当前阶段: {state['current_phase']}\")")
        lines.append("for action in state['next_actions']:")
        lines.append("    print(f\"下一步: {action['description']} ({action['assignee']})\")")
        lines.append("```")
        lines.append("")

        lines.append("### 3. 理解决策推理")
        lines.append("")
        lines.append("```python")
        lines.append("with open('reasoning_traces.json') as f:")
        lines.append("    traces = json.load(f)")
        lines.append("")
        lines.append("cost = traces['reasoning']['cost']")
        lines.append("print(f\"成本: {cost['conclusion']['display']}\")")
        lines.append("print(f\"置信度: {cost['confidence']}\")")
        lines.append("print(f\"原因: {cost['human_explanation']}\")")
        lines.append("```")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 由 DataRecipe 自动生成")

        path = output_mgr.get_path("ai_agent", "README.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(output_mgr.get_relative_path("ai_agent", "README.md"))
