"""Core deep analysis functionality shared between CLI and MCP."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from datarecipe.core.project_layout import (
    DEFAULT_PROJECTS_DIR,
    OUTPUT_SUBDIRS,
    OutputManager,
    ProjectManifest,
)
from datarecipe.core.project_layout import safe_name as _safe_name


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

    # LLM enhancement prompt (for MCP two-step workflow)
    enhancement_prompt: str = ""

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
        output_dir: str = DEFAULT_PROJECTS_DIR,
        region: str = "china",
        use_llm: bool = False,
        llm_provider: str = "anthropic",
        enhance_mode: str = "auto",
        pre_enhanced_context=None,
    ):
        self.output_dir = output_dir
        self.region = region
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.enhance_mode = enhance_mode
        self.pre_enhanced_context = pre_enhanced_context

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
            from pathlib import Path as _Path

            from datasets import load_dataset

            from datarecipe.analyzers import ContextStrategyDetector
            from datarecipe.extractors import PromptExtractor, RubricsAnalyzer
            from datarecipe.generators import HumanMachineSplitter, TaskType
            from datarecipe.integrations.radar import RadarIntegration

            # Detect local file vs HuggingFace dataset
            _local_path = _Path(dataset_id)
            _is_local = _local_path.exists() and _local_path.is_file()

            # Create output directory with organized structure
            _safe_id = _local_path.stem if _is_local else dataset_id
            dataset_output_dir = os.path.join(self.output_dir, _safe_name(_safe_id))
            output_mgr = OutputManager(
                dataset_output_dir,
                subdirs=["decision", "project", "annotation", "guide", "cost", "data", "ai_agent", "samples"],
            )
            result.output_dir = dataset_output_dir

            if _is_local:
                # Local file: load directly with datasets library
                from datarecipe.sources.local import detect_format

                _fmt = detect_format(_local_path)
                ds = load_dataset(
                    _fmt, data_files=str(_local_path), split="train", streaming=True
                )
                split = "train"
            else:
                # HuggingFace dataset: auto-detect config and split
                def _try_load(ds_id, config=None, split_name=None):
                    kwargs = {"streaming": True}
                    if config:
                        kwargs["name"] = config
                    if split_name:
                        kwargs["split"] = split_name
                    return load_dataset(ds_id, **kwargs)

                # Detect available configs
                detected_config = None
                try:
                    from datasets import get_dataset_config_names
                    configs = get_dataset_config_names(dataset_id)
                    if configs and len(configs) > 0:
                        # Prefer 'default' or the first config
                        detected_config = "default" if "default" in configs else configs[0]
                except Exception:
                    pass

                if split is None:
                    try:
                        ds = _try_load(dataset_id, config=detected_config, split_name="train")
                        split = "train"
                    except (ValueError, Exception):
                        for try_split in ["test", "validation", "dev"]:
                            try:
                                ds = _try_load(dataset_id, config=detected_config, split_name=try_split)
                                split = try_split
                                break
                            except (ValueError, Exception):
                                continue
                        else:
                            # Fallback: discover available splits and use the first one
                            try:
                                from datasets import get_dataset_split_names
                                available = get_dataset_split_names(
                                    dataset_id, config_name=detected_config,
                                )
                                if available:
                                    ds = _try_load(dataset_id, config=detected_config, split_name=available[0])
                                    split = available[0]
                                else:
                                    raise ValueError("Cannot find available split")
                            except Exception:
                                raise ValueError("Cannot find available split")
                else:
                    ds = _try_load(dataset_id, config=detected_config, split_name=split)

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

            # === Tier 1: Independent analyzers (parallel) ===
            def _analyze_rubrics():
                files, rr = [], None
                if rubrics:
                    analyzer = RubricsAnalyzer()
                    rr = analyzer.analyze(rubrics, task_count=sample_count)
                    with open(output_mgr.get_path("data", "rubrics_analysis.json"), "w", encoding="utf-8") as f:
                        json.dump(analyzer.to_dict(rr), f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("data", "rubrics_analysis.json"))
                    with open(output_mgr.get_path("annotation", "rubric_template.yaml"), "w", encoding="utf-8") as f:
                        f.write(analyzer.to_yaml_templates(rr))
                    files.append(output_mgr.get_relative_path("annotation", "rubric_template.yaml"))
                    with open(output_mgr.get_path("annotation", "rubric_template.md"), "w", encoding="utf-8") as f:
                        f.write(analyzer.to_markdown_templates(rr))
                    files.append(output_mgr.get_relative_path("annotation", "rubric_template.md"))
                return rr, files

            def _analyze_prompts():
                files, pl = [], None
                if messages:
                    extractor = PromptExtractor()
                    pl = extractor.extract(messages)
                    with open(output_mgr.get_path("data", "prompt_templates.json"), "w", encoding="utf-8") as f:
                        json.dump(extractor.to_dict(pl), f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("data", "prompt_templates.json"))
                return pl, files

            def _analyze_context_strategy():
                files, sr = [], None
                if contexts:
                    detector = ContextStrategyDetector()
                    sr = detector.analyze(contexts[:100])
                    with open(output_mgr.get_path("data", "context_strategy.json"), "w", encoding="utf-8") as f:
                        json.dump(detector.to_dict(sr), f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("data", "context_strategy.json"))
                return sr, files

            def _analyze_preference():
                files = []
                if is_preference_dataset and preference_pairs:
                    pref_data = {
                        "is_preference_dataset": True, "total_pairs": sample_count,
                        "topic_distribution": preference_topics,
                        "patterns": preference_patterns, "examples": preference_pairs[:10],
                    }
                    with open(output_mgr.get_path("data", "preference_analysis.json"), "w", encoding="utf-8") as f:
                        json.dump(pref_data, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("data", "preference_analysis.json"))
                return files

            def _analyze_swe():
                files = []
                if is_swe_dataset and swe_stats["repos"]:
                    avg_patch = (
                        sum(swe_stats["patch_lines"]) / len(swe_stats["patch_lines"])
                        if swe_stats["patch_lines"] else 0
                    )
                    swe_data = {
                        "is_swe_dataset": True, "total_tasks": sample_count,
                        "repos_count": len(swe_stats["repos"]),
                        "repo_distribution": dict(sorted(swe_stats["repos"].items(), key=lambda x: -x[1])[:20]),
                        "language_distribution": swe_stats["languages"],
                        "avg_patch_lines": avg_patch, "examples": swe_stats["examples"],
                    }
                    with open(output_mgr.get_path("data", "swe_analysis.json"), "w", encoding="utf-8") as f:
                        json.dump(swe_data, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("data", "swe_analysis.json"))
                return files

            def _analyze_llm():
                files, warnings, la, dt = [], [], None, None
                is_known_type = is_preference_dataset or is_swe_dataset or rubrics or messages
                if self.use_llm and not is_known_type:
                    try:
                        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer
                        llm_analyzer = LLMDatasetAnalyzer(provider=self.llm_provider)
                        la = llm_analyzer.analyze(
                            dataset_id=dataset_id, schema_info=schema_info,
                            sample_items=sample_items, sample_count=sample_count,
                        )
                        dt = la.dataset_type
                        llm_result_dict = {
                            "dataset_type": la.dataset_type, "purpose": la.purpose,
                            "structure_description": la.structure_description,
                            "key_fields": la.key_fields, "production_steps": la.production_steps,
                            "quality_criteria": la.quality_criteria,
                            "estimated_difficulty": la.estimated_difficulty,
                            "similar_datasets": la.similar_datasets,
                        }
                        with open(output_mgr.get_path("data", "llm_analysis.json"), "w", encoding="utf-8") as f:
                            json.dump(llm_result_dict, f, indent=2, ensure_ascii=False)
                        files.append(output_mgr.get_relative_path("data", "llm_analysis.json"))
                    except Exception as e:
                        warnings.append(f"LLM 数据集分析跳过: {e}")
                return la, dt, files, warnings

            def _analyze_cost():
                files, warnings = [], []
                pac, ts, pe = None, None, None
                try:
                    from datarecipe.cost import PreciseCostCalculator
                    cost_calc = PreciseCostCalculator()
                    pe = cost_calc.calculate(
                        samples=sample_items, target_size=actual_size,
                        model="gpt-4o", iteration_factor=1.2,
                    )
                    pac = pe.adjusted_cost
                    ts = pe.token_stats
                    with open(output_mgr.get_path("cost", "token_analysis.json"), "w", encoding="utf-8") as f:
                        json.dump(pe.to_dict(), f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("cost", "token_analysis.json"))
                    comparisons = cost_calc.compare_models(
                        samples=sample_items, target_size=actual_size,
                        models=["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "deepseek-v3"],
                    )
                    comparison_data = {m: e.to_dict() for m, e in comparisons.items()}
                    with open(output_mgr.get_path("cost", "cost_comparison.json"), "w", encoding="utf-8") as f:
                        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("cost", "cost_comparison.json"))
                except Exception as e:
                    warnings.append(f"Token 成本计算跳过: {e}")
                return pac, ts, files, warnings

            def _analyze_allocation():
                s = HumanMachineSplitter(region=self.region)
                alloc = s.analyze(
                    dataset_size=actual_size,
                    task_types=[
                        TaskType.CONTEXT_CREATION, TaskType.TASK_DESIGN,
                        TaskType.RUBRICS_WRITING, TaskType.DATA_GENERATION,
                        TaskType.QUALITY_REVIEW,
                    ],
                )
                return s, alloc

            # Execute Tier 1 analyzers in parallel
            with ThreadPoolExecutor(max_workers=6) as executor:
                f_rubrics = executor.submit(_analyze_rubrics)
                f_prompts = executor.submit(_analyze_prompts)
                f_strategy = executor.submit(_analyze_context_strategy)
                f_pref = executor.submit(_analyze_preference)
                f_swe = executor.submit(_analyze_swe)
                f_llm = executor.submit(_analyze_llm)
                f_cost = executor.submit(_analyze_cost)
                f_alloc = executor.submit(_analyze_allocation)

            # Collect Tier 1 results
            rubrics_result, rubrics_files = f_rubrics.result()
            result.files_generated.extend(rubrics_files)
            if rubrics_result:
                result.rubric_patterns = rubrics_result.unique_patterns

            prompt_library, prompt_files = f_prompts.result()
            result.files_generated.extend(prompt_files)
            if prompt_library:
                result.prompt_templates = prompt_library.unique_count

            strategy_result, strategy_files = f_strategy.result()
            result.files_generated.extend(strategy_files)

            result.files_generated.extend(f_pref.result())
            result.files_generated.extend(f_swe.result())

            llm_analysis, llm_detected_type, llm_files, llm_warnings = f_llm.result()
            result.files_generated.extend(llm_files)
            result.warnings.extend(llm_warnings)
            if llm_detected_type:
                detected_type = llm_detected_type

            precise_api_cost, token_stats, cost_files, cost_warnings = f_cost.result()
            result.files_generated.extend(cost_files)
            result.warnings.extend(cost_warnings)

            splitter, allocation = f_alloc.result()

            result.dataset_type = detected_type

            # === Tier 2: Complexity analysis (depends on rubrics) ===
            complexity_metrics = None
            try:
                from datarecipe.cost import ComplexityAnalyzer

                complexity_analyzer = ComplexityAnalyzer()
                complexity_metrics = complexity_analyzer.analyze(
                    samples=sample_items,
                    schema_info=schema_info,
                    rubrics=rubrics if rubrics else None,
                )

                with open(
                    output_mgr.get_path("data", "complexity_analysis.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(complexity_metrics.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("data", "complexity_analysis.json")
                )

            except Exception as e:
                result.warnings.append(f"复杂度分析跳过: {e}")

            # === Tier 3: Cost calibration + phased breakdown (depend on Tier 1+2) ===
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

                human_cost = calibration_result.calibrated_human_cost
                api_cost = calibration_result.calibrated_api_cost

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

                with open(
                    output_mgr.get_path("cost", "phased_cost.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(phased_breakdown.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append(
                    output_mgr.get_relative_path("cost", "phased_cost.json")
                )

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

            # Priority 1: Use pre-loaded enhanced context (from MCP two-step workflow)
            if self.pre_enhanced_context is not None:
                enhanced_context = self.pre_enhanced_context
                try:
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
                    result.warnings.append(f"保存 enhanced_context 失败: {e}")

            # Priority 2: Call LLM enhancer directly (CLI with API key)
            elif self.use_llm:
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

            # Always generate enhancement prompt (for MCP two-step workflow)
            try:
                from datarecipe.generators.llm_enhancer import LLMEnhancer

                _enhancer = LLMEnhancer()
                result.enhancement_prompt = _enhancer.get_prompt(
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
            except Exception:
                pass  # enhancement prompt is optional

            # Generate reports in parallel (all generators are independent)
            def _gen_analysis_report():
                files, warnings = [], []
                try:
                    report = self._generate_analysis_report(
                        dataset_id, sample_count, actual_size,
                        rubrics_result, prompt_library, strategy_result,
                        allocation, self.region, enhanced_context=enhanced_context,
                    )
                    path = output_mgr.get_path("guide", "ANALYSIS_REPORT.md")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(report)
                    files.append(output_mgr.get_relative_path("guide", "ANALYSIS_REPORT.md"))
                except Exception as e:
                    warnings.append(f"分析报告生成失败: {e}")
                return files, warnings

            def _gen_reproduction_guide():
                files, warnings = [], []
                try:
                    guide = self._generate_reproduction_guide(
                        dataset_id, schema_info, category_set, sub_category_set,
                        system_prompts_by_domain, rubrics_examples, sample_items,
                        rubrics_result, prompt_library, allocation,
                        is_preference_dataset, preference_pairs,
                        preference_topics, preference_patterns,
                        is_swe_dataset, swe_stats, llm_analysis,
                        enhanced_context=enhanced_context,
                    )
                    path = output_mgr.get_path("guide", "REPRODUCTION_GUIDE.md")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(guide)
                    files.append(output_mgr.get_relative_path("guide", "REPRODUCTION_GUIDE.md"))
                except Exception as e:
                    warnings.append(f"复刻指南生成失败: {e}")
                return files, warnings

            def _gen_annotation_spec():
                files, warnings = [], []
                try:
                    from datarecipe.generators.annotation_spec import AnnotationSpecGenerator
                    spec_generator = AnnotationSpecGenerator()
                    annotation_spec = spec_generator.generate(
                        dataset_id=dataset_id, dataset_type=detected_type or "unknown",
                        schema_info=schema_info, sample_items=sample_items,
                        rubrics_result=rubrics_result, llm_analysis=llm_analysis,
                        complexity_metrics=complexity_metrics, enhanced_context=enhanced_context,
                    )
                    spec_md = spec_generator.to_markdown(annotation_spec)
                    with open(output_mgr.get_path("annotation", "ANNOTATION_SPEC.md"), "w", encoding="utf-8") as f:
                        f.write(spec_md)
                    files.append(output_mgr.get_relative_path("annotation", "ANNOTATION_SPEC.md"))
                    spec_dict = spec_generator.to_dict(annotation_spec)
                    with open(output_mgr.get_path("annotation", "annotation_spec.json"), "w", encoding="utf-8") as f:
                        json.dump(spec_dict, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("annotation", "annotation_spec.json"))
                except Exception as e:
                    warnings.append(f"标注规范生成失败: {e}")
                return files, warnings

            def _gen_milestone_plan():
                files, warnings = [], []
                try:
                    from datarecipe.generators.milestone_plan import MilestonePlanGenerator
                    milestone_generator = MilestonePlanGenerator()
                    plan = milestone_generator.generate(
                        dataset_id=dataset_id, dataset_type=detected_type or "unknown",
                        target_size=actual_size, reproduction_cost=result.reproduction_cost,
                        human_percentage=result.human_percentage,
                        complexity_metrics=complexity_metrics,
                        phased_breakdown=phased_breakdown, enhanced_context=enhanced_context,
                    )
                    milestone_md = milestone_generator.to_markdown(plan)
                    with open(output_mgr.get_path("project", "MILESTONE_PLAN.md"), "w", encoding="utf-8") as f:
                        f.write(milestone_md)
                    files.append(output_mgr.get_relative_path("project", "MILESTONE_PLAN.md"))
                    milestone_dict = milestone_generator.to_dict(plan)
                    with open(output_mgr.get_path("project", "milestone_plan.json"), "w", encoding="utf-8") as f:
                        json.dump(milestone_dict, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("project", "milestone_plan.json"))
                except Exception as e:
                    warnings.append(f"里程碑计划生成失败: {e}")
                return files, warnings

            def _gen_executive_summary():
                files, warnings = [], []
                try:
                    from datarecipe.generators.executive_summary import ExecutiveSummaryGenerator
                    exec_generator = ExecutiveSummaryGenerator()
                    exec_assessment = exec_generator.generate(
                        dataset_id=dataset_id, dataset_type=detected_type or "unknown",
                        sample_count=sample_count, reproduction_cost=result.reproduction_cost,
                        human_percentage=result.human_percentage,
                        complexity_metrics=complexity_metrics,
                        phased_breakdown=phased_breakdown, llm_analysis=llm_analysis,
                        enhanced_context=enhanced_context,
                    )
                    exec_md = exec_generator.to_markdown(
                        assessment=exec_assessment, dataset_id=dataset_id,
                        dataset_type=detected_type or "unknown",
                        reproduction_cost=result.reproduction_cost,
                        phased_breakdown=phased_breakdown,
                    )
                    with open(output_mgr.get_path("decision", "EXECUTIVE_SUMMARY.md"), "w", encoding="utf-8") as f:
                        f.write(exec_md)
                    files.append(output_mgr.get_relative_path("decision", "EXECUTIVE_SUMMARY.md"))
                    exec_dict = exec_generator.to_dict(exec_assessment)
                    with open(output_mgr.get_path("decision", "executive_summary.json"), "w", encoding="utf-8") as f:
                        json.dump(exec_dict, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("decision", "executive_summary.json"))
                except Exception as e:
                    warnings.append(f"执行摘要生成失败: {e}")
                return files, warnings

            def _gen_industry_benchmark():
                files, warnings = [], []
                try:
                    from datarecipe.generators.industry_benchmark import IndustryBenchmarkGenerator
                    benchmark_generator = IndustryBenchmarkGenerator()
                    benchmark_comparison = benchmark_generator.generate(
                        dataset_id=dataset_id, dataset_type=detected_type or "unknown",
                        sample_count=actual_size, reproduction_cost=result.reproduction_cost,
                        human_percentage=result.human_percentage,
                    )
                    benchmark_md = benchmark_generator.to_markdown(benchmark_comparison)
                    with open(output_mgr.get_path("project", "INDUSTRY_BENCHMARK.md"), "w", encoding="utf-8") as f:
                        f.write(benchmark_md)
                    files.append(output_mgr.get_relative_path("project", "INDUSTRY_BENCHMARK.md"))
                    benchmark_dict = benchmark_generator.to_dict(benchmark_comparison)
                    with open(output_mgr.get_path("project", "industry_benchmark.json"), "w", encoding="utf-8") as f:
                        json.dump(benchmark_dict, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("project", "industry_benchmark.json"))
                except Exception as e:
                    warnings.append(f"行业基准对比生成失败: {e}")
                return files, warnings

            def _gen_data_schema():
                """Generate DATA_SCHEMA.json for knowlyr-datalabel integration."""
                files, warnings = [], []
                try:
                    # --- Field definitions ---
                    display_names = {
                        "messages": "对话消息", "instruction": "用户指令",
                        "response": "模型回答", "input": "输入", "output": "输出",
                        "question": "问题", "answer": "回答", "text": "文本内容",
                        "chosen": "优选回答", "rejected": "劣选回答",
                        "source_dataset": "来源数据集", "domain": "领域",
                        "problem_statement": "问题描述", "patch": "补丁",
                        "repo": "代码仓库", "context": "上下文",
                    }
                    fields = []
                    for fld, info in schema_info.items():
                        field_type = info.get("type", "str")
                        type_map = {
                            "str": "string", "int": "integer", "float": "number",
                            "bool": "boolean", "list": "array", "dict": "object",
                            "NoneType": "string",
                        }
                        json_type = type_map.get(field_type, "string")
                        field_def = {"name": fld, "type": json_type}
                        if fld in display_names:
                            field_def["display_name"] = display_names[fld]
                        if info.get("examples"):
                            field_def["examples"] = info["examples"][:2]
                        if info.get("nested_type"):
                            field_def["nested_type"] = info["nested_type"]
                        fields.append(field_def)

                    # --- Annotation config based on dataset type ---
                    dt = detected_type or "unknown"
                    quality_req = (
                        complexity_metrics.quality_requirement
                        if complexity_metrics else "standard"
                    )
                    scoring_rubric = []
                    annotation_config = {}

                    if dt == "preference":
                        # Preference data → ranking annotation
                        annotation_config = {
                            "type": "ranking",
                            "options": [
                                {"value": "chosen", "label": "优选回答"},
                                {"value": "rejected", "label": "劣选回答"},
                            ],
                        }
                    elif dt == "swe_bench":
                        # Code patches → scoring
                        scoring_rubric = [
                            {"score": 1, "label": "通过",
                             "description": "补丁正确解决问题，测试通过"},
                            {"score": 0.5, "label": "部分",
                             "description": "补丁方向正确但不完整或有副作用"},
                            {"score": 0, "label": "失败",
                             "description": "补丁错误或无法应用"},
                        ]
                    else:
                        # SFT / evaluation / unknown → scoring with quality-based granularity
                        if quality_req in ("high", "expert"):
                            scoring_rubric = [
                                {"score": 1, "label": "优秀",
                                 "description": "回答完整准确，逻辑清晰，表达流畅"},
                                {"score": 0.75, "label": "良好",
                                 "description": "回答基本准确，有小瑕疵但不影响理解"},
                                {"score": 0.5, "label": "一般",
                                 "description": "回答部分正确但有明显遗漏或偏差"},
                                {"score": 0.25, "label": "较差",
                                 "description": "回答大部分错误或严重偏题"},
                                {"score": 0, "label": "不可用",
                                 "description": "回答完全错误、无关或有害"},
                            ]
                        else:
                            scoring_rubric = [
                                {"score": 1, "label": "优秀",
                                 "description": "回答完整准确，逻辑清晰，表达流畅"},
                                {"score": 0.5, "label": "一般",
                                 "description": "回答基本正确但有遗漏或表达不够清晰"},
                                {"score": 0, "label": "差",
                                 "description": "回答错误、离题或无实质内容"},
                            ]

                    schema = {
                        "dataset_id": dataset_id,
                        "project_name": (
                            dataset_id.split("/")[-1] if "/" in dataset_id
                            else dataset_id
                        ),
                        "sample_count": sample_count,
                        "fields": fields,
                        "field_names": [f["name"] for f in fields],
                    }
                    if scoring_rubric:
                        schema["scoring_rubric"] = scoring_rubric
                    if annotation_config:
                        schema["annotation_config"] = annotation_config

                    path = output_mgr.get_path("guide", "DATA_SCHEMA.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(schema, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("guide", "DATA_SCHEMA.json"))
                except Exception as e:
                    warnings.append(f"DATA_SCHEMA 生成失败: {e}")
                return files, warnings

            def _gen_samples():
                """Save sample items to 09_样例数据/samples.json for datalabel."""
                files, warnings = [], []
                try:
                    # Convert sample items to serializable format
                    serializable = []
                    for idx, item in enumerate(sample_items):
                        record = {"id": f"SAMPLE_{idx + 1:03d}"}
                        data = {}
                        for k, v in item.items():
                            if isinstance(v, (str, int, float, bool)) or v is None:
                                data[k] = v
                            elif isinstance(v, list):
                                data[k] = [str(x) if not isinstance(x, (str, int, float, bool, dict, list)) else x for x in v[:20]]
                            elif isinstance(v, dict):
                                data[k] = {sk: str(sv) if not isinstance(sv, (str, int, float, bool)) else sv for sk, sv in v.items()}
                            else:
                                data[k] = str(v)
                        record["data"] = data
                        serializable.append(record)
                    path = output_mgr.get_path("samples", "samples.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({"samples": serializable}, f, indent=2, ensure_ascii=False)
                    files.append(output_mgr.get_relative_path("samples", "samples.json"))
                except Exception as e:
                    warnings.append(f"样例数据保存失败: {e}")
                return files, warnings

            # Execute all generators in parallel
            generator_fns = [
                _gen_analysis_report, _gen_reproduction_guide, _gen_annotation_spec,
                _gen_milestone_plan, _gen_executive_summary, _gen_industry_benchmark,
                _gen_data_schema, _gen_samples,
            ]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(fn): fn for fn in generator_fns}
                for future in as_completed(futures):
                    files, warnings = future.result()
                    result.files_generated.extend(files)
                    result.warnings.extend(warnings)

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

            # Update project manifest and generate README
            manifest = ProjectManifest(dataset_output_dir)
            manifest.record_command("deep-analyze")
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

            # Save enhancement state (for MCP enhance_analysis_reports tool)
            try:
                _state = {
                    "dataset_id": dataset_id,
                    "sample_size": sample_size,
                    "target_size": target_size,
                    "split": split,
                    "region": self.region,
                }
                with open(
                    os.path.join(dataset_output_dir, "_enhancement_state.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(_state, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # enhancement state is optional

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
