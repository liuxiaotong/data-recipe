"""Core deep analysis functionality shared between CLI and MCP."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisResult:
    """Result of deep analysis."""

    dataset_id: str
    success: bool = True
    error: str = ""

    # Dataset info
    dataset_type: str = ""
    sample_count: int = 0
    fields: List[str] = field(default_factory=list)

    # Cost info
    reproduction_cost: Dict[str, float] = field(default_factory=dict)
    human_percentage: float = 0.0

    # Analysis stats
    rubric_patterns: int = 0
    prompt_templates: int = 0

    # Output paths
    output_dir: str = ""
    files_generated: List[str] = field(default_factory=list)

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
        }


class DeepAnalyzerCore:
    """Core deep analysis engine shared between CLI and MCP."""

    def __init__(
        self,
        output_dir: str = "./analysis_output",
        region: str = "china",
        use_llm: bool = False,
        llm_provider: str = "anthropic",
    ):
        self.output_dir = output_dir
        self.region = region
        self.use_llm = use_llm
        self.llm_provider = llm_provider

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
            from datarecipe.extractors import RubricsAnalyzer, PromptExtractor
            from datarecipe.analyzers import ContextStrategyDetector
            from datarecipe.generators import HumanMachineSplitter, TaskType
            from datarecipe.integrations.radar import RadarIntegration

            # Create output directory
            safe_name = dataset_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            dataset_output_dir = os.path.join(self.output_dir, safe_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
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
                "chosen_longer": 0, "rejected_longer": 0, "same_length": 0,
                "chosen_more_detailed": 0, "chosen_more_helpful": 0, "chosen_safer": 0,
            }

            # SWE-bench support
            is_swe_dataset = False
            swe_stats = {
                "repos": {}, "languages": {}, "issue_types": {},
                "issue_categories": {}, "patch_lines": [], "examples": [],
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
                                "nested_type": None
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
                    rubrics_examples.append({
                        "rubrics": item_rubrics,
                        "metadata": item.get("metadata", {}),
                        "messages": item.get("messages", [])
                    })

                # Messages and system prompts
                if "messages" in item and isinstance(item["messages"], list):
                    messages.extend(item["messages"])
                    for msg in item["messages"]:
                        if isinstance(msg, dict) and msg.get("role") == "system":
                            content = msg.get("content", "")
                            if content and len(content) > 50:
                                domain = "general"
                                if "metadata" in item and isinstance(item["metadata"], dict):
                                    domain = item["metadata"].get("context_category",
                                             item["metadata"].get("category", "general"))
                                if domain not in system_prompts_by_domain:
                                    system_prompts_by_domain[domain] = []
                                if len(system_prompts_by_domain[domain]) < 3:
                                    system_prompts_by_domain[domain].append({
                                        "content": content,
                                        "metadata": item.get("metadata", {})
                                    })

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
                    self._analyze_preference_pair(item, preference_pairs, preference_topics, preference_patterns)

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

                # Save rubric templates
                with open(os.path.join(dataset_output_dir, "rubrics_analysis.json"), "w", encoding="utf-8") as f:
                    json.dump(analyzer.to_dict(rubrics_result), f, indent=2, ensure_ascii=False)
                result.files_generated.append("rubrics_analysis.json")

                with open(os.path.join(dataset_output_dir, "rubric_templates.yaml"), "w", encoding="utf-8") as f:
                    f.write(analyzer.to_yaml_templates(rubrics_result))
                result.files_generated.append("rubric_templates.yaml")

                with open(os.path.join(dataset_output_dir, "rubric_templates.md"), "w", encoding="utf-8") as f:
                    f.write(analyzer.to_markdown_templates(rubrics_result))
                result.files_generated.append("rubric_templates.md")

            prompt_library = None
            if messages:
                extractor = PromptExtractor()
                prompt_library = extractor.extract(messages)
                result.prompt_templates = prompt_library.unique_count

                with open(os.path.join(dataset_output_dir, "prompt_templates.json"), "w", encoding="utf-8") as f:
                    json.dump(extractor.to_dict(prompt_library), f, indent=2, ensure_ascii=False)
                result.files_generated.append("prompt_templates.json")

            strategy_result = None
            if contexts:
                detector = ContextStrategyDetector()
                strategy_result = detector.analyze(contexts[:100])
                with open(os.path.join(dataset_output_dir, "context_strategy.json"), "w", encoding="utf-8") as f:
                    json.dump(detector.to_dict(strategy_result), f, indent=2, ensure_ascii=False)
                result.files_generated.append("context_strategy.json")

            # Preference analysis
            if is_preference_dataset and preference_pairs:
                preference_analysis = {
                    "is_preference_dataset": True,
                    "total_pairs": sample_count,
                    "topic_distribution": preference_topics,
                    "patterns": preference_patterns,
                    "examples": preference_pairs[:10],
                }
                with open(os.path.join(dataset_output_dir, "preference_analysis.json"), "w", encoding="utf-8") as f:
                    json.dump(preference_analysis, f, indent=2, ensure_ascii=False)
                result.files_generated.append("preference_analysis.json")

            # SWE analysis
            if is_swe_dataset and swe_stats["repos"]:
                avg_patch = sum(swe_stats["patch_lines"]) / len(swe_stats["patch_lines"]) if swe_stats["patch_lines"] else 0
                swe_analysis = {
                    "is_swe_dataset": True,
                    "total_tasks": sample_count,
                    "repos_count": len(swe_stats["repos"]),
                    "repo_distribution": dict(sorted(swe_stats["repos"].items(), key=lambda x: -x[1])[:20]),
                    "language_distribution": swe_stats["languages"],
                    "avg_patch_lines": avg_patch,
                    "examples": swe_stats["examples"],
                }
                with open(os.path.join(dataset_output_dir, "swe_analysis.json"), "w", encoding="utf-8") as f:
                    json.dump(swe_analysis, f, indent=2, ensure_ascii=False)
                result.files_generated.append("swe_analysis.json")

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
                    with open(os.path.join(dataset_output_dir, "llm_analysis.json"), "w", encoding="utf-8") as f:
                        json.dump(llm_result_dict, f, indent=2, ensure_ascii=False)
                    result.files_generated.append("llm_analysis.json")
                except Exception:
                    pass

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

                # Save token analysis
                with open(os.path.join(dataset_output_dir, "token_analysis.json"), "w", encoding="utf-8") as f:
                    json.dump(precise_estimate.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append("token_analysis.json")

                # Model comparison
                comparisons = cost_calc.compare_models(
                    samples=sample_items,
                    target_size=actual_size,
                    models=["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "deepseek-v3"],
                )
                comparison_data = {m: e.to_dict() for m, e in comparisons.items()}
                with open(os.path.join(dataset_output_dir, "cost_comparison.json"), "w", encoding="utf-8") as f:
                    json.dump(comparison_data, f, indent=2, ensure_ascii=False)
                result.files_generated.append("cost_comparison.json")

            except Exception:
                pass

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

                # Save complexity analysis
                with open(os.path.join(dataset_output_dir, "complexity_analysis.json"), "w", encoding="utf-8") as f:
                    json.dump(complexity_metrics.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append("complexity_analysis.json")

            except Exception:
                pass

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
                ]
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

                # Save calibration analysis
                with open(os.path.join(dataset_output_dir, "cost_calibration.json"), "w", encoding="utf-8") as f:
                    json.dump(calibration_result.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append("cost_calibration.json")

            except Exception:
                pass

            total_cost = human_cost + api_cost

            # Phased cost breakdown
            phased_breakdown = None
            try:
                from datarecipe.cost import PhasedCostModel
                phased_model = PhasedCostModel(region=self.region)

                # Calculate API cost per sample for phased model
                api_per_sample = api_cost / actual_size if actual_size > 0 else 0.01
                complexity_mult = complexity_metrics.cost_multiplier if complexity_metrics else 1.0
                quality_req = complexity_metrics.quality_requirement if complexity_metrics else "standard"

                phased_breakdown = phased_model.calculate(
                    target_size=actual_size,
                    dataset_type=detected_type or "unknown",
                    human_percentage=allocation.human_work_percentage,
                    api_cost_per_sample=api_per_sample,
                    complexity_multiplier=complexity_mult,
                    quality_requirement=quality_req,
                )

                # Save phased cost analysis
                with open(os.path.join(dataset_output_dir, "phased_cost.json"), "w", encoding="utf-8") as f:
                    json.dump(phased_breakdown.to_dict(), f, indent=2, ensure_ascii=False)
                result.files_generated.append("phased_cost.json")

                # Save phased cost report
                phased_report = phased_model.format_report(phased_breakdown)
                with open(os.path.join(dataset_output_dir, "COST_BREAKDOWN.md"), "w", encoding="utf-8") as f:
                    f.write(phased_report)
                result.files_generated.append("COST_BREAKDOWN.md")

            except Exception:
                pass

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

            with open(os.path.join(dataset_output_dir, "allocation.json"), "w", encoding="utf-8") as f:
                json.dump(allocation_dict, f, indent=2, ensure_ascii=False)
            result.files_generated.append("allocation.json")

            # Generate reports
            report = self._generate_analysis_report(
                dataset_id, sample_count, actual_size,
                rubrics_result, prompt_library, strategy_result, allocation, self.region
            )
            with open(os.path.join(dataset_output_dir, "ANALYSIS_REPORT.md"), "w", encoding="utf-8") as f:
                f.write(report)
            result.files_generated.append("ANALYSIS_REPORT.md")

            guide = self._generate_reproduction_guide(
                dataset_id, schema_info, category_set, sub_category_set,
                system_prompts_by_domain, rubrics_examples, sample_items,
                rubrics_result, prompt_library, allocation,
                is_preference_dataset, preference_pairs, preference_topics, preference_patterns,
                is_swe_dataset, swe_stats, llm_analysis
            )
            with open(os.path.join(dataset_output_dir, "REPRODUCTION_GUIDE.md"), "w", encoding="utf-8") as f:
                f.write(guide)
            result.files_generated.append("REPRODUCTION_GUIDE.md")

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
                )

                # Save as Markdown
                spec_md = spec_generator.to_markdown(annotation_spec)
                with open(os.path.join(dataset_output_dir, "ANNOTATION_SPEC.md"), "w", encoding="utf-8") as f:
                    f.write(spec_md)
                result.files_generated.append("ANNOTATION_SPEC.md")

                # Save as JSON
                spec_dict = spec_generator.to_dict(annotation_spec)
                with open(os.path.join(dataset_output_dir, "annotation_spec.json"), "w", encoding="utf-8") as f:
                    json.dump(spec_dict, f, indent=2, ensure_ascii=False)
                result.files_generated.append("annotation_spec.json")

            except Exception:
                pass

            # Recipe summary
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
                pass

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
                pass

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
            for h_pat, a_pat in [(r'\n\nHuman:', r'\n\nAssistant:'), (r'\nHuman:', r'\nAssistant:')]:
                if h_pat.replace(r'\n', '\n') in text:
                    parts = re.split(r'(' + h_pat + '|' + a_pat + ')', text)
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
        if any(w in rejected.lower() for w in safety_words) and not any(w in chosen.lower() for w in safety_words):
            patterns["chosen_safer"] += 1

        # Save example
        if len(pairs) < 20:
            pairs.append({
                "topic": topic,
                "turn_count": len(chosen_turns),
                "human_query": chosen_turns[0].get("content", "")[:300] if chosen_turns else "",
            })

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
            stats["examples"].append({
                "repo": repo,
                "language": lang,
                "problem_statement": item.get("problem_statement", "")[:800],
            })

    def _generate_analysis_report(self, dataset_id, sample_count, actual_size,
                                   rubrics_result, prompt_library, strategy_result,
                                   allocation, region) -> str:
        """Generate analysis report markdown."""
        lines = []
        lines.append(f"# 🔬 {dataset_id} 深度逆向分析报告")
        lines.append("")
        lines.append(f"> **分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> **数据集**: {dataset_id}")
        lines.append(f"> **分析样本**: {sample_count} 条")
        lines.append(f"> **目标规模**: {actual_size:,} 条")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## 📊 执行摘要")
        lines.append("")
        lines.append("| 维度 | 发现 |")
        lines.append("|------|------|")

        if rubrics_result:
            lines.append(f"| **评分标准** | {rubrics_result.total_rubrics:,} 条，{rubrics_result.unique_patterns:,} 种独特模式 |")
        if prompt_library:
            lines.append(f"| **Prompt模板** | {prompt_library.unique_count} 个去重后的系统提示模板 |")
        if strategy_result:
            lines.append(f"| **数据来源** | 混合策略（合成 {strategy_result.synthetic_score*100:.0f}% + 改编 {strategy_result.modified_score*100:.0f}%） |")

        lines.append(f"| **复现成本** | 约 ${allocation.total_cost:,.0f}（人工 ${allocation.total_human_cost:,.0f} + API ${allocation.total_machine_cost:,.0f}） |")
        lines.append(f"| **人机分配** | 人工 {allocation.human_work_percentage:.0f}%，机器 {allocation.machine_work_percentage:.0f}% |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*报告由 DataRecipe 自动生成*")

        return "\n".join(lines)

    def _generate_reproduction_guide(self, dataset_id, schema_info, category_set,
                                      sub_category_set, system_prompts_by_domain,
                                      rubrics_examples, sample_items, rubrics_result,
                                      prompt_library, allocation, is_preference_dataset,
                                      preference_pairs, preference_topics, preference_patterns,
                                      is_swe_dataset, swe_stats, llm_analysis) -> str:
        """Generate reproduction guide markdown."""
        lines = []
        lines.append(f"# 📋 {dataset_id} 复刻指南")
        lines.append("")

        if is_swe_dataset:
            lines.append("> **这是一个软件工程评测数据集 (SWE-bench 风格)。**")
        elif is_preference_dataset:
            lines.append("> **这是一个 RLHF 偏好数据集。**")
        elif llm_analysis and llm_analysis.dataset_type != "unknown":
            lines.append(f"> **数据集类型: {llm_analysis.dataset_type}。{llm_analysis.purpose}**")
        else:
            lines.append("> **本指南提供可直接操作的模板和规范。**")
        lines.append("")
        lines.append("---")
        lines.append("")

        # LLM analysis section
        if llm_analysis and llm_analysis.dataset_type != "unknown":
            from datarecipe.analyzers.llm_dataset_analyzer import generate_llm_guide_section
            lines.append(generate_llm_guide_section(llm_analysis))
            lines.append("")

        # Schema section
        lines.append("## 1️⃣ 数据结构规范 (Schema)")
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

        lines.append("---")
        lines.append("")
        lines.append("*指南由 DataRecipe 自动生成*")

        return "\n".join(lines)
