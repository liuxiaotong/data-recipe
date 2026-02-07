"""Generate output documents from specification analysis."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from datarecipe.analyzers.spec_analyzer import FieldDefinition, SpecificationAnalysis
from datarecipe.pipeline import assemble_pipeline
from datarecipe.task_profiles import get_task_profile


@dataclass
class SpecOutputResult:
    """Result of specification output generation."""

    success: bool = True
    error: str = ""
    output_dir: str = ""
    files_generated: List[str] = field(default_factory=list)


class SpecOutputGenerator:
    """Generate all output documents from specification analysis."""

    def __init__(self, output_dir: str = "./spec_output"):
        self.output_dir = output_dir

    def generate(
        self,
        analysis: SpecificationAnalysis,
        target_size: int = 100,
        region: str = "china",
        enhanced_context=None,
    ) -> SpecOutputResult:
        """Generate all output documents.

        Args:
            analysis: SpecificationAnalysis from spec analyzer
            target_size: Target dataset size for cost estimation
            region: Region for cost calculation

        Returns:
            SpecOutputResult with generated files
        """
        result = SpecOutputResult()

        try:
            # Create output directory with structure
            project_name = analysis.project_name or "spec_analysis"
            safe_name = project_name.replace("/", "_").replace(" ", "_")
            output_dir = os.path.join(self.output_dir, safe_name)

            # Create subdirectories
            subdirs = {
                "decision": "01_决策参考",
                "project": "02_项目管理",
                "annotation": "03_标注规范",
                "guide": "04_复刻指南",
                "cost": "05_成本分析",
                "data": "06_原始数据",
                "templates": "07_模板",
                "ai_agent": "08_AI_Agent",
                "samples": "09_样例数据",
            }
            for key, subdir in subdirs.items():
                os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

            result.output_dir = output_dir

            # Generate each document
            self._generate_annotation_spec(analysis, output_dir, subdirs, result, enhanced_context=enhanced_context)
            self._generate_executive_summary(analysis, output_dir, subdirs, target_size, region, result, enhanced_context=enhanced_context)
            self._generate_milestone_plan(analysis, output_dir, subdirs, target_size, region, result, enhanced_context=enhanced_context)
            self._generate_cost_breakdown(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_industry_benchmark(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_raw_analysis(analysis, output_dir, subdirs, result)

            # Generate production documents
            self._generate_training_guide(analysis, output_dir, subdirs, result)
            self._generate_qa_checklist(analysis, output_dir, subdirs, result)
            self._generate_data_template(analysis, output_dir, subdirs, result)
            self._generate_production_sop(analysis, output_dir, subdirs, result)
            self._generate_data_schema(analysis, output_dir, subdirs, result)

            # Generate validation guides for all strategies
            self._generate_validation_guide(analysis, output_dir, subdirs, result)

            # Generate AI Agent layer
            self._generate_ai_agent_context(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_ai_workflow_state(analysis, output_dir, subdirs, result)
            self._generate_ai_reasoning_traces(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_ai_pipeline(analysis, output_dir, subdirs, result)
            self._generate_ai_readme(analysis, output_dir, subdirs, result)

            # Generate sample data
            self._generate_think_po_samples(analysis, output_dir, subdirs, target_size, result)

            self._generate_readme(analysis, output_dir, subdirs, result)

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _generate_annotation_spec(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
        enhanced_context=None,
    ):
        """Generate ANNOTATION_SPEC.md."""
        lines = []

        lines.append(f"# {analysis.project_name} 标注规范")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据类型: {analysis.dataset_type}")
        if analysis.has_images:
            lines.append(f"> 包含图片: 是 ({analysis.image_count} 张)")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 1: Task Type Description
        lines.append("## 一、题目类型描述")
        lines.append("")
        lines.append(f"**任务名称**: {analysis.task_type}")
        lines.append("")
        lines.append(f"**任务说明**: {analysis.task_description}")
        lines.append("")

        if analysis.cognitive_requirements:
            lines.append("**认知要求**:")
            for req in analysis.cognitive_requirements:
                lines.append(f"- {req}")
            lines.append("")

        if analysis.reasoning_chain:
            lines.append("**推理链**:")
            lines.append("")
            lines.append("```")
            lines.append(" → ".join(analysis.reasoning_chain))
            lines.append("```")
            lines.append("")

        # Section 2: Data Requirements
        lines.append("## 二、数据要求")
        lines.append("")

        if analysis.data_requirements:
            for i, req in enumerate(analysis.data_requirements, 1):
                lines.append(f"{i}. {req}")
            lines.append("")

        # Section 3: Quality Constraints
        lines.append("## 三、质量约束")
        lines.append("")

        if analysis.forbidden_items:
            lines.append("### 禁止项 ⚠️")
            lines.append("")
            for item in analysis.forbidden_items:
                lines.append(f"- ❌ {item}")
            lines.append("")

        if analysis.quality_constraints:
            lines.append("### 质量标准")
            lines.append("")
            for constraint in analysis.quality_constraints:
                lines.append(f"- {constraint}")
            lines.append("")

        if analysis.difficulty_criteria:
            lines.append("### 难度验证")
            lines.append("")
            lines.append(f"{analysis.difficulty_criteria}")
            lines.append("")

        # Section 4: Data Structure
        lines.append("## 四、数据结构")
        lines.append("")

        if analysis.fields:
            lines.append("| 字段名 | 类型 | 必填 | 说明 |")
            lines.append("|--------|------|------|------|")
            for f in analysis.fields:
                name = f.get("name", "")
                ftype = f.get("type", "string")
                required = "是" if f.get("required", True) else "否"
                desc = f.get("description", "")
                lines.append(f"| {name} | {ftype} | {required} | {desc} |")
            lines.append("")

        if analysis.field_requirements:
            lines.append("### 字段详细要求")
            lines.append("")
            for fname, freq in analysis.field_requirements.items():
                lines.append(f"**{fname}**: {freq}")
                lines.append("")

        # Section 5: Examples
        lines.append("## 五、示例")
        lines.append("")

        for i, example in enumerate(analysis.examples[:3], 1):
            lines.append(f"### 示例 {i}")
            lines.append("")

            if example.get("has_image"):
                lines.append("**[包含图片]**")
                lines.append("")

            if example.get("question"):
                lines.append("**题目**:")
                lines.append("")
                lines.append(f"> {example['question']}")
                lines.append("")

            if example.get("answer"):
                lines.append(f"**答案**: {example['answer']}")
                lines.append("")

            if example.get("scoring_rubric"):
                lines.append("**打分标准**:")
                lines.append("")
                lines.append(f"{example['scoring_rubric']}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Section 6: Scoring Rubric
        if analysis.scoring_rubric:
            lines.append("## 六、打分标准")
            lines.append("")
            lines.append("| 分数 | 标准 |")
            lines.append("|------|------|")
            for rubric in analysis.scoring_rubric:
                score = rubric.get("score", "")
                criteria = rubric.get("criteria", "")
                lines.append(f"| {score} | {criteria} |")
            lines.append("")

        # Section 7: Domain-specific guidance (LLM enhanced)
        ec = enhanced_context
        if ec and ec.generated and ec.domain_specific_guidelines:
            lines.append("## 七、领域标注指导")
            lines.append("")
            lines.append(ec.domain_specific_guidelines)
            lines.append("")

            if ec.quality_pitfalls:
                lines.append("### 常见错误")
                lines.append("")
                for i, pitfall in enumerate(ec.quality_pitfalls, 1):
                    lines.append(f"{i}. {pitfall}")
                lines.append("")

            if ec.example_analysis:
                lines.append("### 样本分析")
                lines.append("")
                for ex in ec.example_analysis:
                    idx = ex.get("sample_index", "?")
                    lines.append(f"**样本 {idx}**")
                    lines.append(f"- 优点: {ex.get('strengths', '')}")
                    lines.append(f"- 改进: {ex.get('weaknesses', '')}")
                    lines.append(f"- 建议: {ex.get('annotation_tips', '')}")
                    lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本规范由 DataRecipe 从需求文档自动生成")

        # Write file
        spec_path = os.path.join(output_dir, subdirs["annotation"], "ANNOTATION_SPEC.md")
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['annotation']}/ANNOTATION_SPEC.md")

        # Also save as JSON
        spec_dict = {
            "project_name": analysis.project_name,
            "dataset_type": analysis.dataset_type,
            "task_type": analysis.task_type,
            "task_description": analysis.task_description,
            "cognitive_requirements": analysis.cognitive_requirements,
            "reasoning_chain": analysis.reasoning_chain,
            "data_requirements": analysis.data_requirements,
            "quality_constraints": analysis.quality_constraints,
            "forbidden_items": analysis.forbidden_items,
            "difficulty_criteria": analysis.difficulty_criteria,
            "fields": analysis.fields,
            "field_requirements": analysis.field_requirements,
            "examples": analysis.examples,
            "scoring_rubric": analysis.scoring_rubric,
        }
        json_path = os.path.join(output_dir, subdirs["annotation"], "annotation_spec.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(spec_dict, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['annotation']}/annotation_spec.json")

    def _generate_executive_summary(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        target_size: int,
        region: str,
        result: SpecOutputResult,
        enhanced_context=None,
    ):
        """Generate EXECUTIVE_SUMMARY.md."""
        # Calculate cost estimates
        cost_per_item = self._estimate_cost_per_item(analysis, region)
        total_cost = cost_per_item * target_size
        human_cost = total_cost * (analysis.estimated_human_percentage / 100)
        api_cost = total_cost - human_cost

        # Determine recommendation
        if analysis.estimated_difficulty == "expert":
            recommendation = "有条件推荐"
            rec_icon = "🟡"
            score = 5.5
        elif analysis.estimated_difficulty == "hard":
            recommendation = "推荐"
            rec_icon = "🟢"
            score = 6.5
        else:
            recommendation = "强烈推荐"
            rec_icon = "🟢"
            score = 7.5

        lines = []
        lines.append(f"# {analysis.project_name} 执行摘要")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {analysis.dataset_type}")
        lines.append(f"> 目标规模: {target_size} 条")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Decision box
        lines.append(f"## {rec_icon} 决策建议: {recommendation}")
        lines.append("")
        lines.append(f"**评分**: {score}/10")
        lines.append("")
        lines.append(f"**理由**: 数据集价值良好 (评分 {score}/10)，{recommendation}")
        lines.append("")

        # Key metrics
        lines.append("### 关键指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 总成本 | ${total_cost:,.0f} |")
        lines.append(f"| 人工成本 | ${human_cost:,.0f} ({analysis.estimated_human_percentage:.0f}%) |")
        lines.append(f"| 难度 | {analysis.estimated_difficulty} |")
        lines.append(f"| 领域 | {analysis.estimated_domain} |")
        lines.append("")

        # Use cases
        ec = enhanced_context
        lines.append("---")
        lines.append("")
        lines.append("## 用途与价值")
        lines.append("")
        if ec and ec.generated and ec.dataset_purpose_summary:
            lines.append(f"**主要用途**: {ec.dataset_purpose_summary}")
        else:
            lines.append(f"**主要用途**: {analysis.description or analysis.task_description}")
        lines.append("")

        if ec and ec.generated and ec.tailored_use_cases:
            lines.append("### 具体应用场景")
            lines.append("")
            for i, uc in enumerate(ec.tailored_use_cases, 1):
                lines.append(f"{i}. {uc}")
            lines.append("")

        if ec and ec.generated and ec.tailored_roi_scenarios:
            lines.append("### 投资回报分析")
            lines.append("")
            for i, roi in enumerate(ec.tailored_roi_scenarios, 1):
                lines.append(f"{i}. {roi}")
            lines.append("")

        if ec and ec.generated and ec.competitive_positioning:
            lines.append("### 竞争定位")
            lines.append("")
            lines.append(ec.competitive_positioning)
            lines.append("")

        # Risks
        lines.append("---")
        lines.append("")
        lines.append("## 风险评估")
        lines.append("")
        lines.append("| 风险等级 | 描述 | 缓解措施 |")
        lines.append("|----------|------|----------|")

        if ec and ec.generated and ec.tailored_risks:
            for risk in ec.tailored_risks:
                level = risk.get("level", "中")
                desc = risk.get("description", "")
                mit = risk.get("mitigation", "")
                lines.append(f"| {level} | {desc} | {mit} |")
        else:
            if "AI" in str(analysis.forbidden_items) or "ai" in str(analysis.forbidden_items).lower():
                lines.append("| 高 | 禁止使用AI生成内容，全人工成本高 | 严格审核流程，确保数据原创性 |")

            if analysis.estimated_difficulty in ["hard", "expert"]:
                lines.append("| 中 | 难度较高，需要专业人员 | 提前储备人才，加强培训 |")

            if analysis.has_images:
                lines.append("| 中 | 包含图片，制作成本较高 | 建立图片素材库，规范制作流程 |")

            lines.append("| 低 | 标注质量可能波动 | 建立QA流程，定期校准 |")
        lines.append("")

        # Similar datasets
        if analysis.similar_datasets:
            lines.append("---")
            lines.append("")
            lines.append("## 类似数据集")
            lines.append("")
            for ds in analysis.similar_datasets:
                lines.append(f"- {ds}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本摘要由 DataRecipe 从需求文档自动生成")

        # Write file
        path = os.path.join(output_dir, subdirs["decision"], "EXECUTIVE_SUMMARY.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['decision']}/EXECUTIVE_SUMMARY.md")

        # Save JSON
        summary_dict = {
            "project_name": analysis.project_name,
            "recommendation": recommendation,
            "score": score,
            "total_cost": total_cost,
            "human_cost": human_cost,
            "api_cost": api_cost,
            "human_percentage": analysis.estimated_human_percentage,
            "difficulty": analysis.estimated_difficulty,
            "domain": analysis.estimated_domain,
        }
        json_path = os.path.join(output_dir, subdirs["decision"], "executive_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['decision']}/executive_summary.json")

    def _generate_milestone_plan(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        target_size: int,
        region: str,
        result: SpecOutputResult,
        enhanced_context=None,
    ):
        """Generate MILESTONE_PLAN.md."""
        # Estimate duration based on difficulty
        difficulty_days = {
            "easy": 14,
            "medium": 21,
            "hard": 30,
            "expert": 45,
        }
        total_days = difficulty_days.get(analysis.estimated_difficulty, 30)

        lines = []
        lines.append(f"# {analysis.project_name} 里程碑计划")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {analysis.dataset_type}")
        lines.append(f"> 目标规模: {target_size} 条")
        lines.append(f"> 预估工期: {total_days} 工作日")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Progress visualization
        lines.append("## 项目概览")
        lines.append("")
        lines.append("```")
        lines.append("阶段进度:")
        lines.append("M1 项目启动与规范制定    ███                  15%")
        lines.append("M2 试点标注与标准校准    ██                   10%")
        lines.append("M3 主体标注 - 第一批次  ██████               30%")
        lines.append("M4 主体标注 - 第二批次  ██████               30%")
        lines.append("M5 质量审核与交付      ███                  15%")
        lines.append("```")
        lines.append("")

        # Team composition
        lines.append("### 团队配置")
        lines.append("")
        lines.append("| 角色 | 人数 | 说明 |")
        lines.append("|------|------|------|")
        lines.append("| 项目经理 | 1 | 整体协调 |")

        if analysis.estimated_difficulty in ["hard", "expert"]:
            lines.append("| 领域专家 | 2-3 | 规则设计、质量把控 |")
        else:
            lines.append("| 领域专家 | 1-2 | 规则设计、质量把控 |")

        lines.append("| QA | 1-2 | 质量抽检 |")

        if analysis.has_images:
            lines.append("| 图片制作 | 2-3 | 原创图片设计 |")

        annotator_count = max(2, target_size // 50)
        lines.append(f"| 标注员 | {annotator_count}-{annotator_count + 2} | 数据生产 |")
        lines.append("")

        # Milestones
        lines.append("---")
        lines.append("")
        lines.append("## 里程碑详情")
        lines.append("")

        milestones = [
            ("M1", "项目启动与规范制定", "完成项目初始化、制定标注规范和质量标准",
             ["标注指南文档 v1.0", "Schema 定义与示例", "标注工具配置完成", "团队培训材料"]),
            ("M2", "试点标注与标准校准", "完成试点批次，验证标注流程和质量标准",
             [f"试点数据 ({max(5, target_size // 20)} 条)", "标注一致性报告", "流程问题清单与解决方案"]),
            ("M3", "主体标注 - 第一批次", "完成 40% 的标注量",
             [f"已标注数据 ({int(target_size * 0.4)} 条)", "质量周报"]),
            ("M4", "主体标注 - 第二批次", "完成剩余 60% 的标注量",
             [f"已标注数据 ({target_size} 条)", "质量周报"]),
            ("M5", "质量审核与交付", "完成最终质量审核和数据交付",
             ["最终数据集", "质量报告", "数据文档"]),
        ]

        for mid, name, desc, deliverables in milestones:
            lines.append(f"### {mid}: {name}")
            lines.append("")
            lines.append(f"**描述**: {desc}")
            lines.append("")
            lines.append("**交付物**:")
            for d in deliverables:
                lines.append(f"- [ ] {d}")
            lines.append("")

        # Acceptance criteria
        lines.append("---")
        lines.append("")
        lines.append("## 验收标准")
        lines.append("")
        lines.append("| 类别 | 指标 | 阈值 |")
        lines.append("|------|------|------|")
        lines.append("| 一致性 | Cohen's Kappa | ≥ 0.7 |")
        lines.append("| 准确性 | 专家审核通过率 | ≥ 95% |")
        lines.append("| 完整性 | 空值率 | = 0% |")

        if analysis.difficulty_criteria:
            lines.append(f"| 难度 | {analysis.difficulty_criteria[:30]}... | 通过验证 |")

        lines.append("")

        # Risks
        lines.append("---")
        lines.append("")
        lines.append("## 风险管理")
        lines.append("")

        if analysis.forbidden_items:
            lines.append("### R1: 数据合规性风险")
            lines.append("")
            lines.append("- **概率**: 🟡 中")
            lines.append("- **影响**: 🔴 高")
            lines.append("- **缓解措施**: 严格审核流程，确保不含AI内容")
            lines.append("")

        lines.append("### R2: 质量不稳定风险")
        lines.append("")
        lines.append("- **概率**: 🟡 中")
        lines.append("- **影响**: 🟡 中")
        lines.append("- **缓解措施**: 加强培训，定期校准，建立QA流程")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本计划由 DataRecipe 从需求文档自动生成")

        # Write file
        path = os.path.join(output_dir, subdirs["project"], "MILESTONE_PLAN.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['project']}/MILESTONE_PLAN.md")

        # Save JSON
        plan_dict = {
            "project_name": analysis.project_name,
            "target_size": target_size,
            "total_days": total_days,
            "milestones": [
                {"id": mid, "name": name, "description": desc, "deliverables": deliverables}
                for mid, name, desc, deliverables in milestones
            ],
        }
        json_path = os.path.join(output_dir, subdirs["project"], "milestone_plan.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(plan_dict, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['project']}/milestone_plan.json")

    def _generate_cost_breakdown(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        target_size: int,
        region: str,
        result: SpecOutputResult,
    ):
        """Generate COST_BREAKDOWN.md."""
        cost_per_item = self._estimate_cost_per_item(analysis, region)
        total_cost = cost_per_item * target_size
        human_cost = total_cost * (analysis.estimated_human_percentage / 100)

        # Design phase (fixed costs)
        design_cost = 2000 if analysis.estimated_difficulty in ["hard", "expert"] else 1200

        # Production phase (variable costs)
        production_cost = human_cost * 0.7

        # QA phase
        qa_cost = human_cost * 0.2

        # Contingency
        contingency = total_cost * 0.15

        grand_total = design_cost + production_cost + qa_cost + contingency

        lines = []
        lines.append(f"# {analysis.project_name} 成本明细")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 目标规模: {target_size} 条")
        lines.append(f"> 单条成本: ${cost_per_item:.2f}")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## 阶段一：设计阶段（固定成本）")
        lines.append("")
        lines.append("| 项目 | 成本 |")
        lines.append("|------|------|")
        lines.append(f"| Schema 设计 | ${design_cost * 0.3:.0f} |")
        lines.append(f"| 标注指南编写 | ${design_cost * 0.4:.0f} |")
        lines.append(f"| 试点测试 | ${design_cost * 0.2:.0f} |")
        lines.append(f"| 工具配置 | ${design_cost * 0.1:.0f} |")
        lines.append(f"| **小计** | **${design_cost:.0f}** |")
        lines.append("")

        lines.append("## 阶段二：生产阶段（变动成本）")
        lines.append("")
        lines.append("| 项目 | 成本 | 单价 |")
        lines.append("|------|------|------|")
        lines.append(f"| 人工标注 | ${production_cost:.0f} | ${production_cost / target_size:.2f}/条 |")

        if analysis.has_images:
            img_cost = target_size * 5  # $5 per image
            lines.append(f"| 图片制作 | ${img_cost:.0f} | $5/张 |")
            production_cost += img_cost

        lines.append(f"| **小计** | **${production_cost:.0f}** | |")
        lines.append("")

        lines.append("## 阶段三：质量阶段")
        lines.append("")
        lines.append("| 项目 | 成本 |")
        lines.append("|------|------|")
        lines.append(f"| QA 抽检 | ${qa_cost * 0.6:.0f} |")
        lines.append(f"| 返工修正 | ${qa_cost * 0.3:.0f} |")
        lines.append(f"| 专家复核 | ${qa_cost * 0.1:.0f} |")
        lines.append(f"| **小计** | **${qa_cost:.0f}** |")
        lines.append("")

        lines.append("## 汇总")
        lines.append("")
        lines.append("| 阶段 | 成本 | 占比 |")
        lines.append("|------|------|------|")
        lines.append(f"| 设计阶段 | ${design_cost:.0f} | {design_cost / grand_total * 100:.1f}% |")
        lines.append(f"| 生产阶段 | ${production_cost:.0f} | {production_cost / grand_total * 100:.1f}% |")
        lines.append(f"| 质量阶段 | ${qa_cost:.0f} | {qa_cost / grand_total * 100:.1f}% |")
        lines.append(f"| 风险预留 (15%) | ${contingency:.0f} | 15% |")
        lines.append(f"| **总计** | **${grand_total:.0f}** | 100% |")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本成本估算由 DataRecipe 从需求文档自动生成")

        # Write file
        path = os.path.join(output_dir, subdirs["cost"], "COST_BREAKDOWN.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['cost']}/COST_BREAKDOWN.md")

        # Save JSON
        cost_dict = {
            "target_size": target_size,
            "cost_per_item": cost_per_item,
            "design_cost": design_cost,
            "production_cost": production_cost,
            "qa_cost": qa_cost,
            "contingency": contingency,
            "grand_total": grand_total,
        }
        json_path = os.path.join(output_dir, subdirs["cost"], "cost_breakdown.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cost_dict, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['cost']}/cost_breakdown.json")

    def _generate_industry_benchmark(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        target_size: int,
        region: str,
        result: SpecOutputResult,
    ):
        """Generate INDUSTRY_BENCHMARK.md."""
        cost_per_item = self._estimate_cost_per_item(analysis, region)
        total_cost = cost_per_item * target_size

        # Get benchmark data
        benchmarks = {
            "evaluation": {"min": 5, "avg": 15, "max": 50},
            "multimodal": {"min": 10, "avg": 25, "max": 80},
            "reasoning": {"min": 8, "avg": 20, "max": 60},
        }
        benchmark = benchmarks.get(analysis.dataset_type, {"min": 5, "avg": 15, "max": 50})

        lines = []
        lines.append(f"# {analysis.project_name} 行业基准对比")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {analysis.dataset_type}")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## 项目概况")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 样本数量 | {target_size:,} |")
        lines.append(f"| 总成本 | ${total_cost:,.0f} |")
        lines.append(f"| 单条成本 | ${cost_per_item:.2f} |")
        lines.append(f"| 人工占比 | {analysis.estimated_human_percentage:.0f}% |")
        lines.append("")

        lines.append("## 行业基准")
        lines.append("")
        lines.append(f"**数据类型**: {analysis.dataset_type}")
        lines.append("")
        lines.append("### 单条成本基准")
        lines.append("")
        lines.append("```")
        lines.append(f"最低: ${benchmark['min']:.2f}/条")
        lines.append(f"平均: ${benchmark['avg']:.2f}/条")
        lines.append(f"最高: ${benchmark['max']:.2f}/条")
        lines.append("```")
        lines.append("")

        # Rating
        if cost_per_item < benchmark["avg"]:
            rating = "🟢 成本低于行业平均"
        elif cost_per_item <= benchmark["max"]:
            rating = "🟡 成本在合理范围内"
        else:
            rating = "🔴 成本高于行业基准"

        lines.append(f"**成本评级**: {rating}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 基准数据来源于行业调研，仅供参考")

        # Write file
        path = os.path.join(output_dir, subdirs["project"], "INDUSTRY_BENCHMARK.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['project']}/INDUSTRY_BENCHMARK.md")

    def _generate_raw_analysis(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate raw analysis JSON."""
        path = os.path.join(output_dir, subdirs["data"], "spec_analysis.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['data']}/spec_analysis.json")

    def _generate_readme(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate README.md."""
        lines = []
        lines.append(f"# {analysis.project_name} 分析产出")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据类型: {analysis.dataset_type}")
        lines.append(f"> 来源: 需求文档分析")
        lines.append("")
        lines.append("## 目录结构")
        lines.append("")
        lines.append("```")
        lines.append(f"{os.path.basename(output_dir)}/")
        lines.append("├── README.md                    # 本文件")
        lines.append("│")
        lines.append(f"├── {subdirs['decision']}/           # 👔 决策层")
        lines.append("│   └── EXECUTIVE_SUMMARY.md     # 执行摘要")
        lines.append("│")
        lines.append(f"├── {subdirs['project']}/           # 📋 项目管理")
        lines.append("│   ├── MILESTONE_PLAN.md        # 里程碑计划")
        lines.append("│   └── INDUSTRY_BENCHMARK.md    # 行业基准")
        lines.append("│")
        lines.append(f"├── {subdirs['annotation']}/           # 📝 标注团队")
        lines.append("│   ├── ANNOTATION_SPEC.md       # 标注规范")
        lines.append("│   ├── TRAINING_GUIDE.md        # 培训手册")
        lines.append("│   └── QA_CHECKLIST.md          # 质检清单")
        lines.append("│")
        lines.append(f"├── {subdirs['guide']}/           # 📖 复刻指南")
        lines.append("│   ├── PRODUCTION_SOP.md        # 生产流程")
        lines.append("│   ├── DATA_SCHEMA.json         # 数据格式")
        if analysis.has_difficulty_validation():
            lines.append("│   └── DIFFICULTY_VALIDATION.md # 难度验证")
        lines.append("│")
        lines.append(f"├── {subdirs['cost']}/           # 💰 成本分析")
        lines.append("│   └── COST_BREAKDOWN.md        # 成本明细")
        lines.append("│")
        lines.append(f"├── {subdirs['templates']}/              # 📋 模板")
        lines.append("│   └── data_template.json       # 数据模板")
        lines.append("│")
        lines.append(f"├── {subdirs['data']}/           # 📊 原始数据")
        lines.append("│   └── spec_analysis.json       # 分析数据")
        lines.append("│")
        lines.append(f"├── {subdirs['ai_agent']}/            # 🤖 AI Agent")
        lines.append("│   ├── agent_context.json       # 聚合入口")
        lines.append("│   ├── workflow_state.json      # 工作流状态")
        lines.append("│   ├── reasoning_traces.json    # 推理链")
        lines.append("│   └── pipeline.yaml            # 可执行流水线")
        lines.append("│")
        lines.append(f"└── {subdirs['samples']}/           # 🧪 样例数据")
        lines.append("    ├── samples.json             # 样例数据")
        lines.append("    └── SAMPLE_GUIDE.md          # 样例指南")
        lines.append("```")
        lines.append("")
        lines.append("## 快速导航")
        lines.append("")
        lines.append("| 目标 | 查看文件 |")
        lines.append("|------|----------|")
        lines.append(f"| **快速决策** | `{subdirs['decision']}/EXECUTIVE_SUMMARY.md` |")
        lines.append(f"| **项目规划** | `{subdirs['project']}/MILESTONE_PLAN.md` |")
        lines.append(f"| **标注外包** | `{subdirs['annotation']}/ANNOTATION_SPEC.md` |")
        lines.append(f"| **标注培训** | `{subdirs['annotation']}/TRAINING_GUIDE.md` |")
        lines.append(f"| **生产流程** | `{subdirs['guide']}/PRODUCTION_SOP.md` |")
        lines.append(f"| **数据模板** | `{subdirs['templates']}/data_template.json` |")
        lines.append(f"| **成本预算** | `{subdirs['cost']}/COST_BREAKDOWN.md` |")
        lines.append(f"| **AI Agent** | `{subdirs['ai_agent']}/agent_context.json` |")
        lines.append(f"| **样例数据** | `{subdirs['samples']}/SAMPLE_GUIDE.md` |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("> 由 DataRecipe analyze-spec 命令生成")

        path = os.path.join(output_dir, "README.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append("README.md")

    def _generate_training_guide(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate TRAINING_GUIDE.md - annotator training manual."""
        lines = []
        lines.append(f"# {analysis.project_name} 标注员培训手册")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 1: Project Overview
        lines.append("## 一、项目概述")
        lines.append("")
        lines.append("### 1.1 项目目标")
        lines.append("")
        lines.append(f"{analysis.description or analysis.task_description}")
        lines.append("")

        lines.append("### 1.2 核心能力要求")
        lines.append("")
        if analysis.cognitive_requirements:
            for req in analysis.cognitive_requirements:
                lines.append(f"- {req}")
        lines.append("")

        if analysis.reasoning_chain:
            lines.append("### 1.3 推理链")
            lines.append("")
            lines.append("```")
            lines.append(" → ".join(analysis.reasoning_chain))
            lines.append("```")
            lines.append("")

        # Section 2: Data Requirements
        lines.append("---")
        lines.append("")
        lines.append("## 二、数据要求")
        lines.append("")

        if analysis.forbidden_items:
            lines.append("### 2.1 必须遵守的规则")
            lines.append("")
            for item in analysis.forbidden_items:
                lines.append(f"- ❌ {item}")
            lines.append("")

        if analysis.quality_constraints:
            lines.append("### 2.2 质量标准")
            lines.append("")
            for constraint in analysis.quality_constraints:
                lines.append(f"- ✅ {constraint}")
            lines.append("")

        # Section 3: Field Descriptions
        lines.append("---")
        lines.append("")
        lines.append("## 三、字段说明")
        lines.append("")

        if analysis.fields:
            for f in analysis.fields:
                name = f.get("name", "")
                ftype = f.get("type", "string")
                required = "是" if f.get("required", True) else "否"
                desc = f.get("description", "")
                lines.append(f"### {name}")
                lines.append("")
                lines.append(f"- **类型**: {ftype}")
                lines.append(f"- **必填**: {required}")
                lines.append(f"- **说明**: {desc}")
                if analysis.field_requirements.get(name):
                    lines.append(f"- **具体要求**: {analysis.field_requirements[name]}")
                lines.append("")

        # Section 4: Examples
        lines.append("---")
        lines.append("")
        lines.append("## 四、示例讲解")
        lines.append("")

        for i, example in enumerate(analysis.examples[:3], 1):
            lines.append(f"### 4.{i} 优秀示例分析")
            lines.append("")
            lines.append(f"#### 示例 {i}")
            lines.append("")

            if example.get("question"):
                lines.append(f"**题目**: {example['question'][:100]}...")
                lines.append("")

            if example.get("answer"):
                lines.append(f"**答案**: {example['answer']}")
                lines.append("")

            if example.get("scoring_rubric"):
                lines.append(f"**评分标准**: {example['scoring_rubric']}")
                lines.append("")

            lines.append("**优秀原因**:")
            lines.append("- 题意清晰，无歧义")
            lines.append("- 答案明确")
            lines.append("- 评分标准具体")
            lines.append("")

        # Section 5: Common Errors
        lines.append("---")
        lines.append("")
        lines.append("## 五、常见错误")
        lines.append("")

        lines.append("### 5.1 题目设计错误")
        lines.append("")
        lines.append("| 错误类型 | 示例 | 正确做法 |")
        lines.append("|----------|------|----------|")
        lines.append("| 题意模糊 | \"找最好的路线\" | \"找距离最短的路线\" |")
        lines.append("| 信息不足 | 缺少关键数据 | 确保所有必要信息都已给出 |")
        lines.append("| 答案不唯一 | 未说明多解情况 | 列出所有正确答案 |")
        lines.append("")

        if analysis.has_images:
            lines.append("### 5.2 图片制作错误")
            lines.append("")
            lines.append("| 错误类型 | 后果 | 避免方法 |")
            lines.append("|----------|------|----------|")
            lines.append("| 使用 AI 生图 | 数据作废，不支付费用 | 仅使用手绘/软件绘图/照片 |")
            lines.append("| 图片模糊 | 无法评测 | 确保分辨率 ≥ 800x600 |")
            lines.append("| 图文不匹配 | 题目无效 | 核对图片与题目描述一致 |")
            lines.append("")

        lines.append("### 5.3 评分标准错误")
        lines.append("")
        lines.append("| 错误类型 | 示例 | 正确做法 |")
        lines.append("|----------|------|----------|")
        lines.append("| 标准模糊 | \"回答正确得分\" | 明确什么样的回答算正确 |")
        lines.append("| 遗漏情况 | 只写满分条件 | 包含满分、部分分、零分条件 |")
        lines.append("")

        # Section 6: Self-Check List
        lines.append("---")
        lines.append("")
        lines.append("## 六、自检清单")
        lines.append("")
        lines.append("提交前请逐项检查：")
        lines.append("")

        if analysis.has_images:
            lines.append("### 图片")
            lines.append("- [ ] 非 AI 生成")
            lines.append("- [ ] 清晰度达标")
            lines.append("- [ ] 与题目强相关")
            lines.append("")

        lines.append("### 题目")
        lines.append("- [ ] 题意清晰")
        lines.append("- [ ] 无歧义")
        lines.append("- [ ] 格式要求明确")
        lines.append("")

        lines.append("### 答案")
        lines.append("- [ ] 答案正确")
        lines.append("- [ ] 多解已全部列出")
        lines.append("- [ ] 格式符合要求")
        lines.append("")

        lines.append("### 解析")
        lines.append("- [ ] 步骤完整")
        lines.append("- [ ] 逻辑清晰")
        lines.append("- [ ] 与答案一致")
        lines.append("")

        lines.append("### 评分标准")
        lines.append("- [ ] 包含满分条件")
        lines.append("- [ ] 包含零分条件")
        lines.append("- [ ] 条件具体可判")
        lines.append("")

        # Add difficulty validation to checklist if enabled
        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation
            lines.append("### 难度验证")
            lines.append(f"- [ ] 已完成 {diff_val.get('test_count', 3)} 次测试")
            lines.append(f"- [ ] 正确次数 ≤ {diff_val.get('max_correct', 1)}")
            lines.append("- [ ] 记录已保存")
            lines.append("")

        # FAQ
        lines.append("---")
        lines.append("")
        lines.append("## 七、FAQ")
        lines.append("")
        lines.append("**Q: 图片可以用网上下载的吗？**")
        lines.append("A: 不可以，存在版权风险，且可能被 AI 识别。请使用原创图片。")
        lines.append("")

        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation
            max_correct = diff_val.get("max_correct", 1)
            lines.append(f"**Q: 如果模型答对 {max_correct + 1} 次怎么办？**")
            lines.append("A: 题目无效，需要增加难度后重新验证。")
            lines.append("")

        lines.append("**Q: 评分标准必须是 1 分和 0 分吗？**")
        lines.append("A: 可以有部分得分（如 0.5 分），但需明确说明条件。")
        lines.append("")

        lines.append("**Q: 多轮对话题目怎么处理？**")
        lines.append("A: 每轮作为独立字段，明确标注轮次和前后依赖关系。")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本手册由 DataRecipe 自动生成")

        path = os.path.join(output_dir, subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['annotation']}/TRAINING_GUIDE.md")

    def _generate_qa_checklist(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate QA_CHECKLIST.md - quality assurance checklist."""
        lines = []
        lines.append(f"# {analysis.project_name} 质量检查清单")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 1: Single Data Check
        lines.append("## 一、单条数据检查")
        lines.append("")

        if analysis.has_images:
            lines.append("### 1.1 图片检查")
            lines.append("")
            lines.append("| 检查项 | 通过标准 | 检查结果 |")
            lines.append("|--------|----------|----------|")
            lines.append("| 原创性 | 非 AI 生成，无版权问题 | ☐ 通过 ☐ 不通过 |")
            lines.append("| 清晰度 | 分辨率 ≥ 800x600，文字可读 | ☐ 通过 ☐ 不通过 |")
            lines.append("| 相关性 | 图片与题目强相关，无图无法解题 | ☐ 通过 ☐ 不通过 |")
            lines.append("| 格式 | PNG/JPG，大小 ≤ 5MB | ☐ 通过 ☐ 不通过 |")
            lines.append("")

        lines.append("### 1.2 题目检查")
        lines.append("")
        lines.append("| 检查项 | 通过标准 | 检查结果 |")
        lines.append("|--------|----------|----------|")
        lines.append("| 清晰度 | 题意明确，无歧义 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 完整性 | 解题所需信息完整 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 原创性 | 非 AI 生成 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 格式要求 | 已说明答案输出格式 | ☐ 通过 ☐ 不通过 |")
        lines.append("")

        lines.append("### 1.3 答案检查")
        lines.append("")
        lines.append("| 检查项 | 通过标准 | 检查结果 |")
        lines.append("|--------|----------|----------|")
        lines.append("| 正确性 | 答案正确，可验证 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 完整性 | 多解已全部列出 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 格式 | 符合题目要求的格式 | ☐ 通过 ☐ 不通过 |")
        lines.append("")

        lines.append("### 1.4 解析检查")
        lines.append("")
        lines.append("| 检查项 | 通过标准 | 检查结果 |")
        lines.append("|--------|----------|----------|")
        lines.append("| 完整性 | 步骤完整，从题目到答案 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 正确性 | 逻辑正确，与答案一致 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 原创性 | 非 AI 生成 | ☐ 通过 ☐ 不通过 |")
        lines.append("")

        lines.append("### 1.5 评分标准检查")
        lines.append("")
        lines.append("| 检查项 | 通过标准 | 检查结果 |")
        lines.append("|--------|----------|----------|")
        lines.append("| 满分条件 | 已明确说明 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 零分条件 | 已明确说明 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 可操作性 | 条件具体，可客观判定 | ☐ 通过 ☐ 不通过 |")
        lines.append("")

        # Difficulty validation check if enabled
        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation
            lines.append("### 1.6 难度验证检查")
            lines.append("")
            lines.append("| 检查项 | 通过标准 | 检查结果 |")
            lines.append("|--------|----------|----------|")
            lines.append(f"| 测试次数 | 已完成 {diff_val.get('test_count', 3)} 次测试 | ☐ 通过 ☐ 不通过 |")
            lines.append(f"| 正确次数 | ≤ {diff_val.get('max_correct', 1)} 次 | ☐ 通过 ☐ 不通过 |")
            lines.append("| 记录完整 | 三次回答和判定都有记录 | ☐ 通过 ☐ 不通过 |")
            lines.append("")

        # Section 2: Batch Check
        lines.append("---")
        lines.append("")
        lines.append("## 二、批量数据检查")
        lines.append("")

        lines.append("### 2.1 数据完整性")
        lines.append("")
        lines.append("| 检查项 | 通过标准 | 检查结果 |")
        lines.append("|--------|----------|----------|")
        lines.append("| 数量 | 达到目标数量 | ☐ 通过 ☐ 不通过 |")
        lines.append("| 必填字段 | 所有必填字段已填写 | ☐ 通过 ☐ 不通过 |")
        if analysis.has_images:
            lines.append("| 图片文件 | 所有图片文件存在且可访问 | ☐ 通过 ☐ 不通过 |")
        lines.append("")

        lines.append("### 2.2 数据格式")
        lines.append("")
        lines.append("| 检查项 | 通过标准 | 检查结果 |")
        lines.append("|--------|----------|----------|")
        lines.append("| JSON 有效性 | JSON 格式正确，可解析 | ☐ 通过 ☐ 不通过 |")
        lines.append("| Schema 符合 | 符合 DATA_SCHEMA.json | ☐ 通过 ☐ 不通过 |")
        lines.append("| 编码格式 | UTF-8 编码 | ☐ 通过 ☐ 不通过 |")
        lines.append("")

        lines.append("### 2.3 抽检统计")
        lines.append("")
        lines.append("| 指标 | 目标 | 实际 | 达标 |")
        lines.append("|------|------|------|------|")
        lines.append("| 抽检率 | ≥ 20% | ___ % | ☐ |")
        lines.append("| 通过率 | ≥ 95% | ___ % | ☐ |")
        lines.append("| 返工率 | ≤ 10% | ___ % | ☐ |")
        lines.append("")

        # Section 3: Review Process
        lines.append("---")
        lines.append("")
        lines.append("## 三、审核流程")
        lines.append("")

        lines.append("### 3.1 自检（生产者）")
        lines.append("")
        lines.append("```")
        lines.append("完成数据 → 对照清单自检 → 修正问题 → 提交互审")
        lines.append("```")
        lines.append("")

        lines.append("### 3.2 互审（同级）")
        lines.append("")
        lines.append("```")
        lines.append("接收数据 → 交叉检查 → 标记问题 → 反馈/通过")
        lines.append("```")
        lines.append("")

        lines.append("### 3.3 专家抽检（QA）")
        lines.append("")
        lines.append("```")
        lines.append("随机抽取 20% → 深度检查 → 汇总问题 → 反馈/终审")
        lines.append("```")
        lines.append("")

        # Section 4: Issue Tracking
        lines.append("---")
        lines.append("")
        lines.append("## 四、问题记录表")
        lines.append("")
        lines.append("| 题目ID | 问题类型 | 问题描述 | 严重程度 | 处理状态 |")
        lines.append("|--------|----------|----------|----------|----------|")
        lines.append("| | | | ☐高 ☐中 ☐低 | ☐待修 ☐已修 ☐已验 |")
        lines.append("| | | | ☐高 ☐中 ☐低 | ☐待修 ☐已修 ☐已验 |")
        lines.append("| | | | ☐高 ☐中 ☐低 | ☐待修 ☐已修 ☐已验 |")
        lines.append("")

        # Section 5: Acceptance Criteria
        lines.append("---")
        lines.append("")
        lines.append("## 五、验收标准")
        lines.append("")
        lines.append("| 指标 | 阈值 | 说明 |")
        lines.append("|------|------|------|")
        lines.append("| 数据完整率 | 100% | 所有必填字段完整 |")
        lines.append("| 专家审核通过率 | ≥ 95% | 抽检通过比例 |")
        if analysis.has_difficulty_validation():
            lines.append("| 难度验证通过率 | 100% | 所有数据通过模型验证 |")
        lines.append("| 格式正确率 | 100% | JSON 格式和 Schema 符合 |")
        lines.append("")

        # Section: Structured field constraints (Upgrade 6)
        constraints = analysis.parsed_constraints
        if constraints:
            lines.append("---")
            lines.append("")
            lines.append("## 六、结构化字段约束")
            lines.append("")
            # Group by field_name
            from collections import defaultdict
            by_field: dict = defaultdict(list)
            for c in constraints:
                by_field[c.field_name].append(c)

            for fname, fcs in by_field.items():
                display_name = fname if fname != "_global" else "全局约束"
                lines.append(f"### {display_name}")
                lines.append("")
                lines.append("| 约束类型 | 规则 | 严重级别 | 可自动检查 |")
                lines.append("|----------|------|----------|------------|")
                for c in fcs:
                    auto = "是" if c.auto_checkable else "否"
                    lines.append(f"| {c.constraint_type} | {c.rule} | {c.severity} | {auto} |")
                lines.append("")

        # Section: Quality gates (Upgrade 4)
        if analysis.quality_gates:
            lines.append("---")
            lines.append("")
            lines.append("## 七、质量门禁")
            lines.append("")
            lines.append("| 门禁 | 指标 | 条件 | 阈值 | 级别 |")
            lines.append("|------|------|------|------|------|")
            for gate in analysis.quality_gates:
                lines.append(
                    f"| {gate.get('name', gate.get('gate_id', ''))} "
                    f"| {gate.get('metric', '')} "
                    f"| {gate.get('operator', '')} "
                    f"| {gate.get('threshold', '')} "
                    f"| {gate.get('severity', 'blocker')} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本检查清单由 DataRecipe 自动生成")

        path = os.path.join(output_dir, subdirs["annotation"], "QA_CHECKLIST.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['annotation']}/QA_CHECKLIST.md")

    def _generate_difficulty_validation(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate DIFFICULTY_VALIDATION.md - only when difficulty validation is configured."""
        diff_val = analysis.difficulty_validation
        if not diff_val:
            return

        model_name = diff_val.get("model", "未指定模型")
        settings = diff_val.get("settings", "默认设置")
        test_count = diff_val.get("test_count", 3)
        max_correct = diff_val.get("max_correct", 1)
        pass_criteria = diff_val.get("pass_criteria", f"跑 {test_count} 次，正确次数 ≤ {max_correct} 次")

        lines = []
        lines.append(f"# {analysis.project_name} 难度验证流程")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 1: Purpose
        lines.append("## 一、验证目的")
        lines.append("")
        lines.append("确保题目对当前主流大模型具有足够难度，避免生产无效数据。")
        lines.append("")
        lines.append(f"**有效数据标准：** {model_name} {settings}跑 {test_count} 次，正确次数 ≤ {max_correct} 次")
        lines.append("")

        # Section 2: Environment Setup
        lines.append("---")
        lines.append("")
        lines.append("## 二、验证环境配置")
        lines.append("")

        lines.append("### 2.1 模型设置")
        lines.append("")
        lines.append("| 配置项 | 值 |")
        lines.append("|--------|-----|")
        lines.append(f"| 模型名称 | {model_name} |")
        lines.append(f"| 配置设置 | {settings} |")
        lines.append("| 温度 | 默认 |")
        lines.append("| 最大 token | 默认 |")
        lines.append("")

        lines.append("### 2.2 测试平台")
        lines.append("")
        lines.append("推荐使用以下平台进行测试：")
        lines.append(f"- {model_name} 官方 Web 界面")
        lines.append("- API 调用（如有权限）")
        lines.append("")

        # Section 3: Validation Process
        lines.append("---")
        lines.append("")
        lines.append("## 三、验证流程")
        lines.append("")

        lines.append("### 步骤 1: 准备测试输入")
        lines.append("")
        if analysis.has_images:
            lines.append("将题目图片和文字组合成完整的 prompt：")
            lines.append("")
            lines.append("```")
            lines.append("[上传图片]")
            lines.append("")
            lines.append("{题目文字}")
            lines.append("```")
        else:
            lines.append("准备完整的题目文字作为 prompt。")
        lines.append("")

        lines.append(f"### 步骤 2: 执行测试（{test_count}次）")
        lines.append("")
        for i in range(1, test_count + 1):
            lines.append(f"{i}. **第{i}次测试**" + ("（新对话）" if i > 1 else ""))
            if i > 1:
                lines.append("   - 开启新对话")
            lines.append("   - 发送 prompt")
            lines.append("   - 等待模型回复")
            lines.append("   - 记录完整回答")
            lines.append("   - 判定正确/错误")
            lines.append("")

        lines.append("### 步骤 3: 判定结果")
        lines.append("")
        lines.append("| 正确次数 | 判定 | 处理 |")
        lines.append("|----------|------|------|")
        for i in range(test_count + 1):
            if i <= max_correct:
                lines.append(f"| {i} 次 | ✅ 有效 | 可提交 |")
            else:
                lines.append(f"| {i} 次 | ❌ 无效 | 需修改 |")
        lines.append("")

        # Section 4: Recording Template
        lines.append("---")
        lines.append("")
        lines.append("## 四、记录模板")
        lines.append("")

        lines.append("### 4.1 单题测试记录")
        lines.append("")
        lines.append("```markdown")
        lines.append("## 题目 ID: [XXX]")
        lines.append("")
        lines.append("### 测试配置")
        lines.append(f"- 模型: {model_name}")
        lines.append(f"- 配置: {settings}")
        lines.append("- 测试日期: YYYY-MM-DD")
        lines.append("")
        lines.append("### 测试结果")
        lines.append("")
        for i in range(1, test_count + 1):
            lines.append(f"**第 {i} 次测试：**")
            lines.append("- 模型回答: [完整回答]")
            lines.append("- 判定: ✅正确 / ❌错误")
            lines.append("- 原因: [简要说明]")
            lines.append("")
        lines.append("### 最终判定")
        lines.append(f"- 正确次数: X/{test_count}")
        lines.append("- 有效性: ✅有效 / ❌无效")
        lines.append("```")
        lines.append("")

        lines.append("### 4.2 批量记录表格")
        lines.append("")
        header = "| 题目ID |"
        separator = "|--------|"
        for i in range(1, test_count + 1):
            header += f" 测试{i} |"
            separator += "-------|"
        header += " 正确数 | 有效性 |"
        separator += "--------|--------|"
        lines.append(header)
        lines.append(separator)

        # Example rows
        lines.append("| 001 |" + " ❌ |" * test_count + " 0 | ✅ |")
        if max_correct >= 1:
            lines.append("| 002 |" + " ❌ |" * (test_count - 1) + " ✅ | 1 | ✅ |")
        if test_count > 2:
            lines.append("| 003 |" + " ✅ |" * 2 + " ❌ |" * (test_count - 2) + f" 2 | {'❌' if max_correct < 2 else '✅'} |")
        lines.append("")

        # Section 5: Handling Invalid Questions
        lines.append("---")
        lines.append("")
        lines.append("## 五、无效题目处理")
        lines.append("")

        lines.append("### 5.1 常见原因")
        lines.append("")
        lines.append("1. **题目过于简单**：推理步骤少，模型容易猜对")
        lines.append("2. **规则不够复杂**：规则简单，模型能轻松理解")
        lines.append("3. **答案选项有限**：答案空间小，猜中概率高")
        lines.append("")

        lines.append("### 5.2 修改策略")
        lines.append("")
        lines.append("| 问题 | 修改方向 |")
        lines.append("|------|----------|")
        lines.append("| 推理步骤少 | 增加中间步骤，嵌套更多规则 |")
        lines.append("| 规则简单 | 添加例外条件、特殊情况 |")
        lines.append("| 答案空间小 | 设计开放式问题，增加计算量 |")
        if analysis.has_images:
            lines.append("| 图文信息少 | 增加图中信息量，减少文字提示 |")
        lines.append("")

        lines.append("### 5.3 修改后重新验证")
        lines.append("")
        lines.append(f"修改后必须重新执行完整的 {test_count} 次测试流程。")
        lines.append("")

        # Section 6: Notes
        lines.append("---")
        lines.append("")
        lines.append("## 六、注意事项")
        lines.append("")
        lines.append("1. **每次测试使用新对话**：避免上下文影响")
        lines.append("2. **保持 prompt 一致**：三次测试使用完全相同的输入")
        lines.append("3. **客观判定**：根据评分标准判定，不主观放宽")
        lines.append("4. **保留记录**：所有测试记录需保留用于交付")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本流程由 DataRecipe 自动生成")

        path = os.path.join(output_dir, subdirs["guide"], "DIFFICULTY_VALIDATION.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['guide']}/DIFFICULTY_VALIDATION.md")

    def _generate_validation_guide(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate validation guides for all validation strategies.

        For model_test, delegates to the existing _generate_difficulty_validation.
        For other strategy types, generates a corresponding guide document.
        """
        strategies = analysis.parsed_validation_strategies
        if not strategies:
            return

        for strategy in strategies:
            if strategy.strategy_type == "model_test":
                self._generate_difficulty_validation(analysis, output_dir, subdirs, result)
            else:
                lines = []
                lines.append(f"# {analysis.project_name} 验证流程 - {strategy.strategy_type}")
                lines.append("")
                lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                lines.append(f"> 策略类型: {strategy.strategy_type}")
                lines.append("")
                lines.append("---")
                lines.append("")
                lines.append("## 验证说明")
                lines.append("")
                lines.append(f"{strategy.description or '请按以下流程执行验证。'}")
                lines.append("")

                if strategy.config:
                    lines.append("## 配置参数")
                    lines.append("")
                    lines.append("| 参数 | 值 |")
                    lines.append("|------|-----|")
                    for k, v in strategy.config.items():
                        lines.append(f"| {k} | {v} |")
                    lines.append("")

                if strategy.strategy_type == "human_review":
                    lines.append("## 流程")
                    lines.append("")
                    lines.append("1. 随机抽取指定比例的数据")
                    lines.append("2. 由审核员按标注规范逐条检查")
                    lines.append("3. 记录问题并反馈修改")
                    lines.append("4. 达到通过率阈值后放行")
                    lines.append("")
                elif strategy.strategy_type == "format_check":
                    lines.append("## 流程")
                    lines.append("")
                    lines.append("1. 使用 DATA_SCHEMA.json 校验每条数据格式")
                    lines.append("2. 检查必填字段、类型、长度等约束")
                    lines.append("3. 自动过滤不符合格式的数据")
                    lines.append("")
                elif strategy.strategy_type == "cross_validation":
                    lines.append("## 流程")
                    lines.append("")
                    lines.append("1. 同一数据由多名标注员独立标注")
                    lines.append("2. 计算标注者间一致性 (Cohen's Kappa)")
                    lines.append("3. 对低一致性数据进行仲裁")
                    lines.append("")
                elif strategy.strategy_type == "auto_scoring":
                    lines.append("## 流程")
                    lines.append("")
                    lines.append("1. 使用预定义评分函数自动打分")
                    lines.append("2. 低于阈值的数据标记为待审核")
                    lines.append("3. 人工复核标记数据")
                    lines.append("")

                lines.append("---")
                lines.append("")
                lines.append("> 本验证流程由 DataRecipe 自动生成")

                safe_type = strategy.strategy_type.upper()
                filename = f"VALIDATION_{safe_type}.md"
                path = os.path.join(output_dir, subdirs["guide"], filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                result.files_generated.append(f"{subdirs['guide']}/{filename}")

    # --- Schema-driven helpers ---

    def _template_placeholder(self, fd: FieldDefinition) -> Any:
        """Generate a template placeholder for a FieldDefinition (mode=template)."""
        return self._generate_value_from_field(fd, mode="template")

    def _sample_placeholder(self, fd: FieldDefinition, context: dict) -> Any:
        """Generate a sample placeholder for a FieldDefinition (mode=sample)."""
        return self._generate_value_from_field(fd, mode="sample", context=context)

    def _generate_value_from_field(
        self, fd: FieldDefinition, mode: str = "template", context: Optional[dict] = None
    ) -> Any:
        """Recursively generate a value from a FieldDefinition.

        Args:
            fd: field definition
            mode: "template" (placeholder text) or "sample" (example data)
            context: optional dict with task_type, sample_index etc.
        """
        from datarecipe.analyzers.spec_analyzer import _map_type
        json_type = _map_type(fd.type)

        # Enum → pick first value for template, vary for sample
        if fd.enum:
            if mode == "template":
                return fd.enum[0] if fd.enum else ""
            else:
                idx = (context or {}).get("sample_index", 0)
                return fd.enum[idx % len(fd.enum)]

        # Object with properties → recurse
        if json_type == "object" and fd.properties:
            obj = {}
            for p in fd.properties:
                obj[p.name] = self._generate_value_from_field(p, mode=mode, context=context)
            return obj

        # Array with items → wrap one example item
        if json_type == "array" and fd.items:
            item_val = self._generate_value_from_field(fd.items, mode=mode, context=context)
            return [item_val]

        # Scalar types
        desc = fd.description or fd.name
        if json_type == "string":
            if mode == "template":
                return f"[请填写: {desc}]"
            else:
                return f"[{desc}]"
        elif json_type in ("number", "integer"):
            return 0
        elif json_type == "boolean":
            return True
        elif json_type == "array":
            return []
        elif json_type == "object":
            return {}
        return f"[{fd.type}: {desc}]"

    def _generate_data_template(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate data_template.json — schema-driven single data entry template."""
        template: Dict[str, Any] = {"id": "EXAMPLE_001"}

        # Use field_definitions if available, else fall back to profile defaults
        field_defs = analysis.field_definitions
        if not field_defs:
            profile = get_task_profile(analysis.dataset_type)
            field_defs = [FieldDefinition.from_dict(f) for f in profile.default_fields]

        for fd in field_defs:
            if fd.name == "id":
                continue
            template[fd.name] = self._template_placeholder(fd)

        # Add image field if needed and not already present
        if analysis.has_images and "image" not in template:
            template["image"] = {
                "path": "images/example_001.png",
                "type": "software",
                "description": "包含规则定义和待解决问题的图示",
            }

        # Add model test section if difficulty validation is enabled
        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation or {}
            model_name = diff_val.get("model", "未指定模型")
            settings = diff_val.get("settings", "默认设置")
            test_count = diff_val.get("test_count", 3)
            template["model_test"] = {
                "model": model_name,
                "settings": settings,
                "results": [
                    {"attempt": i, "response": f"模型第{i}次的回答...", "is_correct": i == test_count}
                    for i in range(1, test_count + 1)
                ],
                "valid": True,
            }

        template["metadata"] = {
            "category": analysis.estimated_domain or analysis.dataset_type,
            "difficulty": analysis.estimated_difficulty,
            "created_by": "标注员姓名",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "reviewed_by": "",
            "reviewed_at": "",
        }

        path = os.path.join(output_dir, subdirs["templates"], "data_template.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['templates']}/data_template.json")

    def _generate_production_sop(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate PRODUCTION_SOP.md - production standard operating procedure."""
        lines = []
        lines.append(f"# {analysis.project_name} 生产标准操作流程 (SOP)")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据类型: {analysis.dataset_type}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Phase 1: Preparation
        lines.append("## 阶段一：准备阶段")
        lines.append("")
        lines.append("### 1.1 环境准备")
        lines.append("")
        lines.append("- [ ] 创建工作目录结构")
        lines.append("- [ ] 准备标注工具")
        lines.append("- [ ] 配置质检流程")
        lines.append("")

        lines.append("### 1.2 资料准备")
        lines.append("")
        lines.append("- [ ] 阅读 `03_标注规范/ANNOTATION_SPEC.md`")
        lines.append("- [ ] 阅读 `03_标注规范/TRAINING_GUIDE.md`")
        if analysis.has_difficulty_validation():
            lines.append("- [ ] 阅读 `04_复刻指南/DIFFICULTY_VALIDATION.md`")
        lines.append("- [ ] 准备 `07_模板/data_template.json` 模板")
        lines.append("")

        # Phase 2: Content Creation
        lines.append("---")
        lines.append("")
        lines.append("## 阶段二：内容创作")
        lines.append("")

        if analysis.has_images:
            lines.append("### 2.1 图片制作")
            lines.append("")
            lines.append("**要求**：")
            for item in analysis.forbidden_items:
                if "AI" in item or "图" in item:
                    lines.append(f"- ❌ {item}")
            lines.append("- ✅ 使用手绘、软件绘图或照片")
            lines.append("- ✅ 确保分辨率 ≥ 800x600")
            lines.append("")

        lines.append("### 2.2 题目设计")
        lines.append("")
        lines.append("**要求**：")
        lines.append("- 题意清晰，无歧义")
        lines.append("- 与图片强相关（无图无法解题）")
        lines.append("- 明确输出格式要求")
        lines.append("")

        lines.append("### 2.3 答案编写")
        lines.append("")
        lines.append("**要求**：")
        lines.append("- 答案正确且可验证")
        lines.append("- 如有多解，全部列出")
        lines.append("- 格式符合题目要求")
        lines.append("")

        lines.append("### 2.4 解析编写")
        lines.append("")
        lines.append("**要求**：")
        lines.append("- 步骤完整，从题目到答案")
        lines.append("- 逻辑清晰，易于理解")
        lines.append("- 非 AI 生成")
        lines.append("")

        # Phase 3: Difficulty Validation (if enabled)
        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation
            model_name = diff_val.get("model", "未指定模型")
            settings = diff_val.get("settings", "默认设置")
            test_count = diff_val.get("test_count", 3)
            max_correct = diff_val.get("max_correct", 1)

            lines.append("---")
            lines.append("")
            lines.append("## 阶段三：难度验证")
            lines.append("")
            lines.append(f"使用 **{model_name}** ({settings}) 进行验证。")
            lines.append("")
            lines.append("### 验证步骤")
            lines.append("")
            lines.append(f"1. 将题目输入 {model_name}（{settings}）")
            lines.append(f"2. 记录模型回答")
            lines.append(f"3. 判定正确/错误")
            lines.append(f"4. 重复 {test_count} 次（每次新对话）")
            lines.append("")
            lines.append("### 判定标准")
            lines.append("")
            lines.append(f"- ✅ 有效：正确次数 ≤ {max_correct}")
            lines.append(f"- ❌ 无效：正确次数 > {max_correct}，需增加难度后重新验证")
            lines.append("")
            phase_num = 4
        else:
            phase_num = 3

        # Phase 4/3: Quality Check
        lines.append("---")
        lines.append("")
        lines.append(f"## 阶段{phase_num}：质量检查")
        lines.append("")
        lines.append("### 自检清单")
        lines.append("")
        lines.append("对照 `03_标注规范/QA_CHECKLIST.md` 逐项检查：")
        lines.append("")
        if analysis.has_images:
            lines.append("- [ ] 图片：原创、清晰、相关")
        lines.append("- [ ] 题目：清晰、完整、无歧义")
        lines.append("- [ ] 答案：正确、完整、格式规范")
        lines.append("- [ ] 解析：完整、逻辑清晰")
        lines.append("- [ ] 评分标准：具体、可操作")
        if analysis.has_difficulty_validation():
            lines.append("- [ ] 难度验证：已通过")
        lines.append("")

        # Phase 5/4: Submission
        phase_num += 1
        lines.append("---")
        lines.append("")
        lines.append(f"## 阶段{phase_num}：提交")
        lines.append("")
        lines.append("### 提交格式")
        lines.append("")
        lines.append("按照 `04_复刻指南/DATA_SCHEMA.json` 格式提交数据。")
        lines.append("")
        lines.append("### 提交检查")
        lines.append("")
        lines.append("- [ ] JSON 格式正确")
        lines.append("- [ ] 所有必填字段已填写")
        if analysis.has_images:
            lines.append("- [ ] 图片文件已上传")
        if analysis.has_difficulty_validation():
            lines.append("- [ ] 难度验证记录已附上")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本 SOP 由 DataRecipe 自动生成")

        path = os.path.join(output_dir, subdirs["guide"], "PRODUCTION_SOP.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['guide']}/PRODUCTION_SOP.md")

    def _generate_data_schema(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate DATA_SCHEMA.json - JSON schema for data format."""
        if analysis.fields and len(analysis.fields) > 0:
            # Build schema from actual fields via FieldDefinition
            schema = self._build_schema_from_fields(analysis)
        else:
            # Fall back to profile default fields, then generic
            profile = get_task_profile(analysis.dataset_type)
            if profile.default_fields:
                # Temporarily inject profile defaults
                from copy import deepcopy
                tmp = deepcopy(analysis)
                tmp.fields = profile.default_fields
                # Invalidate field_definitions cache
                if hasattr(tmp, "_cached_field_definitions"):
                    object.__setattr__(tmp, "_cached_field_definitions", None)
                schema = self._build_schema_from_fields(tmp)
            else:
                schema = self._build_generic_schema(analysis)

        # Add image field if needed
        if analysis.has_images:
            schema["required"].insert(1, "image")
            schema["properties"]["image"] = {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "图片路径"},
                    "type": {
                        "type": "string",
                        "enum": ["hand_drawn", "software", "photo"],
                        "description": "图片类型",
                    },
                    "description": {"type": "string", "description": "图片描述"},
                },
                "required": ["path", "type"],
            }

        # Add model_test field if difficulty validation is enabled
        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation
            test_count = diff_val.get("test_count", 3)

            schema["required"].append("model_test")
            schema["properties"]["model_test"] = {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "测试使用的模型"},
                    "settings": {"type": "string", "description": "模型配置"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "attempt": {"type": "integer"},
                                "response": {"type": "string"},
                                "is_correct": {"type": "boolean"},
                            },
                            "required": ["attempt", "response", "is_correct"],
                        },
                        "minItems": test_count,
                        "maxItems": test_count,
                    },
                    "valid": {"type": "boolean", "description": "是否通过难度验证"},
                },
                "required": ["model", "results", "valid"],
            }

        path = os.path.join(output_dir, subdirs["guide"], "DATA_SCHEMA.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['guide']}/DATA_SCHEMA.json")

    def _build_schema_from_fields(self, analysis: SpecificationAnalysis) -> dict:
        """Build JSON schema from analysis.field_definitions (FieldDefinition objects)."""
        properties = {}
        required = []

        # Always add id field
        properties["id"] = {
            "type": "string",
            "description": "唯一标识符",
        }
        required.append("id")

        # Convert each FieldDefinition to JSON Schema via to_json_schema()
        for fd in analysis.field_definitions:
            if not fd.name or fd.name == "id":
                continue

            properties[fd.name] = fd.to_json_schema()

            if fd.required:
                required.append(fd.name)

        # Add metadata field
        properties["metadata"] = {
            "type": "object",
            "properties": {
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"]},
                "domain": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
            },
        }

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"{analysis.project_name} 数据格式",
            "type": "object",
            "required": required,
            "properties": properties,
            # Include field definitions for reference
            "x-field-definitions": [fd.to_dict() for fd in analysis.field_definitions],
        }

    def _build_generic_schema(self, analysis: SpecificationAnalysis) -> dict:
        """Build generic fallback schema when no fields are defined."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"{analysis.project_name} 数据格式",
            "type": "object",
            "required": ["id", "question", "answer", "explanation", "scoring_rubric", "metadata"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "唯一标识符",
                    "pattern": "^[A-Z]+_[0-9]+$",
                },
                "question": {
                    "type": "string",
                    "description": "题目文字",
                    "minLength": 10,
                },
                "answer": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "标准答案"},
                        "is_unique": {"type": "boolean", "description": "答案是否唯一"},
                        "alternatives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "其他可接受的答案",
                        },
                    },
                    "required": ["value", "is_unique"],
                },
                "explanation": {
                    "type": "string",
                    "description": "解题过程",
                    "minLength": 20,
                },
                "scoring_rubric": {
                    "type": "object",
                    "properties": {
                        "full_score": {"type": "string", "description": "满分条件"},
                        "partial_score": {"type": "string", "description": "部分得分条件"},
                        "zero_score": {"type": "string", "description": "零分条件"},
                    },
                    "required": ["full_score", "zero_score"],
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"]},
                        "created_by": {"type": "string"},
                        "created_at": {"type": "string", "format": "date"},
                        "reviewed_by": {"type": "string"},
                        "reviewed_at": {"type": "string"},
                    },
                    "required": ["category", "difficulty", "created_by", "created_at"],
                },
            },
        }

    def _estimate_cost_per_item(self, analysis: SpecificationAnalysis, region: str) -> float:
        """Estimate cost per item based on analysis."""
        # Base cost by difficulty
        base_costs = {
            "easy": 5,
            "medium": 10,
            "hard": 20,
            "expert": 40,
        }
        base = base_costs.get(analysis.estimated_difficulty, 15)

        # Multipliers
        multiplier = 1.0

        # Image multiplier
        if analysis.has_images:
            multiplier *= 1.5

        # Complexity multiplier (based on reasoning chain length)
        if len(analysis.reasoning_chain) > 3:
            multiplier *= 1.3

        # Forbidden items multiplier (all human = more expensive)
        if analysis.forbidden_items:
            multiplier *= 1.2

        # Region adjustment
        if region == "china":
            multiplier *= 0.6  # China is cheaper
        elif region == "us":
            multiplier *= 1.0

        return base * multiplier

    # ========== AI Agent Layer ==========

    def _generate_ai_agent_context(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        target_size: int,
        region: str,
        result: SpecOutputResult,
    ):
        """Generate agent_context.json - aggregated entry point for AI agents."""
        cost_per_item = self._estimate_cost_per_item(analysis, region)
        total_cost = cost_per_item * target_size

        context = {
            "_meta": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "generator": "DataRecipe",
                "purpose": "AI Agent 聚合入口，引用其他文件而非复制"
            },
            "project": {
                "name": analysis.project_name,
                "type": analysis.dataset_type,
                "description": analysis.description or analysis.task_description,
                "difficulty": analysis.estimated_difficulty,
                "domain": analysis.estimated_domain,
            },
            "summary": {
                "target_size": target_size,
                "total_cost": round(total_cost, 2),
                "cost_per_item": round(cost_per_item, 2),
                "human_percentage": analysis.estimated_human_percentage,
                "has_images": analysis.has_images,
                "image_count": analysis.image_count,
                "field_count": len(analysis.fields),
                "has_difficulty_validation": analysis.has_difficulty_validation(),
            },
            "key_decisions": [
                {
                    "decision": "difficulty_level",
                    "value": analysis.estimated_difficulty,
                    "reasoning_ref": "#/reasoning/difficulty"
                },
                {
                    "decision": "human_percentage",
                    "value": analysis.estimated_human_percentage,
                    "reasoning_ref": "#/reasoning/human_percentage"
                },
                {
                    "decision": "cost_estimate",
                    "value": round(total_cost, 2),
                    "reasoning_ref": "#/reasoning/cost"
                }
            ],
            "validation": None,
            "file_references": {
                "executive_summary": f"../{subdirs['decision']}/EXECUTIVE_SUMMARY.md",
                "milestone_plan": f"../{subdirs['project']}/MILESTONE_PLAN.md",
                "annotation_spec": f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                "training_guide": f"../{subdirs['annotation']}/TRAINING_GUIDE.md",
                "qa_checklist": f"../{subdirs['annotation']}/QA_CHECKLIST.md",
                "production_sop": f"../{subdirs['guide']}/PRODUCTION_SOP.md",
                "data_schema": f"../{subdirs['guide']}/DATA_SCHEMA.json",
                "cost_breakdown": f"../{subdirs['cost']}/COST_BREAKDOWN.md",
                "data_template": f"../{subdirs['templates']}/data_template.json",
                "raw_analysis": f"../{subdirs['data']}/spec_analysis.json",
            },
            "quick_actions": [
                {
                    "action": "review_spec",
                    "description": "审核标注规范",
                    "file": f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                    "assignee": "human"
                },
                {
                    "action": "setup_tool",
                    "description": "配置标注工具",
                    "config": f"../{subdirs['guide']}/DATA_SCHEMA.json",
                    "assignee": "agent"
                },
                {
                    "action": "create_sample",
                    "description": "创建样本数据",
                    "template": f"../{subdirs['templates']}/data_template.json",
                    "assignee": "human"
                }
            ]
        }

        # Add validation config if present
        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation
            context["validation"] = {
                "enabled": True,
                "model": diff_val.get("model"),
                "settings": diff_val.get("settings"),
                "test_count": diff_val.get("test_count", 3),
                "max_correct": diff_val.get("max_correct", 1),
                "guide_ref": f"../{subdirs['guide']}/DIFFICULTY_VALIDATION.md"
            }
            context["file_references"]["difficulty_validation"] = f"../{subdirs['guide']}/DIFFICULTY_VALIDATION.md"

        path = os.path.join(output_dir, subdirs["ai_agent"], "agent_context.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['ai_agent']}/agent_context.json")

    def _generate_ai_workflow_state(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate workflow_state.json - workflow state tracking."""
        state = {
            "_meta": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "purpose": "工作流状态追踪，供 AI Agent 了解当前进度和下一步"
            },
            "current_phase": "ready_for_review",
            "phases": {
                "analysis": {
                    "status": "completed",
                    "description": "需求文档分析",
                    "outputs": [
                        f"../{subdirs['data']}/spec_analysis.json"
                    ]
                },
                "planning": {
                    "status": "completed",
                    "description": "项目规划与成本估算",
                    "outputs": [
                        f"../{subdirs['decision']}/EXECUTIVE_SUMMARY.md",
                        f"../{subdirs['project']}/MILESTONE_PLAN.md",
                        f"../{subdirs['cost']}/COST_BREAKDOWN.md"
                    ]
                },
                "spec_generation": {
                    "status": "completed",
                    "description": "标注规范与培训材料生成",
                    "outputs": [
                        f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                        f"../{subdirs['annotation']}/TRAINING_GUIDE.md",
                        f"../{subdirs['annotation']}/QA_CHECKLIST.md"
                    ]
                },
                "review": {
                    "status": "pending",
                    "description": "人工审核标注规范",
                    "blocked_by": [],
                    "assignee": "human"
                },
                "pilot": {
                    "status": "pending",
                    "description": "试点标注",
                    "blocked_by": ["review"],
                    "assignee": "human"
                },
                "production": {
                    "status": "pending",
                    "description": "主体标注",
                    "blocked_by": ["pilot"],
                    "assignee": "human"
                },
                "quality_check": {
                    "status": "pending",
                    "description": "质量审核",
                    "blocked_by": ["production"],
                    "assignee": "human"
                }
            },
            "next_actions": [
                {
                    "action": "review_annotation_spec",
                    "description": "审核标注规范是否符合需求",
                    "file": f"../{subdirs['annotation']}/ANNOTATION_SPEC.md",
                    "assignee": "human",
                    "priority": "high"
                },
                {
                    "action": "review_training_guide",
                    "description": "审核培训手册是否清晰",
                    "file": f"../{subdirs['annotation']}/TRAINING_GUIDE.md",
                    "assignee": "human",
                    "priority": "high"
                },
                {
                    "action": "approve_cost_estimate",
                    "description": "确认成本估算并批准预算",
                    "file": f"../{subdirs['cost']}/COST_BREAKDOWN.md",
                    "assignee": "human",
                    "priority": "medium"
                }
            ],
            "blockers": [],
            "decisions_needed": [
                {
                    "question": "标注规范是否需要修改？",
                    "options": ["approved", "needs_revision"],
                    "impact": "影响后续试点阶段启动"
                },
                {
                    "question": "成本预算是否批准？",
                    "options": ["approved", "needs_adjustment", "rejected"],
                    "impact": "影响项目是否继续"
                }
            ]
        }

        path = os.path.join(output_dir, subdirs["ai_agent"], "workflow_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['ai_agent']}/workflow_state.json")

    def _generate_ai_reasoning_traces(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        target_size: int,
        region: str,
        result: SpecOutputResult,
    ):
        """Generate reasoning_traces.json - reasoning chains for all conclusions."""
        cost_per_item = self._estimate_cost_per_item(analysis, region)

        traces = {
            "_meta": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "purpose": "所有结论的推理链，供人类理解和 AI 验证"
            },
            "reasoning": {
                "difficulty": {
                    "conclusion": {
                        "value": analysis.estimated_difficulty,
                        "display": f"难度: {analysis.estimated_difficulty}"
                    },
                    "chain": [],
                    "confidence": 0.0,
                    "assumptions": [
                        "假设标注员无相关领域背景",
                        "假设按照标准培训流程"
                    ],
                    "human_explanation": ""
                },
                "human_percentage": {
                    "conclusion": {
                        "value": analysis.estimated_human_percentage,
                        "display": f"人工比例: {analysis.estimated_human_percentage}%"
                    },
                    "chain": [],
                    "confidence": 0.0,
                    "assumptions": [],
                    "human_explanation": ""
                },
                "cost": {
                    "conclusion": {
                        "value": round(cost_per_item * target_size, 2),
                        "display": f"总成本: ${cost_per_item * target_size:,.0f}"
                    },
                    "chain": [],
                    "confidence": 0.0,
                    "range": {
                        "low": round(cost_per_item * target_size * 0.7, 2),
                        "high": round(cost_per_item * target_size * 1.4, 2)
                    },
                    "assumptions": [],
                    "human_explanation": ""
                }
            }
        }

        # Build difficulty reasoning chain
        difficulty_chain = []
        confidence = 0.5  # Base confidence

        if analysis.reasoning_chain:
            chain_len = len(analysis.reasoning_chain)
            difficulty_chain.append({
                "step": "分析推理链长度",
                "evidence": f"推理链有 {chain_len} 步",
                "impact": "hard" if chain_len > 3 else "medium" if chain_len > 2 else "easy"
            })
            confidence += 0.1

        if analysis.cognitive_requirements:
            req_count = len(analysis.cognitive_requirements)
            difficulty_chain.append({
                "step": "评估认知要求",
                "evidence": f"需要 {req_count} 项认知能力",
                "impact": "expert" if req_count > 3 else "hard" if req_count > 2 else "medium"
            })
            confidence += 0.1

        if analysis.has_images:
            difficulty_chain.append({
                "step": "检测多模态要求",
                "evidence": f"包含 {analysis.image_count} 张图片",
                "impact": "增加难度 - 需要视觉理解能力"
            })
            confidence += 0.1

        if analysis.forbidden_items:
            difficulty_chain.append({
                "step": "检查禁止项",
                "evidence": f"有 {len(analysis.forbidden_items)} 项禁止内容",
                "impact": "增加难度 - 需要更严格的质量控制"
            })
            confidence += 0.05

        traces["reasoning"]["difficulty"]["chain"] = difficulty_chain
        traces["reasoning"]["difficulty"]["confidence"] = min(confidence, 0.95)
        traces["reasoning"]["difficulty"]["human_explanation"] = self._build_difficulty_explanation(analysis)

        # Build human percentage reasoning chain
        human_chain = []
        human_confidence = 0.7

        if analysis.forbidden_items:
            has_ai_restriction = any("AI" in item or "ai" in item.lower() for item in analysis.forbidden_items)
            if has_ai_restriction:
                human_chain.append({
                    "step": "检测 AI 内容限制",
                    "evidence": "禁止使用 AI 生成内容",
                    "impact": "人工比例 100%"
                })
                human_confidence = 0.95
            else:
                human_chain.append({
                    "step": "检测内容限制",
                    "evidence": f"有 {len(analysis.forbidden_items)} 项限制",
                    "impact": "人工比例 > 80%"
                })

        traces["reasoning"]["human_percentage"]["chain"] = human_chain
        traces["reasoning"]["human_percentage"]["confidence"] = human_confidence
        traces["reasoning"]["human_percentage"]["human_explanation"] = (
            f"由于{'禁止使用 AI 生成内容，' if analysis.forbidden_items else ''}"
            f"预估人工比例为 {analysis.estimated_human_percentage}%。"
        )

        # Build cost reasoning chain
        cost_chain = [
            {
                "step": "确定基础成本",
                "evidence": f"难度 {analysis.estimated_difficulty} 对应基础成本",
                "value": {"easy": 5, "medium": 10, "hard": 20, "expert": 40}.get(analysis.estimated_difficulty, 15)
            }
        ]

        if analysis.has_images:
            cost_chain.append({
                "step": "应用图片乘数",
                "evidence": "包含图片，成本 ×1.5",
                "multiplier": 1.5
            })

        if len(analysis.reasoning_chain) > 3:
            cost_chain.append({
                "step": "应用复杂度乘数",
                "evidence": "推理链 > 3 步，成本 ×1.3",
                "multiplier": 1.3
            })

        if analysis.forbidden_items:
            cost_chain.append({
                "step": "应用人工乘数",
                "evidence": "有内容限制，需全人工，成本 ×1.2",
                "multiplier": 1.2
            })

        if region == "china":
            cost_chain.append({
                "step": "应用区域调整",
                "evidence": "中国区域，成本 ×0.6",
                "multiplier": 0.6
            })

        cost_chain.append({
            "step": "计算总成本",
            "evidence": f"单条 ${cost_per_item:.2f} × {target_size} 条",
            "result": round(cost_per_item * target_size, 2)
        })

        traces["reasoning"]["cost"]["chain"] = cost_chain
        traces["reasoning"]["cost"]["confidence"] = 0.75
        traces["reasoning"]["cost"]["assumptions"] = [
            "假设标注效率稳定",
            "假设返工率 < 10%",
            "假设无突发人力短缺"
        ]
        traces["reasoning"]["cost"]["human_explanation"] = (
            f"基于难度({analysis.estimated_difficulty})、图片({analysis.has_images})、"
            f"复杂度({len(analysis.reasoning_chain)}步推理)和区域({region})计算，"
            f"预估总成本 ${cost_per_item * target_size:,.0f}，"
            f"置信区间 ${cost_per_item * target_size * 0.7:,.0f} - ${cost_per_item * target_size * 1.4:,.0f}。"
        )

        path = os.path.join(output_dir, subdirs["ai_agent"], "reasoning_traces.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(traces, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['ai_agent']}/reasoning_traces.json")

    def _build_difficulty_explanation(self, analysis: SpecificationAnalysis) -> str:
        """Build human-readable difficulty explanation."""
        parts = []

        if analysis.cognitive_requirements:
            parts.append(f"需要 {len(analysis.cognitive_requirements)} 项认知能力")

        if analysis.reasoning_chain:
            parts.append(f"推理链长达 {len(analysis.reasoning_chain)} 步")

        if analysis.has_images:
            parts.append("需要视觉理解能力")

        if parts:
            return f"该任务{', '.join(parts)}，属于{analysis.estimated_difficulty}级难度。"
        return f"基于综合评估，难度为{analysis.estimated_difficulty}级。"

    def _generate_ai_pipeline(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate pipeline.yaml - executable pipeline for AI agents."""
        lines = []
        lines.append("# 数据生产流水线")
        lines.append("# 供 AI Agent 执行的可操作步骤")
        lines.append("")
        lines.append("name: 数据生产流水线")
        lines.append("version: '1.0'")
        lines.append(f"project: {analysis.project_name}")
        lines.append(f"generated_at: {datetime.now().isoformat()}")
        lines.append("")

        # Variables section
        lines.append("variables:")
        lines.append(f"  project_name: \"{analysis.project_name}\"")
        lines.append(f"  target_size: 100  # 可调整")
        lines.append(f"  difficulty: \"{analysis.estimated_difficulty}\"")
        if analysis.has_difficulty_validation():
            diff_val = analysis.difficulty_validation
            lines.append(f"  validation_model: \"{diff_val.get('model', '')}\"")
            lines.append(f"  validation_settings: \"{diff_val.get('settings', '')}\"")
            lines.append(f"  validation_test_count: {diff_val.get('test_count', 3)}")
            lines.append(f"  validation_max_correct: {diff_val.get('max_correct', 1)}")
        lines.append("")

        # Phases
        lines.append("phases:")
        lines.append("")

        # Phase 1: Setup
        lines.append("  - name: setup")
        lines.append("    description: 环境准备")
        lines.append("    steps:")
        lines.append("      - action: validate_schema")
        lines.append("        description: 验证数据格式定义")
        lines.append(f"        input: ../{subdirs['guide']}/DATA_SCHEMA.json")
        lines.append("        assignee: agent")
        lines.append("")
        lines.append("      - action: prepare_template")
        lines.append("        description: 准备数据模板")
        lines.append(f"        input: ../{subdirs['templates']}/data_template.json")
        lines.append("        assignee: agent")
        lines.append("")
        lines.append("      - action: review_training_guide")
        lines.append("        description: 审核培训手册")
        lines.append(f"        input: ../{subdirs['annotation']}/TRAINING_GUIDE.md")
        lines.append("        assignee: human")
        lines.append("        required: true")
        lines.append("")

        # Phase 2: Pilot
        lines.append("  - name: pilot")
        lines.append("    description: 试点标注")
        lines.append("    depends_on: [setup]")
        lines.append("    steps:")
        lines.append("      - action: create_pilot_samples")
        lines.append("        description: 创建试点样本 (5-10 条)")
        lines.append(f"        template: ../{subdirs['templates']}/data_template.json")
        lines.append(f"        spec: ../{subdirs['annotation']}/ANNOTATION_SPEC.md")
        lines.append("        count: 10")
        lines.append("        assignee: human")
        lines.append("")
        lines.append("      - action: quality_review_pilot")
        lines.append("        description: 试点质量审核")
        lines.append(f"        checklist: ../{subdirs['annotation']}/QA_CHECKLIST.md")
        lines.append("        assignee: human")
        lines.append("")

        # Phase 3+: Validation phases for each strategy
        last_validation_phase = "pilot"
        for strategy in analysis.parsed_validation_strategies:
            phase_name = f"validation_{strategy.strategy_type}"
            strategy_desc = {
                "model_test": "难度验证",
                "human_review": "人工审核",
                "format_check": "格式校验",
                "cross_validation": "交叉验证",
                "auto_scoring": "自动评分",
            }.get(strategy.strategy_type, strategy.strategy_type)

            lines.append(f"  - name: {phase_name}")
            lines.append(f"    description: {strategy_desc}")
            lines.append(f"    depends_on: [{last_validation_phase}]")
            lines.append("    steps:")

            if strategy.strategy_type == "model_test":
                cfg = strategy.config
                lines.append("      - action: run_model_test")
                lines.append("        description: 执行模型测试")
                lines.append("        config:")
                lines.append(f"          model: \"{cfg.get('model', '')}\"")
                lines.append(f"          settings: \"{cfg.get('settings', '')}\"")
                lines.append(f"          test_count: {cfg.get('test_count', 3)}")
                lines.append(f"          max_correct: {cfg.get('max_correct', 1)}")
                lines.append(f"        reference: ../{subdirs['guide']}/DIFFICULTY_VALIDATION.md")
                lines.append("        assignee: human")
                lines.append("")
                lines.append("      - action: validate_difficulty_result")
                lines.append("        description: 验证难度测试结果")
                lines.append("        on_failure: revise_question")
                lines.append("        assignee: agent")
            elif strategy.strategy_type == "human_review":
                lines.append("      - action: human_review")
                lines.append(f"        description: {strategy.description or '人工审核抽检'}")
                lines.append(f"        checklist: ../{subdirs['annotation']}/QA_CHECKLIST.md")
                lines.append(f"        sample_rate: {strategy.config.get('sample_rate', 0.2)}")
                lines.append("        assignee: human")
            elif strategy.strategy_type == "format_check":
                lines.append("      - action: format_check")
                lines.append("        description: 自动格式校验")
                lines.append(f"        schema: ../{subdirs['guide']}/DATA_SCHEMA.json")
                lines.append("        assignee: agent")
            elif strategy.strategy_type == "cross_validation":
                lines.append("      - action: cross_validation")
                lines.append(f"        description: {strategy.description or '交叉验证'}")
                lines.append(f"        min_annotators: {strategy.config.get('min_annotators', 2)}")
                lines.append(f"        min_kappa: {strategy.config.get('min_kappa', 0.7)}")
                lines.append("        assignee: human")
            elif strategy.strategy_type == "auto_scoring":
                lines.append("      - action: auto_scoring")
                lines.append(f"        description: {strategy.description or '自动评分'}")
                lines.append(f"        threshold: {strategy.config.get('threshold', 0.6)}")
                lines.append("        assignee: agent")
            else:
                lines.append(f"      - action: {strategy.strategy_type}")
                lines.append(f"        description: {strategy.description}")
                lines.append("        assignee: human")

            lines.append("")
            last_validation_phase = phase_name

        # Production phase
        lines.append("  - name: production")
        lines.append("    description: 主体标注")
        lines.append(f"    depends_on: [{last_validation_phase}]")
        lines.append("    steps:")
        lines.append("      - action: batch_annotation")
        lines.append("        description: 批量标注")
        lines.append(f"        template: ../{subdirs['templates']}/data_template.json")
        lines.append(f"        spec: ../{subdirs['annotation']}/ANNOTATION_SPEC.md")
        lines.append("        count: \"{{ target_size }}\"")
        lines.append("        assignee: human")
        lines.append("")
        lines.append("      - action: incremental_qa")
        lines.append("        description: 增量质检")
        lines.append(f"        checklist: ../{subdirs['annotation']}/QA_CHECKLIST.md")
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
        lines.append(f"        checklist: ../{subdirs['annotation']}/QA_CHECKLIST.md")
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
        lines.append("  on_validation_failure:")
        lines.append("    action: increase_difficulty")
        lines.append("    retry: true")
        lines.append("    max_retries: 3")

        path = os.path.join(output_dir, subdirs["ai_agent"], "pipeline.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['ai_agent']}/pipeline.yaml")

    def _generate_ai_readme(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        result: SpecOutputResult,
    ):
        """Generate README.md for AI Agent directory."""
        lines = []
        lines.append(f"# {analysis.project_name} - AI Agent 入口")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
        lines.append("print(f\"项目: {context['project']['name']}\")")
        lines.append("print(f\"难度: {context['project']['difficulty']}\")")
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
        lines.append("difficulty = traces['reasoning']['difficulty']")
        lines.append("print(f\"难度: {difficulty['conclusion']['value']}\")")
        lines.append("print(f\"置信度: {difficulty['confidence']}\")")
        lines.append("print(f\"原因: {difficulty['human_explanation']}\")")
        lines.append("```")
        lines.append("")

        lines.append("### 4. 执行流水线")
        lines.append("")
        lines.append("```python")
        lines.append("import yaml")
        lines.append("")
        lines.append("with open('pipeline.yaml') as f:")
        lines.append("    pipeline = yaml.safe_load(f)")
        lines.append("")
        lines.append("for phase in pipeline['phases']:")
        lines.append("    print(f\"阶段: {phase['name']}\")")
        lines.append("    for step in phase['steps']:")
        lines.append("        print(f\"  - {step['action']}: {step['description']}\")")
        lines.append("```")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 与人类文档的关系")
        lines.append("")
        lines.append("```")
        lines.append("AI Agent 文件              人类文档")
        lines.append("─────────────────────────────────────────────")
        lines.append(f"agent_context.json    →  ../README.md (导航)")
        lines.append(f"workflow_state.json   →  ../{subdirs['project']}/MILESTONE_PLAN.md")
        lines.append(f"reasoning_traces.json →  ../{subdirs['decision']}/EXECUTIVE_SUMMARY.md")
        lines.append(f"pipeline.yaml         →  ../{subdirs['guide']}/PRODUCTION_SOP.md")
        lines.append("```")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("> 本目录由 DataRecipe 自动生成")

        path = os.path.join(output_dir, subdirs["ai_agent"], "README.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['ai_agent']}/README.md")

    def _generate_think_po_samples(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        target_size: int,
        result: SpecOutputResult,
    ):
        """Generate sample data with automation analysis.

        Generates up to 50 sample data entries with:
        - Actual sample data for automatable tasks
        - Clear markers for manual steps when automation isn't possible
        - Reasoning traces for each sample
        """
        samples = []
        max_samples = min(50, target_size)

        # Analyze automation feasibility
        automation_analysis = self._analyze_automation_feasibility(analysis)

        # Get task types from analysis
        task_types = self._extract_task_types(analysis)

        # Generate samples for each task type
        samples_per_type = max(1, max_samples // max(len(task_types), 1))

        for task_type in task_types:
            type_automation = automation_analysis.get(task_type, automation_analysis.get("default", {}))

            for i in range(samples_per_type):
                if len(samples) >= max_samples:
                    break

                sample = self._generate_single_sample(
                    analysis=analysis,
                    task_type=task_type,
                    sample_index=len(samples) + 1,
                    automation_info=type_automation,
                )
                samples.append(sample)

        # Build the complete samples document
        samples_doc = {
            "_meta": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "generator": "DataRecipe Sample Generator",
                "purpose": "生产样例数据，支持人机协同理解",
                "total_samples": len(samples),
                "target_size": target_size,
            },
            "automation_summary": {
                "overall_automation_rate": automation_analysis.get("overall_rate", 0),
                "fully_automated_tasks": automation_analysis.get("fully_automated", []),
                "partially_automated_tasks": automation_analysis.get("partially_automated", []),
                "manual_tasks": automation_analysis.get("manual_only", []),
                "automation_blockers": automation_analysis.get("blockers", []),
            },
            "samples": samples,
            "production_notes": self._generate_production_notes(analysis, automation_analysis),
        }

        # Write JSON file
        json_path = os.path.join(output_dir, subdirs["samples"], "samples.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples_doc, f, indent=2, ensure_ascii=False)
        result.files_generated.append(f"{subdirs['samples']}/samples.json")

        # Generate human-readable guide
        self._generate_samples_guide(analysis, output_dir, subdirs, samples_doc, result)

    def _analyze_automation_feasibility(self, analysis: SpecificationAnalysis) -> dict:
        """Analyze which parts of the pipeline can be automated."""
        result = {
            "overall_rate": 0,
            "fully_automated": [],
            "partially_automated": [],
            "manual_only": [],
            "blockers": [],
            "default": {
                "can_automate": False,
                "automation_rate": 0,
                "manual_steps": [],
                "reasons": [],
            }
        }

        # Check for automation blockers
        blockers = []

        # Check forbidden items
        forbidden_lower = [f.lower() for f in analysis.forbidden_items]
        if any("ai" in f or "机器" in f or "自动" in f for f in forbidden_lower):
            blockers.append({
                "type": "forbidden_ai",
                "description": "需求明确禁止使用AI生成",
                "impact": "所有内容必须人工创作"
            })

        # Check if needs human creativity
        cognitive_lower = " ".join(analysis.cognitive_requirements).lower()
        if "创意" in cognitive_lower or "创作" in cognitive_lower or "原创" in cognitive_lower:
            blockers.append({
                "type": "creativity_required",
                "description": "任务需要人类创意",
                "impact": "核心内容必须人工创作，AI可辅助格式化"
            })

        # Check if needs expert knowledge
        if analysis.estimated_difficulty in ["expert", "hard"]:
            if "专业" in cognitive_lower or "领域" in cognitive_lower:
                blockers.append({
                    "type": "expert_knowledge",
                    "description": "需要专业领域知识",
                    "impact": "需要领域专家参与内容审核"
                })

        # Check if has difficulty validation
        if analysis.has_difficulty_validation():
            blockers.append({
                "type": "difficulty_validation",
                "description": f"需要使用 {analysis.difficulty_validation.get('model', '指定模型')} 进行难度验证",
                "impact": "每条数据需要额外的模型测试步骤"
            })

        result["blockers"] = blockers

        # Determine automation rate based on blockers
        if not blockers:
            result["overall_rate"] = 80
            result["default"]["can_automate"] = True
            result["default"]["automation_rate"] = 80
            result["default"]["manual_steps"] = [
                {"step": "quality_review", "reason": "最终质量把关需要人工确认"}
            ]
        elif any(b["type"] == "forbidden_ai" for b in blockers):
            result["overall_rate"] = 10
            result["default"]["can_automate"] = False
            result["default"]["automation_rate"] = 10
            result["default"]["manual_steps"] = [
                {"step": "content_creation", "reason": "需求禁止AI生成，必须人工创作"},
                {"step": "quality_review", "reason": "人工质检"}
            ]
            result["default"]["reasons"] = ["需求明确禁止AI参与内容生成"]
        elif any(b["type"] == "creativity_required" for b in blockers):
            result["overall_rate"] = 30
            result["default"]["can_automate"] = False
            result["default"]["automation_rate"] = 30
            result["default"]["manual_steps"] = [
                {"step": "content_creation", "reason": "需要人类创意"},
                {"step": "quality_review", "reason": "创意内容需人工评估"}
            ]
            result["default"]["reasons"] = ["任务需要人类创意，AI仅可辅助格式化"]
        else:
            result["overall_rate"] = 50
            result["default"]["can_automate"] = True
            result["default"]["automation_rate"] = 50
            result["default"]["manual_steps"] = [
                {"step": "expert_review", "reason": "专业内容需要专家审核"},
                {"step": "difficulty_validation", "reason": "需要进行难度验证测试"}
            ]

        return result

    def _extract_task_types(self, analysis: SpecificationAnalysis) -> list:
        """Extract task types from analysis."""
        task_types = []

        # Check fields for task_type field
        for field in analysis.fields:
            if field.get("name") == "task_type":
                desc = field.get("description", "")
                # Extract types from description like "understanding/editing/generation"
                if "/" in desc:
                    parts = desc.split("：")[-1] if "：" in desc else desc
                    task_types = [t.strip() for t in parts.split("/")]
                    break

        # If no task types found, use examples
        if not task_types and analysis.examples:
            seen = set()
            for ex in analysis.examples:
                if ex.get("task_type") and ex["task_type"] not in seen:
                    task_types.append(ex["task_type"])
                    seen.add(ex["task_type"])

        # Default to generic if nothing found
        if not task_types:
            task_types = ["default"]

        return task_types

    def _generate_single_sample(
        self,
        analysis: SpecificationAnalysis,
        task_type: str,
        sample_index: int,
        automation_info: dict,
    ) -> dict:
        """Generate a single sample entry (schema-driven)."""
        sample_id = f"SAMPLE_{sample_index:03d}"

        # Use field_definitions for schema-driven generation
        field_defs = analysis.field_definitions
        if not field_defs:
            profile = get_task_profile(analysis.dataset_type)
            field_defs = [FieldDefinition.from_dict(f) for f in profile.default_fields]

        context = {"task_type": task_type, "sample_index": sample_index}
        data_fields: Dict[str, Any] = {}
        for fd in field_defs:
            name = fd.name
            if name == "task_type":
                data_fields[name] = task_type
            elif fd.type == "image":
                data_fields[name] = f"[图片占位符: {fd.description}]"
            elif "svg" in name.lower() or "code" in name.lower():
                data_fields[name] = self._generate_sample_svg(task_type, sample_index)
            elif "instruction" in name.lower() or "question" in name.lower():
                data_fields[name] = self._generate_sample_instruction(analysis, task_type, sample_index)
            else:
                data_fields[name] = self._sample_placeholder(fd, context)

        # Determine automation status for this sample
        can_automate = automation_info.get("can_automate", False)
        automation_rate = automation_info.get("automation_rate", 0)

        sample = {
            "id": sample_id,
            "task_type": task_type,
            "data": data_fields,
            "think_process": {
                "step_1_parse_input": f"解析输入数据，识别任务类型为 {task_type}",
                "step_2_understand_task": f"理解任务要求: {analysis.task_description[:100] if analysis.task_description else '执行指定任务'}...",
                "step_3_execute": "执行任务逻辑，生成输出",
                "step_4_validate": "验证输出符合质量约束",
                "step_5_format": "格式化输出为目标格式",
            },
            "automation_status": {
                "can_fully_automate": can_automate and automation_rate >= 80,
                "automation_rate": automation_rate,
                "automated_steps": [],
                "manual_steps": [],
            },
            "metadata": {
                "difficulty": analysis.estimated_difficulty,
                "domain": analysis.estimated_domain,
                "generated_at": datetime.now().isoformat(),
            }
        }

        # Fill in automation details
        if can_automate and automation_rate >= 80:
            sample["automation_status"]["automated_steps"] = [
                {"step": "input_parsing", "method": "规则解析"},
                {"step": "task_execution", "method": "自动化脚本"},
                {"step": "output_formatting", "method": "模板生成"},
            ]
            sample["automation_status"]["manual_steps"] = [
                {
                    "step": "quality_review",
                    "reason": "最终质量需人工确认",
                    "effort": "低 (抽检即可)"
                }
            ]
        else:
            manual_steps = automation_info.get("manual_steps", [])
            sample["automation_status"]["manual_steps"] = [
                {
                    "step": ms.get("step", "unknown"),
                    "reason": ms.get("reason", "需要人工参与"),
                    "effort": "中" if ms.get("step") != "content_creation" else "高"
                }
                for ms in manual_steps
            ]

            # What can still be automated
            sample["automation_status"]["automated_steps"] = [
                {"step": "input_parsing", "method": "规则解析"},
                {"step": "output_formatting", "method": "模板生成"},
            ]

        return sample

    def _generate_sample_svg(self, task_type: str, index: int) -> str:
        """Generate a sample SVG code."""
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        color = colors[index % len(colors)]

        if "edit" in task_type.lower() or "editing" in task_type.lower():
            return f'''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="40" fill="{color}" />
</svg>'''
        elif "generation" in task_type.lower() or "生成" in task_type:
            return "[待生成的SVG代码]"
        else:
            return f'''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="80" height="80" fill="{color}" />
  <circle cx="50" cy="50" r="20" fill="white" />
</svg>'''

    def _generate_sample_instruction(self, analysis: SpecificationAnalysis, task_type: str, index: int) -> str:
        """Generate a sample instruction based on task type."""
        # Try to use examples from analysis
        for ex in analysis.examples:
            if ex.get("task_type") == task_type and ex.get("question"):
                return ex["question"]

        # Generate based on task type
        instructions = {
            "understanding": [
                "分析以下SVG代码，描述其中包含的图形元素",
                "解释这段SVG代码实现的视觉效果",
                "识别SVG中使用的颜色和形状",
            ],
            "editing": [
                "将圆形的颜色改为红色",
                "增加矩形的宽度为原来的1.5倍",
                "为所有形状添加2px的黑色边框",
            ],
            "generation": [
                "创建一个包含蓝色矩形和红色圆形的SVG",
                "生成一个简单的笑脸图标",
                "绘制一个三角形警告标志",
            ],
            "default": [
                f"执行任务 {index}",
            ]
        }

        task_key = task_type.lower()
        for key in instructions:
            if key in task_key:
                return instructions[key][index % len(instructions[key])]

        return instructions["default"][0]

    def _generate_production_notes(self, analysis: SpecificationAnalysis, automation_analysis: dict) -> dict:
        """Generate production notes based on analysis."""
        overall_rate = automation_analysis.get("overall_rate", 0)
        blockers = automation_analysis.get("blockers", [])

        if overall_rate >= 80:
            recommendation = "可以批量自动化生产"
            workflow = "自动生成 → 抽检审核 → 批量提交"
        elif overall_rate >= 50:
            recommendation = "半自动化生产，需要人工参与关键环节"
            workflow = "自动生成初稿 → 人工审核/修改 → 质检 → 提交"
        elif overall_rate >= 30:
            recommendation = "以人工为主，AI辅助"
            workflow = "人工创作 → AI辅助格式化 → 质检 → 提交"
        else:
            recommendation = "需要全人工生产"
            workflow = "人工创作 → 交叉审核 → 质检 → 提交"

        return {
            "recommendation": recommendation,
            "suggested_workflow": workflow,
            "key_blockers": [b["description"] for b in blockers],
            "optimization_suggestions": self._get_optimization_suggestions(analysis, automation_analysis),
        }

    def _get_optimization_suggestions(self, analysis: SpecificationAnalysis, automation_analysis: dict) -> list:
        """Get suggestions for optimizing the production process."""
        suggestions = []

        if automation_analysis.get("overall_rate", 0) < 50:
            suggestions.append({
                "area": "模板化",
                "suggestion": "创建标准模板减少重复劳动",
                "impact": "可提升10-20%效率"
            })

        if analysis.has_difficulty_validation():
            suggestions.append({
                "area": "难度验证",
                "suggestion": "批量运行难度验证而非逐条测试",
                "impact": "可节省50%验证时间"
            })

        if len(analysis.fields) > 5:
            suggestions.append({
                "area": "字段简化",
                "suggestion": "考虑使用默认值减少必填项",
                "impact": "可提升标注效率"
            })

        suggestions.append({
            "area": "质检抽样",
            "suggestion": "使用分层抽样代替全量检查",
            "impact": "质检效率提升60%以上"
        })

        return suggestions

    def _generate_samples_guide(
        self,
        analysis: SpecificationAnalysis,
        output_dir: str,
        subdirs: dict,
        samples_doc: dict,
        result: SpecOutputResult,
    ):
        """Generate human-readable SAMPLE_GUIDE.md."""
        lines = []
        lines.append(f"# {analysis.project_name} 样例数据指南")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 样例数量: {samples_doc['_meta']['total_samples']} 条")
        lines.append(f"> 目标规模: {samples_doc['_meta']['target_size']} 条")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Automation summary
        auto_summary = samples_doc["automation_summary"]
        lines.append("## 自动化评估")
        lines.append("")

        rate = auto_summary["overall_automation_rate"]
        if rate >= 80:
            status = "🟢 高度自动化"
        elif rate >= 50:
            status = "🟡 半自动化"
        elif rate >= 30:
            status = "🟠 低自动化"
        else:
            status = "🔴 需人工生产"

        lines.append(f"**自动化程度**: {status} ({rate}%)")
        lines.append("")

        # Blockers
        if auto_summary["automation_blockers"]:
            lines.append("### 自动化阻塞因素")
            lines.append("")
            for blocker in auto_summary["automation_blockers"]:
                lines.append(f"- **{blocker['type']}**: {blocker['description']}")
                lines.append(f"  - 影响: {blocker['impact']}")
            lines.append("")

        # Production notes
        prod_notes = samples_doc["production_notes"]
        lines.append("### 生产建议")
        lines.append("")
        lines.append(f"**建议**: {prod_notes['recommendation']}")
        lines.append("")
        lines.append(f"**工作流**: `{prod_notes['suggested_workflow']}`")
        lines.append("")

        # Manual steps explanation
        lines.append("---")
        lines.append("")
        lines.append("## 人工参与说明")
        lines.append("")
        lines.append("以下步骤需要人工参与:")
        lines.append("")

        # Collect all manual steps from samples
        manual_steps_seen = {}
        for sample in samples_doc["samples"][:5]:  # Check first 5 samples
            for ms in sample["automation_status"]["manual_steps"]:
                step = ms["step"]
                if step not in manual_steps_seen:
                    manual_steps_seen[step] = ms

        if manual_steps_seen:
            lines.append("| 步骤 | 原因 | 工作量 |")
            lines.append("|------|------|--------|")
            for step, info in manual_steps_seen.items():
                lines.append(f"| {step} | {info['reason']} | {info.get('effort', '中')} |")
            lines.append("")
        else:
            lines.append("无需人工参与核心生产步骤，仅需抽检审核。")
            lines.append("")

        # Sample examples
        lines.append("---")
        lines.append("")
        lines.append("## 样例展示")
        lines.append("")
        lines.append("以下是生成的样例数据示例:")
        lines.append("")

        for i, sample in enumerate(samples_doc["samples"][:3], 1):
            lines.append(f"### 样例 {i}: {sample['id']}")
            lines.append("")
            lines.append(f"**任务类型**: `{sample['task_type']}`")
            lines.append("")

            # Show data fields
            lines.append("**数据字段**:")
            lines.append("")
            lines.append("```json")
            # Pretty print the data
            import json
            lines.append(json.dumps(sample["data"], indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

            # Show think process
            lines.append("**推理步骤**:")
            lines.append("")
            for step_name, step_desc in sample["think_process"].items():
                step_num = step_name.split("_")[1]
                lines.append(f"{step_num}. {step_desc}")
            lines.append("")

            # Automation status
            auto_status = sample["automation_status"]
            if auto_status["can_fully_automate"]:
                lines.append("**自动化状态**: ✅ 可完全自动化")
            else:
                lines.append(f"**自动化状态**: ⚠️ 部分自动化 ({auto_status['automation_rate']}%)")
                if auto_status["manual_steps"]:
                    lines.append("")
                    lines.append("需人工处理:")
                    for ms in auto_status["manual_steps"]:
                        lines.append(f"- **{ms['step']}**: {ms['reason']}")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Optimization suggestions
        if prod_notes.get("optimization_suggestions"):
            lines.append("## 优化建议")
            lines.append("")
            for sug in prod_notes["optimization_suggestions"]:
                lines.append(f"### {sug['area']}")
                lines.append("")
                lines.append(f"- **建议**: {sug['suggestion']}")
                lines.append(f"- **预期效果**: {sug['impact']}")
                lines.append("")

        # File reference
        lines.append("---")
        lines.append("")
        lines.append("## 文件说明")
        lines.append("")
        lines.append("| 文件 | 用途 | 消费者 |")
        lines.append("|------|------|--------|")
        lines.append("| `samples.json` | 机器可解析的完整样例 | AI Agent |")
        lines.append("| `SAMPLE_GUIDE.md` | 人类可读的样例指南 | 标注团队/项目经理 |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("> 本指南由 DataRecipe 自动生成")

        path = os.path.join(output_dir, subdirs["samples"], "SAMPLE_GUIDE.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append(f"{subdirs['samples']}/SAMPLE_GUIDE.md")
