"""Generate output documents from specification analysis."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from datarecipe.analyzers.spec_analyzer import SpecificationAnalysis


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
            }
            for key, subdir in subdirs.items():
                os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

            result.output_dir = output_dir

            # Generate each document
            self._generate_annotation_spec(analysis, output_dir, subdirs, result)
            self._generate_executive_summary(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_milestone_plan(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_cost_breakdown(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_industry_benchmark(analysis, output_dir, subdirs, target_size, region, result)
            self._generate_raw_analysis(analysis, output_dir, subdirs, result)
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

        lines.append("---")
        lines.append("")
        lines.append("*本规范由 DataRecipe 从需求文档自动生成*")

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
        lines.append("---")
        lines.append("")
        lines.append("## 用途与价值")
        lines.append("")
        lines.append(f"**主要用途**: {analysis.description or analysis.task_description}")
        lines.append("")

        # Risks
        lines.append("---")
        lines.append("")
        lines.append("## 风险评估")
        lines.append("")
        lines.append("| 风险等级 | 描述 | 缓解措施 |")
        lines.append("|----------|------|----------|")

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
        lines.append("*本摘要由 DataRecipe 从需求文档自动生成*")

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
        lines.append("*本计划由 DataRecipe 从需求文档自动生成*")

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
        lines.append("*本成本估算由 DataRecipe 从需求文档自动生成*")

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
        lines.append("*基准数据来源于行业调研，仅供参考*")

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
        lines.append("│   └── ANNOTATION_SPEC.md       # 标注规范")
        lines.append("│")
        lines.append(f"├── {subdirs['cost']}/           # 💰 成本分析")
        lines.append("│   └── COST_BREAKDOWN.md        # 成本明细")
        lines.append("│")
        lines.append(f"└── {subdirs['data']}/           # 📊 原始数据")
        lines.append("    └── spec_analysis.json       # 分析数据")
        lines.append("```")
        lines.append("")
        lines.append("## 快速导航")
        lines.append("")
        lines.append("| 目标 | 查看文件 |")
        lines.append("|------|----------|")
        lines.append(f"| **快速决策** | `{subdirs['decision']}/EXECUTIVE_SUMMARY.md` |")
        lines.append(f"| **项目规划** | `{subdirs['project']}/MILESTONE_PLAN.md` |")
        lines.append(f"| **标注外包** | `{subdirs['annotation']}/ANNOTATION_SPEC.md` |")
        lines.append(f"| **成本预算** | `{subdirs['cost']}/COST_BREAKDOWN.md` |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*由 DataRecipe analyze-spec 命令生成*")

        path = os.path.join(output_dir, "README.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.files_generated.append("README.md")

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
