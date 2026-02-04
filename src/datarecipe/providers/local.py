"""本地文件 Provider——默认实现

将投产配置输出到本地文件系统，生成完整的项目结构。
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml

from datarecipe.schema import (
    DataRecipe,
    ProductionConfig,
    AnnotatorProfile,
    ValidationResult,
    AnnotatorMatch,
    ProjectHandle,
    DeploymentResult,
    ProjectStatus,
)


class LocalFilesProvider:
    """本地文件 Provider"""

    @property
    def name(self) -> str:
        return "local"

    @property
    def description(self) -> str:
        return "输出到本地文件（默认 provider）"

    def validate_config(self, config: ProductionConfig) -> ValidationResult:
        """验证配置"""
        errors = []
        warnings = []

        if not config.annotation_guide:
            warnings.append("标注指南为空")

        if not config.quality_rules:
            warnings.append("未定义质检规则")

        if not config.acceptance_criteria:
            warnings.append("未定义验收标准")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def match_annotators(
        self,
        profile: AnnotatorProfile,
        limit: int = 10,
    ) -> list[AnnotatorMatch]:
        """本地模式不支持匹配标注者"""
        return []

    def create_project(
        self,
        recipe: DataRecipe,
        config: Optional[ProductionConfig] = None,
        output_dir: str = "./project",
    ) -> ProjectHandle:
        """创建本地项目结构

        Args:
            recipe: 数据配方
            config: 投产配置
            output_dir: 输出目录

        Returns:
            ProjectHandle 项目句柄
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files_created = []

        # 1. recipe.yaml
        recipe_path = output_path / "recipe.yaml"
        recipe_path.write_text(
            yaml.dump(recipe.to_dict(), default_flow_style=False, allow_unicode=True),
            encoding="utf-8"
        )
        files_created.append(str(recipe_path))

        # 2. annotator_profile.yaml
        if recipe.annotator_profile:
            profile_path = output_path / "annotator_profile.yaml"
            profile_path.write_text(recipe.annotator_profile.to_yaml(), encoding="utf-8")
            files_created.append(str(profile_path))

        # 3. cost_estimate.yaml
        if recipe.enhanced_cost:
            cost_path = output_path / "cost_estimate.yaml"
            cost_path.write_text(
                yaml.dump(recipe.enhanced_cost.to_dict(), default_flow_style=False, allow_unicode=True),
                encoding="utf-8"
            )
            files_created.append(str(cost_path))

        # 4. annotation_guide.md
        if config and config.annotation_guide:
            guide_path = output_path / "annotation_guide.md"
            guide_path.write_text(config.annotation_guide, encoding="utf-8")
            files_created.append(str(guide_path))

        # 5. quality_rules.yaml + quality_rules.md (人类可读版本)
        if config and config.quality_rules:
            rules_path = output_path / "quality_rules.yaml"
            rules_data = [r.to_dict() for r in config.quality_rules]
            rules_path.write_text(
                yaml.dump(rules_data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8"
            )
            files_created.append(str(rules_path))

            # 生成人类可读的 Markdown 版本
            rules_md_path = output_path / "quality_rules.md"
            rules_md_content = self._generate_quality_rules_md(config.quality_rules)
            rules_md_path.write_text(rules_md_content, encoding="utf-8")
            files_created.append(str(rules_md_path))

        # 6. acceptance_criteria.yaml + acceptance_criteria.md (人类可读版本)
        if config and config.acceptance_criteria:
            criteria_path = output_path / "acceptance_criteria.yaml"
            criteria_data = [c.to_dict() for c in config.acceptance_criteria]
            criteria_path.write_text(
                yaml.dump(criteria_data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8"
            )
            files_created.append(str(criteria_path))

            # 生成人类可读的 Markdown 版本
            criteria_md_path = output_path / "acceptance_criteria.md"
            criteria_md_content = self._generate_acceptance_criteria_md(config.acceptance_criteria)
            criteria_md_path.write_text(criteria_md_content, encoding="utf-8")
            files_created.append(str(criteria_md_path))

        # 7. timeline.md
        if config and config.milestones:
            timeline_path = output_path / "timeline.md"
            timeline_content = self._generate_timeline_md(config)
            timeline_path.write_text(timeline_content, encoding="utf-8")
            files_created.append(str(timeline_path))

        # 8. README.md
        readme_path = output_path / "README.md"
        readme_content = self._generate_readme(recipe, config)
        readme_path.write_text(readme_content, encoding="utf-8")
        files_created.append(str(readme_path))

        # 9. scripts/ 目录
        scripts_dir = output_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # 生成脚本文件
        script_files = self._generate_scripts(recipe, scripts_dir)
        files_created.extend(script_files)

        return ProjectHandle(
            project_id=f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            provider="local",
            created_at=datetime.now().isoformat(),
            status="created",
            metadata={
                "output_dir": str(output_path),
                "files_created": files_created,
            },
        )

    def submit(self, project: ProjectHandle) -> DeploymentResult:
        """本地模式不需要提交"""
        return DeploymentResult(
            success=True,
            project_handle=project,
            details={"message": "本地项目创建成功"},
        )

    def get_status(self, project: ProjectHandle) -> ProjectStatus:
        """获取项目状态"""
        return ProjectStatus(
            status="completed",
            progress=100.0,
            completed_count=0,
            total_count=0,
        )

    def cancel(self, project: ProjectHandle) -> bool:
        """取消项目"""
        return True

    def _generate_readme(self, recipe: DataRecipe, config: Optional[ProductionConfig]) -> str:
        """生成 README"""
        lines = []
        lines.append(f"# {recipe.name} 投产项目")
        lines.append("")
        lines.append("## 项目概述")
        lines.append("")
        if recipe.description:
            lines.append(recipe.description)
        lines.append("")

        # 基本信息
        lines.append("## 基本信息")
        lines.append("")
        lines.append("| 属性 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| **数据集名称** | {recipe.name} |")
        lines.append(f"| **数据来源** | {recipe.source_type.value} |")
        if recipe.num_examples:
            lines.append(f"| **目标数量** | {recipe.num_examples:,} |")
        if recipe.synthetic_ratio is not None:
            lines.append(f"| **合成比例** | {recipe.synthetic_ratio * 100:.0f}% |")
        if recipe.teacher_models:
            lines.append(f"| **教师模型** | {', '.join(recipe.teacher_models)} |")
        lines.append("")

        # 成本估算
        if recipe.enhanced_cost:
            lines.append("## 成本估算")
            lines.append("")
            lines.append(f"- **API 成本**: ${recipe.enhanced_cost.api_cost:,.2f}")
            lines.append(f"- **人力成本**: ${recipe.enhanced_cost.human_cost:,.2f}")
            lines.append(f"- **总成本**: ${recipe.enhanced_cost.total_cost:,.2f}")
            lines.append(f"- **成本区间**: ${recipe.enhanced_cost.total_range['low']:,.2f} - ${recipe.enhanced_cost.total_range['high']:,.2f}")
            lines.append("")

        # 项目结构
        lines.append("## 项目结构")
        lines.append("")
        lines.append("```")
        lines.append("./")
        lines.append("├── README.md              # 本文件")
        lines.append("├── recipe.yaml            # 数据配方")
        lines.append("├── annotator_profile.yaml # 标注专家画像")
        lines.append("├── cost_estimate.yaml     # 成本估算")
        lines.append("├── annotation_guide.md    # 标注指南")
        lines.append("├── quality_rules.yaml     # 质检规则")
        lines.append("├── acceptance_criteria.yaml # 验收标准")
        lines.append("├── timeline.md            # 项目时间线")
        lines.append("└── scripts/               # 脚本目录")
        lines.append("    ├── 01_prepare_data.py")
        lines.append("    ├── 02_generate.py")
        lines.append("    └── 03_validate.py")
        lines.append("```")
        lines.append("")

        # 快速开始
        lines.append("## 快速开始")
        lines.append("")
        lines.append("1. 阅读 `annotation_guide.md` 了解标注要求")
        lines.append("2. 根据 `annotator_profile.yaml` 组建团队")
        lines.append("3. 按照 `timeline.md` 执行项目")
        lines.append("4. 使用 `quality_rules.yaml` 进行质检")
        lines.append("")

        # 时间规划
        if config and config.milestones:
            lines.append("## 里程碑")
            lines.append("")
            for i, m in enumerate(config.milestones, 1):
                lines.append(f"{i}. **{m.name}** ({m.estimated_days} 天)")
                for d in m.deliverables:
                    lines.append(f"   - {d}")
            lines.append("")

        lines.append("---")
        lines.append("*由 DataRecipe 生成*")
        return "\n".join(lines)

    def _generate_timeline_md(self, config: ProductionConfig) -> str:
        """生成时间线文档"""
        lines = []
        lines.append("# 项目时间线")
        lines.append("")
        lines.append(f"**预计总工期**: {config.estimated_timeline_days} 天")
        lines.append("")
        lines.append(f"**审核流程**: {config.review_workflow.value}")
        lines.append(f"**抽检比例**: {config.review_sample_rate * 100:.0f}%")
        lines.append("")

        lines.append("## 里程碑")
        lines.append("")

        total_days = 0
        for i, m in enumerate(config.milestones, 1):
            lines.append(f"### M{i}: {m.name}")
            lines.append("")
            lines.append(f"**描述**: {m.description}")
            lines.append(f"**预计天数**: {m.estimated_days}")
            lines.append(f"**累计天数**: {total_days + m.estimated_days}")
            total_days += m.estimated_days
            lines.append("")
            if m.dependencies:
                lines.append(f"**依赖**: {', '.join(m.dependencies)}")
                lines.append("")
            lines.append("**交付物**:")
            for d in m.deliverables:
                lines.append(f"- [ ] {d}")
            lines.append("")

        lines.append("## 甘特图")
        lines.append("")
        lines.append("```")
        current = 0
        for m in config.milestones:
            bar = " " * current + "=" * m.estimated_days
            lines.append(f"{m.name[:15]:<15} |{bar}")
            current += m.estimated_days
        lines.append(f"{'总计':<15} |{'=' * current}")
        lines.append("```")

        return "\n".join(lines)

    def _generate_quality_rules_md(self, rules: list) -> str:
        """生成人类可读的质检规则文档"""
        lines = []
        lines.append("# 质检规则说明")
        lines.append("")
        lines.append("本文档定义了数据标注的质量检查规则，确保交付数据符合质量标准。")
        lines.append("")

        # 规则严重程度说明
        lines.append("## 严重程度说明")
        lines.append("")
        lines.append("| 级别 | 图标 | 含义 | 处理方式 |")
        lines.append("|------|------|------|----------|")
        lines.append("| **error** | 🔴 | 严重问题 | 必须修复后才能通过 |")
        lines.append("| **warning** | 🟡 | 潜在问题 | 建议修复，可酌情放过 |")
        lines.append("| **info** | 🔵 | 提示信息 | 仅供参考 |")
        lines.append("")

        # 检查类型说明
        lines.append("## 检查类型说明")
        lines.append("")
        lines.append("| 类型 | 含义 | 示例 |")
        lines.append("|------|------|------|")
        lines.append("| **format** | 格式检查 | 字段是否为空、长度是否合规、JSON 格式是否正确 |")
        lines.append("| **content** | 内容检查 | 事实是否准确、是否有 AI 痕迹、是否符合主题 |")
        lines.append("| **consistency** | 一致性检查 | 是否与其他数据重复、风格是否统一 |")
        lines.append("")

        # 规则详情
        lines.append("## 规则详情")
        lines.append("")

        # 按类型分组
        rules_by_type = {}
        for r in rules:
            check_type = r.check_type if hasattr(r, 'check_type') else r.get('type', 'other')
            if check_type not in rules_by_type:
                rules_by_type[check_type] = []
            rules_by_type[check_type].append(r)

        type_names = {
            'format': '📋 格式检查规则',
            'content': '📝 内容检查规则',
            'consistency': '🔗 一致性检查规则',
        }

        for check_type, type_rules in rules_by_type.items():
            lines.append(f"### {type_names.get(check_type, check_type)}")
            lines.append("")

            for r in type_rules:
                rule_id = r.rule_id if hasattr(r, 'rule_id') else r.get('id', '')
                name = r.name if hasattr(r, 'name') else r.get('name', '')
                desc = r.description if hasattr(r, 'description') else r.get('description', '')
                severity = r.severity if hasattr(r, 'severity') else r.get('severity', 'warning')
                auto = r.auto_check if hasattr(r, 'auto_check') else r.get('auto_check', False)

                # 严重程度图标
                severity_icon = {'error': '🔴', 'warning': '🟡', 'info': '🔵'}.get(severity, '⚪')
                auto_label = "✅ 自动检查" if auto else "👤 人工检查"

                lines.append(f"#### {severity_icon} {rule_id}: {name}")
                lines.append("")
                lines.append(f"**描述**: {desc}")
                lines.append("")
                lines.append(f"- **严重程度**: {severity}")
                lines.append(f"- **检查方式**: {auto_label}")
                lines.append("")

        # 使用指南
        lines.append("## 使用指南")
        lines.append("")
        lines.append("### 质检流程")
        lines.append("")
        lines.append("1. **自动检查**: 系统自动执行标记为「自动检查」的规则")
        lines.append("2. **人工抽检**: 质检员随机抽取样本进行人工检查")
        lines.append("3. **问题标记**: 发现问题的数据标记对应规则 ID")
        lines.append("4. **修正反馈**: 标注员根据问题反馈进行修正")
        lines.append("5. **复核验收**: 修正后重新检查直至通过")
        lines.append("")

        lines.append("### 常见问题")
        lines.append("")
        lines.append("**Q: 同时触发多条规则怎么办？**")
        lines.append("A: 优先处理 error 级别的问题，warning 可在后续批量处理。")
        lines.append("")
        lines.append("**Q: 规则判断有歧义怎么办？**")
        lines.append("A: 联系项目负责人确认，必要时更新规则说明。")
        lines.append("")

        lines.append("---")
        lines.append("*由 DataRecipe 生成*")

        return "\n".join(lines)

    def _generate_acceptance_criteria_md(self, criteria: list) -> str:
        """生成人类可读的验收标准文档"""
        lines = []
        lines.append("# 验收标准说明")
        lines.append("")
        lines.append("本文档定义了项目验收的量化标准，所有指标达标后方可正式交付。")
        lines.append("")

        # 总览表
        lines.append("## 验收指标总览")
        lines.append("")
        lines.append("| 指标 | 阈值 | 优先级 | 说明 |")
        lines.append("|------|------|--------|------|")

        for c in criteria:
            name = c.name if hasattr(c, 'name') else c.get('name', '')
            threshold = c.threshold if hasattr(c, 'threshold') else c.get('threshold', 0)
            priority = c.priority if hasattr(c, 'priority') else c.get('priority', 'required')
            desc = c.description if hasattr(c, 'description') else c.get('description', '')

            # 优先级图标
            priority_icon = "🔴 必须" if priority == 'required' else "🟢 建议"
            threshold_str = f"{threshold * 100:.0f}%" if threshold <= 1 else str(threshold)

            lines.append(f"| **{name}** | ≥ {threshold_str} | {priority_icon} | {desc} |")

        lines.append("")

        # 详细说明
        lines.append("## 指标详细说明")
        lines.append("")

        metric_explanations = {
            'completeness': {
                'title': '📊 完成率',
                'what': '衡量标注任务的完成程度',
                'how': '完成率 = 已完成条数 / 总任务条数 × 100%',
                'tips': [
                    '确保所有分配的任务都已处理',
                    '「无法标注」的数据也计入已完成',
                    '每日同步进度，及时发现落后情况',
                ],
            },
            'accuracy': {
                'title': '🎯 准确率',
                'what': '衡量标注结果的正确程度',
                'how': '准确率 = 抽检正确数 / 抽检总数 × 100%',
                'tips': [
                    '由质检员随机抽样检查',
                    '与标准答案或专家判断对比',
                    '低于阈值需要返工修正',
                ],
            },
            'agreement': {
                'title': '🤝 一致性',
                'what': '衡量不同标注者之间的标注一致程度',
                'how': '使用 Cohen\'s Kappa 系数衡量，值域 [-1, 1]',
                'tips': [
                    'Kappa ≥ 0.8: 几乎完全一致（优秀）',
                    'Kappa ≥ 0.6: 基本一致（良好）',
                    'Kappa < 0.4: 一致性差（需改进）',
                ],
            },
            'format': {
                'title': '📋 格式合规',
                'what': '衡量数据格式的规范程度',
                'how': '格式合规率 = 格式正确条数 / 总条数 × 100%',
                'tips': [
                    '使用自动化脚本检查',
                    'JSON 格式、字段完整性、编码规范',
                    '格式错误必须 100% 修复',
                ],
            },
            'timeliness': {
                'title': '⏰ 时效性',
                'what': '衡量按时完成的情况',
                'how': '时效性 = 按时完成的里程碑数 / 总里程碑数 × 100%',
                'tips': [
                    '每个里程碑有预定完成日期',
                    '提前预警风险，及时调整',
                    '合理延期需提前申请',
                ],
            },
        }

        for c in criteria:
            metric_type = c.metric_type if hasattr(c, 'metric_type') else c.get('type', '')
            name = c.name if hasattr(c, 'name') else c.get('name', '')
            threshold = c.threshold if hasattr(c, 'threshold') else c.get('threshold', 0)
            desc = c.description if hasattr(c, 'description') else c.get('description', '')

            exp = metric_explanations.get(metric_type, {})
            title = exp.get('title', f"📌 {name}")

            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"**定义**: {desc}")
            lines.append("")

            threshold_str = f"{threshold * 100:.0f}%" if threshold <= 1 else str(threshold)
            lines.append(f"**验收阈值**: ≥ **{threshold_str}**")
            lines.append("")

            if 'what' in exp:
                lines.append(f"**含义**: {exp['what']}")
                lines.append("")

            if 'how' in exp:
                lines.append(f"**计算方式**: {exp['how']}")
                lines.append("")

            if 'tips' in exp:
                lines.append("**实践建议**:")
                for tip in exp['tips']:
                    lines.append(f"- {tip}")
                lines.append("")

        # 验收流程
        lines.append("## 验收流程")
        lines.append("")
        lines.append("```")
        lines.append("┌─────────────┐    ┌─────────────┐    ┌─────────────┐")
        lines.append("│  自检提交   │ -> │  质检审核   │ -> │  最终验收   │")
        lines.append("└─────────────┘    └─────────────┘    └─────────────┘")
        lines.append("       │                  │                  │")
        lines.append("       v                  v                  v")
        lines.append("  标注团队自查      质检员抽检审核      项目负责人确认")
        lines.append("  所有指标达标      出具质检报告       签字交付")
        lines.append("```")
        lines.append("")

        lines.append("## 不达标处理")
        lines.append("")
        lines.append("| 情况 | 处理方式 |")
        lines.append("|------|----------|")
        lines.append("| 必须指标未达标 | 返工修正，直至达标 |")
        lines.append("| 建议指标未达标 | 评估影响，协商处理 |")
        lines.append("| 多项指标未达标 | 组织复盘，制定改进计划 |")
        lines.append("")

        lines.append("---")
        lines.append("*由 DataRecipe 生成*")

        return "\n".join(lines)

    def _generate_scripts(self, recipe: DataRecipe, scripts_dir: Path) -> list[str]:
        """生成脚本文件"""
        files = []

        # 01_prepare_data.py
        prepare_script = scripts_dir / "01_prepare_data.py"
        prepare_script.write_text(f'''#!/usr/bin/env python3
"""
步骤 1: 准备数据

准备种子数据和配置。
"""

import json
from pathlib import Path

# 配置
OUTPUT_DIR = Path("../data")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = {recipe.num_examples or 10000}

def main():
    print("准备数据...")

    # 加载种子数据或创建初始数据
    seeds = []

    # TODO: 在此添加种子数据加载逻辑

    # 保存种子数据
    with open(OUTPUT_DIR / "seed_data.jsonl", "w", encoding="utf-8") as f:
        for seed in seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + "\\n")

    print(f"准备完成，共 {{len(seeds)}} 条种子数据")


if __name__ == "__main__":
    main()
''', encoding="utf-8")
        files.append(str(prepare_script))

        # 02_generate.py
        generate_script = scripts_dir / "02_generate.py"
        model = recipe.teacher_models[0] if recipe.teacher_models else "gpt-4o"
        generate_script.write_text(f'''#!/usr/bin/env python3
"""
步骤 2: 数据生成

使用 LLM 生成数据。
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# 配置
MODEL = "{model}"
INPUT_FILE = Path("../data/seed_data.jsonl")
OUTPUT_FILE = Path("../data/generated.jsonl")

PROMPT_TEMPLATE = """
请根据以下种子数据生成高质量的训练样本：

种子数据：
{{seed}}

要求：
1. 保持内容准确性
2. 格式规范
3. 语言流畅

生成：
"""


def generate_single(client, seed):
    """生成单条数据"""
    prompt = PROMPT_TEMPLATE.format(seed=json.dumps(seed, ensure_ascii=False))

    # TODO: 根据实际使用的 API 修改
    # response = client.chat.completions.create(
    #     model=MODEL,
    #     messages=[{{"role": "user", "content": prompt}}],
    #     temperature=0.7,
    # )
    # return response.choices[0].message.content

    return None


def main():
    print(f"开始生成，使用模型: {{MODEL}}")

    # 加载种子数据
    seeds = []
    if INPUT_FILE.exists():
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                seeds.append(json.loads(line))

    print(f"加载了 {{len(seeds)}} 条种子数据")

    # TODO: 初始化 API 客户端
    # from openai import OpenAI
    # client = OpenAI()

    generated = []
    for seed in tqdm(seeds, desc="生成中"):
        # result = generate_single(client, seed)
        # if result:
        #     generated.append({{"seed": seed, "generated": result}})
        pass

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")

    print(f"生成完成，共 {{len(generated)}} 条")


if __name__ == "__main__":
    main()
''', encoding="utf-8")
        files.append(str(generate_script))

        # 03_validate.py
        validate_script = scripts_dir / "03_validate.py"
        validate_script.write_text('''#!/usr/bin/env python3
"""
步骤 3: 数据验证

验证生成的数据质量。
"""

import json
from pathlib import Path
from collections import Counter

# 配置
INPUT_FILE = Path("../data/generated.jsonl")
OUTPUT_FILE = Path("../data/validated.jsonl")
REPORT_FILE = Path("../data/validation_report.json")

# 质量规则
MIN_LENGTH = 50
MAX_LENGTH = 10000


def validate_item(item):
    """验证单条数据"""
    errors = []
    warnings = []

    text = item.get("generated", "")

    # 长度检查
    if len(text) < MIN_LENGTH:
        errors.append("文本过短")
    if len(text) > MAX_LENGTH:
        warnings.append("文本过长")

    # 格式检查
    if not text.strip():
        errors.append("内容为空")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def main():
    print("开始验证...")

    # 加载数据
    items = []
    if INPUT_FILE.exists():
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))

    print(f"加载了 {len(items)} 条数据")

    # 验证
    validated = []
    stats = Counter()

    for item in items:
        result = validate_item(item)
        if result["valid"]:
            validated.append(item)
            stats["passed"] += 1
        else:
            stats["failed"] += 1
            for e in result["errors"]:
                stats[f"error:{e}"] += 1

    # 保存验证通过的数据
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in validated:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")

    # 保存报告
    report = {
        "total": len(items),
        "passed": stats["passed"],
        "failed": stats["failed"],
        "pass_rate": stats["passed"] / len(items) if items else 0,
        "details": dict(stats),
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"验证完成: {stats['passed']}/{len(items)} 通过 ({report['pass_rate']*100:.1f}%)")
    print(f"报告保存至: {REPORT_FILE}")


if __name__ == "__main__":
    main()
''', encoding="utf-8")
        files.append(str(validate_script))

        return files
