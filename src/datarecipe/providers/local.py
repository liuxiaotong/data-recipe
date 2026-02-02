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

        # 5. quality_rules.yaml
        if config and config.quality_rules:
            rules_path = output_path / "quality_rules.yaml"
            rules_data = [r.to_dict() for r in config.quality_rules]
            rules_path.write_text(
                yaml.dump(rules_data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8"
            )
            files_created.append(str(rules_path))

        # 6. acceptance_criteria.yaml
        if config and config.acceptance_criteria:
            criteria_path = output_path / "acceptance_criteria.yaml"
            criteria_data = [c.to_dict() for c in config.acceptance_criteria]
            criteria_path.write_text(
                yaml.dump(criteria_data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8"
            )
            files_created.append(str(criteria_path))

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
