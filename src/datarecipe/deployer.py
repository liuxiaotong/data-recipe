"""投产部署生成器

根据数据配方和标注画像，生成投产配置：
- 标注指南
- 质检规则
- 验收标准
- 项目结构

支持输出到本地文件或调用 Provider API 部署到平台。
"""

from typing import Optional

from datarecipe.providers import ProviderNotFoundError, discover_providers
from datarecipe.schema import (
    AcceptanceCriterion,
    AnnotatorProfile,
    DataRecipe,
    DeploymentProvider,
    DeploymentResult,
    Milestone,
    ProductionConfig,
    QualityRule,
    Recipe,
    ReviewWorkflow,
)


class ProductionDeployer:
    """投产部署生成器"""

    def __init__(self):
        """初始化"""
        self._providers: dict[str, DeploymentProvider] = {}
        self._load_providers()

    def _load_providers(self):
        """加载已安装的 providers"""
        provider_classes = discover_providers()
        for name, provider_class in provider_classes.items():
            try:
                self._providers[name] = provider_class()
            except Exception as e:
                print(f"Warning: Failed to instantiate provider {name}: {e}")

    def list_providers(self) -> list[dict]:
        """列出可用的 providers"""
        return [{"name": name, "description": p.description} for name, p in self._providers.items()]

    def get_provider(self, name: str) -> DeploymentProvider:
        """获取指定 provider"""
        if name not in self._providers:
            raise ProviderNotFoundError(
                f"Provider '{name}' not found. Available: {list(self._providers.keys())}"
            )
        return self._providers[name]

    def generate_config(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile] = None,
    ) -> ProductionConfig:
        """生成投产配置

        Args:
            recipe: 数据配方
            profile: 标注专家画像（可选）

        Returns:
            ProductionConfig 投产配置
        """
        # 1. 生成标注指南
        guide = self._generate_annotation_guide(recipe, profile)

        # 2. 生成质检规则
        quality_rules = self._generate_quality_rules(recipe)

        # 3. 生成验收标准
        acceptance = self._generate_acceptance_criteria(recipe)

        # 4. 确定审核流程
        workflow = self._determine_review_workflow(recipe)

        # 5. 生成里程碑
        milestones = self._generate_milestones(recipe, profile)

        # 6. 估算时间
        days = self._estimate_timeline(recipe, profile)

        return ProductionConfig(
            annotation_guide=guide,
            quality_rules=quality_rules,
            acceptance_criteria=acceptance,
            review_workflow=workflow,
            estimated_timeline_days=days,
            milestones=milestones,
        )

    def deploy(
        self,
        recipe: DataRecipe,
        output: str,
        provider: str = "local",
        config: Optional[ProductionConfig] = None,
        profile: Optional[AnnotatorProfile] = None,
        submit: bool = True,
    ) -> DeploymentResult:
        """部署投产项目

        Args:
            recipe: 数据配方
            output: 输出路径（本地）或项目名称（平台）
            provider: Provider 名称
            config: 投产配置（可选，自动生成）
            profile: 标注画像（可选，自动生成）

        Returns:
            DeploymentResult 部署结果
        """
        # 自动生成缺失的配置
        if profile is None:
            from datarecipe.profiler import AnnotatorProfiler

            profiler = AnnotatorProfiler()
            # 创建临时 Recipe 用于 profiler
            temp_recipe = Recipe(
                name=recipe.name,
                description=recipe.description,
                tags=recipe.tags,
                languages=recipe.languages,
                num_examples=recipe.num_examples,
                synthetic_ratio=recipe.synthetic_ratio,
                human_ratio=recipe.human_ratio,
            )
            profile = profiler.generate_profile(temp_recipe)
            recipe.annotator_profile = profile

        if config is None:
            config = self.generate_config(recipe, profile)
            recipe.production_config = config

        # 获取 provider
        try:
            p = self.get_provider(provider)
        except ProviderNotFoundError as e:
            return DeploymentResult(
                success=False,
                error=str(e),
            )

        # 验证配置
        validation = p.validate_config(config)
        if not validation.valid:
            return DeploymentResult(
                success=False,
                error=f"配置验证失败: {validation.errors}",
                details={"warnings": validation.warnings},
            )

        # 创建项目
        try:
            # 对于 local provider，需要传递 output_dir
            if provider == "local":
                handle = p.create_project(recipe, config, output_dir=output)
            else:
                handle = p.create_project(recipe, config)

            # 对于 local provider，handle 已包含所有信息
            if provider == "local":
                return DeploymentResult(
                    success=True,
                    project_handle=handle,
                    details={
                        "output_path": output,
                        "files_created": handle.metadata.get("files_created", []),
                        "warnings": validation.warnings,
                    },
                )

            # 对于平台 provider，提交项目
            if submit:
                result = p.submit(handle)
                return result

            return DeploymentResult(
                success=True,
                project_handle=handle,
                details={
                    "warnings": validation.warnings,
                    "message": "配置已生成，未自动提交。使用 provider API 手动提交。",
                },
            )

        except Exception as e:
            return DeploymentResult(
                success=False,
                error=str(e),
            )

    def _generate_annotation_guide(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile],
    ) -> str:
        """生成标注指南"""
        lines = []

        lines.append(f"# 标注指南：{recipe.name}")
        lines.append("")

        # 任务概述
        lines.append("## 1. 任务概述")
        lines.append("")
        if recipe.description:
            lines.append(f"> {recipe.description}")
            lines.append("")

        # 数据集信息
        lines.append("### 数据集信息")
        lines.append("")
        lines.append("| 属性 | 值 |")
        lines.append("|------|-----|")
        if recipe.num_examples:
            lines.append(f"| 目标数量 | {recipe.num_examples:,} |")
        if recipe.synthetic_ratio is not None:
            lines.append(f"| 合成比例 | {recipe.synthetic_ratio * 100:.0f}% |")
        if recipe.languages:
            valid_langs = [l for l in recipe.languages if l]
            if valid_langs:
                lines.append(f"| 语言 | {', '.join(valid_langs)} |")
        lines.append("")

        # 标注目标
        lines.append("## 2. 标注目标")
        lines.append("")
        lines.append("请根据以下标准对每条数据进行标注：")
        lines.append("")

        # 根据数据集类型生成具体目标
        if recipe.generation_type.value == "synthetic":
            lines.append("### 合成数据验证")
            lines.append("- 验证生成内容的准确性和相关性")
            lines.append("- 检查是否存在事实性错误")
            lines.append("- 评估内容质量和流畅度")
            lines.append("- 标记需要修正的内容")
        elif recipe.generation_type.value == "human":
            lines.append("### 人工标注要求")
            lines.append("- 按照预定义类别进行分类")
            lines.append("- 标注关键信息和实体")
            lines.append("- 评估内容质量")
            lines.append("- 记录不确定的情况")
        else:
            lines.append("### 混合数据处理")
            lines.append("- 区分合成数据和人工数据")
            lines.append("- 验证合成数据的准确性")
            lines.append("- 按标准进行人工标注")
            lines.append("- 确保整体一致性")
        lines.append("")

        # 标注标准
        lines.append("## 3. 标注标准")
        lines.append("")

        lines.append("### 3.1 质量要求")
        lines.append("")
        lines.append("| 维度 | 要求 | 说明 |")
        lines.append("|------|------|------|")
        lines.append("| 准确性 | 必须 | 标注必须符合实际内容 |")
        lines.append("| 完整性 | 必须 | 所有必填字段都需填写 |")
        lines.append("| 一致性 | 必须 | 相似内容应有相似的标注 |")
        lines.append("| 时效性 | 建议 | 标注应在规定时间内完成 |")
        lines.append("")

        lines.append("### 3.2 格式要求")
        lines.append("")
        lines.append("- 遵循字段定义中的格式说明")
        lines.append("- 使用规定的标签/类别")
        lines.append("- 避免拼写和语法错误")
        lines.append("- 保持格式统一")
        lines.append("")

        # 示例
        lines.append("## 4. 示例")
        lines.append("")

        lines.append("### 4.1 正确示例")
        lines.append("")
        lines.append("```json")
        lines.append("{")
        lines.append('  "input": "示例输入内容...",')
        lines.append('  "output": "标注结果...",')
        lines.append('  "label": "正确的标签",')
        lines.append('  "confidence": "high"')
        lines.append("}")
        lines.append("```")
        lines.append("")
        lines.append("**说明**: 此示例展示了完整、准确的标注。")
        lines.append("")

        lines.append("### 4.2 错误示例")
        lines.append("")
        lines.append("```json")
        lines.append("{")
        lines.append('  "input": "示例输入内容...",')
        lines.append('  "output": "",  // 错误：输出为空')
        lines.append('  "label": "错误标签"  // 错误：标签不正确')
        lines.append("}")
        lines.append("```")
        lines.append("")
        lines.append("**问题**: 输出内容缺失，标签选择错误。")
        lines.append("")

        # 特殊情况处理
        lines.append("## 5. 特殊情况处理")
        lines.append("")
        lines.append("### 5.1 模糊情况")
        lines.append("- 如果内容含义不明确，选择最可能的标签并在备注中说明")
        lines.append("- 如果多个标签都适用，选择最主要的一个")
        lines.append("")
        lines.append("### 5.2 无法标注")
        lines.append("- 如果内容完全无法理解，标记为「无法标注」")
        lines.append("- 在备注中说明具体原因")
        lines.append("")
        lines.append("### 5.3 内容问题")
        lines.append("- 如发现违规内容，立即标记并跳过")
        lines.append("- 如发现数据错误，记录并报告")
        lines.append("")

        # FAQ
        lines.append("## 6. 常见问题")
        lines.append("")
        lines.append("**Q: 遇到不确定的情况怎么办？**")
        lines.append("A: 先尝试根据上下文判断，如仍不确定，在备注中说明后选择最可能的选项。")
        lines.append("")
        lines.append("**Q: 发现数据本身有问题怎么办？**")
        lines.append("A: 标记该条数据为「数据异常」，在备注中详细描述问题。")
        lines.append("")
        lines.append("**Q: 标注速度和质量如何平衡？**")
        lines.append("A: 质量优先。确保每条标注的准确性后再追求速度。")
        lines.append("")

        # 联系方式
        lines.append("## 7. 联系与反馈")
        lines.append("")
        lines.append("- 如有疑问，请联系项目负责人")
        lines.append("- 定期检查更新通知")
        lines.append("- 欢迎提出改进建议")
        lines.append("")

        lines.append("---")
        lines.append("> 由 DataRecipe 自动生成，请根据实际情况补充完善")

        return "\n".join(lines)

    def _generate_quality_rules(self, recipe: DataRecipe) -> list[QualityRule]:
        """生成质检规则"""
        rules = []

        # 通用规则
        rules.append(
            QualityRule(
                rule_id="QR001",
                name="非空检查",
                description="必填字段不能为空",
                check_type="format",
                severity="error",
                auto_check=True,
            )
        )

        rules.append(
            QualityRule(
                rule_id="QR002",
                name="长度检查",
                description="文本长度在合理范围内（50-10000字符）",
                check_type="format",
                severity="warning",
                auto_check=True,
            )
        )

        rules.append(
            QualityRule(
                rule_id="QR003",
                name="重复检查",
                description="不能与已有数据高度重复（相似度<90%）",
                check_type="consistency",
                severity="error",
                auto_check=True,
            )
        )

        rules.append(
            QualityRule(
                rule_id="QR004",
                name="格式规范",
                description="JSON/文本格式必须规范",
                check_type="format",
                severity="error",
                auto_check=True,
            )
        )

        # 根据数据集类型添加特定规则
        if recipe.generation_type.value in ["synthetic", "mixed"]:
            rules.append(
                QualityRule(
                    rule_id="QR005",
                    name="事实性检查",
                    description="生成内容不能包含明显的事实错误",
                    check_type="content",
                    severity="error",
                    auto_check=False,
                )
            )

            rules.append(
                QualityRule(
                    rule_id="QR006",
                    name="AI痕迹检查",
                    description="检查是否存在明显的AI生成痕迹",
                    check_type="content",
                    severity="warning",
                    auto_check=False,
                )
            )

        # 多语言数据集
        if recipe.languages and len(recipe.languages) > 1:
            rules.append(
                QualityRule(
                    rule_id="QR007",
                    name="语言一致性",
                    description="标注语言与数据语言必须一致",
                    check_type="consistency",
                    severity="error",
                    auto_check=True,
                )
            )

        return rules

    def _generate_acceptance_criteria(self, recipe: DataRecipe) -> list[AcceptanceCriterion]:
        """生成验收标准"""
        criteria = []

        criteria.append(
            AcceptanceCriterion(
                criterion_id="AC001",
                name="完成率",
                description="标注任务完成比例",
                threshold=0.98,
                metric_type="completeness",
                priority="required",
            )
        )

        criteria.append(
            AcceptanceCriterion(
                criterion_id="AC002",
                name="准确率",
                description="抽检准确率",
                threshold=0.95,
                metric_type="accuracy",
                priority="required",
            )
        )

        criteria.append(
            AcceptanceCriterion(
                criterion_id="AC003",
                name="一致性",
                description="标注者间一致性（Cohen's Kappa ≥ 0.7）",
                threshold=0.7,
                metric_type="agreement",
                priority="required",
            )
        )

        criteria.append(
            AcceptanceCriterion(
                criterion_id="AC004",
                name="格式合规",
                description="数据格式符合规范的比例",
                threshold=1.0,
                metric_type="format",
                priority="required",
            )
        )

        criteria.append(
            AcceptanceCriterion(
                criterion_id="AC005",
                name="时效性",
                description="按时完成率",
                threshold=0.9,
                metric_type="timeliness",
                priority="recommended",
            )
        )

        return criteria

    def _determine_review_workflow(self, recipe: DataRecipe) -> ReviewWorkflow:
        """确定审核流程"""
        # 高人工比例 -> 双人审核
        if recipe.human_ratio and recipe.human_ratio > 0.5:
            return ReviewWorkflow.DOUBLE

        # 专业领域 -> 专家审核
        tags = " ".join(recipe.tags or []).lower()
        if any(kw in tags for kw in ["medical", "legal", "financial", "医疗", "法律", "金融"]):
            return ReviewWorkflow.EXPERT

        # 大规模数据集 -> 双人审核
        if recipe.num_examples and recipe.num_examples > 50000:
            return ReviewWorkflow.DOUBLE

        return ReviewWorkflow.SINGLE

    def _generate_milestones(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile],
    ) -> list[Milestone]:
        """生成项目里程碑"""
        milestones = []

        # 估算总时间
        total_days = self._estimate_timeline(recipe, profile)

        # M1: 项目启动
        milestones.append(
            Milestone(
                name="项目启动",
                description="完成项目设置和团队组建",
                deliverables=[
                    "标注指南定稿",
                    "团队培训完成",
                    "工具配置完成",
                    "样例数据准备",
                ],
                estimated_days=max(2, int(total_days * 0.1)),
            )
        )

        # M2: 试标注
        milestones.append(
            Milestone(
                name="试标注",
                description="小规模试标注验证流程",
                deliverables=[
                    "100条试标注完成",
                    "问题清单",
                    "指南修订",
                    "质量基准建立",
                ],
                estimated_days=max(3, int(total_days * 0.15)),
                dependencies=["项目启动"],
            )
        )

        # M3: 正式标注
        milestones.append(
            Milestone(
                name="正式标注",
                description="大规模标注执行",
                deliverables=[
                    "全量数据标注完成",
                    "进度日报",
                    "质量周报",
                ],
                estimated_days=max(10, int(total_days * 0.5)),
                dependencies=["试标注"],
            )
        )

        # M4: 质检验收
        milestones.append(
            Milestone(
                name="质检验收",
                description="质量检查和验收",
                deliverables=[
                    "质检报告",
                    "问题数据修正",
                    "最终数据集",
                    "项目总结",
                ],
                estimated_days=max(3, int(total_days * 0.15)),
                dependencies=["正式标注"],
            )
        )

        # M5: 交付归档
        milestones.append(
            Milestone(
                name="交付归档",
                description="数据交付和项目归档",
                deliverables=[
                    "数据交付",
                    "文档归档",
                    "复盘报告",
                ],
                estimated_days=max(2, int(total_days * 0.1)),
                dependencies=["质检验收"],
            )
        )

        return milestones

    def _estimate_timeline(
        self,
        recipe: DataRecipe,
        profile: Optional[AnnotatorProfile],
    ) -> int:
        """估算项目时间"""
        base_days = 30

        if profile and profile.estimated_person_days:
            # 假设团队并行工作
            team_size = profile.team_size or 10
            base_days = int(profile.estimated_person_days / max(1, team_size * 0.8)) + 10

        # 根据数据量调整
        if recipe.num_examples:
            if recipe.num_examples < 1000:
                base_days = max(14, base_days)
            elif recipe.num_examples > 100000:
                base_days = max(60, base_days)

        return max(14, min(180, base_days))  # 最少2周，最多6个月
