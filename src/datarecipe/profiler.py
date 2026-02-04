"""标注专家画像生成器

根据数据集分析结果，自动推导所需的标注专家画像：
- 技能要求
- 经验等级
- 人力规模
- 团队结构
- 费率参考
"""

from typing import Optional

from datarecipe.schema import (
    Recipe,
    AnnotatorProfile,
    SkillRequirement,
    ExperienceLevel,
    EducationLevel,
)


# 数据集类型 → 技能要求映射
DATASET_TYPE_SKILLS = {
    "code": [
        SkillRequirement("programming", "Python", "advanced", True),
        SkillRequirement("programming", "代码审查", "intermediate", True),
        SkillRequirement("domain", "软件工程", "intermediate", True),
    ],
    "code_review": [
        SkillRequirement("programming", "目标编程语言", "advanced", True),
        SkillRequirement("domain", "代码审查", "advanced", True),
        SkillRequirement("tool", "Git", "intermediate", True),
    ],
    "rlhf": [
        SkillRequirement("domain", "内容理解", "advanced", True),
        SkillRequirement("language", "目标语言", "native", True),
        SkillRequirement("tool", "标注平台", "basic", True),
    ],
    "preference": [
        SkillRequirement("domain", "内容评估", "advanced", True),
        SkillRequirement("language", "目标语言", "native", True),
        SkillRequirement("domain", "比较判断", "intermediate", True),
    ],
    "medical": [
        SkillRequirement("domain", "医学", "expert", True),
        SkillRequirement("certification", "医师资格证", "required", True),
        SkillRequirement("language", "医学术语", "advanced", True),
    ],
    "legal": [
        SkillRequirement("domain", "法律", "expert", True),
        SkillRequirement("certification", "法律从业资格", "required", True),
    ],
    "financial": [
        SkillRequirement("domain", "金融", "advanced", True),
        SkillRequirement("certification", "金融从业资格", "recommended", False),
    ],
    "math": [
        SkillRequirement("domain", "数学", "advanced", True),
        SkillRequirement("education", "理工科背景", "required", True),
    ],
    "reasoning": [
        SkillRequirement("domain", "逻辑推理", "advanced", True),
        SkillRequirement("education", "理工科背景", "recommended", False),
    ],
    "agent": [
        SkillRequirement("domain", "AI工具使用", "advanced", True),
        SkillRequirement("programming", "Python", "intermediate", False),
        SkillRequirement("tool", "API调试", "intermediate", True),
    ],
    "multilingual": [
        SkillRequirement("language", "目标语言", "native", True),
        SkillRequirement("language", "英语", "C1", False),
    ],
    "translation": [
        SkillRequirement("language", "源语言", "C1", True),
        SkillRequirement("language", "目标语言", "native", True),
        SkillRequirement("domain", "翻译", "advanced", True),
    ],
    "general": [
        SkillRequirement("domain", "通用理解", "intermediate", True),
        SkillRequirement("language", "目标语言", "native", True),
    ],
}

# 数据集类型 → 经验等级映射
DATASET_TYPE_EXPERIENCE = {
    "code": (ExperienceLevel.SENIOR, 3),
    "code_review": (ExperienceLevel.SENIOR, 5),
    "rlhf": (ExperienceLevel.MID, 1),
    "preference": (ExperienceLevel.MID, 1),
    "medical": (ExperienceLevel.EXPERT, 5),
    "legal": (ExperienceLevel.EXPERT, 5),
    "financial": (ExperienceLevel.SENIOR, 3),
    "math": (ExperienceLevel.SENIOR, 3),
    "reasoning": (ExperienceLevel.MID, 2),
    "agent": (ExperienceLevel.MID, 2),
    "multilingual": (ExperienceLevel.MID, 1),
    "translation": (ExperienceLevel.SENIOR, 3),
    "general": (ExperienceLevel.JUNIOR, 0),
}

# 数据集类型 → 学历要求
DATASET_TYPE_EDUCATION = {
    "code": EducationLevel.BACHELOR,
    "code_review": EducationLevel.BACHELOR,
    "medical": EducationLevel.PROFESSIONAL,
    "legal": EducationLevel.PROFESSIONAL,
    "financial": EducationLevel.BACHELOR,
    "math": EducationLevel.MASTER,
    "reasoning": EducationLevel.BACHELOR,
    "general": EducationLevel.HIGH_SCHOOL,
}

# 地区费率系数
REGION_MULTIPLIERS = {
    "us": 1.0,
    "uk": 0.9,
    "eu": 0.85,
    "china": 0.4,
    "cn": 0.4,
    "india": 0.25,
    "in": 0.25,
    "latam": 0.35,
    "sea": 0.3,
}

# 基础小时费率（美元）
BASE_HOURLY_RATES = {
    ExperienceLevel.JUNIOR: {"min": 10, "max": 20},
    ExperienceLevel.MID: {"min": 20, "max": 35},
    ExperienceLevel.SENIOR: {"min": 35, "max": 60},
    ExperienceLevel.EXPERT: {"min": 60, "max": 150},
}

# 每条数据的平均标注时间（分钟）
ANNOTATION_TIME_PER_TYPE = {
    "code": 15,
    "code_review": 30,
    "rlhf": 5,
    "preference": 3,
    "medical": 20,
    "legal": 25,
    "financial": 15,
    "math": 20,
    "reasoning": 10,
    "agent": 10,
    "multilingual": 8,
    "translation": 15,
    "general": 5,
}


class AnnotatorProfiler:
    """标注专家画像生成器"""

    def __init__(self, custom_rules: Optional[dict] = None):
        """初始化

        Args:
            custom_rules: 自定义推导规则，覆盖默认规则
        """
        self.custom_rules = custom_rules or {}

    def generate_profile(
        self,
        recipe: Recipe,
        target_size: Optional[int] = None,
        region: str = "us",
        budget: Optional[float] = None,
    ) -> AnnotatorProfile:
        """根据数据集配方生成标注专家画像

        Args:
            recipe: 数据集配方
            target_size: 目标数据量
            region: 目标地区
            budget: 预算限制

        Returns:
            AnnotatorProfile 标注专家画像
        """
        # 1. 检测数据集类型
        dataset_type = self._detect_dataset_type(recipe)

        # 2. 推导技能要求
        skills = self._derive_skills(dataset_type, recipe)

        # 3. 推导经验等级
        exp_level, min_years = self._derive_experience(dataset_type)

        # 4. 推导学历要求
        education = self._derive_education(dataset_type)

        # 5. 推导语言要求
        languages = self._derive_languages(recipe)

        # 6. 推导领域知识
        domains = self._derive_domains(recipe, dataset_type)

        # 7. 计算团队规模和工作量
        target = target_size or recipe.num_examples or 10000
        team_size, person_days, hours_per_example = self._calculate_workload(
            target, dataset_type, recipe
        )

        # 8. 如果有预算限制，调整团队规模
        if budget:
            team_size, person_days = self._adjust_for_budget(
                budget, team_size, person_days, exp_level, region
            )

        # 9. 计算费率
        hourly_rate = self._calculate_hourly_rate(exp_level, region)

        # 10. 生成筛选标准
        screening = self._generate_screening_criteria(skills, exp_level, domains)

        # 11. 推荐平台
        platforms = self._recommend_platforms(dataset_type, region)

        return AnnotatorProfile(
            skill_requirements=skills,
            experience_level=exp_level,
            min_experience_years=min_years,
            language_requirements=languages,
            domain_knowledge=domains,
            education_level=education,
            team_size=team_size,
            team_structure={
                "annotator": max(1, int(team_size * 0.8)),
                "reviewer": max(1, int(team_size * 0.2)),
            },
            estimated_person_days=person_days,
            estimated_hours_per_example=hours_per_example,
            hourly_rate_range=hourly_rate,
            screening_criteria=screening,
            recommended_platforms=platforms,
        )

    def _detect_dataset_type(self, recipe: Recipe) -> str:
        """检测数据集类型"""
        # 基于标签检测
        tags = " ".join(recipe.tags).lower() if recipe.tags else ""
        description = (recipe.description or "").lower()
        text = tags + " " + description

        # 优先级从高到低检测
        type_keywords = {
            "medical": ["medical", "health", "clinical", "医疗", "医学", "临床"],
            "legal": ["legal", "law", "法律", "合同", "诉讼"],
            "financial": ["financial", "banking", "金融", "投资", "银行"],
            "code_review": ["code review", "代码审查", "pr review"],
            "code": ["code", "programming", "github", "代码", "编程"],
            "math": ["math", "数学", "arithmetic", "calculation"],
            "reasoning": ["reasoning", "logic", "推理", "逻辑"],
            "rlhf": ["rlhf", "preference", "reward", "alignment", "偏好"],
            "preference": ["preference", "ranking", "comparison", "比较", "排序"],
            "agent": ["agent", "tool", "function", "工具", "代理"],
            "translation": ["translation", "translate", "翻译"],
            "multilingual": ["multilingual", "多语言"],
        }

        for dtype, keywords in type_keywords.items():
            if any(kw in text for kw in keywords):
                return dtype

        # 检查语言数量
        if recipe.languages and len(recipe.languages) > 2:
            return "multilingual"

        return "general"

    def _derive_skills(self, dataset_type: str, recipe: Recipe) -> list[SkillRequirement]:
        """推导技能要求"""
        # 检查自定义规则
        if dataset_type in self.custom_rules.get("skills", {}):
            return self.custom_rules["skills"][dataset_type]

        # 获取基础技能
        base_skills = DATASET_TYPE_SKILLS.get(
            dataset_type, DATASET_TYPE_SKILLS["general"]
        )

        # 复制一份避免修改原始数据
        skills = [
            SkillRequirement(
                skill_type=s.skill_type,
                name=s.name,
                level=s.level,
                required=s.required,
                details=s.details,
            )
            for s in base_skills
        ]

        # 根据生成类型添加额外技能
        if recipe.synthetic_ratio and recipe.synthetic_ratio > 0.5:
            skills.append(SkillRequirement(
                "domain", "AI生成内容识别", "intermediate", False
            ))

        return skills

    def _derive_experience(self, dataset_type: str) -> tuple[ExperienceLevel, int]:
        """推导经验要求"""
        return DATASET_TYPE_EXPERIENCE.get(
            dataset_type,
            (ExperienceLevel.MID, 1)
        )

    def _derive_education(self, dataset_type: str) -> EducationLevel:
        """推导学历要求"""
        return DATASET_TYPE_EDUCATION.get(dataset_type, EducationLevel.BACHELOR)

    def _derive_languages(self, recipe: Recipe) -> list[str]:
        """推导语言要求"""
        langs = []
        for lang in recipe.languages or ["en"]:
            if not lang:
                continue
            lang_lower = lang.lower()
            if lang_lower in ["zh", "zh-cn", "zh-tw", "chinese", "中文"]:
                langs.append("zh-CN:native")
            elif lang_lower in ["en", "english", "英语"]:
                langs.append("en:C1")
            elif lang_lower in ["ja", "japanese", "日语"]:
                langs.append("ja:native")
            elif lang_lower in ["ko", "korean", "韩语"]:
                langs.append("ko:native")
            else:
                langs.append(f"{lang}:B2")
        return langs if langs else ["en:C1"]

    def _derive_domains(self, recipe: Recipe, dataset_type: str) -> list[str]:
        """推导领域知识"""
        domains = []
        text = ((recipe.description or "") + " ".join(recipe.tags or [])).lower()

        domain_keywords = {
            "医疗": ["medical", "health", "clinical", "医"],
            "金融": ["financial", "banking", "金融", "投资"],
            "法律": ["legal", "law", "法律", "合同"],
            "技术": ["code", "programming", "software", "技术", "开发"],
            "教育": ["education", "learning", "教育"],
            "电商": ["e-commerce", "retail", "电商", "购物"],
            "客服": ["customer", "service", "客服", "支持"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                domains.append(domain)

        # 根据数据集类型补充
        type_to_domain = {
            "medical": "医疗",
            "legal": "法律",
            "financial": "金融",
            "code": "技术",
            "code_review": "技术",
        }
        if dataset_type in type_to_domain:
            domain = type_to_domain[dataset_type]
            if domain not in domains:
                domains.append(domain)

        return domains or ["通用"]

    def _calculate_workload(
        self,
        target_size: int,
        dataset_type: str,
        recipe: Recipe,
    ) -> tuple[int, float, float]:
        """计算工作量

        Returns:
            (team_size, person_days, hours_per_example)
        """
        # 每个样本的平均标注时间（分钟）
        minutes = ANNOTATION_TIME_PER_TYPE.get(dataset_type, 5)

        # 如果是混合数据集，只需要标注人工部分
        human_ratio = 1.0 if recipe.human_ratio is None else recipe.human_ratio
        effective_size = int(target_size * human_ratio)

        hours_per_example = minutes / 60

        # 总工时
        total_hours = effective_size * hours_per_example

        # 加上审核时间（20%）
        total_hours *= 1.2

        # 人天（8小时/天）
        person_days = total_hours / 8

        # 团队规模（假设项目周期30天）
        team_size = max(2, int(person_days / 25) + 1)

        # 加上审核人员（约20%）
        team_size = int(team_size * 1.25)

        return team_size, round(person_days, 1), round(hours_per_example, 3)

    def _adjust_for_budget(
        self,
        budget: float,
        team_size: int,
        person_days: float,
        exp_level: ExperienceLevel,
        region: str,
    ) -> tuple[int, float]:
        """根据预算调整团队规模"""
        # 计算当前预算需求
        hourly_rate = BASE_HOURLY_RATES[exp_level]["max"]
        region_mult = REGION_MULTIPLIERS.get(region, 1.0)
        effective_rate = hourly_rate * region_mult

        # 当前预算 = 人天 * 8小时 * 费率
        current_budget = person_days * 8 * effective_rate

        if current_budget <= budget:
            return team_size, person_days

        # 需要缩减
        ratio = budget / current_budget
        adjusted_days = person_days * ratio
        adjusted_team = max(2, int(team_size * ratio))

        return adjusted_team, round(adjusted_days, 1)

    def _calculate_hourly_rate(
        self,
        exp_level: ExperienceLevel,
        region: str,
    ) -> dict:
        """计算小时费率"""
        base = BASE_HOURLY_RATES[exp_level]
        multiplier = REGION_MULTIPLIERS.get(region, 1.0)

        return {
            "min": round(base["min"] * multiplier, 2),
            "max": round(base["max"] * multiplier, 2),
            "currency": "USD",
            "region": region,
        }

    def _generate_screening_criteria(
        self,
        skills: list[SkillRequirement],
        exp_level: ExperienceLevel,
        domains: list[str],
    ) -> list[str]:
        """生成筛选标准"""
        criteria = []

        # 技能要求
        for skill in skills:
            if skill.required:
                criteria.append(f"具备{skill.name}能力（{skill.level}级）")

        # 经验要求
        exp_text = {
            ExperienceLevel.JUNIOR: "初级",
            ExperienceLevel.MID: "中级",
            ExperienceLevel.SENIOR: "高级",
            ExperienceLevel.EXPERT: "专家级",
        }
        if exp_level in [ExperienceLevel.SENIOR, ExperienceLevel.EXPERT]:
            criteria.append(f"至少 {exp_text[exp_level]} 工作经验")

        # 领域要求
        for domain in domains:
            if domain != "通用":
                criteria.append(f"具有{domain}领域背景")

        # 通用要求
        criteria.append("通过平台资质审核")
        criteria.append("完成样例任务测试（正确率≥80%）")

        return criteria

    def _recommend_platforms(self, dataset_type: str, region: str) -> list[str]:
        """推荐标注平台"""
        # 根据地区推荐
        if region in ["china", "cn"]:
            base_platforms = ["集识光年"]
        else:
            base_platforms = []

        # 根据类型补充
        type_platforms = {
            "code": ["Scale AI", "Surge AI"],
            "code_review": ["Scale AI", "Surge AI"],
            "medical": ["Scale AI"],
            "legal": ["Scale AI"],
            "general": ["Amazon MTurk", "Prolific"],
            "multilingual": ["Appen", "Lionbridge"],
            "translation": ["Gengo", "Lionbridge"],
        }

        platforms = base_platforms + type_platforms.get(dataset_type, ["Amazon MTurk"])

        return list(dict.fromkeys(platforms))  # 去重保序


def profile_to_markdown(profile: AnnotatorProfile, dataset_name: str = "") -> str:
    """将画像转换为 Markdown 格式"""
    lines = []

    lines.append(f"# 标注专家画像{f'：{dataset_name}' if dataset_name else ''}")
    lines.append("")

    # 概览
    lines.append("## 概览")
    lines.append("")
    lines.append("| 属性 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| **经验等级** | {profile.experience_level.value} |")
    lines.append(f"| **最低经验年限** | {profile.min_experience_years} 年 |")
    lines.append(f"| **学历要求** | {profile.education_level.value} |")
    lines.append(f"| **团队规模** | {profile.team_size} 人 |")
    lines.append(f"| **预估人天** | {profile.estimated_person_days:.1f} 天 |")
    lines.append("")

    # 团队结构
    lines.append("## 团队结构")
    lines.append("")
    for role, count in profile.team_structure.items():
        role_zh = {"annotator": "标注员", "reviewer": "审核员"}.get(role, role)
        lines.append(f"- {role_zh}: {count} 人")
    lines.append("")

    # 技能要求
    lines.append("## 技能要求")
    lines.append("")
    lines.append("| 技能类型 | 技能名称 | 级别 | 必需 |")
    lines.append("|----------|----------|------|------|")
    for skill in profile.skill_requirements:
        required = "是" if skill.required else "否"
        lines.append(f"| {skill.skill_type} | {skill.name} | {skill.level} | {required} |")
    lines.append("")

    # 语言要求
    if profile.language_requirements:
        lines.append("## 语言要求")
        lines.append("")
        for lang in profile.language_requirements:
            lines.append(f"- {lang}")
        lines.append("")

    # 领域知识
    if profile.domain_knowledge:
        lines.append("## 领域知识")
        lines.append("")
        for domain in profile.domain_knowledge:
            lines.append(f"- {domain}")
        lines.append("")

    # 费率参考
    lines.append("## 费率参考")
    lines.append("")
    lines.append(f"- **地区**: {profile.hourly_rate_range.get('region', 'us')}")
    lines.append(f"- **小时费率**: ${profile.hourly_rate_range['min']:.2f} - ${profile.hourly_rate_range['max']:.2f} {profile.hourly_rate_range['currency']}")
    lines.append(f"- **每条数据耗时**: {profile.estimated_hours_per_example * 60:.1f} 分钟")
    lines.append("")

    # 筛选标准
    if profile.screening_criteria:
        lines.append("## 筛选标准")
        lines.append("")
        for criterion in profile.screening_criteria:
            lines.append(f"- [ ] {criterion}")
        lines.append("")

    # 推荐平台
    if profile.recommended_platforms:
        lines.append("## 推荐平台")
        lines.append("")
        for platform in profile.recommended_platforms:
            lines.append(f"- {platform}")
        lines.append("")

    lines.append("---")
    lines.append("*由 DataRecipe 生成*")

    return "\n".join(lines)
