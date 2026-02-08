"""Unified project layout constants and utilities.

All output pipelines (deep-analyze, analyze-spec, deploy, integrate-report)
share this module so that a single dataset produces one project folder.
"""

import json
import os
from datetime import datetime

# Canonical subdirectory mapping — every numbered folder lives here.
OUTPUT_SUBDIRS = {
    "decision": "01_决策参考",
    "project": "02_项目管理",
    "annotation": "03_标注规范",
    "guide": "04_复刻指南",
    "cost": "05_成本分析",
    "data": "06_原始数据",
    "templates": "07_模板",
    "ai_agent": "08_AI_Agent",
    "samples": "09_样例数据",
    "deploy": "10_生产部署",
    "reports": "11_综合报告",
}

DEFAULT_PROJECTS_DIR = "./projects"


def safe_name(raw: str) -> str:
    """Sanitize a dataset/project name for use as a directory name."""
    return raw.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


class ProjectManifest:
    """Read/write .project_manifest.json at the project root.

    Tracks which commands have been executed and when, so the README
    generator can show only relevant sections.
    """

    FILENAME = ".project_manifest.json"

    def __init__(self, project_dir: str):
        self.path = os.path.join(project_dir, self.FILENAME)
        self._data: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {"commands_run": {}}

    def record_command(self, command: str, *, version: str = "0.2.0"):
        """Record that *command* was executed now."""
        self._data.setdefault("commands_run", {})[command] = {
            "timestamp": datetime.now().isoformat(),
            "version": version,
        }
        self._save()

    def has_command(self, command: str) -> bool:
        return command in self._data.get("commands_run", {})

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)


class OutputManager:
    """Manage organized output directory structure.

    Parameters
    ----------
    base_dir : str
        Project root (e.g. ``./projects/tencent_CL-bench``).
    subdirs : list[str] | None
        Category keys to create.  ``None`` means create all of them.
    """

    def __init__(self, base_dir: str, subdirs: list[str] | None = None):
        self.base_dir = base_dir
        self.subdirs: dict[str, str] = {}
        self._create_structure(subdirs)

    def _create_structure(self, keys: list[str] | None = None):
        os.makedirs(self.base_dir, exist_ok=True)
        target_keys = keys if keys is not None else list(OUTPUT_SUBDIRS.keys())
        for key in target_keys:
            if key in OUTPUT_SUBDIRS:
                path = os.path.join(self.base_dir, OUTPUT_SUBDIRS[key])
                os.makedirs(path, exist_ok=True)
                self.subdirs[key] = path

    def get_path(self, category: str, filename: str) -> str:
        """Get full path for a file in a category."""
        if category in self.subdirs:
            return os.path.join(self.subdirs[category], filename)
        return os.path.join(self.base_dir, filename)

    def get_relative_path(self, category: str, filename: str) -> str:
        """Get relative path for display."""
        if category in OUTPUT_SUBDIRS:
            return f"{OUTPUT_SUBDIRS[category]}/{filename}"
        return filename

    # ------------------------------------------------------------------
    # README generation — only shows directories that actually exist
    # ------------------------------------------------------------------

    # Human-readable metadata per subdirectory
    _DIR_META: dict[str, tuple[str, str, list[str]]] = {
        # key: (emoji+label, description, key files)
        "decision": (
            "决策层",
            "执行摘要、价值评分、ROI 分析",
            ["EXECUTIVE_SUMMARY.md", "executive_summary.json"],
        ),
        "project": (
            "项目管理",
            "里程碑计划、行业基准对比",
            ["MILESTONE_PLAN.md", "INDUSTRY_BENCHMARK.md"],
        ),
        "annotation": (
            "标注规范",
            "标注规范、评分标准模板",
            ["ANNOTATION_SPEC.md"],
        ),
        "guide": (
            "技术指南",
            "复刻指南、分析报告",
            ["REPRODUCTION_GUIDE.md", "ANALYSIS_REPORT.md"],
        ),
        "cost": (
            "成本分析",
            "成本明细、人机分配、Token 分析",
            ["COST_BREAKDOWN.md"],
        ),
        "data": (
            "原始数据",
            "复杂度分析、Prompt 模板等原始 JSON",
            [],
        ),
        "templates": (
            "模板",
            "培训指南、QA 清单、数据模板、SOP",
            ["training_guide.md", "qa_checklist.md"],
        ),
        "ai_agent": (
            "AI Agent",
            "Agent 上下文、工作流、推理链、流水线",
            ["agent_context.json", "pipeline.yaml"],
        ),
        "samples": (
            "样例数据",
            "Think-PO 样例等示范数据",
            [],
        ),
        "deploy": (
            "生产部署",
            "recipe.yaml、质检规则、验收标准、自动化脚本",
            ["recipe.yaml", "README.md"],
        ),
        "reports": (
            "综合报告",
            "Radar + Recipe 综合分析报告",
            [],
        ),
    }

    def generate_readme(self, project_name: str, dataset_type: str = "") -> str:
        """Generate README.md that reflects only the directories actually present."""
        # Scan which subdirectories actually exist on disk
        present = []
        for key, dirname in OUTPUT_SUBDIRS.items():
            dirpath = os.path.join(self.base_dir, dirname)
            if os.path.isdir(dirpath) and os.listdir(dirpath):
                present.append(key)

        type_line = f"\n> 数据类型: {dataset_type}" if dataset_type else ""

        lines = [
            f"# {project_name} 项目产出",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}{type_line}",
            "",
            "## 目录导航",
            "",
            "| 目录 | 说明 | 关键文件 |",
            "|------|------|----------|",
        ]

        for key in present:
            dirname = OUTPUT_SUBDIRS[key]
            label, desc, key_files = self._DIR_META.get(
                key, (key, "", [])
            )
            files_str = ", ".join(f"`{f}`" for f in key_files) if key_files else "-"
            lines.append(f"| `{dirname}/` | {label} — {desc} | {files_str} |")

        lines += [
            "",
            "## 快速导航",
            "",
        ]

        # Build quick-nav based on what's present
        nav_map = {
            "decision": ("快速决策", "EXECUTIVE_SUMMARY.md"),
            "project": ("项目规划", "MILESTONE_PLAN.md"),
            "annotation": ("外包标注", "ANNOTATION_SPEC.md"),
            "guide": ("技术复刻", "REPRODUCTION_GUIDE.md"),
            "cost": ("成本预算", "COST_BREAKDOWN.md"),
            "ai_agent": ("AI Agent", "agent_context.json"),
            "deploy": ("生产部署", "README.md"),
        }

        lines.append("| 目标 | 查看文件 |")
        lines.append("|------|----------|")
        for key, (label, fname) in nav_map.items():
            if key in present:
                lines.append(
                    f"| **{label}** | `{OUTPUT_SUBDIRS[key]}/{fname}` |"
                )

        lines += [
            "",
            "---",
            "",
            "> 由 DataRecipe 自动生成",
            "",
        ]

        return "\n".join(lines)
