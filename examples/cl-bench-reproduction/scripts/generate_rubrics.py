#!/usr/bin/env python3
"""
Rubric 自动生成器

基于逆向分析的模式，自动生成验证标准。
"""

import random
from typing import Optional

# 基于分析结果的模板库
RUBRIC_TEMPLATES = {
    # 定义型
    "define": [
        "The response should define what {concept} is and its role in {context}.",
        "The response should clearly explain the meaning of {concept}.",
        "The response should provide a definition of {concept} according to the provided context.",
    ],
    # 列举型
    "list": [
        "The response should name the {count} different {items}, namely: {list}.",
        "The response should list all {items} mentioned in the context.",
        "The response should identify at least {count} {items}.",
        "The response should enumerate the {items} in order.",
    ],
    # 解释型
    "explain": [
        "The response should explain how {subject} works.",
        "The response should explain the {process} in detail.",
        "The response should explain why {reason}.",
        "The response should provide a clear explanation of {concept}.",
    ],
    # 包含型
    "include": [
        "The response should include {content}.",
        "The response should include a {section} section that {description}.",
        "The response should contain information about {topic}.",
    ],
    # 陈述型
    "state": [
        "The response should state that {fact}.",
        "The response should state the requirements for {action}.",
        "The response should clearly state {information}.",
    ],
    # 否定型
    "avoid": [
        "The response should not {action}.",
        "The response should avoid {behavior}.",
        "The response should not assume {assumption}.",
        "The response should not include {excluded_content}.",
    ],
    # 条件型
    "condition": [
        "The response should {action} when {condition}.",
        "If {condition}, the response should {action}.",
        "The response should handle the case when {condition} by {action}.",
    ],
    # 格式型
    "format": [
        "The response should present {content} in the form of {format}.",
        "The response should format {content} as {format}.",
        "The response should structure the answer with {structure}.",
    ],
    # 验证型
    "verify": [
        "The response should correctly identify {element}.",
        "The response should accurately {action} based on the provided context.",
        "The response should verify that {condition} before {action}.",
    ],
    # 引用型
    "cite": [
        "The response should cite {source} when {action}.",
        "According to the context, the response should {action}.",
        "Based on the provided {document}, the response should {action}.",
    ],
}

# 常用动词
VERBS = {
    "include": ["include", "contain", "incorporate", "feature"],
    "explain": ["explain", "describe", "clarify", "elaborate on"],
    "state": ["state", "mention", "indicate", "specify"],
    "provide": ["provide", "give", "offer", "present"],
    "identify": ["identify", "recognize", "name", "list"],
    "avoid": ["not", "avoid", "refrain from", "exclude"],
}

# 增强词
ENHANCERS = {
    "specificity": ["specifically", "explicitly", "clearly", "precisely"],
    "completeness": ["all", "every", "complete", "full", "entire"],
    "quantity": ["at least", "exactly", "no more than", "approximately"],
    "examples": ["for example", "such as", "including", "e.g."],
}


class RubricGenerator:
    """Rubric 生成器"""

    def __init__(self):
        self.templates = RUBRIC_TEMPLATES

    def generate_definition_rubric(
        self, concept: str, context: Optional[str] = None
    ) -> str:
        """生成定义型 Rubric"""
        template = random.choice(self.templates["define"])
        return template.format(
            concept=concept,
            context=context or "the given context"
        )

    def generate_list_rubric(
        self, items: str, count: Optional[int] = None, item_list: Optional[str] = None
    ) -> str:
        """生成列举型 Rubric"""
        if item_list:
            template = "The response should name the {count} different {items}, namely: {list}."
            return template.format(
                count=count or len(item_list.split(",")),
                items=items,
                list=item_list
            )
        elif count:
            template = "The response should identify at least {count} {items}."
            return template.format(count=count, items=items)
        else:
            template = "The response should list all {items} mentioned in the context."
            return template.format(items=items)

    def generate_explanation_rubric(
        self, subject: str, detail: Optional[str] = None
    ) -> str:
        """生成解释型 Rubric"""
        if detail:
            return f"The response should explain how {subject} works. {detail}"
        return f"The response should explain {subject} in detail."

    def generate_inclusion_rubric(
        self, content: str, section: Optional[str] = None
    ) -> str:
        """生成包含型 Rubric"""
        if section:
            return f"The response should include a '{section}' section that {content}."
        return f"The response should include {content}."

    def generate_negation_rubric(self, action: str) -> str:
        """生成否定型 Rubric"""
        return f"The response should not {action}."

    def generate_condition_rubric(self, condition: str, action: str) -> str:
        """生成条件型 Rubric"""
        return f"When {condition}, the response should {action}."

    def generate_format_rubric(self, content: str, format_type: str) -> str:
        """生成格式型 Rubric"""
        return f"The response should present {content} in the form of {format_type}."

    def generate_from_context(
        self,
        context_summary: str,
        rubric_type: str = "mixed",
        count: int = 5,
    ) -> list[str]:
        """基于上下文生成 Rubrics（需要 LLM）"""
        # 这里返回模板，实际使用时可以用 LLM 填充
        rubrics = []

        type_methods = {
            "define": self.generate_definition_rubric,
            "list": self.generate_list_rubric,
            "explain": self.generate_explanation_rubric,
            "include": self.generate_inclusion_rubric,
            "avoid": self.generate_negation_rubric,
        }

        if rubric_type == "mixed":
            types = list(type_methods.keys())
        else:
            types = [rubric_type]

        for i in range(count):
            t = random.choice(types)
            # 生成占位符 Rubric
            if t == "define":
                rubrics.append(self.generate_definition_rubric(f"[CONCEPT_{i+1}]"))
            elif t == "list":
                rubrics.append(self.generate_list_rubric(f"[ITEMS_{i+1}]", count=3))
            elif t == "explain":
                rubrics.append(self.generate_explanation_rubric(f"[SUBJECT_{i+1}]"))
            elif t == "include":
                rubrics.append(self.generate_inclusion_rubric(f"[CONTENT_{i+1}]"))
            elif t == "avoid":
                rubrics.append(self.generate_negation_rubric(f"[ACTION_{i+1}]"))

        return rubrics


def demo_generation():
    """演示 Rubric 生成"""
    gen = RubricGenerator()

    print("=" * 70)
    print("Rubric 自动生成示例")
    print("=" * 70)

    # 示例：游戏规则
    print("\n【示例 1: 游戏规则】")
    print("-" * 50)
    rubrics = [
        gen.generate_definition_rubric("Sighting card", "playing Twisted Cryptids"),
        gen.generate_list_rubric("sighting card types", 4, "Decoys, Hoaxes, Silhouettes, Genuine Sightings"),
        gen.generate_explanation_rubric("sighting cards are scored once revealed"),
        gen.generate_condition_rubric("a player reveals a sighting card", "check if all requirements are met"),
        gen.generate_negation_rubric("assume knowledge not provided in the rulebook"),
    ]
    for i, r in enumerate(rubrics, 1):
        print(f"  {i}. {r}")

    # 示例：技术文档
    print("\n【示例 2: 技术文档】")
    print("-" * 50)
    rubrics = [
        gen.generate_definition_rubric("API endpoint", "the authentication flow"),
        gen.generate_list_rubric("required parameters", 3),
        gen.generate_inclusion_rubric("explains the error handling mechanism", "error handling"),
        gen.generate_format_rubric("the request body", "JSON with proper indentation"),
        gen.generate_negation_rubric("expose sensitive information like API keys"),
    ]
    for i, r in enumerate(rubrics, 1):
        print(f"  {i}. {r}")

    # 示例：法规制度
    print("\n【示例 3: 法规制度】")
    print("-" * 50)
    rubrics = [
        gen.generate_definition_rubric("data controller", "GDPR regulations"),
        gen.generate_explanation_rubric("the penalty calculation process"),
        gen.generate_condition_rubric("a data breach occurs", "describe the notification requirements"),
        gen.generate_list_rubric("exceptions to the consent requirement"),
        gen.generate_negation_rubric("provide legal advice beyond the scope of the regulation"),
    ]
    for i, r in enumerate(rubrics, 1):
        print(f"  {i}. {r}")

    print("\n" + "=" * 70)
    print("Rubric 生成公式")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  Rubric = 结构词 + 动词 + 对象 + [条件/细节]                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  结构词:                                                            │
│    "The response should..."  (79.5% 使用)                          │
│    "Does the response..."    (问句形式)                            │
│    "Did the model..."        (过去式)                              │
│                                                                     │
│  动词选择:                                                          │
│    ┌──────────┬──────────┬────────────────────────────┐            │
│    │ 动词     │ 频率     │ 用途                       │            │
│    ├──────────┼──────────┼────────────────────────────┤            │
│    │ include  │ 2.5%     │ 检查是否包含内容           │            │
│    │ state    │ 2.4%     │ 检查是否陈述事实           │            │
│    │ provide  │ 1.9%     │ 检查是否提供信息           │            │
│    │ explain  │ 1.1%     │ 检查是否解释清楚           │            │
│    │ not      │ 3.2%     │ 检查是否避免某行为         │            │
│    │ identify │ 0.9%     │ 检查是否识别元素           │            │
│    └──────────┴──────────┴────────────────────────────┘            │
│                                                                     │
│  增强具体性:                                                        │
│    • namely: A, B, C       → 明确列举 (+具体性 33%)                │
│    • at least N            → 指定最小数量                          │
│    • for example, ...      → 要求举例                              │
│    • according to context  → 限定信息来源                          │
│                                                                     │
│  难度调节:                                                          │
│    简单: 单一验证点，无条件                                        │
│    中等: 多验证点，含条件                                          │
│    困难: 复杂条件 + 具体细节 + 多步骤                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    demo_generation()
