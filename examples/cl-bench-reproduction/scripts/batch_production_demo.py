#!/usr/bin/env python3
"""
CL-bench 批量生产完整演示

演示从零开始生产类似 CL-bench 的数据：
1. Context 构建
2. Task 设计
3. Rubrics 生成
4. 质量验证
5. 输出最终数据

无需 API Key，使用模板和规则生成。
"""

import json
import random
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

OUTPUT_DIR = Path(__file__).parent.parent / "production_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 第一部分：Context 模板库
# ============================================================

CONTEXT_TEMPLATES = {
    "game_rules": {
        "name": "虚构桌游规则",
        "template": """# {game_name} 规则书

## 游戏概述
{game_name} 是一款 {player_count} 人策略游戏。游戏目标是 {goal}。

## 游戏组件
- {component_1}：共 {count_1} 张/个
- {component_2}：共 {count_2} 张/个
- {component_3}：共 {count_3} 张/个
- {component_4}：共 {count_4} 张/个

## 游戏准备
1. {setup_1}
2. {setup_2}
3. {setup_3}

## 回合流程

### 阶段 1：{phase_1_name}
{phase_1_desc}

规则要点：
- {phase_1_rule_1}
- {phase_1_rule_2}
- {phase_1_rule_3}

### 阶段 2：{phase_2_name}
{phase_2_desc}

规则要点：
- {phase_2_rule_1}
- {phase_2_rule_2}

### 阶段 3：{phase_3_name}
{phase_3_desc}

规则要点：
- {phase_3_rule_1}
- {phase_3_rule_2}

## 特殊规则

### {special_rule_1_name}
{special_rule_1_desc}

触发条件：{special_rule_1_trigger}
效果：{special_rule_1_effect}

### {special_rule_2_name}
{special_rule_2_desc}

触发条件：{special_rule_2_trigger}
效果：{special_rule_2_effect}

## 胜利条件
- 主要胜利：{win_condition_1}
- 次要胜利：{win_condition_2}
- 平局条件：{draw_condition}

## 计分规则
| 项目 | 分值 |
|------|------|
| {score_item_1} | {score_value_1} 分 |
| {score_item_2} | {score_value_2} 分 |
| {score_item_3} | {score_value_3} 分 |

## 常见问题

Q: {faq_1_q}
A: {faq_1_a}

Q: {faq_2_q}
A: {faq_2_a}
""",
        "variables": {
            "game_name": ["星际殖民者", "符文之战", "迷雾森林", "量子交易所", "时空编织者"],
            "player_count": ["2-4", "3-5", "2-6", "2-3"],
            "goal": ["率先获得 50 点胜利分", "控制最多的领地", "收集全部 5 种神器", "成为最富有的玩家"],
            "component_1": ["资源卡", "行动卡", "角色卡", "事件卡"],
            "component_2": ["领地板块", "建筑标记", "单位棋子", "道具token"],
            "component_3": ["骰子", "分数标记", "回合指示物", "资源cube"],
            "component_4": ["玩家图板", "中央游戏板", "计分轨道", "卡牌市场板"],
            "count_1": ["60", "80", "100", "45"],
            "count_2": ["24", "36", "48", "20"],
            "count_3": ["6", "8", "4", "12"],
            "count_4": ["4", "1", "2", "6"],
        },
    },

    "technical_doc": {
        "name": "技术规范文档",
        "template": """# {system_name} 技术规范 v{version}

## 1. 概述

### 1.1 目的
本规范定义了 {system_name} 的 {purpose}。

### 1.2 适用范围
本规范适用于 {scope}。

### 1.3 术语定义

| 术语 | 定义 |
|------|------|
| {term_1} | {def_1} |
| {term_2} | {def_2} |
| {term_3} | {def_3} |
| {term_4} | {def_4} |

## 2. 数据格式

### 2.1 请求格式

```json
{{
  "{field_1}": "{type_1}",  // {field_1_desc}
  "{field_2}": "{type_2}",  // {field_2_desc}
  "{field_3}": {{
    "{nested_1}": "{nested_type_1}",
    "{nested_2}": "{nested_type_2}"
  }},
  "{field_4}": [{type_4}]  // {field_4_desc}
}}
```

### 2.2 响应格式

```json
{{
  "status": "string",      // {status_desc}
  "code": "integer",       // {code_desc}
  "data": {{...}},         // 响应数据
  "timestamp": "datetime"  // ISO 8601 格式
}}
```

## 3. 处理流程

### 3.1 标准流程

```
{flow_step_1}
    ↓
{flow_step_2}
    ↓
{flow_step_3}
    ↓
{flow_step_4}
    ↓
{flow_step_5}
```

### 3.2 流程说明

**步骤 1：{flow_step_1}**
{flow_desc_1}

**步骤 2：{flow_step_2}**
{flow_desc_2}

**步骤 3：{flow_step_3}**
{flow_desc_3}

## 4. 错误处理

| 错误码 | 名称 | 描述 | 处理建议 |
|--------|------|------|----------|
| {err_code_1} | {err_name_1} | {err_desc_1} | {err_action_1} |
| {err_code_2} | {err_name_2} | {err_desc_2} | {err_action_2} |
| {err_code_3} | {err_name_3} | {err_desc_3} | {err_action_3} |
| {err_code_4} | {err_name_4} | {err_desc_4} | {err_action_4} |

## 5. 安全要求

### 5.1 认证
{auth_requirement}

### 5.2 加密
{encryption_requirement}

### 5.3 访问控制
{access_control}

## 6. 限制与配额

| 限制项 | 限制值 | 说明 |
|--------|--------|------|
| {limit_1} | {limit_value_1} | {limit_desc_1} |
| {limit_2} | {limit_value_2} | {limit_desc_2} |
""",
        "variables": {
            "system_name": ["DataSync Pro", "CloudBridge API", "SecureVault", "StreamFlow"],
            "version": ["2.0", "3.1", "1.5", "4.0"],
            "purpose": ["数据同步协议", "服务间通信规范", "安全存储接口", "流处理标准"],
        },
    },
}

# ============================================================
# 第二部分：System Prompt 模板
# ============================================================

SYSTEM_PROMPT_TEMPLATES = {
    "game_rules": """You are an AI assistant specialized in the board game "{game_name}". Your role is to:

1. Explain game rules accurately based on the provided rulebook
2. Answer questions about game mechanics
3. Clarify edge cases and special situations
4. Help players understand strategic implications

Important guidelines:
- Only use information from the provided rulebook
- Do not invent or assume rules not explicitly stated
- When uncertain, acknowledge the limitation
- Use specific rule references when explaining

You should be helpful, precise, and thorough in your explanations.""",

    "technical_doc": """You are a technical support assistant for {system_name}. Your responsibilities:

1. Explain technical specifications accurately
2. Help users understand data formats and protocols
3. Diagnose errors using the error code reference
4. Guide users through proper implementation

Guidelines:
- Reference specific sections of the documentation
- Provide code examples when helpful
- Be precise about requirements and constraints
- Warn about common mistakes

Always base your answers on the provided technical specification.""",
}

# ============================================================
# 第三部分：Rubric 生成器
# ============================================================

class RubricGenerator:
    """基于模式的 Rubric 生成器"""

    TEMPLATES = {
        "define": [
            "The response should define what {concept} is according to the provided documentation.",
            "The response should clearly explain the meaning of {concept} as specified in the context.",
        ],
        "list": [
            "The response should list all {count} {items}, namely: {item_list}.",
            "The response should identify at least {count} {items} from the context.",
            "The response should enumerate the {items} mentioned in the documentation.",
        ],
        "explain": [
            "The response should explain how {process} works based on the provided rules.",
            "The response should describe the {process} process step by step.",
        ],
        "state": [
            "The response should state that {fact}.",
            "The response should mention that {fact} according to the documentation.",
        ],
        "condition": [
            "The response should explain what happens when {condition}.",
            "The response should describe the correct procedure if {condition}.",
        ],
        "avoid": [
            "The response should not {wrong_action}.",
            "The response should avoid {wrong_action}.",
        ],
        "include": [
            "The response should include {required_content}.",
            "The response should contain information about {required_content}.",
        ],
        "format": [
            "The response should present {content} in {format_type} format.",
            "The response should structure the answer with {structure}.",
        ],
    }

    def generate(self, rubric_type: str, **kwargs) -> str:
        """生成单条 Rubric"""
        templates = self.TEMPLATES.get(rubric_type, self.TEMPLATES["state"])
        template = random.choice(templates)
        return template.format(**kwargs)

    def generate_set(self, context_info: dict, count: int = 10) -> list[str]:
        """根据上下文信息生成一组 Rubrics"""
        rubrics = []

        # 1. 定义型 Rubrics
        for concept in context_info.get("key_concepts", [])[:2]:
            rubrics.append(self.generate("define", concept=concept))

        # 2. 列举型 Rubrics
        for item_info in context_info.get("lists", [])[:2]:
            rubrics.append(self.generate(
                "list",
                count=item_info["count"],
                items=item_info["name"],
                item_list=item_info["items"]
            ))

        # 3. 解释型 Rubrics
        for process in context_info.get("processes", [])[:2]:
            rubrics.append(self.generate("explain", process=process))

        # 4. 陈述型 Rubrics
        for fact in context_info.get("facts", [])[:2]:
            rubrics.append(self.generate("state", fact=fact))

        # 5. 否定型 Rubrics
        for wrong in context_info.get("common_mistakes", [])[:2]:
            rubrics.append(self.generate("avoid", wrong_action=wrong))

        return rubrics[:count]


# ============================================================
# 第四部分：数据生成流程
# ============================================================

@dataclass
class GeneratedSample:
    """生成的样本"""
    messages: list
    rubrics: list
    metadata: dict


def generate_game_context():
    """生成游戏规则 Context"""
    template_config = CONTEXT_TEMPLATES["game_rules"]

    # 随机选择变量值
    game_name = random.choice(template_config["variables"]["game_name"])
    player_count = random.choice(template_config["variables"]["player_count"])

    # 构建完整 Context (简化版)
    context = f"""# {game_name} 规则书

## 游戏概述
{game_name} 是一款 {player_count} 人策略游戏。

## 游戏组件
- 资源卡：共 60 张，分为木材、石材、金属、水晶四种
- 建筑卡：共 24 张，分为基础建筑和高级建筑
- 角色卡：共 12 张，每个角色有独特能力
- 事件卡：共 20 张，触发随机事件

## 回合流程

### 阶段 1：资源收集
玩家从资源堆中抽取 3 张资源卡。如果资源堆为空，则跳过此阶段。

规则要点：
- 每回合只能抽取 3 张
- 必须先抽取再进行其他行动
- 手牌上限为 10 张

### 阶段 2：建造
玩家可以使用资源卡建造建筑。每种建筑有不同的资源需求。

建筑列表：
| 建筑名 | 需要资源 | 效果 |
|--------|----------|------|
| 伐木场 | 2木材 | 每回合+1木材 |
| 矿场 | 2石材+1金属 | 每回合+1金属 |
| 魔法塔 | 3水晶 | 解锁高级卡牌 |
| 堡垒 | 2石材+2金属 | 防御值+3 |

### 阶段 3：行动
玩家可以执行角色卡的特殊能力或发起攻击。

特殊规则：
- "连锁反应"：当建造第3个同类建筑时，立即获得该建筑的双倍效果
- "资源枯竭"：当任一资源堆抽完时，触发资源重置事件

## 胜利条件
- 率先获得 50 分胜利点
- 或者：控制 5 个不同类型的建筑
- 平局：回合数达到 30 且无人满足胜利条件

## 计分规则
| 项目 | 分值 |
|------|------|
| 每个基础建筑 | 2 分 |
| 每个高级建筑 | 5 分 |
| 控制最多资源 | 10 分 |
| 角色能力达成 | 3-8 分 |
"""

    # 提取用于生成 Rubrics 的信息
    context_info = {
        "key_concepts": ["资源卡", "建造阶段", "连锁反应规则", "胜利条件"],
        "lists": [
            {"name": "资源类型", "count": 4, "items": "木材、石材、金属、水晶"},
            {"name": "建筑", "count": 4, "items": "伐木场、矿场、魔法塔、堡垒"},
        ],
        "processes": ["回合流程", "建造过程", "计分方式"],
        "facts": [
            "每回合只能抽取 3 张资源卡",
            "手牌上限为 10 张",
            "建造第3个同类建筑时触发连锁反应",
            "率先获得 50 分可以获胜",
        ],
        "common_mistakes": [
            "假设规则书中未提及的规则",
            "在资源收集阶段进行建造",
            "忽略手牌上限限制",
        ],
    }

    return game_name, context, context_info


def generate_question(context_info: dict, question_type: str) -> str:
    """生成问题"""
    questions = {
        "definition": [
            f"What is {random.choice(context_info['key_concepts'])} and how does it work?",
            f"Can you explain the {random.choice(context_info['key_concepts'])}?",
        ],
        "list": [
            f"What are all the {context_info['lists'][0]['name']}?",
            f"List the {context_info['lists'][1]['name']} and their effects.",
        ],
        "procedure": [
            f"How does the {random.choice(context_info['processes'])} work?",
            f"Walk me through the {random.choice(context_info['processes'])}.",
        ],
        "condition": [
            "What happens when a player builds their third building of the same type?",
            "What are the victory conditions?",
        ],
    }
    return random.choice(questions.get(question_type, questions["definition"]))


def generate_single_sample(category: str = "game_rules") -> GeneratedSample:
    """生成单条完整样本"""

    if category == "game_rules":
        game_name, context, context_info = generate_game_context()

        # 生成 System Prompt
        system_prompt = SYSTEM_PROMPT_TEMPLATES["game_rules"].format(game_name=game_name)

        # 生成问题
        question_type = random.choice(["definition", "list", "procedure", "condition"])
        question = generate_question(context_info, question_type)

        # 组合 user message
        user_content = f"""## Game Rules

{context}

## Question

{question}"""

        # 生成 Rubrics
        generator = RubricGenerator()
        rubrics = generator.generate_set(context_info, count=random.randint(5, 12))

        # 构建样本
        return GeneratedSample(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            rubrics=rubrics,
            metadata={
                "task_id": str(uuid.uuid4()),
                "context_category": "Rule System Application",
                "sub_category": "Game Mechanics",
                "generated_at": datetime.now().isoformat(),
                "generator": "batch_production_demo",
            }
        )

    # 可以扩展其他类别...
    return None


def validate_sample(sample: GeneratedSample) -> dict:
    """验证样本质量"""
    issues = []
    warnings = []

    # 检查 messages
    if len(sample.messages) < 2:
        issues.append("消息数量不足")

    if len(sample.messages[0]["content"]) < 100:
        warnings.append("System prompt 较短")

    if len(sample.messages[1]["content"]) < 1000:
        warnings.append("Context 较短，建议 > 5000 字符")

    # 检查 rubrics
    if len(sample.rubrics) < 3:
        issues.append("Rubrics 数量不足 (< 3)")

    for r in sample.rubrics:
        if not r.startswith("The response should"):
            warnings.append(f"Rubric 格式不标准: {r[:50]}...")

    # 检查 metadata
    if not sample.metadata.get("task_id"):
        issues.append("缺少 task_id")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "stats": {
            "system_prompt_length": len(sample.messages[0]["content"]),
            "context_length": len(sample.messages[1]["content"]),
            "rubrics_count": len(sample.rubrics),
        }
    }


def run_batch_production(count: int = 10):
    """运行批量生产"""
    print("=" * 70)
    print("CL-bench 批量生产演示")
    print("=" * 70)

    samples = []
    validation_results = []

    print(f"\n正在生成 {count} 条样本...")
    print("-" * 50)

    for i in range(count):
        # 生成样本
        sample = generate_single_sample("game_rules")
        samples.append(sample)

        # 验证质量
        validation = validate_sample(sample)
        validation_results.append(validation)

        # 显示进度
        status = "✓" if validation["valid"] else "✗"
        print(f"  [{i+1:>2}/{count}] {status} | "
              f"Context: {validation['stats']['context_length']:>5} chars | "
              f"Rubrics: {validation['stats']['rubrics_count']:>2}")

    # 统计
    valid_count = sum(1 for v in validation_results if v["valid"])
    total_rubrics = sum(v["stats"]["rubrics_count"] for v in validation_results)
    avg_context = sum(v["stats"]["context_length"] for v in validation_results) / count

    print("\n" + "=" * 70)
    print("生产统计")
    print("=" * 70)
    print(f"""
  总样本数:      {count}
  有效样本:      {valid_count} ({valid_count/count*100:.0f}%)
  总 Rubrics:    {total_rubrics}
  平均 Rubrics:  {total_rubrics/count:.1f}
  平均 Context:  {avg_context:,.0f} 字符
""")

    # 保存结果
    output_path = OUTPUT_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            data = {
                "messages": sample.messages,
                "rubrics": sample.rubrics,
                "metadata": sample.metadata,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"  输出文件: {output_path}")

    # 展示第一个样本
    print("\n" + "=" * 70)
    print("样本预览 (第1条)")
    print("=" * 70)

    sample = samples[0]
    print(f"\n【System Prompt】({len(sample.messages[0]['content'])} 字符)")
    print("-" * 40)
    print(sample.messages[0]["content"][:500] + "...")

    print(f"\n【Context + Question】({len(sample.messages[1]['content'])} 字符)")
    print("-" * 40)
    print(sample.messages[1]["content"][:800] + "...")

    print(f"\n【Rubrics】({len(sample.rubrics)} 条)")
    print("-" * 40)
    for i, r in enumerate(sample.rubrics[:5], 1):
        print(f"  {i}. {r}")
    if len(sample.rubrics) > 5:
        print(f"  ... 还有 {len(sample.rubrics) - 5} 条")

    print(f"\n【Metadata】")
    print("-" * 40)
    print(json.dumps(sample.metadata, indent=2, ensure_ascii=False))

    return samples, output_path


if __name__ == "__main__":
    run_batch_production(10)
