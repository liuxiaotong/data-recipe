#!/usr/bin/env python3
"""
Rubrics 逆向分析

分析 CL-bench 的 31,607 条 Rubrics，提取：
1. 句式模板
2. 动词模式
3. 验证类型
4. 难度分布
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"


def load_all_rubrics():
    """加载所有 rubrics"""
    rubrics = []
    metadata = []

    with open(DATA_DIR / "cl_bench_full.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            for r in sample["rubrics"]:
                rubrics.append(r)
                metadata.append(sample["metadata"])

    return rubrics, metadata


def extract_patterns(rubrics):
    """提取句式模板"""
    # 常见开头模式
    starts = Counter()
    for r in rubrics:
        # 提取前 6 个词
        words = r.split()[:6]
        start = " ".join(words)
        starts[start] += 1

    return starts


def extract_verbs(rubrics):
    """提取动词模式"""
    # "The response should [VERB]" 模式
    verb_pattern = re.compile(r"The response should (\w+)")
    verbs = Counter()

    for r in rubrics:
        match = verb_pattern.search(r)
        if match:
            verbs[match.group(1)] += 1

    return verbs


def extract_keywords(rubrics):
    """提取关键词"""
    # 验证类型关键词
    keywords = {
        "定义型": ["define", "explain", "describe", "clarify"],
        "列举型": ["list", "name", "enumerate", "identify", "mention"],
        "判断型": ["determine", "assess", "evaluate", "judge"],
        "计算型": ["calculate", "compute", "count", "sum"],
        "比较型": ["compare", "contrast", "differentiate", "distinguish"],
        "条件型": ["if", "when", "condition", "unless", "provided"],
        "否定型": ["not", "avoid", "without", "never", "exclude"],
        "引用型": ["according to", "based on", "refer to", "cite"],
        "格式型": ["format", "structure", "organize", "present"],
        "完整型": ["all", "every", "complete", "full", "entire"],
    }

    type_counts = Counter()
    type_examples = defaultdict(list)

    for r in rubrics:
        r_lower = r.lower()
        for type_name, kws in keywords.items():
            for kw in kws:
                if kw in r_lower:
                    type_counts[type_name] += 1
                    if len(type_examples[type_name]) < 3:
                        type_examples[type_name].append(r)
                    break

    return type_counts, type_examples


def extract_specificity(rubrics):
    """分析具体性（包含具体数字/名称）"""
    specific = []
    general = []

    # 包含具体数字或引号内容的视为具体
    specific_pattern = re.compile(r'(\d+|"[^"]+"|\'[^\']+\'|namely|specifically)')

    for r in rubrics:
        if specific_pattern.search(r):
            specific.append(r)
        else:
            general.append(r)

    return specific, general


def extract_templates(rubrics):
    """提取模板结构"""
    templates = Counter()

    for r in rubrics:
        # 替换具体内容为占位符
        template = r
        # 替换引号内容
        template = re.sub(r'"[^"]+"', '"[VALUE]"', template)
        template = re.sub(r"'[^']+'", "'[VALUE]'", template)
        # 替换数字
        template = re.sub(r'\b\d+\b', '[NUM]', template)
        # 替换括号内容
        template = re.sub(r'\([^)]+\)', '([DETAIL])', template)

        # 只保留前 50 个字符作为模板
        template = template[:80]
        templates[template] += 1

    return templates


def analyze_by_category(rubrics, metadata):
    """按类别分析"""
    category_rubrics = defaultdict(list)

    for r, m in zip(rubrics, metadata):
        cat = m["context_category"]
        category_rubrics[cat].append(r)

    category_patterns = {}
    for cat, cat_rubrics in category_rubrics.items():
        verbs = extract_verbs(cat_rubrics)
        category_patterns[cat] = {
            "count": len(cat_rubrics),
            "top_verbs": verbs.most_common(10),
            "avg_length": sum(len(r) for r in cat_rubrics) / len(cat_rubrics),
        }

    return category_patterns


def main():
    print("正在加载 Rubrics...")
    rubrics, metadata = load_all_rubrics()
    print(f"共加载 {len(rubrics):,} 条 Rubrics\n")

    # 1. 句式开头分析
    print("=" * 70)
    print("1. 句式开头模式 (Top 20)")
    print("=" * 70)
    starts = extract_patterns(rubrics)
    for start, count in starts.most_common(20):
        pct = count / len(rubrics) * 100
        print(f"  {count:>5} ({pct:>5.1f}%) | {start}")

    # 2. 动词分析
    print("\n" + "=" * 70)
    print("2. 核心动词分布")
    print("=" * 70)
    verbs = extract_verbs(rubrics)
    print("\n动词频率:")
    for verb, count in verbs.most_common(30):
        pct = count / len(rubrics) * 100
        bar = "█" * int(pct)
        print(f"  {verb:<20} {count:>5} ({pct:>5.1f}%) {bar}")

    # 3. 验证类型分析
    print("\n" + "=" * 70)
    print("3. 验证类型分布")
    print("=" * 70)
    type_counts, type_examples = extract_keywords(rubrics)
    for type_name, count in type_counts.most_common():
        pct = count / len(rubrics) * 100
        print(f"\n  【{type_name}】 {count} 条 ({pct:.1f}%)")
        for ex in type_examples[type_name][:2]:
            print(f"      例: {ex[:80]}...")

    # 4. 具体性分析
    print("\n" + "=" * 70)
    print("4. 具体性分析")
    print("=" * 70)
    specific, general = extract_specificity(rubrics)
    print(f"  具体型 (含数字/引号/namely): {len(specific):,} ({len(specific)/len(rubrics)*100:.1f}%)")
    print(f"  一般型:                      {len(general):,} ({len(general)/len(rubrics)*100:.1f}%)")
    print("\n  具体型示例:")
    for ex in specific[:5]:
        print(f"    - {ex[:100]}...")

    # 5. 模板提取
    print("\n" + "=" * 70)
    print("5. 常见模板 (Top 15)")
    print("=" * 70)
    templates = extract_templates(rubrics)
    for template, count in templates.most_common(15):
        print(f"  {count:>4}x | {template}")

    # 6. 按类别分析
    print("\n" + "=" * 70)
    print("6. 按领域类别分析")
    print("=" * 70)
    category_patterns = analyze_by_category(rubrics, metadata)
    for cat, stats in category_patterns.items():
        print(f"\n  【{cat}】")
        print(f"    Rubrics 数量: {stats['count']}")
        print(f"    平均长度: {stats['avg_length']:.0f} 字符")
        print(f"    常用动词: {', '.join([v[0] for v in stats['top_verbs'][:5]])}")

    # 7. 生成 Rubric 构建指南
    print("\n" + "=" * 70)
    print("7. Rubric 构建模板总结")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     Rubric 构建黄金模板                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  基础结构:                                                          │
│    "The response should [动词] [对象] [条件/细节]"                   │
│                                                                     │
│  常用动词 (按频率):                                                  │
│    1. include/contain  - 检查是否包含某内容                         │
│    2. explain/describe - 检查是否解释清楚                           │
│    3. identify/name    - 检查是否识别/列举                          │
│    4. provide          - 检查是否提供某信息                         │
│    5. state/mention    - 检查是否陈述某事实                         │
│    6. not/avoid        - 检查是否避免某行为                         │
│                                                                     │
│  增强具体性:                                                        │
│    - 使用 "namely: X, Y, Z" 明确列举                               │
│    - 使用 "at least N" 指定数量                                    │
│    - 使用 "specifically" 强调具体内容                              │
│    - 使用引号 "term" 指明确切术语                                  │
│                                                                     │
│  条件限定:                                                          │
│    - "according to the context" 限定信息来源                       │
│    - "based on the provided rules" 限定依据                        │
│    - "for example" 要求举例说明                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

    # 8. 导出分析结果
    results = {
        "total_rubrics": len(rubrics),
        "top_starts": starts.most_common(50),
        "top_verbs": verbs.most_common(50),
        "type_distribution": dict(type_counts),
        "specificity": {
            "specific": len(specific),
            "general": len(general),
        },
        "category_patterns": category_patterns,
    }

    output_path = DATA_DIR / "rubrics_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n分析结果已保存: {output_path}")


if __name__ == "__main__":
    main()
