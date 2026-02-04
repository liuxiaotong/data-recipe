# CL-bench 完整复现与批量生产项目

基于 DataRecipe 对腾讯混元 CL-bench 数据集的深度逆向工程，生成的**可直接用于批量生产**的完整资料包。

---

## 人机分工总览

> **核心原则**：机器提供模板和框架，人类提供创意和专业判断。

### 人类必须完成的工作

| 任务 | 工作量 | 为什么不能自动化 |
|------|--------|-----------------|
| **Context 内容创作** | 40% | 需要原创性，确保不在模型预训练数据中 |
| **任务问题设计** | 25% | 需要教学设计思维，判断什么问题有区分度 |
| **Rubrics 定制** | 20% | 需要理解 Context 细节，写出精确验证标准 |
| **质量审核** | 10% | 需要判断数据是否真正测试"上下文学习"能力 |
| **边界案例处理** | 5% | 需要经验判断模糊情况 |

### 机器可自动完成的工作

| 任务 | 自动化程度 | 工具 |
|------|-----------|------|
| System Prompt 生成 | 100% | 495 个模板直接复用 |
| Rubrics 句式生成 | 80% | `generate_rubrics.py` |
| 数据格式组装 | 100% | `batch_production_demo.py` |
| 统计分析 | 100% | `analyze_rubrics.py` |
| 批量导出 | 100% | 脚本自动输出 JSONL |

### 人力估算 (生产 500 Context / 1,899 Task)

```
总工时: ~33,780 小时 ≈ 20 人 × 6 个月

其中:
├── Rubrics 编写: 47% (15,804 小时) ← 最耗人力
├── Context 创作: 30% (10,000 小时)
├── Task 设计:    22% (7,596 小时)
└── 质量审核:     1%  (380 小时)
```

详见 [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) 第二章。

---

## 快速开始

```bash
# 1. 查看生产指南
cat PRODUCTION_GUIDE.md

# 2. 运行批量生产演示
python scripts/batch_production_demo.py

# 3. 查看生成的数据
cat production_output/batch_*.jsonl | head -1 | python -m json.tool
```

## 项目结构

```
tencent_cl-bench_full/
│
├── PRODUCTION_GUIDE.md          # 📋 【核心】完整生产指南 (512行)
├── methodology.md               # 🔬 方法论详解
├── README.md                    # 本文件
│
├── reproduction_kit/            # 🔧 模板资源包
│   ├── sample_*.json (4个)      #    每个类别的完整样本
│   ├── system_prompt_templates.json  # 495个 System Prompt 模板
│   ├── subcategory_analysis.json     # 18个子类别详细分析
│   ├── context_patterns.json         # Context 构建模式
│   ├── judge_prompts.json            # 评估 Prompt 模板
│   └── reproduction_checklist.md     # 检查清单
│
├── scripts/                     # 🛠 自动化工具
│   ├── 01_download_data.py      #    下载原始数据
│   ├── 02_inference.py          #    模型推理
│   ├── 03_evaluate.py           #    评估脚本
│   ├── 04_generate_benchmark.py #    LLM 生成数据
│   ├── analyze_rubrics.py       #    Rubrics 逆向分析
│   ├── generate_rubrics.py      #    Rubrics 生成器
│   ├── batch_production_demo.py #    批量生产演示
│   └── demo_full_pipeline.py    #    完整流程演示
│
├── data/                        # 📊 数据目录
│   ├── cl_bench_full.jsonl      #    原始数据 (86MB, 1,899条)
│   ├── cl_bench_sample.jsonl    #    样本数据 (100条)
│   ├── rubrics_analysis.json    #    31,607条 Rubrics 分析
│   └── statistics.json          #    统计信息
│
└── production_output/           # 📦 生产输出
    └── batch_*.jsonl            #    批量生产的数据
```

## CL-bench 数据集概述

| 指标 | 数值 |
|------|------|
| 总任务数 | 1,899 |
| 总 Rubrics | 31,607 |
| 平均 Context 长度 | 35,604 字符 |
| 平均 Rubrics/任务 | 16.6 |
| 最高模型通过率 | 23.7% (GPT-5.1) |

### 四大领域分布

```
Domain Knowledge Reasoning    ████████████████████  34.9%
Rule System Application       ████████████████      29.8%
Procedural Task Execution     █████████████         24.8%
Empirical Discovery           ██████                10.5%
```

## 核心发现

### Rubrics 构建模式

```
The response should [动词] [对象] [条件/细节]
```

**Top 动词**：
- `not` (3.2%) - 否定检查
- `include` (2.5%) - 包含检查
- `state` (2.4%) - 陈述检查
- `provide` (1.9%) - 提供检查
- `explain` (1.1%) - 解释检查

### Context 构建策略

| 策略 | 适用场景 | 占比 |
|------|----------|------|
| 虚构创作 | 游戏规则、虚拟法规 | ~30% |
| 修改现实 | 技术文档、医疗指南 | ~40% |
| 小众来源 | 新产品手册、前沿研究 | ~30% |

## 使用指南

### 1. 下载原始数据

```bash
python scripts/01_download_data.py
```

### 2. 分析 Rubrics 模式

```bash
python scripts/analyze_rubrics.py
```

### 3. 批量生产数据

```bash
# 无需 API，使用模板生产
python scripts/batch_production_demo.py

# 使用 LLM 增强（需要 API Key）
export OPENAI_API_KEY=your_key
python scripts/04_generate_benchmark.py --domain game_rules --num-contexts 10
```

### 4. 验证数据质量

```bash
# 演示完整流程（无需 API）
python scripts/demo_full_pipeline.py

# 实际评估（需要 API Key）
python scripts/02_inference.py --model gpt-4o --input data/produced.jsonl
python scripts/03_evaluate.py --input data/responses.jsonl
```

## 交付给团队

如果你要让团队开始生产，只需要给他们：

1. **`PRODUCTION_GUIDE.md`** - 完整生产指南
2. **`reproduction_kit/`** - 模板和样本
3. **`scripts/batch_production_demo.py`** - 生产工具

## 数据格式

```json
{
  "messages": [
    {"role": "system", "content": "系统指令"},
    {"role": "user", "content": "上下文 + 问题"}
  ],
  "rubrics": [
    "The response should define what X is...",
    "The response should list all Y...",
    "The response should not assume Z..."
  ],
  "metadata": {
    "task_id": "uuid",
    "context_category": "Rule System Application",
    "sub_category": "Game Mechanics"
  }
}
```

## 参考资源

- [CL-bench GitHub](https://github.com/Tencent-Hunyuan/CL-bench)
- [CL-bench HuggingFace](https://huggingface.co/datasets/tencent/CL-bench)
- [官方排行榜](https://www.clbench.com)
- [DataRecipe](https://github.com/liuxiaotong/data-recipe)

---

*由 DataRecipe 生成 | 2026-02-04*
