# CL-bench 复现完整检查清单

## 第一阶段：准备工作

### 1.1 数据理解
- [ ] 阅读 methodology.md 理解方法论
- [ ] 查看 sample_*.json 理解数据格式
- [ ] 分析 rubrics_analysis.json 理解验证标准模式
- [ ] 研究 subcategory_analysis.json 理解子类别特点

### 1.2 工具准备
- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 配置 LLM API (用于生成和评估)
- [ ] 准备标注工具/平台

## 第二阶段：Context 构建

### 2.1 选择领域
- [ ] Domain Knowledge Reasoning (34.9%)
- [ ] Rule System Application (29.8%)
- [ ] Procedural Task Execution (24.8%)
- [ ] Empirical Discovery & Simulation (10.5%)

### 2.2 构建策略选择
- [ ] 策略 A: 虚构创作 (游戏规则、虚拟法规)
- [ ] 策略 B: 修改现实 (修改技术文档、流程)
- [ ] 策略 C: 小众来源 (新产品手册、前沿研究)

### 2.3 Context 质量检查
- [ ] 完整性：包含解决任务所需的所有信息
- [ ] 自洽性：内部规则不矛盾
- [ ] 独特性：不能从预训练知识推断
- [ ] 复杂度：支持多层次、多步骤推理
- [ ] 长度：平均 35K 字符，参考 context_patterns.json

## 第三阶段：Task 设计

### 3.1 System Prompt 编写
- [ ] 参考 system_prompt_templates.json
- [ ] 明确角色定义
- [ ] 设定行为规范
- [ ] 指定输出格式

### 3.2 任务设计
- [ ] 设计 2-12 轮对话
- [ ] 50% 任务需要序列依赖
- [ ] 覆盖多种难度级别

## 第四阶段：Rubrics 编写

### 4.1 使用模板
- [ ] 参考 rubrics_analysis.json 的动词频率
- [ ] 使用 generate_rubrics.py 辅助生成
- [ ] 每任务 3-114 条，平均 16.6 条

### 4.2 Rubrics 类型覆盖
- [ ] 定义型：检查概念理解
- [ ] 列举型：检查完整列举
- [ ] 解释型：检查解释深度
- [ ] 否定型：检查避免错误
- [ ] 条件型：检查条件处理

### 4.3 质量检查
- [ ] 可验证性：每条 rubric 可以明确判断 yes/no
- [ ] 独立性：rubrics 之间不重复
- [ ] 覆盖性：覆盖任务的关键点

## 第五阶段：验证

### 5.1 无上下文测试
- [ ] 随机抽取 100 条任务
- [ ] 移除 context，只给 question
- [ ] 运行模型，通过率应 < 1%

### 5.2 有上下文基线
- [ ] 使用完整数据测试
- [ ] 参考 GPT-5.1 的 23.7% 通过率
- [ ] 分析失败原因

### 5.3 人工抽检
- [ ] 抽检 10% 的 rubrics 判断
- [ ] 确保 rubrics 判断一致性 > 90%

## 第六阶段：规模化生产

### 6.1 流程固化
- [ ] 编写详细的标注指南
- [ ] 建立质检流程
- [ ] 培训标注团队

### 6.2 成本控制
- [ ] Context: ~20 小时/个
- [ ] Task: ~4 小时/个
- [ ] Rubrics: ~0.5 小时/条

---

## 输出物检查

- [ ] data/*.jsonl - 原始数据
- [ ] reproduction_kit/sample_*.json - 类别样本
- [ ] reproduction_kit/system_prompt_templates.json - 模板库
- [ ] reproduction_kit/subcategory_analysis.json - 子类别分析
- [ ] reproduction_kit/judge_prompts.json - 评估提示词
