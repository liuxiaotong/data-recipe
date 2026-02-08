"""Pipeline extraction and production guide generation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipelineStepType(Enum):
    """Types of pipeline steps."""

    DATA_COLLECTION = "data_collection"
    SEED_DATA = "seed_data"
    PROMPT_DESIGN = "prompt_design"
    LLM_GENERATION = "llm_generation"
    HUMAN_ANNOTATION = "human_annotation"
    QUALITY_FILTER = "quality_filter"
    DEDUPLICATION = "deduplication"
    FORMAT_CONVERSION = "format_conversion"
    VALIDATION = "validation"
    POST_PROCESSING = "post_processing"


@dataclass
class PipelineStep:
    """A single step in the data production pipeline."""

    step_number: int
    step_type: PipelineStepType
    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    estimated_cost: float | None = None
    estimated_time: str | None = None
    code_snippet: str | None = None
    tips: list[str] = field(default_factory=list)


@dataclass
class ProductionPipeline:
    """Complete data production pipeline."""

    name: str
    description: str
    target_size: int | None = None
    estimated_total_cost: float | None = None
    estimated_total_time: str | None = None
    prerequisites: list[str] = field(default_factory=list)
    steps: list[PipelineStep] = field(default_factory=list)
    quality_criteria: list[str] = field(default_factory=list)
    common_pitfalls: list[str] = field(default_factory=list)


# Pre-defined pipeline templates for common dataset types
PIPELINE_TEMPLATES = {
    "distillation": ProductionPipeline(
        name="LLM 蒸馏数据集生产流程",
        description="通过大型语言模型生成高质量训练数据",
        prerequisites=[
            "OpenAI/Anthropic/其他 LLM API 访问权限",
            "种子数据或提示词模板",
            "Python 环境 + requests/openai 库",
            "数据存储空间",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.SEED_DATA,
                name="准备种子数据",
                description="收集或创建初始数据样本，作为生成的基础",
                inputs=["领域知识", "示例数据"],
                outputs=["seed_data.jsonl"],
                tips=[
                    "种子数据质量直接影响生成质量",
                    "建议人工审核种子数据",
                    "多样性比数量更重要",
                ],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.PROMPT_DESIGN,
                name="设计提示词模板",
                description="创建用于调用 LLM 的提示词模板",
                inputs=["任务需求", "输出格式要求"],
                outputs=["prompt_templates.yaml"],
                code_snippet='''PROMPT_TEMPLATE = """
你是一个专业的数据生成助手。请根据以下要求生成数据：

任务：{task_description}
输入：{input_data}
要求：
1. 输出格式为 JSON
2. 确保数据多样性
3. 保持逻辑一致性

请生成：
"""''',
                tips=[
                    "使用 few-shot 示例提高质量",
                    "明确输出格式要求",
                    "添加质量约束条件",
                ],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.LLM_GENERATION,
                name="调用 LLM 生成数据",
                description="批量调用 LLM API 生成数据",
                inputs=["seed_data.jsonl", "prompt_templates.yaml"],
                outputs=["raw_generated.jsonl"],
                tools=["OpenAI API", "Anthropic API", "本地模型"],
                estimated_cost=0.01,  # per 1k tokens
                code_snippet="""import openai
from tqdm import tqdm

def generate_batch(prompts, model="gpt-4"):
    results = []
    for prompt in tqdm(prompts):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        results.append(response.choices[0].message.content)
    return results""",
                tips=[
                    "使用批量 API 降低成本",
                    "设置合理的 temperature",
                    "添加重试机制处理 API 错误",
                    "记录所有请求用于复现",
                ],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.QUALITY_FILTER,
                name="质量过滤",
                description="过滤低质量或不符合要求的生成结果",
                inputs=["raw_generated.jsonl"],
                outputs=["filtered_data.jsonl"],
                code_snippet="""def quality_filter(data):
    filtered = []
    for item in data:
        # 长度检查
        if len(item["text"]) < 50:
            continue
        # 格式检查
        if not is_valid_json(item):
            continue
        # 去重检查
        if is_duplicate(item, filtered):
            continue
        filtered.append(item)
    return filtered""",
                tips=[
                    "定义明确的质量标准",
                    "保留过滤日志用于分析",
                    "考虑使用 LLM 进行质量评分",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.VALIDATION,
                name="人工抽样验证",
                description="人工检查部分数据确保质量",
                inputs=["filtered_data.jsonl"],
                outputs=["validated_data.jsonl", "quality_report.md"],
                tips=[
                    "抽样比例建议 5-10%",
                    "建立评分标准",
                    "记录常见问题用于迭代改进",
                ],
            ),
            PipelineStep(
                step_number=6,
                step_type=PipelineStepType.FORMAT_CONVERSION,
                name="格式转换与发布",
                description="转换为目标格式并准备发布",
                inputs=["validated_data.jsonl"],
                outputs=["final_dataset/"],
                tools=["HuggingFace datasets", "pandas"],
                code_snippet="""from datasets import Dataset

dataset = Dataset.from_json("validated_data.jsonl")
dataset.push_to_hub("your-org/dataset-name")""",
            ),
        ],
        quality_criteria=[
            "生成内容与种子数据主题一致",
            "无事实性错误",
            "格式符合要求",
            "无重复或高度相似内容",
            "语言流畅自然",
        ],
        common_pitfalls=[
            "种子数据质量差导致生成质量差",
            "提示词不够明确导致输出不一致",
            "未设置 temperature 导致生成单一",
            "跳过人工验证导致质量问题",
            "未记录生成参数导致无法复现",
        ],
    ),
    "human_annotation": ProductionPipeline(
        name="人工标注数据集生产流程",
        description="通过众包或专家标注创建高质量数据",
        prerequisites=[
            "标注平台账号（Scale AI/Labelbox/Amazon MTurk）",
            "标注指南文档",
            "原始数据",
            "质量控制机制",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.DATA_COLLECTION,
                name="收集原始数据",
                description="收集需要标注的原始数据",
                inputs=["数据来源"],
                outputs=["raw_data.jsonl"],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.PROMPT_DESIGN,
                name="编写标注指南",
                description="创建详细的标注指南和示例",
                inputs=["任务需求", "质量标准"],
                outputs=["annotation_guidelines.md", "examples.json"],
                tips=[
                    "包含正面和负面示例",
                    "定义边界情况处理方式",
                    "使用截图和可视化",
                ],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.HUMAN_ANNOTATION,
                name="执行标注",
                description="通过标注平台分发任务并收集结果",
                inputs=["raw_data.jsonl", "annotation_guidelines.md"],
                outputs=["annotated_data.jsonl"],
                tools=["Scale AI", "Labelbox", "Amazon MTurk", "Prolific"],
                estimated_cost=0.10,  # per annotation
                tips=[
                    "先进行小规模试标注",
                    "设置标注者资格要求",
                    "使用多人标注提高质量",
                ],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.QUALITY_FILTER,
                name="质量控制",
                description="检查标注一致性和质量",
                inputs=["annotated_data.jsonl"],
                outputs=["quality_checked.jsonl"],
                code_snippet='''def check_agreement(annotations):
    """计算标注者一致性"""
    from sklearn.metrics import cohen_kappa_score
    # 计算 Cohen's Kappa
    kappa = cohen_kappa_score(annotator1, annotator2)
    return kappa > 0.7  # 阈值''',
                tips=[
                    "计算标注者间一致性（Inter-annotator agreement）",
                    "对低一致性样本进行仲裁",
                    "定期校准标注者",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.VALIDATION,
                name="专家审核",
                description="专家审核最终数据集",
                inputs=["quality_checked.jsonl"],
                outputs=["final_dataset.jsonl"],
            ),
        ],
        quality_criteria=[
            "标注者间一致性 > 0.7",
            "无遗漏标注",
            "标注符合指南要求",
            "边界情况处理一致",
        ],
        common_pitfalls=[
            "标注指南不够清晰",
            "未进行试标注直接大规模标注",
            "忽略标注者反馈",
            "未计算标注一致性",
            "标注者疲劳导致质量下降",
        ],
    ),
    "hybrid": ProductionPipeline(
        name="混合数据集生产流程（LLM + 人工）",
        description="结合 LLM 生成和人工验证/修正",
        prerequisites=[
            "LLM API 访问权限",
            "标注平台或内部标注团队",
            "明确的质量标准",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.SEED_DATA,
                name="准备种子数据",
                description="收集高质量种子样本",
                inputs=["领域数据"],
                outputs=["seed_data.jsonl"],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.LLM_GENERATION,
                name="LLM 批量生成",
                description="使用 LLM 生成初始数据",
                inputs=["seed_data.jsonl"],
                outputs=["llm_generated.jsonl"],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.QUALITY_FILTER,
                name="自动质量过滤",
                description="使用规则或模型过滤明显低质量数据",
                inputs=["llm_generated.jsonl"],
                outputs=["auto_filtered.jsonl"],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.HUMAN_ANNOTATION,
                name="人工验证与修正",
                description="人工验证 LLM 生成的数据并修正错误",
                inputs=["auto_filtered.jsonl"],
                outputs=["human_verified.jsonl"],
                tips=[
                    "重点关注事实性验证",
                    "允许标注者修改而非仅判断",
                    "收集修改原因用于改进提示词",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.POST_PROCESSING,
                name="最终处理",
                description="去重、格式化、准备发布",
                inputs=["human_verified.jsonl"],
                outputs=["final_dataset/"],
            ),
        ],
        quality_criteria=[
            "LLM 生成通过率 > 70%",
            "人工修正率 < 30%",
            "最终数据无事实错误",
        ],
        common_pitfalls=[
            "LLM 生成质量差导致人工成本过高",
            "人工修正标准不一致",
            "未利用修正反馈改进 LLM 提示词",
        ],
    ),
    "programmatic": ProductionPipeline(
        name="程序化数据集生产流程",
        description="通过程序化/组合式方法生成可验证的多样化任务数据",
        prerequisites=[
            "领域知识文档",
            "环境规范定义",
            "原子操作库",
            "Python 环境 + 测试框架",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.DATA_COLLECTION,
                name="设计领域环境",
                description="定义可交互的共享世界：实体、属性、状态",
                inputs=["领域知识", "需求文档"],
                outputs=["environment_spec.yaml"],
                code_snippet="""# environment_spec.yaml 示例
domain: telecom
entities:
  - user_account:
      attributes: [plan_type, balance, usage, status]
  - device:
      attributes: [model, imei, sim_status, network_status]
  - service:
      attributes: [name, status, start_date, end_date]

world_state:
  observable_by_user: [account_balance, current_plan, device_status]
  observable_by_agent: [full_account_history, internal_notes, system_status]
  shared: [conversation_history, action_results]""",
                tips=[
                    "区分用户可见状态和代理可见状态",
                    "定义共享的可观测信息",
                    "确保环境足够复杂以产生有意义的交互",
                ],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.PROMPT_DESIGN,
                name="定义原子操作",
                description="定义用户和代理可执行的基本操作及其前置条件和效果",
                inputs=["environment_spec.yaml"],
                outputs=["actions.yaml"],
                code_snippet="""# actions.yaml 示例
user_actions:
  - check_balance:
      preconditions: [logged_in]
      effects: [display_balance]
  - restart_device:
      preconditions: [has_device]
      effects: [device_restart, potential_fix]
  - provide_info:
      params: [info_type, value]
      effects: [update_context]

agent_actions:
  - lookup_account:
      params: [user_id]
      effects: [retrieve_full_account]
  - apply_credit:
      params: [amount, reason]
      preconditions: [verified_issue]
      effects: [update_balance]
  - guide_user:
      params: [action_to_take, instructions]
      effects: [user_receives_guidance]""",
                tips=[
                    "用户操作应该对解决问题有实际影响",
                    "代理需要能够指导用户执行操作",
                    "定义清晰的前置条件和效果",
                ],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.LLM_GENERATION,
                name="组合任务生成",
                description="程序化生成多样、可验证的任务",
                inputs=["environment_spec.yaml", "actions.yaml"],
                outputs=["tasks.jsonl"],
                code_snippet="""class TaskGenerator:
    def __init__(self, env_spec, actions):
        self.env = env_spec
        self.actions = actions

    def generate_task(self, complexity_level):
        # 1. 随机初始化世界状态
        initial_state = self.random_initial_state()

        # 2. 定义目标状态
        goal_state = self.define_goal(initial_state, complexity_level)

        # 3. 计算最优解（用于验证）
        optimal_solution = self.compute_solution(initial_state, goal_state)

        # 4. 生成用户意图描述
        user_intent = self.generate_intent(initial_state, goal_state)

        return {
            "initial_state": initial_state,
            "goal_state": goal_state,
            "user_intent": user_intent,
            "optimal_solution": optimal_solution,
            "requires_user_action": self.check_user_action_needed(optimal_solution),
            "complexity": complexity_level,
        }""",
                tips=[
                    "确保每个任务都有明确的解决方案",
                    "平衡需要/不需要用户操作的任务",
                    "保持任务多样性",
                ],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.POST_PROCESSING,
                name="构建模拟器",
                description="创建与环境耦合的用户/代理模拟器",
                inputs=["tasks.jsonl", "environment_spec.yaml"],
                outputs=["user_simulator.py", "env_simulator.py"],
                code_snippet="""class UserSimulator:
    def __init__(self, task, behavior_params):
        self.task = task
        self.observable_state = task["initial_state"]["user_visible"]
        self.intent = task["user_intent"]
        self.cooperation_level = behavior_params.get("cooperation", 0.9)
        self.comprehension_level = behavior_params.get("comprehension", 0.8)

    def respond(self, agent_message, available_actions):
        # 1. 理解代理消息
        parsed = self.parse_agent_message(agent_message)

        # 2. 决定是否配合
        if random.random() > self.cooperation_level:
            return self.generate_uncooperative_response()

        # 3. 如果代理请求用户操作
        if parsed["requests_user_action"]:
            action = parsed["requested_action"]
            if random.random() < self.comprehension_level:
                result = self.execute_action(action)
                return self.report_action_result(result)
            else:
                return self.generate_confused_response(action)

        return self.generate_response(parsed)""",
                tips=[
                    "定义用户行为参数：配合度、理解能力、耐心",
                    "模拟真实用户的错误和误解",
                    "用户模拟器应能执行操作并更新状态",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.VALIDATION,
                name="人工验证",
                description="确保生成的任务对人类可解",
                inputs=["tasks.jsonl"],
                outputs=["human_baseline.json", "validated_tasks.jsonl"],
                code_snippet="""# human_baseline.json 示例
{
  "task_id": "task_001",
  "human_solvers": 2,
  "attempts_needed": [1, 2],
  "success": true,
  "time_taken_seconds": [120, 180],
  "notes": "需要指导用户重启路由器"
}""",
                tips=[
                    "每个任务至少 2 名人类测试者",
                    "在 2 次尝试内解决",
                    "记录人类平均表现作为基准",
                ],
            ),
            PipelineStep(
                step_number=6,
                step_type=PipelineStepType.FORMAT_CONVERSION,
                name="校准与发布",
                description="数据集划分、复杂度校准、发布",
                inputs=["validated_tasks.jsonl", "human_baseline.json"],
                outputs=["final_dataset/"],
                tips=[
                    "划分训练集/公开评估集/私有评估集",
                    "确保复杂度分布均匀",
                    "覆盖所有操作类型",
                ],
            ),
        ],
        quality_criteria=[
            "每个任务有明确的最优解",
            "需要双控制协作的任务占比 > 50%",
            "人类测试通过率 > 90%",
            "任务多样性：覆盖所有原子操作",
            "用户模拟器行为符合真实用户分布",
        ],
        common_pitfalls=[
            "环境设计过于简单 → 无法产生有意义的双控制交互",
            "用户模拟器太完美 → 不能反映真实用户的错误和误解",
            "任务无解或有多解 → 导致评估不公平",
            "跳过人工验证 → 可能包含人类也无法解决的任务",
            "忽略效率指标 → 只看完成率会鼓励低效的暴力尝试",
        ],
    ),
    "simulation": ProductionPipeline(
        name="模拟器驱动的数据集生产流程",
        description="基于环境模拟器生成交互式评估数据",
        prerequisites=[
            "领域专家输入",
            "模拟器框架（如 Gym/PettingZoo）",
            "用户行为模型",
            "Python 环境",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.DATA_COLLECTION,
                name="设计双控制环境",
                description="建模为 Dec-POMDP：用户和代理都能影响环境状态",
                inputs=["领域分析", "交互模式"],
                outputs=["env_spec.yaml", "state_space.py"],
                code_snippet='''# Dec-POMDP 环境建模
class DualControlEnv:
    """双控制环境：用户和代理都能使用工具"""

    def __init__(self, config):
        # 状态空间
        self.state = self.init_state(config)

        # 可观测性
        self.user_observable = config["user_observable"]
        self.agent_observable = config["agent_observable"]

    def get_user_observation(self):
        """返回用户可见的部分状态"""
        return {k: self.state[k] for k in self.user_observable}

    def get_agent_observation(self):
        """返回代理可见的部分状态"""
        return {k: self.state[k] for k in self.agent_observable}

    def execute(self, action, actor):
        """执行动作并更新状态"""
        if actor == "user":
            return self._execute_user_action(action)
        else:
            return self._execute_agent_action(action)''',
                tips=[
                    "定义清晰的部分可观测性",
                    "用户和代理的观测可以重叠",
                    "考虑动作的前置条件和效果",
                ],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.PROMPT_DESIGN,
                name="定义动作空间",
                description="定义用户和代理的完整动作空间",
                inputs=["env_spec.yaml"],
                outputs=["actions.py"],
                tips=[
                    "用户动作应有实际环境效果",
                    "代理需要能指导用户执行操作",
                    "定义动作的成功/失败条件",
                ],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.LLM_GENERATION,
                name="任务实例生成",
                description="基于状态空间生成多样化任务实例",
                inputs=["env_spec.yaml", "actions.py"],
                outputs=["tasks.jsonl"],
                tips=[
                    "控制任务复杂度分布",
                    "确保每个任务有可达的目标状态",
                    "生成最优解路径用于评估",
                ],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.POST_PROCESSING,
                name="用户模拟器构建",
                description="创建参数化的用户行为模型",
                inputs=["tasks.jsonl", "actions.py"],
                outputs=["user_sim.py"],
                code_snippet='''class ParameterizedUserSim:
    """参数化用户模拟器"""

    def __init__(self, params):
        self.cooperation = params.get("cooperation", 0.9)
        self.comprehension = params.get("comprehension", 0.8)
        self.patience = params.get("patience", 5)  # max turns
        self.verbosity = params.get("verbosity", 0.5)

    def should_cooperate(self):
        return random.random() < self.cooperation

    def understands_instruction(self, complexity):
        threshold = self.comprehension * (1 - complexity * 0.2)
        return random.random() < threshold''',
                tips=[
                    "模拟真实用户的不完美行为",
                    "参数化便于控制实验变量",
                    "收集真实用户数据校准参数",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.VALIDATION,
                name="人类基准测试",
                description="人类参与者完成任务以验证可解性和建立基准",
                inputs=["tasks.jsonl", "user_sim.py"],
                outputs=["human_baseline.json"],
                tips=[
                    "记录成功率、尝试次数、时间",
                    "收集定性反馈",
                    "剔除人类无法解决的任务",
                ],
            ),
            PipelineStep(
                step_number=6,
                step_type=PipelineStepType.FORMAT_CONVERSION,
                name="数据集打包发布",
                description="划分数据集并准备发布",
                inputs=["human_baseline.json", "tasks.jsonl"],
                outputs=["final_dataset/"],
                tips=[
                    "保留私有测试集防止过拟合",
                    "提供详细的评估协议",
                    "开源模拟器代码",
                ],
            ),
        ],
        quality_criteria=[
            "任务可解性：人类成功率 > 60%",
            "复杂度分布均匀",
            "模拟器与真实用户行为匹配",
            "评估指标明确可计算",
        ],
        common_pitfalls=[
            "模拟器与真实环境差距过大",
            "用户模型过于简单或完美",
            "忽略效率/成本指标",
            "测试集泄露",
        ],
    ),
    "benchmark": ProductionPipeline(
        name="评估基准数据集生产流程",
        description="创建标准化的模型评估基准",
        prerequisites=[
            "评估目标定义",
            "参考数据或任务来源",
            "评估协议文档",
            "人类基准测试资源",
        ],
        steps=[
            PipelineStep(
                step_number=1,
                step_type=PipelineStepType.DATA_COLLECTION,
                name="定义评估维度",
                description="明确要评估的能力维度和指标",
                inputs=["评估目标"],
                outputs=["eval_spec.yaml"],
                tips=[
                    "区分不同能力维度",
                    "定义明确的成功标准",
                    "考虑效率和成本指标",
                ],
            ),
            PipelineStep(
                step_number=2,
                step_type=PipelineStepType.SEED_DATA,
                name="任务收集/设计",
                description="收集或设计评估任务",
                inputs=["eval_spec.yaml"],
                outputs=["raw_tasks.jsonl"],
                tips=[
                    "任务应覆盖所有评估维度",
                    "控制难度分布",
                    "避免数据污染",
                ],
            ),
            PipelineStep(
                step_number=3,
                step_type=PipelineStepType.QUALITY_FILTER,
                name="任务筛选与验证",
                description="确保任务质量和可解性",
                inputs=["raw_tasks.jsonl"],
                outputs=["filtered_tasks.jsonl"],
                tips=[
                    "人工审核关键任务",
                    "确保答案唯一或可验证",
                    "检查任务表述清晰度",
                ],
            ),
            PipelineStep(
                step_number=4,
                step_type=PipelineStepType.VALIDATION,
                name="人类基准建立",
                description="人类完成任务以建立性能基准",
                inputs=["filtered_tasks.jsonl"],
                outputs=["human_baseline.json"],
                tips=[
                    "多人测试取平均",
                    "记录时间和错误类型",
                    "作为模型性能的参照",
                ],
            ),
            PipelineStep(
                step_number=5,
                step_type=PipelineStepType.FORMAT_CONVERSION,
                name="数据集划分与发布",
                description="划分公开/私有测试集并发布",
                inputs=["filtered_tasks.jsonl", "human_baseline.json"],
                outputs=["benchmark_dataset/"],
                tips=[
                    "保留私有测试集",
                    "提供标准评估脚本",
                    "建立排行榜",
                ],
            ),
        ],
        quality_criteria=[
            "任务覆盖所有评估维度",
            "人类基准可靠",
            "评估协议标准化",
            "防止数据泄露",
        ],
        common_pitfalls=[
            "任务难度分布不均",
            "评估指标定义模糊",
            "测试集被污染",
            "缺乏人类基准",
        ],
    ),
}


def get_pipeline_template(
    generation_type: str, synthetic_ratio: float = None, category: str = None
) -> ProductionPipeline:
    """Get appropriate pipeline template based on generation type or category.

    Args:
        generation_type: The type of data generation (synthetic, human, etc.)
        synthetic_ratio: Ratio of synthetic data (0.0-1.0)
        category: Dataset category from deep analysis (programmatic, simulation, etc.)

    Returns:
        Appropriate ProductionPipeline template
    """
    # First check category from deep analysis
    if category:
        category_lower = category.lower()
        if category_lower in ["programmatic", "programmatic_generation"]:
            return PIPELINE_TEMPLATES["programmatic"]
        elif category_lower in ["simulation", "simulator"]:
            return PIPELINE_TEMPLATES["simulation"]
        elif category_lower in ["benchmark", "evaluation"]:
            return PIPELINE_TEMPLATES["benchmark"]
        elif category_lower in ["llm_distillation", "distillation"]:
            return PIPELINE_TEMPLATES["distillation"]
        elif category_lower in ["human_annotation", "human"]:
            return PIPELINE_TEMPLATES["human_annotation"]

    # Fall back to synthetic ratio
    if synthetic_ratio is not None:
        if synthetic_ratio >= 0.9:
            return PIPELINE_TEMPLATES["distillation"]
        elif synthetic_ratio <= 0.1:
            return PIPELINE_TEMPLATES["human_annotation"]
        else:
            return PIPELINE_TEMPLATES["hybrid"]

    # Fall back to generation type
    if generation_type in ["synthetic", "distillation"]:
        return PIPELINE_TEMPLATES["distillation"]
    elif generation_type in ["human", "human_annotation"]:
        return PIPELINE_TEMPLATES["human_annotation"]
    elif generation_type in ["programmatic"]:
        return PIPELINE_TEMPLATES["programmatic"]
    elif generation_type in ["simulation"]:
        return PIPELINE_TEMPLATES["simulation"]
    elif generation_type in ["benchmark"]:
        return PIPELINE_TEMPLATES["benchmark"]
    else:
        return PIPELINE_TEMPLATES["hybrid"]


def pipeline_to_markdown(pipeline: ProductionPipeline, dataset_name: str = None) -> str:
    """Convert pipeline to production guide markdown."""
    lines = []

    # Title
    title = f"数据生产指南：{dataset_name}" if dataset_name else f"数据生产指南：{pipeline.name}"
    lines.append(f"# {title}")
    lines.append("")

    # Overview
    lines.append("## 概述")
    lines.append("")
    lines.append(f"> {pipeline.description}")
    lines.append("")

    if pipeline.estimated_total_cost:
        lines.append(f"**预估总成本**: ${pipeline.estimated_total_cost:,.0f}")
    if pipeline.estimated_total_time:
        lines.append(f"**预估时间**: {pipeline.estimated_total_time}")
    if pipeline.target_size:
        lines.append(f"**目标数据量**: {pipeline.target_size:,} 条")
    lines.append("")

    # Prerequisites
    lines.append("## 前置准备")
    lines.append("")
    for prereq in pipeline.prerequisites:
        lines.append(f"- [ ] {prereq}")
    lines.append("")

    # Pipeline steps
    lines.append("## 生产流程")
    lines.append("")

    # Flow diagram
    lines.append("### 流程图")
    lines.append("")
    lines.append("```")
    step_names = [f"[{s.step_number}. {s.name}]" for s in pipeline.steps]
    lines.append(" → ".join(step_names))
    lines.append("```")
    lines.append("")

    # Detailed steps
    lines.append("### 详细步骤")
    lines.append("")

    for step in pipeline.steps:
        lines.append(f"#### 步骤 {step.step_number}: {step.name}")
        lines.append("")
        lines.append(f"**描述**: {step.description}")
        lines.append("")

        if step.inputs:
            lines.append(f"**输入**: {', '.join(step.inputs)}")
        if step.outputs:
            lines.append(f"**输出**: {', '.join(step.outputs)}")
        if step.tools:
            lines.append(f"**工具**: {', '.join(step.tools)}")
        if step.estimated_cost:
            lines.append(f"**成本**: ${step.estimated_cost} per item")
        lines.append("")

        if step.code_snippet:
            lines.append("**代码示例**:")
            lines.append("")
            lines.append("```python")
            lines.append(step.code_snippet.strip())
            lines.append("```")
            lines.append("")

        if step.tips:
            lines.append("**提示**:")
            lines.append("")
            for tip in step.tips:
                lines.append(f"- {tip}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Quality criteria
    lines.append("## 质量标准")
    lines.append("")
    for criterion in pipeline.quality_criteria:
        lines.append(f"- [ ] {criterion}")
    lines.append("")

    # Common pitfalls
    lines.append("## 常见问题与避坑指南")
    lines.append("")
    for i, pitfall in enumerate(pipeline.common_pitfalls, 1):
        lines.append(f"{i}. ⚠️ {pitfall}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("> 由 DataRecipe 生成 — 数据生产指南")

    return "\n".join(lines)


# =============================================================================
# Composable Pipeline (Upgrade 5)
# =============================================================================


@dataclass
class PhaseDefinition:
    """A reusable pipeline phase definition."""

    phase_id: str
    name: str
    description: str
    default_steps: list[dict] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    condition: str | None = None  # e.g. "has_difficulty_validation"
    assignee: str = "human"  # human, agent, mixed

    def to_dict(self) -> dict:
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "description": self.description,
            "default_steps": self.default_steps,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "assignee": self.assignee,
        }


# ---- Phase Registry ----

_PHASE_REGISTRY: dict[str, PhaseDefinition] = {}


def register_phase(phase: PhaseDefinition) -> None:
    """Register a phase definition."""
    _PHASE_REGISTRY[phase.phase_id] = phase


def get_phase(phase_id: str) -> PhaseDefinition | None:
    """Get a phase by id."""
    return _PHASE_REGISTRY.get(phase_id)


def list_phases() -> list[PhaseDefinition]:
    """List all registered phases."""
    return list(_PHASE_REGISTRY.values())


# ---- Built-in phases ----

register_phase(
    PhaseDefinition(
        phase_id="setup",
        name="环境准备",
        description="验证数据格式、准备模板、审核培训手册",
        default_steps=[
            {"action": "validate_schema", "description": "验证数据格式定义", "assignee": "agent"},
            {"action": "prepare_template", "description": "准备数据模板", "assignee": "agent"},
            {
                "action": "review_training_guide",
                "description": "审核培训手册",
                "assignee": "human",
                "required": True,
            },
        ],
        depends_on=[],
        assignee="mixed",
    )
)

register_phase(
    PhaseDefinition(
        phase_id="pilot",
        name="试点标注",
        description="创建试点样本并进行质量审核",
        default_steps=[
            {
                "action": "create_pilot_samples",
                "description": "创建试点样本 (5-10 条)",
                "count": 10,
                "assignee": "human",
            },
            {"action": "quality_review_pilot", "description": "试点质量审核", "assignee": "human"},
        ],
        depends_on=["setup"],
        assignee="human",
    )
)

register_phase(
    PhaseDefinition(
        phase_id="model_test",
        name="难度验证",
        description="使用模型进行难度验证测试",
        default_steps=[
            {"action": "run_model_test", "description": "执行模型测试", "assignee": "human"},
            {
                "action": "validate_difficulty_result",
                "description": "验证难度测试结果",
                "assignee": "agent",
            },
        ],
        depends_on=["pilot"],
        condition="has_difficulty_validation",
        assignee="mixed",
    )
)

register_phase(
    PhaseDefinition(
        phase_id="human_review",
        name="人工审核",
        description="人工审核抽检数据质量",
        default_steps=[
            {
                "action": "human_review",
                "description": "人工抽检审核",
                "sample_rate": 0.2,
                "assignee": "human",
            },
        ],
        depends_on=["pilot"],
        condition="has_strategy:human_review",
        assignee="human",
    )
)

register_phase(
    PhaseDefinition(
        phase_id="production",
        name="主体标注",
        description="批量标注和增量质检",
        default_steps=[
            {"action": "batch_annotation", "description": "批量标注", "assignee": "human"},
            {
                "action": "incremental_qa",
                "description": "增量质检",
                "sample_rate": 0.2,
                "assignee": "human",
            },
        ],
        depends_on=["pilot"],  # will be dynamically updated
        assignee="human",
    )
)

register_phase(
    PhaseDefinition(
        phase_id="final_qa",
        name="最终质量审核",
        description="全量质检、生成报告、最终审批",
        default_steps=[
            {"action": "full_qa_review", "description": "全量质检", "assignee": "human"},
            {"action": "generate_qa_report", "description": "生成质检报告", "assignee": "agent"},
            {
                "action": "final_approval",
                "description": "最终审批",
                "assignee": "human",
                "required": True,
            },
        ],
        depends_on=["production"],
        assignee="mixed",
    )
)


DEFAULT_PHASE_SEQUENCE = ["setup", "pilot", "model_test", "human_review", "production", "final_qa"]


def assemble_pipeline(
    phase_ids: list[str] | None = None,
    analysis: Any = None,
) -> list[PhaseDefinition]:
    """Assemble a pipeline from phase IDs, filtering by conditions.

    Args:
        phase_ids: ordered list of phase IDs (defaults to DEFAULT_PHASE_SEQUENCE)
        analysis: SpecificationAnalysis for condition evaluation

    Returns:
        Ordered list of PhaseDefinition that passed condition checks,
        with depends_on resolved to only include present phases.
    """
    if phase_ids is None:
        phase_ids = list(DEFAULT_PHASE_SEQUENCE)

    # Gather phases
    phases: list[PhaseDefinition] = []
    for pid in phase_ids:
        phase = get_phase(pid)
        if phase is None:
            continue

        # Evaluate condition
        if phase.condition and analysis is not None:
            if not _evaluate_condition(phase.condition, analysis):
                continue

        phases.append(phase)

    # Resolve depends_on: remove references to phases not in the assembled list
    present_ids = {p.phase_id for p in phases}
    resolved: list[PhaseDefinition] = []
    for phase in phases:
        # Create a copy with resolved depends_on
        from copy import copy

        p = copy(phase)
        p.depends_on = [d for d in phase.depends_on if d in present_ids]

        # For production phase: make it depend on the last validation phase
        if p.phase_id == "production":
            # Find the latest phase that comes before production (i.e. any validation)
            validation_phases = [
                pp.phase_id
                for pp in resolved
                if pp.phase_id not in ("setup", "production", "final_qa")
            ]
            if validation_phases:
                p.depends_on = [validation_phases[-1]]
            elif "pilot" in present_ids:
                p.depends_on = ["pilot"]

        resolved.append(p)

    return resolved


def _evaluate_condition(condition: str, analysis: Any) -> bool:
    """Evaluate a phase condition against an analysis object."""
    if condition == "has_difficulty_validation":
        return getattr(analysis, "has_difficulty_validation", lambda: False)()
    if condition.startswith("has_strategy:"):
        strategy_type = condition.split(":", 1)[1]
        return getattr(analysis, "has_strategy", lambda t: False)(strategy_type)
    # Default: assume true
    return True
