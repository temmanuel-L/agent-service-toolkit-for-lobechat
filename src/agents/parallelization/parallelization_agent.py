# -*- coding: utf-8 -*-
"""
@Time ： 2026/2/13 13:56
@Auth ： luanxing
@File ：parallelization_agent.py
@IDE ：PyCharm
"""

import operator
from typing import Annotated, List, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from core import get_model, settings
from memory.long_term_concat_for_agents import build_llm_messages, prepare_long_term_entry
from utils.log_utils import get_logger
from agents.utils import get_silent_config

logger = get_logger(__name__)


# 定义状态 (State) ---
class AgentState(TypedDict):
    # 具备自动追加能力的消息流
    messages: Annotated[List[BaseMessage], add_messages]
    topic: str
    # 存储并行任务的结果
    summary: str
    questions: str
    key_terms: str
    did_interrupted_asking: bool    # topic属性，是否通过interrupt机制，从用户处进行过询问


# --- 定义抽取主题的节点 ---
async def extract_topic_node(state: AgentState, config: RunnableConfig):
    """节点：从最新的 HumanMessage 中提取话题意图"""
    print("--- NODE: TOPIC EXTRACTION ---")
    sub_config = get_silent_config(config)

    # 异常处理：获取最后一条用户消息
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        raise ValueError("Last message must be a HumanMessage to extract topic.")

    # 让 LLM 判断内容是否包含明确的可供分析的话题
    prompt = [
        SystemMessage(content="""Analyze the user input. If it contains a clear topic for research, 
        output the topic concisely. If it's too vague or empty, output 'INVALID'."""),
        last_msg
    ]
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await llm.ainvoke(prompt, config=sub_config)
    content = response.content.strip()

    if "INVALID" in content.upper():
        return {
            "topic": None,
            "did_interrupted_asking": False,
        }

    return {
        "topic": content,
        "did_interrupted_asking": True,
    }


# --- 定义话题缺失补充提问节点 ---
def ask_missing_info(state: AgentState, config: RunnableConfig) -> dict:
    """通过 interrupt 动态询问缺失的必填字段。"""
    topic_value = state.get('topic')
    did_interrupted_asking = state.get('did_interrupted_asking')
    if topic_value is not None:
        return {}
    if did_interrupted_asking:
        return {}
    ask_user_prompt = "无法提取有效的话题信息，请您再确认一下需要讨论什么话题："

    # 通过 interrupt 获取用户输入
    user_response = interrupt(ask_user_prompt)
    logger.info(f"用户回复: {user_response}")

    return {
        "messages": [HumanMessage(content=user_response)],
        "did_interrupted_asking": True
    }


# --- 定义topic属性检测的路由 ---
def route_by_completeness(state: AgentState) -> Literal["complete", "incomplete"]:
    """根据必填字段是否已全部填写进行路由。"""
    topic_value = state.get('topic')
    did_interrupted_asking = state.get('did_interrupted_asking')
    if topic_value is None:
        return "incomplete"
    if not did_interrupted_asking:
        return "incomplete"
    return "complete"


# --- 定义一个空的中间节点用于分发并行任务 ---
def parallel_distributor(state: AgentState):
    """仅仅作为一个逻辑中转站，触发下游并行"""
    return {}


# --- 定义并行节点 (Parallel Nodes / Fan-out) ---
async def summarize_node(state: AgentState, config: RunnableConfig) -> dict:
    """任务 A：摘要"""
    logger.info("--- NODE: SUMMARIZER (Parallel) ---")
    # 深度合并配置：确保 skip_stream 进入 config 的最底层
    # 这样既能保留 Langfuse 回调，又能确保该调用产生的每一个 Token 事件都携带此标签
    sub_config = get_silent_config(config)

    prompt = [
        SystemMessage(content="Summarize the topic concisely:"),
        HumanMessage(content=state["topic"])
    ]
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await llm.ainvoke(prompt, config=sub_config)
    # logger.info(f"生成关于此话题的摘要内容: {response.content}")
    return {"summary": response.content}


# --- 定义提问节点 ---
async def questions_node(state: AgentState, config: RunnableConfig) -> dict:
    """任务 B：提问"""
    logger.info("--- NODE: QUESTION GENERATOR (Parallel) ---")

    sub_config = get_silent_config(config)

    prompt = [
        SystemMessage(content="Generate three interesting questions about this topic:"),
        HumanMessage(content=state["topic"])
    ]
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await llm.ainvoke(prompt, config=sub_config)
    # logger.info(f"生成关于此话题的提问内容: {response.content}")
    return {"questions": response.content}


# --- 定义述语提取节点 ---
async def terms_node(state: AgentState, config: RunnableConfig) -> dict:
    """任务 C：术语提取"""
    print("--- NODE: TERMS EXTRACTOR (Parallel) ---")

    sub_config = get_silent_config(config)

    prompt = [
        SystemMessage(content="Identify 5-10 key terms, separated by commas:"),
        HumanMessage(content=state["topic"])
    ]
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await llm.ainvoke(prompt, config=sub_config)
    # logger.info(f"生成关于此话题的专业术语内容: {response.content}")
    return {"key_terms": response.content}


# --- 定义综合节点 (Synthesis Node / Fan-in) ---
async def synthesis_node(state: AgentState, config: RunnableConfig) -> dict:
    """汇聚所有并行结果进行最终创作 - 深度优化版"""

    # 1. 核心门禁, 必须放在函数最开始
    # 只要有一个素材没齐，立即退出，绝对不要碰 LLM
    summary = state.get("summary")
    terms = state.get("key_terms")
    questions = state.get("questions")

    logger.info(f'summary: {summary}')
    logger.info(f'questions: {questions}')
    logger.info(f'terms: {terms}')

    if not (summary and terms and questions):
        # logger.info("素材未齐，Synthesizer 节点跳过执行")
        return {}

    logger.info("--- NODE: SYNTHESIZER (Fan-in) ---")

    # 2. 构建结构化的背景资料prompt
    # 通过明确的分隔符和角色设定，防止模型直接复制
    context = f"""
    ### 调研素材库 ###
    【初步摘要】: 
    {state['summary']}

    【关键术语】: 
    {state['key_terms']}

    【核心待解答问题】: 
    {state['questions']}
    ##################

    你现在是一名资深研究专家。请基于上述素材，为话题“{state['topic']}”撰写一份深度分析报告。

    ### 创作要求 ###
    1. 逻辑重组：严禁按照素材出现的先后顺序机械堆砌。
    2. 语序优化：确保全篇行文流畅，消除素材拼接感。
    3. 长度控制：内容精炼，确保最终输出在 1500 Tokens 以内（为了兼容后续的记忆存储）。
    4. 格式：使用清晰的 Markdown 层级。
    """

    prompt = build_llm_messages(
        [HumanMessage(content=context)],
        config,
        agent_system="你是一位不直接复述素材，而是进行二次深度创作的专业分析师。",
    )
    # 3. 真正执行生成
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await llm.ainvoke(prompt, config)

    # 4. 返回结果并清理状态
    return {
        "messages": [AIMessage(content=response.content)],
        # 为下一次生成，重置以下字段内容
        "topic": None,
        "summary": None,
        "key_terms": None,
        "questions": None,
        "did_interrupted_asking": None,
    }


# --- 构建图并定义路由及并行流 ---
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("prepare_long_term", prepare_long_term_entry)
workflow.add_node("extract_topic", extract_topic_node)
workflow.add_node("ask_missing", ask_missing_info)
workflow.add_node("distributor", parallel_distributor) # 新增中转节点
workflow.add_node("summarizer", summarize_node)
workflow.add_node("questions_gen", questions_node)
workflow.add_node("terms_ext", terms_node)
workflow.add_node("synthesizer", synthesis_node)

workflow.add_edge(START, "prepare_long_term")
workflow.add_edge("prepare_long_term", "extract_topic")

# 设置topic判断的路由
workflow.add_conditional_edges(
    "extract_topic",
    route_by_completeness,
    {
        "complete": "distributor",   # 并行化拓扑结构
        "incomplete": "ask_missing"
    }
)

# 从中转节点并行扇出 (Static Edge 支持列表)
workflow.add_edge("distributor", "summarizer")
workflow.add_edge("distributor", "questions_gen")
workflow.add_edge("distributor", "terms_ext")

# 询问后回到抽取节点以处理新输入
workflow.add_edge("ask_missing", "extract_topic")

# 三个节点同时指向同一个汇聚节点 (Fan-in)
# LangGraph 会自动等待所有前置节点完成后才执行下一个节点
workflow.add_edge("summarizer", "synthesizer")
workflow.add_edge("questions_gen", "synthesizer")
workflow.add_edge("terms_ext", "synthesizer")

# 汇聚节点指向 END
workflow.add_edge("synthesizer", END)

# 编译应用
parallelization_agent = workflow.compile().with_config({'recursion_limit': 10})

# try:
#     graph_obj = parallelization_agent.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     with open('parallelization_assistant_state_graph.png', 'wb') as f:
#         f.write(pic)
# except Exception as e:
#     logger.warning(f"生成图例失败: {e}")