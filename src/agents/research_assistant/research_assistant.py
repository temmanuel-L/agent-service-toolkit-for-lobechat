from datetime import datetime
from typing import Annotated, Literal

from langchain_community.tools import OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator, vector_search_tool, web_search
from core import get_model, settings
from memory.long_term_concat_for_agents import build_llm_messages, prepare_long_term_entry
from rag import (
    create_model_to_tools_router,
    create_reset_rounds_node,
    create_tools_node_with_rounds_increment,
    create_force_done_node,
    create_rag_evaluator_node,
    tool_rounds_add_reducer,
)
from utils.log_utils import get_logger

logger = get_logger(__name__)


class AgentState(MessagesState, total=False):
    """`total=False` 来自 PEP589（TypedDict）规范。

    文档： https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    tool_rounds 使用显式 reducer 确保合并/持久化后不被丢失。
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    tool_rounds: Annotated[int, tool_rounds_add_reducer]  # 本轮工具轮数，用于轮次上限
    retrieval_eval: Literal["sufficient", "insufficient", "not_found"]


tools = [web_search, calculator, vector_search_tool]

# 若配置了 API Key，则添加天气工具
# 申请 API Key： https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    你是一位乐于助人的研究助手，能够进行网页搜索并使用其他工具。
    今天的日期是 {current_date}。

    注意：用户可以看到工具的返回结果与执行步骤。

    请牢记：
    - 在回答中加入用于引用的 Markdown 链接。除非确有必要，否则每次回答只给 1-2 个引用。
      只允许使用工具返回的链接。
    - 使用 calculator（numexpr）工具回答数学问题。用户不理解 numexpr，最终回复请用人类可读格式，
      例如写成 "300 * 200"，不要写成 "(300 \\times 200)"。
    """


def _preprocess_state(state: AgentState, config: RunnableConfig):
    return build_llm_messages(state["messages"], config, agent_system=instructions)


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        _preprocess_state,
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"本对话被标记为不安全内容：{', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    
    # 追踪当前状态
    current_rounds = state.get("tool_rounds", 0)
    logger.info("[State Check][acall_model] entering with tool_rounds=%s", current_rounds)
    
    response = await model_runnable.ainvoke(state, config)

    # 在此处运行 llama guard，避免返回不安全内容
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
            "tool_rounds": 0, # 这里 0 表示本节点不对 tool_rounds 做增量累加（加 0）
        }

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="抱歉，需要更多步骤才能处理该请求。",
                )
            ],
            "tool_rounds": 0,
        }
    
    # 显式回传 0，在累加式 reducer 模式下 (current + 0 = current)，确保状态不丢失且不增加轮次
    return {"messages": [response], "tool_rounds": 0}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# 每轮对话内最多执行的工具轮数（含 WebSearch 等）；至少 3 轮可让模型在两次检索后仍有一次机会做总结或说明“未找到相关结果”
MAX_TOOL_ROUNDS = 3

# 定义图
agent = StateGraph(AgentState)
agent.add_node("prepare_long_term", prepare_long_term_entry)
agent.add_node("model", acall_model)
agent.add_node("tools", create_tools_node_with_rounds_increment(ToolNode(tools)))
agent.add_node("evaluator", create_rag_evaluator_node(tool_names=["web_search", "vector_search_tool"]))
agent.add_node("force_done", create_force_done_node())
agent.add_node("guard_input", llama_guard_input)
agent.add_node("reset_rounds", create_reset_rounds_node())
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_edge(START, "prepare_long_term")
agent.add_edge("prepare_long_term", "guard_input")


# 检测不安全输入，若命中则阻断后续处理
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "reset_rounds"}
)
agent.add_edge("reset_rounds", "model")

# 阻断不安全内容后直接结束
agent.add_edge("block_unsafe_content", END)

# tools 返回时先经过 evaluator，再回到 model
agent.add_edge("tools", "evaluator")
agent.add_edge("evaluator", "model")
agent.add_edge("force_done", END)

agent.add_conditional_edges(
    "model",
    create_model_to_tools_router(max_tool_rounds=MAX_TOOL_ROUNDS),
    {"tools": "tools", "done": END, "force_done": "force_done"},
)


research_assistant = agent.compile().with_config({"recursion_limit": 10})

# try:
#     graph_obj = research_assistant.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     with open('state_graph_research_assistant.png', 'wb') as f:
#         f.write(pic)
# except Exception as e:
#     logger.warning(f"生成图例失败: {e}")
