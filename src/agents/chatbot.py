from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, add_messages
from core import get_model, settings
from memory.long_term_concat_for_agents import (
    build_llm_messages,
    prepare_long_term_entry,
)


# 第一步：定义状态（State）
# 使用 Annotated 和 add_messages。这告诉 LangGraph：
# 当有新消息进来时，执行“追加/合并”操作，而不是“全量覆盖”。
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 第二步：定义节点（Node）
async def call_model(state: ChatState, config: RunnableConfig):
    # 这里的 state["messages"] 会自动包含该 thread_id 下的所有历史记录
    # LangGraph 会自动从 Postgres 中帮你加载并合并好
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    messages = build_llm_messages(state["messages"], config)
    response = await model.ainvoke(messages)

    # 关键：只需返回【新增】的部分
    # Reducer 会自动将这个 [response] 追加到数据库的 messages 列表中
    return {"messages": [response]}


def build_workflow():
    graph = StateGraph(ChatState)

    graph.add_node("prepare_long_term", prepare_long_term_entry)
    graph.add_node("chatbot", call_model)

    graph.add_edge(START, "prepare_long_term")
    graph.add_edge("prepare_long_term", "chatbot")
    graph.add_edge("chatbot", END)
    return graph

workflow = build_workflow()
chatbot = workflow.compile()
