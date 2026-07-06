import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import StreamWriter

from agents.bg_task_agent.task import Task
from core import get_model, settings
from memory.long_term_concat_for_agents import build_llm_messages, prepare_long_term_entry


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


def _preprocess_state(state: AgentState, config: RunnableConfig):
    return build_llm_messages(state["messages"], config)


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        _preprocess_state,
        name="StateModifier",
    )
    return preprocessor | model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def bg_task(state: AgentState, writer: StreamWriter) -> AgentState:
    task1 = Task("Simple task 1...", writer)
    task2 = Task("Simple task 2...", writer)

    task1.start()
    await asyncio.sleep(2)
    task2.start()
    await asyncio.sleep(2)
    task1.write_data(data={"status": "Still running..."})
    await asyncio.sleep(2)
    task2.finish(result="error", data={"output": 42})
    await asyncio.sleep(2)
    task1.finish(result="success", data={"output": 42})
    return {"messages": []}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("prepare_long_term", prepare_long_term_entry)
agent.add_node("model", acall_model)
agent.add_node("bg_task", bg_task)
agent.add_edge(START, "prepare_long_term")
agent.add_edge("prepare_long_term", "bg_task")

agent.add_edge("bg_task", "model")
agent.add_edge("model", END)

bg_task_agent = agent.compile()
