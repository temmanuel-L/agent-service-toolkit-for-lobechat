from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from memory.long_term_concat_for_agents import (
    LONG_TERM_MEMORY_CONTENT_KEY,
    LONG_TERM_MEMORY_SKIP_KEY,
    LONG_TERM_MEMORY_SOURCE,
    build_llm_messages,
    is_long_term_memory_message,
    should_attach_long_term_memory,
    strip_legacy_long_term_messages,
)


def _ltm(content: str) -> SystemMessage:
    return SystemMessage(
        content=content,
        additional_kwargs={"source": LONG_TERM_MEMORY_SOURCE},
        id=f"ltm-{content}",
    )


def test_build_llm_messages_merges_memory_and_agent_system():
    messages = [
        _ltm("old"),
        SystemMessage(content="frontend system"),
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]
    config = RunnableConfig(
        configurable={
            LONG_TERM_MEMORY_CONTENT_KEY: "[长期记忆]\nnew memory",
        }
    )
    result = build_llm_messages(messages, config, agent_system="agent prompt")

    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert "new memory" in result[0].content
    assert "agent prompt" in result[0].content
    assert result[0].content.index("new memory") < result[0].content.index("agent prompt")
    assert result[1].content == "hi"
    assert result[2].content == "hello"


def test_build_llm_messages_skips_when_skip_flag():
    messages = [HumanMessage(content="hi")]
    config = RunnableConfig(
        configurable={
            LONG_TERM_MEMORY_CONTENT_KEY: "memory",
            LONG_TERM_MEMORY_SKIP_KEY: True,
        }
    )
    assert not should_attach_long_term_memory(config)
    assert build_llm_messages(messages, config) == messages


def test_build_llm_messages_without_memory_or_agent_system():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert build_llm_messages(messages, None) == messages


def test_strip_legacy_long_term_messages():
    messages = [_ltm("a"), HumanMessage(content="b")]
    removals = strip_legacy_long_term_messages(messages)
    assert len(removals) == 1
    assert removals[0].id == "ltm-a"


def test_is_long_term_memory_message():
    assert is_long_term_memory_message(_ltm("x"))
    assert not is_long_term_memory_message(SystemMessage(content="other"))
