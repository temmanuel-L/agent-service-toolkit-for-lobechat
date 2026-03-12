# -*- coding: utf-8 -*-
"""Web 搜索子智能体。"""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from core import get_model, settings

from agents.multi_agent_medical_assistant.state import MedicalAgentState, get_input_text
from utils.log_utils import get_logger

logger = get_logger(__name__)


def run_web_search_agent(
    state: MedicalAgentState, config: RunnableConfig | None = None
) -> dict:
    """Web 搜索。"""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        text = get_input_text(state.get("current_input"))
        result = search.invoke(text)
        model = get_model(
            (config or {}).get("configurable", {}).get("model", settings.DEFAULT_MODEL)
        )
        resp = model.invoke(
            f"Based on web search results, summarize a helpful medical information response:\n\n{result}"
        )
        content = resp.content if hasattr(resp, "content") else str(resp)
        return {"output": AIMessage(content=content), "agent_name": "WEB_SEARCH_PROCESSOR_AGENT"}
    except Exception as e:
        logger.warning("Web search fallback: %s", e)
        return {
            "output": AIMessage(
                content="Web search is not available. Please try a different question."
            ),
            "agent_name": "WEB_SEARCH_PROCESSOR_AGENT",
        }
