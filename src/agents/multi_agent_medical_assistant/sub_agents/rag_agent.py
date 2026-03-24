# -*- coding: utf-8 -*-
"""RAG 知识库问答子智能体。"""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.utils import get_silent_config
from core import get_model, settings

from agents.multi_agent_medical_assistant.state import MedicalAgentState, get_input_text
from rag import SearchKnowledgeTool
from utils.log_utils import get_logger

logger = get_logger(__name__)


async def run_rag_agent(
    state: MedicalAgentState, config: RunnableConfig | None = None
) -> dict:
    """RAG 知识库问答。使用 SearchKnowledgeTool + LLM 生成。"""
    text = get_input_text(state.get("current_input"))
    cfg = (config or {}).get("configurable", {}) or {}
    kb_ids = cfg.get("kb_ids")
    if not kb_ids:
        return {
            "output": AIMessage(content="请指定知识库 (kb_ids)。"),
            "agent_name": "RAG_AGENT",
            "retrieval_confidence": 0.0,
        }
    tool = SearchKnowledgeTool()
    try:
        context = await tool.ainvoke({"query": text}, config=config)
    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        context = "检索失败，请稍后重试。"
    model = get_model(cfg.get("model", settings.DEFAULT_MODEL))
    prompt = f"""基于以下检索结果回答用户问题。若检索结果不足以回答，请如实说明。

检索结果:
{context[:8000]}

用户问题: {text}

请给出准确、简洁的回答。必须使用与用户问题相同的语言作答（用户用中文则用中文，用英文则用英文）。"""
    silent_cfg = get_silent_config(config or {})
    resp = model.invoke(prompt, config=silent_cfg)
    content = resp.content if hasattr(resp, "content") else str(resp)
    return {
        "output": AIMessage(content=content),
        "agent_name": "RAG_AGENT",
        "retrieval_confidence": 0.9,
        "insufficient_info": "don't have enough" in content.lower()
        or "insufficient" in content.lower(),
    }
