# -*- coding: utf-8 -*-
"""
Graph RAG Agent

基于 Neo4j + neo4j-graphrag 的图增强 RAG 智能体。
流程：
1. 一级意图识别：闲聊 / RAG（LLM 判断，参考图数据库 schema）
2. RAG 二级意图识别：匹配 VectorCypherRetriever 或 Text2CypherRetriever
3. VectorCypherRetriever：通过 LLM 匹配 JSON 中的 intention，返回预置 retrieval_query
4. Text2CypherRetriever：传入 schema + few-shot examples，由 GraphRAG 一站式检索并生成回答
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from neo4j_graphrag.retrievers import VectorCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.schema import get_schema

from core import get_model, get_model_for_neo4j_graphrag, get_embedding_model, settings
from memory.long_term_concat_for_agents import build_llm_messages, prepare_long_term_entry
from utils.log_utils import get_logger
from agents.graph_rag_agent.neo4j_client import get_neo4j_driver

logger = get_logger(__name__)

_SCHEMA_CACHE: str | None = None

_EMPTY_CONTEXT_REPLY = (
    "根据当前知识库中的图数据，无法找到与您问题直接相关的记录。"
    "本库目前主要包含 NetApp 的 10-K 文档片段，以及部分机构在特定报告日期的持股关系；"
    "请勿使用外部公开资料臆测答案。如需查询其他公司或日期，请确认数据是否已入库。"
)

_RAG_ANSWER_SYSTEM_PROMPT = """你是 Graph RAG 助手。请仅依据下方 Context 中的检索结果回答用户问题。

规则：
1. 只能使用 Context 中出现的信息，禁止引用外部常识或公开资料补全。
2. OWNS_STOCK_IN 边上的 value 字段单位为美元（USD），shares 为持股股数；不要额外乘以 1000 或 1000000。
3. 若 Context 为空或与问题无关，明确说明知识库中无相关数据，不要编造。
4. 回答使用中文，结构清晰；不要输出 Cypher 语句或 MATCH 查询。"""

# 含具体时间/季度/比较/排除/item 章节 → 必须走 Text2Cypher
_TEXT2CYPHER_HINT = re.compile(
    r"(\d{4}[-年/]"
    r"|第[一二三四1234]季度"
    r"|Q[1-4]"
    r"|item\s*\d"
    r"|除了"
    r"|比较|对比|更高|更低|多少家"
    r"|苹果|Apple"
    r"|BlackRock.*Vanguard|Vanguard.*BlackRock)",
    re.IGNORECASE,
)

_CYPHER_LINE_PREFIX = re.compile(
    r"^(MATCH|WHERE|RETURN|ORDER|LIMIT|WITH|CALL|AND|OR|OPTIONAL|UNWIND|\(|\)-\[:)",
    re.IGNORECASE,
)


# =============================================================================
# 状态定义
# =============================================================================
class GraphRAGState(MessagesState, total=False):
    """Graph RAG 智能体状态。"""

    # 一级意图：chat 或 rag
    intent: Literal["chat", "rag"]
    # RAG 子意图：vector_cypher 或 text2cypher
    rag_sub_intent: Literal["vector_cypher", "text2cypher"]
    # 当前用户问题快照
    current_user_text: Optional[str]
    # VectorCypherRetriever 使用的 retrieval_query
    retrieval_query: Optional[str]
    # 最终检索结果 / 回答
    final_answer: Optional[str]
    # 错误信息
    error: Optional[str]


# =============================================================================
# 结构化意图分类模型（用于 bind_tools 可靠输出）
# =============================================================================
class ChatVsRagIntent(BaseModel):
    """判断用户问题是否需要使用图数据库进行 RAG 检索。"""
    intent: Literal["chat", "rag"] = Field(
        ...,
        description="chat=一般闲聊或与图数据无关的问题；rag=涉及公司、投资者、持股、10-K 表单等问题"
    )


class RagSubIntent(BaseModel):
    """判断 RAG 检索应使用 VectorCypherRetriever 还是 Text2CypherRetriever。"""
    sub_intent: Literal["vector_cypher", "text2cypher"] = Field(
        ...,
        description="vector_cypher=问题与预定义意图高度匹配，可使用固定 retrieval_query；text2cypher=需要 LLM 动态生成 Cypher"
    )


class VectorCypherSelection(BaseModel):
    """从预定义意图列表中选择最匹配的 intention 索引。"""
    selected_index: int = Field(
        ...,
        description="最匹配用户问题的 intention 在列表中的索引（从0开始）",
        ge=0
    )


# =============================================================================
# 辅助函数
# =============================================================================
def _last_human_message_text(state: GraphRAGState) -> str:
    """从 messages 中提取最后一条 HumanMessage 的文本。"""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _should_force_text2cypher(user_text: str) -> bool:
    """规则优先：带日期/季度/比较/排除/item/库外公司 → Text2Cypher。"""
    return bool(_TEXT2CYPHER_HINT.search(user_text))


def _get_cached_neo4j_schema(driver) -> str:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = get_schema(driver=driver, is_enhanced=True, sanitize=True)
    return _SCHEMA_CACHE


def _retriever_items(retriever_result: Any) -> list:
    if retriever_result is None:
        return []
    if hasattr(retriever_result, "items"):
        return list(retriever_result.items or [])
    if isinstance(retriever_result, dict):
        return list(retriever_result.get("items") or [])
    return []


def _format_retriever_context(items: list) -> str:
    if not items:
        return ""
    parts: list[str] = []
    for item in items:
        if hasattr(item, "content"):
            parts.append(str(item.content))
        elif isinstance(item, dict):
            parts.append(str(item.get("content", item)))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _sanitize_answer(text: str) -> str:
    """去掉模型偶发泄露到正文中的 Cypher 片段。"""
    if not text:
        return text
    stripped = text.strip()
    if not stripped.upper().startswith("MATCH"):
        return stripped

    if "\n" not in stripped:
        match = re.search(
            r"(根据|基于|从|目前|知识库|无法|抱歉|BlackRock|Vanguard|NetApp|Netapp|机构|持股)",
            stripped,
            re.IGNORECASE,
        )
        if match:
            return stripped[match.start() :].strip()
        return _EMPTY_CONTEXT_REPLY

    kept: list[str] = []
    past_cypher = False
    for line in stripped.splitlines():
        if not past_cypher:
            if line.strip() and not _CYPHER_LINE_PREFIX.match(line.strip()):
                past_cypher = True
                kept.append(line)
        else:
            kept.append(line)
    cleaned = "\n".join(kept).strip()
    return cleaned or _EMPTY_CONTEXT_REPLY


def _make_node_result(answer: str) -> dict:
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


async def _run_retriever_search(retriever, query_text: str, top_k: int = 10):
    if isinstance(retriever, Text2CypherRetriever):
        return await asyncio.to_thread(retriever.search, query_text=query_text)
    return await asyncio.to_thread(retriever.search, query_text=query_text, top_k=top_k)


async def _generate_answer_from_context(
    user_text: str,
    context: str,
    config: RunnableConfig,
    model_name: str,
) -> str:
    if not context.strip():
        return _EMPTY_CONTEXT_REPLY

    llm = get_model(model_name).with_config(tags=["skip_stream"])
    response = await llm.ainvoke(
        [
            SystemMessage(content=_RAG_ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{user_text}\n\nAnswer:"
            ),
        ],
        config,
    )
    content = response.content if hasattr(response, "content") else str(response)
    return _sanitize_answer(content if isinstance(content, str) else str(content))


# =============================================================================
# 一级意图识别节点：闲聊 vs RAG
# =============================================================================
async def chat_vs_rag_intent_node(state: GraphRAGState, config: RunnableConfig) -> dict:
    """使用 LLM 判断用户问题是一级意图：闲聊 或 RAG（通过 bind_tools 结构化输出）。

    判断依据：图数据库主要存储 10-K 申报、机构投资者持股（Vanguard、BlackRock 等对 Netapp 等公司的持股数据）。
    如果问题与这些领域相关 → RAG，否则 → chat。
    """
    user_text = _last_human_message_text(state)
    if not user_text:
        return {"intent": "chat"}

    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_with_tools = llm.bind_tools([ChatVsRagIntent])

    try:
        response = await llm_with_tools.with_config(tags=["skip_stream"]).ainvoke(user_text, config)
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = dict(tool_call.get("args") or {})
            parsed = ChatVsRagIntent(**args)
            intent = parsed.intent
        else:
            # 回退：尝试解析 content
            content = (response.content or "").strip().lower() if hasattr(response, "content") else ""
            intent = content if content in ("chat", "rag") else "chat"

        logger.info(f"一级意图识别结果: {intent}")
        return {"intent": intent, "current_user_text": user_text}
    except Exception as e:
        logger.error(f"意图识别失败: {e}")
        return {"intent": "chat", "current_user_text": user_text}


# =============================================================================
# RAG 二级意图识别：VectorCypherRetriever vs Text2CypherRetriever
# =============================================================================
async def rag_sub_intent_node(state: GraphRAGState, config: RunnableConfig) -> dict:
    """二级意图识别：判断走 VectorCypherRetriever（预置 retrieval_query）还是 Text2CypherRetriever。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)

    if _should_force_text2cypher(user_text):
        logger.info("RAG 二级意图: text2cypher（规则强制）")
        return {"rag_sub_intent": "text2cypher"}

    from agents.graph_rag_agent.retrieval_query.loader import load_intention_statements

    intentions = load_intention_statements()
    intention_list = "\n".join([f"- {item['intention']}" for item in intentions])

    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_with_tools = llm.bind_tools([RagSubIntent])

    system_prompt = f"""你是一个 RAG 路由器。请判断用户问题应使用哪种检索方式。

                    仅当同时满足以下条件时，输出 vector_cypher：
                    - 问题是在宽泛地「介绍某公司的投资者/机构股东及持股概况」
                    - 不涉及具体日期、季度、比较、排除某机构、item 章节、或库中未收录的公司
                    - 与下方某条预定义意图语义高度一致
                    
                    以下情况必须输出 text2cypher：
                    - 含具体日期/季度（如 2020-12-31、2023年第一季度、Q2）
                    - 比较两家机构、排除某机构、聚合统计
                    - 查询 10-K 某 item 章节内容
                    - 查询 Apple/苹果等非 NetApp 公司
                    
                    预定义意图列表（仅 vector_cypher 可匹配）：
                    {intention_list}"""

    try:
        response = await llm_with_tools.with_config(tags=["skip_stream"]).ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_text)],
            config,
        )
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = dict(tool_call.get("args") or {})
            parsed = RagSubIntent(**args)
            sub_intent = parsed.sub_intent
        else:
            content = (response.content or "").strip().lower() if hasattr(response, "content") else ""
            sub_intent = content if content in ("vector_cypher", "text2cypher") else "text2cypher"

        logger.info(f"RAG 二级意图: {sub_intent}")
        return {"rag_sub_intent": sub_intent}
    except Exception as e:
        logger.error(f"二级意图识别失败: {e}")
        return {"rag_sub_intent": "text2cypher"}


# =============================================================================
# VectorCypherRetriever 分支
# =============================================================================
async def vector_cypher_retrieval_node(state: GraphRAGState, config: RunnableConfig) -> dict:
    """通过 LLM 从预定义意图列表中选择最匹配的索引，直接定位 statement 并执行检索。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)

    from agents.graph_rag_agent.retrieval_query.loader import load_intention_statements

    intentions = load_intention_statements()
    # 带编号的意图列表，方便 LLM 输出索引
    numbered_intentions = "\n".join(
        [f"{i}. {item['intention']}" for i, item in enumerate(intentions)]
    )

    llm = get_model_for_neo4j_graphrag(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_with_tools = llm.bind_tools([VectorCypherSelection])

    system_prompt = f"""请判断用户问题最匹配下面哪个预定义意图，并输出该意图在列表中的索引（从0开始）。
                    预定义意图列表：
                    {numbered_intentions}
                    
                    仅输出最匹配的索引数字。"""

    try:
        response = await llm_with_tools.with_config(tags=["skip_stream"]).ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_text)],
            config
        )

        selected_index = 0  # 默认回退
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = dict(tool_call.get("args") or {})
            parsed = VectorCypherSelection(**args)
            selected_index = parsed.selected_index

        # 校验索引范围
        if not (0 <= selected_index < len(intentions)):
            logger.warning(f"LLM 返回的索引 {selected_index} 越界，回退到 0")
            selected_index = 0

        statement = intentions[selected_index]["statement"]
        logger.info(f"VectorCypherRetriever 选中索引 {selected_index}，意图: {intentions[selected_index]['intention'][:50]}...")

        driver = get_neo4j_driver()
        embeddings = get_embedding_model()

        retriever = VectorCypherRetriever(
            driver=driver,
            neo4j_database=os.getenv("NEO4J_DB"),
            index_name="form_10k_chunks",
            embedder=embeddings,
            retrieval_query=statement,
        )

        model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
        retriever_result = await _run_retriever_search(retriever, user_text, top_k=10)
        items = _retriever_items(retriever_result)
        context = _format_retriever_context(items)
        answer = await _generate_answer_from_context(user_text, context, config, model_name)

        logger.info("VectorCypherRetriever 检索完成")
        return {**_make_node_result(answer), "retrieval_query": statement}
    except Exception as e:
        logger.error(f"VectorCypherRetriever 执行失败: {e}")
        return {**_make_node_result("检索失败，请稍后重试。"), "error": str(e)}


# =============================================================================
# Text2CypherRetriever 分支
# =============================================================================
async def text2cypher_retrieval_node(state: GraphRAGState, config: RunnableConfig) -> dict:
    """Text2CypherRetriever 检索 + 受控生成（空结果不幻觉）。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)
    driver = get_neo4j_driver()
    llm = get_model_for_neo4j_graphrag(config["configurable"].get(
        "model", settings.DEFAULT_MODEL)).with_config(tags=["skip_stream"])
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)

    from agents.graph_rag_agent.text2cypher_examples.examples import load_examples

    try:
        neo4j_schema = _get_cached_neo4j_schema(driver)
        examples = load_examples()

        retriever = Text2CypherRetriever(
            driver=driver,
            neo4j_database=os.getenv("NEO4J_DB"),
            llm=llm,
            neo4j_schema=neo4j_schema,
            examples=examples,
        )

        retriever_result = await _run_retriever_search(retriever, user_text)
        items = _retriever_items(retriever_result)
        if not items:
            logger.info("Text2CypherRetriever 无命中记录")
            return _make_node_result(_EMPTY_CONTEXT_REPLY)

        context = _format_retriever_context(items)
        answer = await _generate_answer_from_context(user_text, context, config, model_name)

        logger.info("Text2CypherRetriever 检索完成")
        return _make_node_result(answer)
    except Exception as e:
        logger.error(f"Text2CypherRetriever 执行失败: {e}")
        return {**_make_node_result("检索失败，请稍后重试。"), "error": str(e)}


# =============================================================================
# 闲聊分支
# =============================================================================
async def chat_node(state: GraphRAGState, config: RunnableConfig) -> dict:
    """直接调用 LLM 进行普通对话。"""
    user_text = _last_human_message_text(state)
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    try:
        messages = build_llm_messages([HumanMessage(content=user_text)], config)
        resp = await llm.ainvoke(messages, config)
        answer = resp.content if hasattr(resp, "content") else str(resp)
        return _make_node_result(answer if isinstance(answer, str) else str(answer))
    except Exception as e:
        logger.error(f"LLM 对话失败: {e}")
        return _make_node_result("抱歉，暂时无法回复。")


# =============================================================================
# 条件路由函数
# =============================================================================
def route_after_first_intent(state: GraphRAGState) -> str:
    """一级意图路由。"""
    intent = state.get("intent")
    if intent == "rag":
        return "rag"          # 返回 path_map 的 key
    return "chat"


def route_after_sub_intent(state: GraphRAGState) -> str:
    """RAG 二级意图路由。"""
    sub = state.get("rag_sub_intent")
    if sub == "vector_cypher":
        return "vector_cypher"
    return "text2cypher"


# =============================================================================
# 构建 StateGraph
# =============================================================================
def build_graph_rag_agent():
    """构建并返回编译后的 Graph RAG Agent。"""
    graph = StateGraph(GraphRAGState)

    # 节点
    graph.add_node("prepare_long_term", prepare_long_term_entry)
    graph.add_node("intent_router", chat_vs_rag_intent_node)
    graph.add_node("rag_sub_intent", rag_sub_intent_node)
    graph.add_node("vector_cypher", vector_cypher_retrieval_node)
    graph.add_node("text2cypher", text2cypher_retrieval_node)
    graph.add_node("chat", chat_node)

    # 边（使用显式 path_map，确保 Mermaid 渲染器能正确识别所有分支和循环）
    graph.add_edge(START, "prepare_long_term")
    graph.add_edge("prepare_long_term", "intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_after_first_intent,
        {"rag": "rag_sub_intent", "chat": "chat"},
    )

    graph.add_conditional_edges(
        "rag_sub_intent",
        route_after_sub_intent,
        {"vector_cypher": "vector_cypher", "text2cypher": "text2cypher"},
    )

    # VectorCypher / Text2Cypher 路径
    graph.add_edge("vector_cypher", END)
    graph.add_edge("text2cypher", END)

    # Chat 路径
    graph.add_edge("chat", END)

    compiled = graph.compile()
    return compiled


# 导出编译后的图（供 agents.py 注册使用）
graph_rag_agent = build_graph_rag_agent()

# try:
#     graph_obj = graph_rag_agent.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     with open('graph_rag_agent.png', 'wb') as f:
#         f.write(pic)
# except Exception as e:
#     logger.warning(f"生成图例失败: {e}")