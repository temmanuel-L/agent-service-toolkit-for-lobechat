from typing import Optional
import asyncio
import os
import numexpr
import math
import re

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search.tool import TavilyInput, TavilySearchResults
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from core import settings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


class FormattedDuckDuckGoSearchResults(DuckDuckGoSearchResults):
    """
    增强版 DuckDuckGo 搜索工具，支持相关性筛选、去重与数量控制。

    数量、相关性与超时均从 core.settings 读取（对应 .env 中的 DDGS_* 配置）。

    超时与条数逻辑：
    - 每次调用 WebSearch(query) = 只发 1 次搜索请求；DDGS_TIMEOUT 限制的是这一次调用的总耗时。
    - DDGS_TOP_K 是「这一次搜索」返回的结果里，经 BM25/去重/过滤后最多保留几条，不会因 TOP_K=5 而发 5 次请求或等 5 倍超时。

    相关性保障：
    - 仅当结果通过 BM25 且分数 >= DDGS_MIN_SCORE 时才返回；若全部低于阈值（如 API 返回无关内容或多引擎超时后的脏数据），
      返回明确提示「未找到与您问题相关的结果…」，不再回退到低相关条目，避免答非所问。
    """

    name: str = "WebSearch"

    def _format_tool_return(self, text: str, raw_results: list | None = None) -> str | tuple:
        """在 ``response_format='content_and_artifact'`` 时统一返回 (content, artifact)，避免 invoke/run 校验失败。"""
        if getattr(self, "response_format", None) == "content_and_artifact":
            return text, raw_results or []
        return text

    def _run(self, query: str, run_manager=None) -> str | tuple:
        """执行网页搜索工具（单次请求），并返回 Markdown 格式结果。"""
        # 单次 api_wrapper.results = 单次网络请求，返回最多 max_results 条；后续仅内存内 BM25/去重/截断
        try:
            results = self.api_wrapper.results(query, self.max_results)
            if not results:
                return self._format_tool_return("未找到相关结果。")

            max_out = settings.DDGS_MAX_RESULTS
            top_k = settings.DDGS_TOP_K
            min_score = settings.DDGS_MIN_SCORE
            bm25_k1 = settings.DDGS_BM25_K1
            bm25_b = settings.DDGS_BM25_B

            q_tokens = self._tokenize_list(query)
            scored = self._score_with_bm25(results, q_tokens, k1=bm25_k1, b=bm25_b)
            dedup = self._deduplicate_by_link(scored)
            picked = self._filter_and_rank(dedup, top_k=top_k, min_score=min_score)
            # 若全部低于 min_score（如 API 返回无关内容、多引擎超时后的脏数据），不再回退到低相关结果，避免返回与问题无关的条目
            if not picked:
                return self._format_tool_return(
                    "未找到与您问题相关的结果，建议更换关键词或稍后重试。"
                )

            results = picked[:max(1, max_out)]

            formatted_results = []
            for res in results:
                title = res.get("title", "No Title")
                link = res.get("link", "")
                snippet = res.get("snippet", "")

                # 构建 Markdown 格式
                formatted_results.append(f"### [{title}]({link})\n> {snippet}")

            formatted_res = "\n\n".join(formatted_results)

            return self._format_tool_return(formatted_res, results)
        except Exception:
            # 若出错，回退到默认行为
            return super()._run(query, run_manager)

    async def _arun(self, query: str, run_manager=None) -> str | tuple:
        """异步版本，带超时以避免因引擎超时导致长时间阻塞。"""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run, query, run_manager),
                timeout=settings.DDGS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return self._format_tool_return("网络搜索超时，请稍后重试或换一种问法。")

    def _tokenize_list(self, text: str) -> list[str]:
        if not text:
            return []
        # 兼容中英文：英文按词切分；中文按连续中文片段 + 单字补充
        text = text.lower()
        en = re.findall(r"[a-z0-9]+", text)
        zh_segments = re.findall(r"[\u4e00-\u9fff]+", text)
        zh = []
        for seg in zh_segments:
            zh.extend(list(seg))
        return en + zh

    def _score_with_bm25(
        self,
        results: list[dict],
        query_tokens: list[str],
        k1: float,
        b: float,
    ) -> list[tuple[float, dict]]:
        # 使用 BM25 计算相关性分数
        if not query_tokens:
            return [(0.0, r) for r in results]

        docs = []
        df: dict[str, int] = {}
        for res in results:
            title = (res.get("title") or "").strip()
            snippet = (res.get("snippet") or "").strip()
            tokens = self._tokenize_list(f"{title} {snippet}")
            docs.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        n_docs = len(docs)
        avgdl = (sum(len(d) for d in docs) / n_docs) if n_docs else 0.0
        if n_docs == 0 or avgdl == 0:
            return [(0.0, r) for r in results]

        scored: list[tuple[float, dict]] = []
        for res, doc_tokens in zip(results, docs):
            tf: dict[str, int] = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1

            score = 0.0
            dl = len(doc_tokens)
            for q in query_tokens:
                if q not in tf:
                    continue
                df_q = df.get(q, 0)
                # BM25 IDF
                idf = math.log((n_docs - df_q + 0.5) / (df_q + 0.5) + 1.0)
                freq = tf[q]
                denom = freq + k1 * (1 - b + b * (dl / avgdl))
                score += idf * (freq * (k1 + 1) / denom)

            scored.append((score, res))

        return scored

    def _deduplicate_by_link(self, scored: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
        # 按 link 去重，保留更相关的一条
        dedup: dict[str, tuple[float, dict]] = {}
        for s, res in scored:
            link = (res.get("link") or "").strip()
            if not link:
                link = f"__no_link__::{res.get('title','')[:30]}::{id(res)}"
            prev = dedup.get(link)
            if prev is None or s > prev[0]:
                dedup[link] = (s, res)
        return list(dedup.values())

    def _filter_and_rank(
        self,
        scored: list[tuple[float, dict]],
        top_k: int,
        min_score: float,
    ) -> list[dict]:
        # 过滤低相关性，并按分数降序重排
        filtered = [(s, r) for (s, r) in scored if s >= min_score]
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in filtered[:max(0, top_k)]]


web_search = FormattedDuckDuckGoSearchResults(max_results=settings.DDGS_MAX_RESULTS)


def _resolve_tavily_api_key() -> str | None:
    if settings.TAVILY_API_KEY:
        v = settings.TAVILY_API_KEY.get_secret_value()
        if v and str(v).strip():
            return str(v).strip()
    env_v = (os.environ.get("TAVILY_API_KEY") or "").strip()
    return env_v or None


class FormattedTavilySearchResults(TavilySearchResults):
    """
    基于 LangChain ``TavilySearchResults`` 的网页搜索，输出与 DDG 工具一致的 Markdown 片段格式。

    条数、搜索深度与超时从 ``core.settings`` 读取（``.env`` 中 ``TAVILY_*``）。
    ``name`` 仍为 ``WebSearch``，便于子智能体沿用同一工具调用约定。
    """

    name: str = "WebSearch"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    include_answer: bool = False
    include_raw_content: bool = False
    include_images: bool = False

    def _format_tool_return(self, text: str, raw_results: list | None = None) -> str | tuple:
        if getattr(self, "response_format", None) == "content_and_artifact":
            return text, raw_results or []
        return text

    def _run(self, query: str, run_manager=None) -> str | tuple:
        q = (query or "").strip()
        if not q:
            return self._format_tool_return("未找到相关结果。")
        try:
            raw = self.api_wrapper.raw_results(
                q,
                self.max_results,
                self.search_depth,
                self.include_domains,
                self.exclude_domains,
                self.include_answer,
                self.include_raw_content,
                self.include_images,
            )
        except Exception as e:
            return self._format_tool_return(f"Tavily 检索失败: {e}")
        results_list = raw.get("results") or []
        if not results_list:
            return self._format_tool_return("未找到相关结果。")
        cleaned = self.api_wrapper.clean_results(results_list)
        if not cleaned:
            return self._format_tool_return("未找到相关结果。")
        formatted_results = []
        for res in cleaned:
            title = res.get("title", "No Title")
            link = res.get("url", "")
            snippet = res.get("content", "")
            formatted_results.append(f"### [{title}]({link})\n> {snippet}")
        formatted_res = "\n\n".join(formatted_results)
        return self._format_tool_return(formatted_res, cleaned)

    async def _arun(self, query: str, run_manager=None) -> str | tuple:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run, query, run_manager),
                timeout=settings.TAVILY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return self._format_tool_return("网络搜索超时，请稍后重试或换一种问法。")


class _TavilyKeyMissingWebSearch(BaseTool):
    """未配置 ``TAVILY_API_KEY`` 时的占位工具，避免 import 阶段崩溃。"""

    name: str = "WebSearch"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    args_schema: type[BaseModel] = TavilyInput

    def _run(self, query: str, run_manager=None) -> str:
        return (
            "未配置 TAVILY_API_KEY，无法使用 Tavily 联网检索。"
            "请在环境变量或 .env 中设置 TAVILY_API_KEY 后重启服务。"
        )

    async def _arun(self, query: str, run_manager=None) -> str:
        return self._run(query, run_manager)


_tavily_key = _resolve_tavily_api_key()
if _tavily_key:
    tavily_search: BaseTool = FormattedTavilySearchResults(
        tavily_api_key=_tavily_key,
        max_results=settings.TAVILY_MAX_RESULTS,
        search_depth=settings.TAVILY_SEARCH_DEPTH,
    )
else:
    tavily_search = _TavilyKeyMissingWebSearch()


class VectorSearchInput(BaseModel):
    query: str = Field(description="The search query for similarity search")
    user_id: Optional[str] = Field(description="The user ID to filter results", default=None)
    thread_id: Optional[str] = Field(description="The thread ID to filter results", default=None)
    k: int = Field(description="Number of results to return", default=4)


class VectorSearchTool(BaseTool):
    """A tool for performing similarity search in vector store."""
    
    name: str = "vector_search"
    description: str = "Useful for searching similar messages or content in the conversation history."
    args_schema: type[BaseModel] = VectorSearchInput
    vector_manager: Optional[object] = None

    def __init__(self, vector_manager=None):
        super().__init__()
        # We'll get the vector manager from the FastAPI app state when the tool is called
        # This requires passing the app state to the tool, which we'll handle in the agent
        object.__setattr__(self, 'vector_manager', vector_manager)

    def _run(self, query: str, user_id: Optional[str] = None, thread_id: Optional[str] = None, k: int = 4) -> str:
        """Synchronous version - not used in async context."""
        raise NotImplementedError("This tool only supports async execution")

    async def _arun(self, query: str, user_id: Optional[str] = None, thread_id: Optional[str] = None, k: int = 4) -> str:
        """Asynchronous version of the tool."""
        # This will be set by the agent when the tool is called
        if self.vector_manager is None:
            return "Error: Vector manager not initialized"
        
        try:
            results = await self.vector_manager.asimilarity_search(
                query=query,
                user_id=user_id,
                thread_id=thread_id,
                k=k
            )
            
            if not results:
                return "No similar content found."
                
            # Format the results
            formatted_results = []
            for i, doc in enumerate(results, 1):
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                metadata = doc.metadata
                formatted_results.append(
                    f"{i}. {content} (type: {metadata.get('message_type', 'unknown')}, "
                    f"index: {metadata.get('message_index', 'unknown')})"
                )
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error during vector search: {str(e)}"


# Global instance of the vector search tool (without vector manager initially)
vector_search_tool = VectorSearchTool()


def database_search(query: str) -> str:
    """
    A placeholder function for database search.
    In a real implementation, this would connect to your database and perform the search.
    """
    # This is a placeholder implementation
    # In a real system, you would implement actual database search logic here
    return f"Database search for: {query} - No results found (placeholder implementation)"