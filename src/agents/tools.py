from typing import Optional
import asyncio
import os
import numexpr
import math
import re

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


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
    """增强版 DuckDuckGo 搜索工具，支持相关性筛选、去重与数量控制。"""

    name: str = "WebSearch"

    def _run(self, query: str, run_manager=None) -> str | tuple:
        """执行网页搜索工具，并返回 Markdown 格式结果。"""
        # 直接使用 api_wrapper 获取结构化数据，而不是解析字符串输出
        try:
            results = self.api_wrapper.results(query, self.max_results)
            if not results:
                return "未找到相关结果。"

            max_out = int(os.getenv("DDGS_MAX_RESULTS", str(self.max_results or 5)))
            top_k = int(os.getenv("DDGS_TOP_K", str(max_out)))
            min_score = float(os.getenv("DDGS_MIN_SCORE", "0.08"))
            bm25_k1 = float(os.getenv("DDGS_BM25_K1", "1.5"))
            bm25_b = float(os.getenv("DDGS_BM25_B", "0.75"))

            q_tokens = self._tokenize_list(query)
            scored = self._score_with_bm25(results, q_tokens, k1=bm25_k1, b=bm25_b)
            dedup = self._deduplicate_by_link(scored)
            picked = self._filter_and_rank(dedup, top_k=top_k, min_score=min_score)
            if not picked:
                # 若全部被过滤，回退到去重后的前 max_out 条，避免无输出
                picked = [r for _, r in dedup][:max(1, max_out)]

            results = picked[:max(1, max_out)]

            formatted_results = []
            for res in results:
                title = res.get("title", "No Title")
                link = res.get("link", "")
                snippet = res.get("snippet", "")

                # 构建 Markdown 格式
                formatted_results.append(f"### [{title}]({link})\n> {snippet}")

            formatted_res = "\n\n".join(formatted_results)

            # 兼容 content_and_artifact 格式（lobe-chat）
            if getattr(self, "response_format", None) == "content_and_artifact":
                return formatted_res, results

            return formatted_res
        except Exception:
            # 若出错，回退到默认行为
            return super()._run(query, run_manager)

    async def _arun(self, query: str, run_manager=None) -> str | tuple:
        """异步版本，必要时将同步逻辑放到线程池执行。"""
        return await asyncio.to_thread(self._run, query, run_manager)

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


web_search = FormattedDuckDuckGoSearchResults()


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