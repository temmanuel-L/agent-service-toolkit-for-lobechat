from typing import Type, List
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from rag.service import rag_service
from rag.utils import truncate_rag_result
from utils.log_utils import get_logger

logger = get_logger(__name__)


class SearchKnowledgeInput(BaseModel):
    """仅包含模型应提供的参数。kb_ids 由请求绑定，经 config 注入，不暴露给模型，避免编造。"""
    query: str = Field(description="The search query to look up in the knowledge base.")


class SearchKnowledgeTool(BaseTool):
    name: str = "search_knowledge"
    description: str = "Search the official knowledge base for relevant information about policies, procedures, or technical documentation."
    args_schema: Type[BaseModel] = SearchKnowledgeInput

    def _run(self, query: str) -> str:
        """Use the tool synchronously (not recommended for this async service)."""
        raise NotImplementedError("Use _arun instead")

    async def _arun(self, query: str, config: RunnableConfig = None) -> str:
        """Query the rag_service. kb_ids 仅从 config.configurable 读取（由 handler 从请求注入），不来自模型参数。"""
        try:
            kb_ids: List[str] | None = (config or {}).get("configurable", {}).get("kb_ids") if config else None
            if not kb_ids:
                return "Error: No knowledge base IDs (kb_ids) provided for search. Please specify which knowledge base to search."
            
            logger.info(f"Tool search_knowledge: query='{query}', kb_ids={kb_ids}")
            context = await rag_service.query_knowledge(query, kb_ids)
            
            if not context:
                return "The knowledge base did not return any relevant segments for this specific query."
            
            # 动态计算截断上限：完全绑定在 RAG_CHUNK_SIZE 与 RAG_DEFAULT_TOP_K 上，
            # 方便通过这两个参数统一控制上下文长度与性能。
            from core.settings import settings
            from rag.postprocess.truncate import truncate_rag_result_token_aware

            dynamic_max_tokens = int(settings.RAG_CHUNK_SIZE * settings.RAG_DEFAULT_TOP_K)
            dynamic_max_segments = settings.RAG_DEFAULT_TOP_K

            # 基于 token 的截断，防止超出模型上下文限制
            truncated_context, truncated = truncate_rag_result_token_aware(
                context,
                max_tokens=dynamic_max_tokens,
                max_segments=dynamic_max_segments,
            )

            if truncated:
                logger.info(
                    "RAG结果已截断（token-aware）: 原长度=%d, 新长度=%d, Limit: %d tokens, %d segments",
                    len(context),
                    len(truncated_context),
                    dynamic_max_tokens,
                    dynamic_max_segments,
                )

            return truncated_context
        except Exception as e:
            logger.error(f"Error in search_knowledge tool: {str(e)}")
            return f"Error performing search: {str(e)}"
