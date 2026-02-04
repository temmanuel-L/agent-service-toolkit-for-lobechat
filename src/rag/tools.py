from typing import Optional, Type, List
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from rag.service import rag_service
from rag.utils import truncate_rag_result
from utils.log_utils import get_logger

logger = get_logger(__name__)

class SearchKnowledgeInput(BaseModel):
    query: str = Field(description="The search query to look up in the knowledge base.")
    kb_ids: Optional[List[str]] = Field(default=None, description="List of knowledge base IDs to search. If not provided, uses configured defaults.")

class SearchKnowledgeTool(BaseTool):
    name: str = "search_knowledge"
    description: str = "Search the official knowledge base for relevant information about policies, procedures, or technical documentation."
    args_schema: Type[BaseModel] = SearchKnowledgeInput

    def _run(self, query: str, kb_ids: Optional[List[str]] = None) -> str:
        """Use the tool synchronously (not recommended for this async service)."""
        raise NotImplementedError("Use _arun instead")

    async def _arun(self, query: str, kb_ids: Optional[List[str]] = None, config: RunnableConfig = None) -> str:
        """Query the rag_service for relevant context."""
        try:
            # Fallback to kb_ids from config if not provided in arguments
            if not kb_ids and config:
                kb_ids = config.get("configurable", {}).get("kb_ids")
                
            if not kb_ids:
                return "Error: No knowledge base IDs (kb_ids) provided for search. Please specify which knowledge base to search."
            
            logger.info(f"Tool search_knowledge: query='{query}', kb_ids={kb_ids}")
            context = await rag_service.query_knowledge(query, kb_ids)
            
            if not context:
                return "The knowledge base did not return any relevant segments for this specific query."
            
            # 截断过长的结果，防止超出模型上下文限制
            truncated_context = truncate_rag_result(context)
            if len(truncated_context) < len(context):
                logger.info(f"RAG结果已截断: {len(context)} -> {len(truncated_context)}")
            
            return truncated_context
        except Exception as e:
            logger.error(f"Error in search_knowledge tool: {str(e)}")
            return f"Error performing search: {str(e)}"
