"""Vector storage manager for agent conversations and RAG."""
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from memory.qdrant import get_qdrant_store

logger = logging.getLogger(__name__)


class VectorManager:
    """Manages vector storage for agent conversations and RAG."""
    
    def __init__(self, collection_name: str = "agent_conversations"):
        self.collection_name = collection_name
        self.qdrant_store = None
        
    async def ainitialize(self):
        """Initialize the vector manager."""
        self.qdrant_store = await get_qdrant_store(
            collection_name=self.collection_name
        )
        
    async def aadd_messages(
        self, 
        messages: List[BaseMessage], 
        user_id: Optional[str] = None, 
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """Add messages to vector store with metadata."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        # 将 BaseMessage 转换为 Document
        documents = []
        for i, message in enumerate(messages):
            # 创建包含消息内容和元数据的文档
            metadata = {
                "user_id": user_id,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "message_type": message.type,
                "message_index": i,
            }
            
            # 添加时间戳
            metadata["timestamp"] = message.additional_kwargs.get("timestamp") if hasattr(message, 'additional_kwargs') else None
            
            document = Document(
                page_content=str(message.content),
                metadata=metadata
            )
            documents.append(document)
            
        return await self.qdrant_store.aadd_documents(documents)
        
    async def aadd_documents(
        self,
        documents: List[Document],
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """Add documents to vector store with metadata."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        # 为每个文档添加元数据
        for doc in documents:
            if user_id:
                doc.metadata["user_id"] = user_id
            if thread_id:
                doc.metadata["thread_id"] = thread_id
            if agent_id:
                doc.metadata["agent_id"] = agent_id
                
        return await self.qdrant_store.aadd_documents(documents)
        
    async def asimilarity_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        k: int = 4,
    ) -> List[Document]:
        """Perform similarity search with optional filters."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        # 构建过滤器
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if thread_id:
            filter_dict["thread_id"] = thread_id
        if agent_id:
            filter_dict["agent_id"] = agent_id
            
        return await self.qdrant_store.asimilarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict
        )
        
    async def acleanup_user_data(self, user_id: str):
        """Clean up all vector data for a specific user."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        await self.qdrant_store.adelete_points_by_metadata({"user_id": user_id})
        logger.info(f"Cleaned up vector data for user: {user_id}")
        
    async def acleanup_thread_data(self, thread_id: str):
        """Clean up all vector data for a specific thread."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        await self.qdrant_store.adelete_points_by_metadata({"thread_id": thread_id})
        logger.info(f"Cleaned up vector data for thread: {thread_id}")