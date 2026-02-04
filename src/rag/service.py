"""
RAG (Retrieval-Augmented Generation) 服务模块

该模块提供知识库的文档摄入、向量化存储和检索功能。

Architecture:
- 使用 LlamaIndex 作为文档处理和索引框架
- 使用 Qdrant 作为向量数据库后端
- 支持多种文档格式（PDF、DOCX 等）

Best Practices:
- chunk_size 和 chunk_overlap 的选择对 RAG 质量有重大影响
- 对于技术文档，建议使用较大的 chunk_size 以保留完整上下文
- similarity_top_k 应根据问题复杂度和文档特性调整
"""
import os
import httpx
import tempfile
from typing import Optional, List
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import SimpleDirectoryReader
from urllib.parse import urlparse

from core.settings import settings
from core.llm import get_embedding_model
from utils.log_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# RAG 配置参数
# ============================================================================
# 这些参数对 RAG 检索质量有重大影响，根据文档类型和使用场景可能需要调整
#
# chunk_size: 每个文本块的目标大小（字符数）
#   - 较小值（512-1024）: 适合简单问答、FAQ 类文档
#   - 较大值（2048-4096）: 适合技术文档、学术论文等需要完整上下文的场景
#   - 当前设置 2048: 平衡了上下文完整性和检索精确性
#
# chunk_overlap: 相邻块之间的重叠字符数
#   - 目的是确保跨块边界的信息不会丢失
#   - 建议为 chunk_size 的 10-20%
#   - 当前设置 256: 约 12.5%，确保句子级别的连续性
#
# similarity_top_k: 检索返回的最相关文本块数量
#   - 较小值（3-5）: 精确匹配，适合简单直接的问题
#   - 较大值（8-15）: 覆盖更广，适合概述性问题或复杂查询
#   - 当前默认 8: 为复杂问题提供足够上下文
#
# 注意：增大这些参数会增加 LLM 的输入 token 消耗
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "2048"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "256"))
RAG_DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "8"))

class RagService:
    def __init__(self):
        # Initialize Qdrant clients
        self.api_key = settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
        self.url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
        self.aclient = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key
        )
        
        # Wrap existing LangChain embedding model into LlamaIndex
        # This ensures consistency with other parts of the system
        lc_embeddings = get_embedding_model()
        self.embed_model = LangchainEmbedding(lc_embeddings)
        
    def _map_url_internally(self, url: str) -> tuple[str, dict]:
        """
        Map a URL from 'localhost' to a Docker-reachable host if necessary.
        Returns (mapped_url, headers_with_original_host).
        """
        parsed = urlparse(url)
        original_host = parsed.netloc
        
        # Allow user to override via environment variable if they have a specific minio service name
        internal_host = os.getenv("S3_INTERNAL_HOST", "host.docker.internal")
        
        headers = {}
        new_url = url
        
        if "localhost" in original_host or "127.0.0.1" in original_host:
            new_url = url.replace(original_host.split(':')[0], internal_host)
            # CRITICAL: We MUST preserve the original Host header
            # because S3 presigned URLs include the 'host' in their signature.
            headers["Host"] = original_host
            logger.info(f"Mapping external URL to internal: {url} -> {new_url} (Preserving Host: {original_host})")
            
        return new_url, headers

    async def ingest_file(self, file_url: str, kb_id: str, file_name: Optional[str] = None) -> int:
        """
        Download file from URL, parse it using LlamaIndex, chunk it, and store in Qdrant.
        The collection name will be the kb_id.
        """
        # Map URL to internal Docker network if needed and get necessary headers
        internal_url, headers = self._map_url_internally(file_url)
        
        collection_name = kb_id
        logger.info(f"Starting ingestion: file={file_name or internal_url}, kb_id={kb_id}")
        
        # 1. Download the file
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(internal_url, headers=headers)
            response.raise_for_status()
            file_content = response.content
            
        # 2. Save content to a temporary file
        # LlamaIndex readers often need a file path
        suffix = os.path.splitext(file_name or file_url.split('?')[0])[1].lower()
        if not suffix:
            suffix = ".tmp"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
            
        try:
            logger.info(f"Loading document from temp file: {tmp_path}")
            # 3. Load data using SimpleDirectoryReader (handles PDF, Docx, etc. automatically)
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            documents = reader.load_data()
            
            if not documents:
                logger.warning(f"No content extracted from {file_name or file_url}")
                return 0
            
            # Inject metadata to all document segments
            for doc in documents:
                if file_name:
                    doc.metadata["file_name"] = file_name
                # Ensure metadata is useful for both embedding and LLM
                doc.excluded_embed_metadata_keys = []
                doc.excluded_llm_metadata_keys = []
            
            logger.info(f"Extracted {len(documents)} document pages/segments. Starting indexing into Qdrant collection: {collection_name}")
                
            # 4. Setup Qdrant Vector Store and Transformations
            vector_store = QdrantVectorStore(
                collection_name=collection_name,
                client=self.client,
                aclient=self.aclient
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 5. Define Transformations (Chunking)
            # Use SentenceSplitter for more natural text boundaries
            # 
            # 参数选择说明：
            # - chunk_size=2048: 较大的块可以保留更完整的上下文，
            #   对于技术文档中的公式、表格、详细描述尤为重要
            # - chunk_overlap=256: 约 12.5% 的重叠，确保句子和段落
            #   不会在块边界处被截断丢失关键信息
            from llama_index.core.node_parser import SentenceSplitter
            transformations = [
                SentenceSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
            ]
            logger.info(f"使用分块参数: chunk_size={RAG_CHUNK_SIZE}, chunk_overlap={RAG_CHUNK_OVERLAP}")
            
            # 6. Create index (Parsing + Embedding + Upserting)
            # VectorStoreIndex.from_documents handles the pipeline
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=transformations,
                show_progress=False # Set to False for cleaner logs in production
            )
            
            logger.info(f"Successfully ingested {len(documents)} pages/nodes into collection '{collection_name}'")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to ingest file {file_name or file_url} into collection {collection_name}: {str(e)}", exc_info=True)
            raise e
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"Removed temp file: {tmp_path}")

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """
        Delete an entire collection (knowledge base) from Qdrant.
        """
        try:
            if self.client.collection_exists(kb_id):
                logger.info(f"Deleting collection: {kb_id}")
                self.client.delete_collection(kb_id)
                return True
            else:
                logger.warning(f"Collection {kb_id} does not exist, nothing to delete.")
                return False
        except Exception as e:
            logger.error(f"Error deleting collection {kb_id}: {str(e)}")
            return False

    async def query_knowledge(self, query_str: str, kb_ids: List[str], similarity_top_k: int = None) -> str:
        """
        跨多个知识库（collection）进行查询
        
        返回带有段落标记的相关上下文拼接字符串。
        
        Args:
            query_str: 查询字符串
            kb_ids: 要查询的知识库 ID 列表
            similarity_top_k: 返回的最相关文本块数量，默认使用 RAG_DEFAULT_TOP_K
                             - 对于简单直接的问题（如"作者是谁"），可以使用较小值（3-5）
                             - 对于概述性问题（如"文章讲了什么"），建议使用较大值（8-15）
        
        Returns:
            str: 格式化的检索结果，每个段落带有 [Knowledge Segment N] 标记
        """
        # 使用默认值或传入的值
        if similarity_top_k is None:
            similarity_top_k = RAG_DEFAULT_TOP_K
        if not kb_ids:
            return ""
        
        logger.info(f"知识库检索: query='{query_str[:50]}...', kb_ids={kb_ids}, top_k={similarity_top_k}")
        all_segments = []
        segment_count = 1
        
        for kb_id in kb_ids:
            try:
                # Check if collection exists
                if not self.client.collection_exists(kb_id):
                    logger.debug(f"Collection {kb_id} does not exist, skipping query.")
                    continue
                
                vector_store = QdrantVectorStore(
                    collection_name=kb_id,
                    client=self.client,
                    aclient=self.aclient
                )
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=self.embed_model
                )
                
                retriever = index.as_retriever(similarity_top_k=similarity_top_k)
                nodes = await retriever.aretrieve(query_str)
                
                for node in nodes:
                    segment_header = f"[Knowledge Segment {segment_count}]"
                    all_segments.append(f"{segment_header}\n{node.text}")
                    segment_count += 1
                    
            except Exception as e:
                logger.error(f"Error querying collection {kb_id}: {str(e)}")
                continue
                
        if not all_segments:
            return ""
            
        return "\n\n---\n\n".join(all_segments)

# Singleton instance
rag_service = RagService()
