"""
RAG (Retrieval-Augmented Generation) 服务模块

该模块提供知识库的文档摄入、向量化存储和检索功能。

架构：
- 使用 LlamaIndex 作为文档处理和索引框架
- 使用 Qdrant 作为向量数据库后端
- 支持多种文档格式（PDF、DOCX 等）

最佳实践：
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
from llama_index.core.node_parser import SentenceSplitter
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
        # 初始化 Qdrant 客户端
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
        
        # 将现有 LangChain 嵌入模型封装为 LlamaIndex 使用
        # 以保证与系统其他部分的一致性
        lc_embeddings = get_embedding_model()
        self.embed_model = LangchainEmbedding(lc_embeddings)
        
    def _map_url_internally(self, url: str) -> tuple[str, dict]:
        """
        在需要时将 URL 从 'localhost' 映射为 Docker 可访问的主机。
        返回 (映射后的 url, 带原始 Host 的请求头)。
        """
        parsed = urlparse(url)
        original_host = parsed.netloc
        
        # 允许用户通过环境变量覆盖，以指定特定的 minio 服务名
        internal_host = os.getenv("S3_INTERNAL_HOST", "host.docker.internal")
        
        headers = {}
        new_url = url
        
        if "localhost" in original_host or "127.0.0.1" in original_host:
            new_url = url.replace(original_host.split(':')[0], internal_host)
            # 关键：必须保留原始 Host 请求头，因为 S3 预签名 URL 的签名中包含 'host'
            headers["Host"] = original_host
            logger.info(f"Mapping external URL to internal: {url} -> {new_url} (Preserving Host: {original_host})")
            
        return new_url, headers

    async def ingest_file(self, file_url: str, kb_id: str, file_name: Optional[str] = None) -> int:
        """
        从 URL 下载文件，用 LlamaIndex 解析、分块，并存入 Qdrant。
        集合名即为 kb_id。
        """
        # 如需要则将 URL 映射到 Docker 内网并获取必要请求头
        internal_url, headers = self._map_url_internally(file_url)
        
        collection_name = kb_id
        logger.info(f"Starting ingestion: file={file_name or internal_url}, kb_id={kb_id}")
        
        # 1. 下载文件
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(internal_url, headers=headers)
            response.raise_for_status()
            file_content = response.content
            
        # 2. 将内容保存到临时文件（LlamaIndex 的 reader 通常需要文件路径）
        suffix = os.path.splitext(file_name or file_url.split('?')[0])[1].lower()
        if not suffix:
            suffix = ".tmp"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
            
        try:
            logger.info(f"Loading document from temp file: {tmp_path}")
            # 3. 使用 SimpleDirectoryReader 加载数据（自动支持 PDF、Docx 等）
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            documents = reader.load_data()
            
            if not documents:
                logger.warning(f"No content extracted from {file_name or file_url}")
                return 0
            
            # 为所有文档片段注入元数据
            for doc in documents:
                if file_name:
                    doc.metadata["file_name"] = file_name
                # 确保元数据对嵌入和 LLM 都有用
                doc.excluded_embed_metadata_keys = []
                doc.excluded_llm_metadata_keys = []
            
            logger.info(f"Extracted {len(documents)} document pages/segments. Starting indexing into Qdrant collection: {collection_name}")
                
            # 4. 配置 Qdrant 向量存储与转换
            vector_store = QdrantVectorStore(
                collection_name=collection_name,
                client=self.client,
                aclient=self.aclient
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 5. 定义转换（分块），使用 SentenceSplitter 获得更自然的文本边界
            # 
            # 参数选择说明：
            # - chunk_size=2048: 较大的块可以保留更完整的上下文，
            #   对于技术文档中的公式、表格、详细描述尤为重要
            # - chunk_overlap=256: 约 12.5% 的重叠，确保句子和段落
            #   不会在块边界处被截断丢失关键信息
            transformations = [
                SentenceSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
            ]
            logger.info(f"使用分块参数: chunk_size={RAG_CHUNK_SIZE}, chunk_overlap={RAG_CHUNK_OVERLAP}")
            
            # 6. 创建索引（解析 + 嵌入 + 写入），由 VectorStoreIndex.from_documents 完成整条流水线
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=transformations,
                show_progress=False  # 生产环境设为 False 以保持日志简洁
            )
            
            logger.info(f"Successfully ingested {len(documents)} pages/nodes into collection '{collection_name}'")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to ingest file {file_name or file_url} into collection {collection_name}: {str(e)}", exc_info=True)
            raise e
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"Removed temp file: {tmp_path}")

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """
        从 Qdrant 中删除整个集合（知识库）。
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

# 单例实例
rag_service = RagService()
