# RAG 模块详细设计文档

## 1. 概述

### 1.1 设计目标

- **知识库检索增强**：支持多格式文档摄入、向量化存储与按 query 检索，为对话提供“先检索再回答”的能力。
- **混合检索**：向量检索（语义）+ BM25（关键词）双路检索，通过 Reciprocal Rank Fusion (RRF) 融合，兼顾语义与精确匹配（标题、人名、术语等）。
- **与长期记忆共用能力**：与 Memory 共用 Embedding 模型与 EmbeddingCache，减少重复 API 调用与上下文溢出风险。

### 1.2 适用范围

- 模块路径：`src/rag`
- 主要对外接口：`RagService.ingest_file`、`RagService.query_knowledge`、`SearchKnowledgeTool`
- 调用方：知识库类 Agent（如 rag-assistant、knowledge-base-agent）、LobeChat 等前端通过 API 触发摄入与对话中的工具调用。

---

## 2. 设计理念

| 原则 | 说明 |
|------|------|
| **文档元数据增强** | 摄入时提取文档标题并注入每个 chunk 的 metadata（如 doc_title），使标题参与 embedding，解决“标题页文本短小、向量检索排名低”的问题。 |
| **基于 Token 的分块** | 使用 tiktoken (cl100k_base) 控制 chunk_size/chunk_overlap，确保每块严格适配 Embedding 模型上下文窗口，避免截断歧义。 |
| **混合检索 + RRF** | 向量检索捕获语义；BM25 精确命中关键词；RRF 无需分数归一化即可合并两路结果，并可配置 BM25 权重（RAG_BM25_WEIGHT）。 |
| **BM25 按需构建与失效** | 首次查询某 kb_id 时从 Qdrant scroll 加载节点并构建 BM25 索引并缓存；新文档摄入或删除知识库时使对应缓存失效，保证结果一致。 |
| **结果长度与安全** | 检索结果经 truncate_rag_result 截断（字符数 + 段落数），并做 is_low_quality_text 安检，防止脏数据进入 LLM 上下文。 |
| **最低相关性阈值（可选）** | 知识库与问题域不符时（如库内仅有薪酬报告、用户问「值班安排」），可设 RAG_MIN_RELEVANCE_SCORE（RRF 分约 0.008～0.01），最高分低于阈值则该库不返回片段；无任何片段时返回友好提示。 |

---

## 3. 架构

### 3.1 模块结构

```
src/rag/
├── __init__.py
├── service.py         # RagService：摄入、混合检索、BM25 缓存、RRF
├── nodes.py           # create_rag_model_node、create_rag_system_prompt、语言检测
├── tools.py           # SearchKnowledgeTool（对外工具）
├── chunking/
│   ├── core.py                  # 分块策略（简单分块 / 预留父子分块）
│   └── embedding_batcher.py     # TokenAwareEmbedding、CacheAwareEmbedding（与 Memory 共用缓存）
├── utils.py           # truncate_rag_result、format_rag_fallback_response 等
└── ...
```

### 3.2 架构图

```
                    ┌─────────────────────────────────┐
                    │         RagService              │
                    │  ingest_file / query_knowledge   │
                    └────┬──────────────────┬─────────┘
                         │                  │
        ┌────────────────▼──────┐   ┌───────▼──────────────────┐
        │ 文档摄入流程            │   │ 混合检索流程             │
        │ URL → 下载 → 解析 →     │   │ query → Vector + BM25   │
        │ 标题提取 → 分块 →       │   │ → RRF → 格式化/安检     │
        │ Embedding → Qdrant     │   └───────┬──────────────────┘
        └───────────────────────┘           │
                         │                  │
        ┌────────────────▼──────────────────▼─────────────────┐
        │  Embedding 链                                         │
        │  get_embedding_model() → TokenAwareEmbedding          │
        │  → CacheAwareEmbedding(EmbeddingCache) → LangchainEmbedding │
        └──────────────────────────────────────────────────────┘
                         │
        ┌────────────────▼──────────────────────────────────────┐
        │  Qdrant（按 kb_id = collection_name 隔离）              │
        │  BM25：内存索引，由 Qdrant scroll 加载节点按需构建/失效  │
        └──────────────────────────────────────────────────────┘
```

### 3.3 与 Memory 的共用关系

- **EmbeddingCache**：RAG 通过 `rag.chunking.embedding_batcher.CacheAwareEmbedding` 使用 `memory.embedding_cache.get_embedding_cache()`，与长期记忆共用同一缓存。
- **Embedding 模型**：均通过 `core.llm.get_embedding_model()` 获取，保证向量空间一致。
- **Qdrant 连接**：RAG 使用独立 QdrantClient/AsyncQdrantClient（同一 host/port），按 collection_name=kb_id 存储；长期记忆使用独立 collection（如 agent_conversations），二者数据隔离。

---

## 4. 核心流程

### 4.1 文档摄入流程（ingest_file）

```
ingest_file(file_url, kb_id, file_name?)
    │
    ├─► _map_url_internally(file_url)  # Docker 内网映射、Host 头
    │
    ├─► httpx 下载文件 → 临时文件（按后缀）
    │
    ├─► SimpleDirectoryReader.load_data()  # 多格式解析：PDF、DOCX、PPTX、HTML 等
    │
    ├─► _extract_doc_title()  # PDF 元数据 / 首页启发式 / 文件名
    │       → doc.metadata["doc_title"]、doc.excluded_embed_metadata_keys=[]
    │
    ├─► SentenceSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP,
    │                    tokenizer=tiktoken.cl100k_base.encode)
    │
    ├─► QdrantVectorStore(collection_name=kb_id) + StorageContext
    │
    ├─► VectorStoreIndex.from_documents(docs, embed_model=self.embed_model, transformations=[splitter])
    │       → 内部调用 CacheAwareEmbedding → TokenAwareEmbedding → 写入 Qdrant
    │
    └─► invalidate_bm25_cache(kb_id)  # 下次查询时 BM25 重新从 Qdrant 加载
```

### 4.2 混合检索流程（query_knowledge）

```
query_knowledge(query_str, kb_ids, similarity_top_k?)
    │
    ├─► similarity_top_k = similarity_top_k or RAG_DEFAULT_TOP_K
    │
    └─► for kb_id in kb_ids:
            │
            ├─► 若 !collection_exists(kb_id) → continue
            │
            ├─► 向量检索（始终执行）
            │       VectorStoreIndex.from_vector_store(embed_model=self.embed_model)
            │       → vector_retriever.aretrieve(query_str) → vector_nodes
            │
            ├─► 若 RAG_HYBRID_SEARCH:
            │       _get_or_build_bm25(kb_id, top_k)
            │           → 缓存命中则返回；否则 scroll 加载节点 → BM25Retriever.from_defaults(tokenizer=_hybrid_tokenize)
            │       bm25.retrieve(query_str) → bm25_nodes
            │       _reciprocal_rank_fusion(vector_nodes, bm25_nodes, top_k, bm25_weight=RAG_BM25_WEIGHT) → final_nodes
            │   else: final_nodes = vector_nodes
            │
            ├─► 若 RAG_MIN_RELEVANCE_SCORE > 0 且 max(score) < 阈值 → 本库跳过（不加入 all_segments），避免问题域与知识库不符时返回无关片段
            │
            ├─► 格式化与安检
            │       for node: is_low_quality_text(content) → 剔除
            │       输出 "[Knowledge Segment N]\n{content}"，用 "\n\n---\n\n" 拼接
            │
            └─► 返回 all_segments 拼接结果
```

### 4.3 RRF 融合公式（加权）

- `Score = (1 - bm25_weight) * RR_vector + bm25_weight * RR_bm25`
- `RR = 1 / (rrf_k + rank + 1)`，默认 `rrf_k=60`
- 按 Score 降序取 top_k，去重（同一 node_id 只保留一条）。

### 4.4 BM25 构建与失效

- **构建**：`_get_or_build_bm25(kb_id, top_k)` 内通过 `client.scroll(collection_name=kb_id, ...)` 分页拉取 payload，解析出 `text` 或 `_node_content`，构造 `TextNode` 列表，再用 `BM25Retriever.from_defaults(nodes=..., tokenizer=_hybrid_tokenize)` 构建并缓存。
- **失效**：`invalidate_bm25_cache(kb_id)` 在 `ingest_file` 成功、`delete_knowledge_base` 时调用，下次查询该 kb_id 时重新 scroll 并构建。

---

## 5. 实现要点

### 5.1 分词与分块

- **BM25 分词**：`_hybrid_tokenize(text)` — CJK 按字、英文/数字按词（含连字符），统一小写。确保中英文混合文档下关键词与标题可被匹配。
- **分块**：`SentenceSplitter` + `tiktoken.get_encoding("cl100k_base").encode`，`RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` 为字符数配置，内部按 token 数控制，避免超长 chunk。

### 5.2 文档标题提取（_extract_doc_title）

1. PDF：`pypdf.PdfReader(file_path).metadata.title`。
2. 通用：取首页前若干行，长度适中、非纯数字开头的行作为标题。
3. 兜底：文件名去扩展名并规范化。  
标题写入 `doc.metadata["doc_title"]`，且不排除在 embed/llm metadata 之外，使标题参与向量与展示。

### 5.3 Embedding 链（RagService 初始化）

- `get_embedding_model()` → LangChain Embeddings。
- `TokenAwareEmbedding(lc_embeddings, max_batch_tokens=6000)`：按 token 数拆批，防止超过模型上下文。
- `CacheAwareEmbedding(token_aware)`：先查 EmbeddingCache，未命中再调底层并回写，与 Memory 共用。
- `LangchainEmbedding(self.token_aware_embed, embed_batch_size=100)`：供 LlamaIndex 使用。

### 5.4 工具层（SearchKnowledgeTool）

- 输入：`query`、可选 `kb_ids`（可从 configurable.kb_ids 回退）。
- 调用：`await rag_service.query_knowledge(query, kb_ids)`。
- 后处理：`truncate_rag_result(context)`（按字符数 + 段落数截断），避免超出模型上下文。
- 异常与空结果：返回明确错误或“未找到相关段落”的提示，不暴露原始 segment。

### 5.5 节点与系统提示（nodes.py）

- **create_rag_system_prompt**：根据 `kb_ids` 是否存在提示“已绑定知识库”或“需先绑定”；强调“先 search_knowledge 再回答、禁止重复调用”。
- **detect_user_language**：根据最近几条用户消息是否含中文决定“中文”或“English”，用于系统提示中的输出语言要求。
- **create_rag_model_node**：将 RAG 系统提示、工具、模型组合为 LangGraph 节点，供具体 Agent 图复用。

---

## 6. 配置项

### 6.1 环境变量 / 代码常量

| 配置项 | 含义 | 默认 |
|--------|------|------|
| RAG_CHUNK_SIZE | 分块目标大小（字符数，内部按 token 控制） | 512 |
| RAG_CHUNK_OVERLAP | 块间重叠字符数 | 64 |
| RAG_DEFAULT_TOP_K | 检索返回的最相关块数 | 8 |
| RAG_HYBRID_SEARCH | 是否启用混合检索（关闭则仅向量） | True（settings） |
| RAG_BM25_WEIGHT | BM25 在 RRF 中的权重 (0.0–1.0) | 0.4（settings） |
| RAG_MIN_RELEVANCE_SCORE | 最高 RRF 分低于此值则该库不返回片段（0=不启用）；混合检索时 RRF 分约 0.01 量级 | 0.0（settings） |

### 6.2 utils 中的常量

> 说明：长度与段落数上限不再通过单独的常量配置，而是由 `SearchKnowledgeTool`
> 基于 `RAG_CHUNK_SIZE * RAG_DEFAULT_TOP_K` 动态计算并传入 `truncate_rag_result`。

| 常量 | 含义 | 默认 |
|------|------|------|
| RAG_SEGMENT_SEPARATOR | 段落分隔符 | "\n\n---\n\n" |

### 6.3 其他依赖

- Qdrant：`settings.QDRANT_HOST`、`settings.QDRANT_PORT`、`settings.QDRANT_API_KEY`。
- 内网映射：`S3_INTERNAL_HOST`（如 host.docker.internal）用于将 localhost URL 映射为 Docker 内可访问地址。

---

## 7. 知识库删除与数据隔离

- **delete_knowledge_base(kb_id)**：若 collection 存在则 `client.delete_collection(kb_id)`，并 `invalidate_bm25_cache(kb_id)`。
- **数据隔离**：每个知识库对应一个 Qdrant collection（collection_name = kb_id），检索时按 kb_ids 列表逐库查询，结果按 segment 序号与 kb 维度拼接，无跨库混合存储。

---

## 8. 日志与可观测性

- 摄入：`Starting ingestion: file=..., kb_id=...`、`文档标题: ...`、`Successfully ingested N pages/nodes into collection '...'`。
- BM25：`BM25 索引构建完成: collection=..., nodes=..., elapsed=...ms`、`BM25 缓存已失效: collection=...`。
- 检索：`知识库检索: query='...', kb_ids=..., top_k=..., mode=hybrid|vector-only`、`混合检索: kb=..., vector=...(ms), bm25=...(ms), fused=..., total=...ms`。
- 安全：`检测到 RAG 检索结果包含脏数据 (Score: ..., 已剔除)`；截断时 `RAG结果已截断: ... -> ...`。

---

## 9. 变更与扩展建议

- **重排序**：可在 RRF 之后增加 cross-encoder 或 LLM 重排，进一步提升 top-k 质量。
- **多语言**：BM25 分词已支持中英混合；若支持更多语言可考虑按语言选择分词器。
- **流式/分页**：当前一次返回全部 segment；若单库体量极大，可考虑按库或按段分页返回并控制总 token。
- **Embedding 与 Memory 一致性**：保持与 Memory 共用 EmbeddingCache 与模型，避免同一文本在 RAG 与长期记忆中向量不一致。
