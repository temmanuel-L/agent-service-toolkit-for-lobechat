# 长期记忆模块详细设计文档（Memory）

## 1. 概述

### 1.1 设计目标

- **跨会话持久化**：按用户维度保存对话摘要与可检索片段，供后续请求注入上下文。
- **不阻塞首包**：读写带超时与降级，检索可配置超时后仅用摘要，确保首包延迟可控。
- **与 RAG 共用能力**：与 RAG 共用 Embedding 缓存与 VectorManager 的 Qdrant 连接，减少重复建连与重复 embedding。

### 1.2 适用范围

- 模块路径：`src/memory`
- 主要对外接口：`MemoryManager.abuild_system_message`、`MemoryManager.arecord_turn`
- 调用方：`src/service/handlers.py`（请求入口）、lifespan（依赖注入）

---

## 2. 设计理念

| 原则 | 说明 |
|------|------|
| **摘要 + 片段双轨** | 摘要：LLM 压缩的滚动梗概，存 Postgres；片段：原始对话向量化存 Qdrant，按 query 语义检索。 |
| **按间隔压缩** | 摘要不每轮都调 LLM，而是每 N 轮（`LONG_TERM_MEMORY_COMPRESSION_INTERVAL`）将缓冲的对话一次性压缩，缩短回写时间。 |
| **复用 query embedding** | 长期记忆检索时先算一次 query embedding，再按向量检索 + 仅对 snippet 做一次 embedding，避免重复调用。 |
| **超时降级** | 片段检索可配置 `LONG_TERM_MEMORY_MAX_WAIT_MS`，超时则仅使用摘要，保证首包不因 embedding 慢而拖死。 |
| **缓存与重试** | Embedding 经 EmbeddingCache 与 VectorManager 的带重试 embedding，与 RAG 共用同一缓存。 |

---

## 3. 架构

### 3.1 模块结构

```
src/memory/
├── __init__.py
├── long_term.py        # MemoryManager、摘要/片段编排、回写
├── vector_manager.py   # 向量写入与检索（与 RAG 共用实例）
├── embedding_cache.py  # 进程内 + 磁盘 LRU 缓存，与 RAG 共用
├── qdrant.py          # Qdrant 连接与 LangChain Store 封装
├── postgres.py         # Postgres Store 实现（摘要/缓冲）
├── utils.py            # 文本质量校验等工具
├── mongodb.py
├── sqlite.py
└── ...
```

### 3.2 架构图

```
                    ┌──────────────────────┐
                    │    MemoryManager      │
                    │  （长期记忆编排器）     │
                    └────┬────────────┬─────┘
                         │            │
              ┌──────────▼──┐   ┌─────▼──────────┐
              │ Postgres    │   │ VectorManager   │
              │ store       │   │ (片段向量检索   │
              │ (摘要/缓冲)  │   │  与写入)       │
              └─────────────┘   └────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │ EmbeddingCache        │
                         │ get_cache_aware_embed  │
                         └───────────────────────┘
```

### 3.3 依赖注入（lifespan）

- **MemoryManager**：模块级单例 `memory_manager`，由 lifespan 注入：
  - `memory_manager.set_store(shared_postgres_store)`
  - `memory_manager.vector_manager = global_vector_manager`（与 vector_search_tool / RAG 共用）
- **VectorManager**：全局单例，由 lifespan 创建并注入到 MemoryManager 与 RAG 相关逻辑。

---

## 4. 核心流程

### 4.1 读路径：构建长期记忆上下文（注入到 SystemMessage）

```
abuild_system_message(user_id, query)
    │
    ├─► abuild_context(user_id, query)
    │       │
    │       ├─► _aget_summary(user_id)                    # Postgres 读摘要
    │       │
    │       ├─► [可选] asyncio.wait_for(
    │       │       _aretrieve_snippets(query, user_id),
    │       │       timeout=LONG_TERM_MEMORY_MAX_WAIT_MS/1000
    │       │   )  # 超时则 snippets=[]
    │       │       │
    │       │       └─► _aretrieve_snippets:
    │       │               query_embed → asimilarity_search_by_vector
    │       │               → 哈希去重 → snippet_embed → 语义去重+相关性过滤
    │       │
    │       └─► 按 LONG_TERM_MEMORY_MAX_CONTEXT_TOKENS 裁剪摘要与 snippets
    │
    └─► MemoryContext.format_for_prompt() → SystemMessage("[长期记忆]\n...")
```

### 4.2 写路径：回写本轮对话（arecord_turn）

```
arecord_turn(user_id, thread_id, user_message, assistant_message)
    │
    ├─► 组装 messages[]、summary_inputs[]
    │
    ├─► [若 backend 含向量] _vector_manager.aadd_messages(...)
    │       │
    │       └─► _embed_with_cache_and_retry → _add_vectors_to_qdrant
    │
    └─► [若 backend 含摘要]
            │
            ├─► interval <= 1 → _aupdate_summary(user_id, summary_inputs)
            │
            └─► interval > 1  → _aget_pending → pending.extend(summary_inputs)
                                → 若 len(pending) >= interval：
                                    _aupdate_summary(user_id, pending)
                                    _aput_pending(user_id, [])
                                否则：_aput_pending(user_id, pending)
```

### 4.3 片段检索详细流程（_aretrieve_snippets）

```
query
  │
  ├─► query_embed = embed_model.embed_query(query)   # 可走缓存
  │
  ├─► docs = vm.asimilarity_search_by_vector(query_embed, user_id, k=retrieve_k)
  │
  ├─► 校验 + 截断 → snippets[]
  │
  ├─► 哈希去重（MD5）
  │
  ├─► snippet_embs = embed_model.aembed_documents(snippets)  # 可走缓存
  │
  ├─► 语义去重（cosine_similarity 矩阵，DEDUP_THRESHOLD）
  │
  ├─► 相关性过滤（cosine_similarity(snippets, query_emb) >= MIN_RELEVANCE_SCORE）
  │
  └─► 按分数排序取 top-k
```

---

## 5. 数据结构与配置

### 5.1 MemoryContext

```python
@dataclass
class MemoryContext:
    summary: str | None   # 滚动摘要
    snippets: list[str]   # 检索得到的片段列表

    def format_for_prompt(self) -> str  # 格式化为可注入的文本
```

### 5.2 Store 命名空间约定（Postgres）

| 用途 | namespace | key | value 示例 |
|------|-----------|-----|------------|
| 摘要 | `(user_id, "long_term_summary")` | `"summary"` | `{"summary": "...", "updated_at": "..."}` |
| 待压缩缓冲 | `(user_id, "long_term_pending")` | `"inputs"` | `{"messages": ["用户: ...", "助手: ..."]}` |

### 5.3 主要配置项（settings）

| 配置项 | 含义 | 默认 |
|--------|------|------|
| LONG_TERM_MEMORY_ENABLED | 是否启用长期记忆 | True |
| LONG_TERM_MEMORY_BACKEND | 存储组合：pg_plus_qdrant / postgres_only / qdrant_only | pg_plus_qdrant |
| LONG_TERM_MEMORY_COMPRESSION_INTERVAL | 每 N 轮执行一次摘要压缩 | 10 |
| LONG_TERM_MEMORY_MAX_WAIT_MS | 片段检索最大等待 ms，0 表示不限制 | 1000 |
| LONG_TERM_MEMORY_MAX_CONTEXT_TOKENS | 注入上下文的 token 上限 | 4000 |
| LONG_TERM_MEMORY_DEDUP_THRESHOLD | 语义去重相似度阈值 | 0.95 |
| LONG_TERM_MEMORY_MIN_RELEVANCE_SCORE | 片段最低相关性分数 | 0.3 |
| LONG_TERM_MEMORY_SUMMARY_MAX_CHARS | 摘要最大字符数 | 1500 |
| LONG_TERM_MEMORY_MODEL | 摘要压缩使用的模型，None 则用 DEFAULT_MODEL | None |

---

## 6. 实现要点

### 6.1 VectorManager 与长期记忆

- **aadd_messages**：将 HumanMessage/AIMessage 转为 Document，经 `_embed_with_cache_and_retry` 后写入 Qdrant，payload 含 `page_content`、`metadata`（user_id、thread_id 等）。
- **asimilarity_search_by_vector**：按已算好的 query 向量检索，供长期记忆复用 query embedding，避免多一次 embedding 请求。
- **Qdrant 过滤**：检索时按 `metadata.user_id`（及可选 thread_id）过滤，保证用户隔离。

### 6.2 EmbeddingCache 共用

- **EmbeddingCache**：进程内单例 `get_embedding_cache()`，内存 LRU + 可选磁盘持久化。
- **get_cache_aware_embedding()**：返回包装了缓存的 embedding，供 VectorManager 与 RAG 使用，保证长期记忆与 RAG 共用同一缓存。

### 6.3 摘要压缩（_aupdate_summary）

- 输入：当前摘要（_aget_summary）+ 新增对话（或 pending 列表）。
- 单条长度限制：`LONG_TERM_MEMORY_MAX_ITEM_CHARS` 截断后再送 LLM。
- 使用 `LONG_TERM_MEMORY_MODEL or DEFAULT_MODEL` 调用 LLM，输出经 `_truncate(..., LONG_TERM_MEMORY_SUMMARY_MAX_CHARS)` 及低质量校验后写入 `_aput_summary`。

### 6.4 质量与安全

- **is_low_quality_text**：过滤循环、乱码等，用于摘要写入前校验与片段内容校验。
- **长度上限**：单条记忆内容超过 10000 字符视为异常丢弃。
- **超时**：摘要/缓冲的 store 读写使用 `LONG_TERM_MEMORY_STORE_TIMEOUT_MS`；片段检索使用 `LONG_TERM_MEMORY_MAX_WAIT_MS` 做整体超时。

---

## 7. 与 RAG 的关系

- **共用**：EmbeddingCache 单例、VectorManager 实例（部分场景）、get_embedding_model / get_cache_aware_embedding。
- **隔离**：长期记忆使用固定 collection（如 `agent_conversations`）且按 user_id 过滤；RAG 按 kb_id 使用不同 collection。
- **一致性**：两者均通过同一套 embedding 模型与缓存，保证向量空间一致、缓存命中率最大化。

---

## 8. 日志与可观测性

- 摘要读取/写入：`摘要读取: user=... found=... elapsed=...ms`、`摘要写入成功/失败/超时`。
- 片段检索：`长期记忆耗时: query_embed=...ms qdrant=...ms snippet_embed=...ms dedup_filter=...ms total=...ms raw=... -> ...`。
- 超时：`长期记忆片段检索超时(limit=...ms)，仅使用摘要以保证响应速度`。
- 回写：`记忆回写: user=... vector=... summary=... elapsed=...ms`。

---

## 9. 变更与扩展建议

- 摘要模型：可通过 `LONG_TERM_MEMORY_MODEL` 指定更小/更快模型以进一步缩短回写时间。
- 片段检索：若需更强检索，可考虑多向量或重排序，当前为单向量 + 余弦相似度 + 阈值过滤。
- 缓冲持久化：pending 已存 Postgres，服务重启不丢；若需跨实例共享，需保证 store 为共享库。
