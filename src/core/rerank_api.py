"""
Rerank 通过外部 API 调用（TEI / 智谱 / 其它服务），不占用 agent_service 宿主机算力。

设计目标
--------
1. **对上层统一**：
   暴露给 RAG 层的是一个 LlamaIndex 的 `BaseNodePostprocessor`，
   只关心「给我候选节点 + query，我返回按相关性排序后的前 N 个节点」。

2. **对下层可扩展**：
   具体如何调用外部服务（TEI / 智谱 / 未来其它厂商）都封装在本文件，
   通过 `base_url` 做路由，避免在 RAG 流程或其它模块里散落 HTTP 细节。

3. **零侵入配置**：
   用户只需在 .env 里通过以下变量配置即可切换实现：
   - `RAG_RERANK_ENABLED`：是否启用 Rerank。
   - `RAG_RERANK_BASE_URL`：Rerank 服务的 **Base URL**，不包含具体 path。
   - `RAG_RERANK_MODEL`：可选，若不配置则由 settings 自动推断合适的默认值。
   - `RAG_RERANK_API_KEY`：可选，若不配置且为智谱，则会复用 `ZHIPU_API_KEY`。

当前内置兼容两类接口
--------------------
1) TEI 风格 / 自建服务（默认分支）
   - 环境变量示例：
       `RAG_RERANK_BASE_URL="http://tei-host:port"`  # 注意：不包含 `/rerank`
   - 请求:
       `POST {base_url}/rerank`
       body = `{"query": str, "texts": [str, ...], "model": str?}`
   - 响应（宽松约定）：
       - 纯数组: `[{"index": int, "score": float}, ...]`
       - 或带包装: `{"results": [...]} / {"data": [...]}`

2) 智谱 AI 文本重排序 API（Zhipu Rerank）
   - 官方文档：
       https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%9C%AC%E9%87%8D%E6%8E%92%E5%BA%8F
   - 环境变量示例：
       `RAG_RERANK_BASE_URL="https://open.bigmodel.cn/api"`  # 注意：不包含 `/paas/v4/rerank`
       `RAG_RERANK_MODEL="rerank"`                           # 若不配置，settings 会自动填充
   - 请求:
       `POST {base_url}/paas/v4/rerank`
       body = `{"model": "rerank", "query": str, "documents": [str, ...], "top_n": int?}`
   - 响应（根据官方 OpenAPI）：
       ```json
       {
         "results": [
           {"index": 1, "relevance_score": 0.99, "document": "..."},
           ...
         ],
         "usage": {...},
         ...
       }
       ```

为什么 RAG_RERANK_BASE_URL 不直接写完整 path？
----------------------------------------------
- 约定：
    - TEI / 自建：代码自动在 `base_url` 后拼接 `/rerank`。
    - 智谱：代码自动在 `base_url` 后拼接 `/paas/v4/rerank`。
- 好处：
    1. 切换供应商时，只改 host/base 即可，path 逻辑由代码统一维护。
    2. 避免在 .env 里复制完整路径字符串，减少运维和迁移时的易错点。

位置说明
--------
- 本模块位于 core 下，避免 import rag.* 触发 `rag/__init__.py`，从而引发循环导入。
"""

from typing import Any, List, Optional

import httpx
from llama_index.core.schema import NodeWithScore, QueryBundle

from utils.log_utils import get_logger

logger = get_logger(__name__)


class RerankAPIPostprocessor:
    """
    通过 HTTP 调用外部 Rerank 服务（TEI / 智谱等）。

    对调用方（RAG 层）的抽象：
    --------------------------
    - 输入：一批 `NodeWithScore`（召回阶段的结果）和 `QueryBundle`（用户查询）。
    - 输出：按相关性重新排序后的前 `top_n` 个 `NodeWithScore`。

    对实现方（本文件维护者）的扩展约定：
    ----------------------------------
    - 若未来要接入新的 Rerank 厂商，例如 FooRerank：
      1. 在 `__init__` 中根据 `base_url` 识别 provider 类型，记录到布尔标记或枚举。
      2. 新增一个 `_call_foo_rerank()` 私有方法，统一封装该厂商的 HTTP 协议。
      3. 在 `_postprocess_nodes()` 中，根据标记选择对应的 `_call_xxx_rerank()`。
      4. 若该厂商的响应结构与现有不兼容，可在其私有方法中统一转换为列表结构，
         交给后续通用解析逻辑处理。
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "default",
        top_n: int = 5,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self.timeout = timeout

        # 根据 base_url 粗粒度判断当前使用的 Rerank 服务提供商
        # - 目前仅区分「智谱 open.bigmodel.cn」与「其它（TEI / 自建）」两类
        # - 若未来支持更多厂商，可在此处扩展为枚举 / 多个布尔标记
        self._is_zhipu = "open.bigmodel.cn" in self.base_url

    # ------------------------------------------------------------------
    # 私有辅助方法：负责真正的 HTTP 调用，每个厂商一个方法，便于扩展与调试
    # ------------------------------------------------------------------
    def _call_zhipu_rerank(self, query_str: str, texts: list[str]) -> Any:
        """
        调用智谱开放平台的文本重排序接口。

        文档参考：
        https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%9C%AC%E9%87%8D%E6%8E%92%E5%BA%8F
        """
        url = f"{self.base_url}/paas/v4/rerank"

        # 智谱要求字段：model / query / documents / top_n
        payload: dict[str, Any] = {
            "model": self.model or "rerank",
            "query": query_str,
            "documents": texts,
        }
        # top_n = 0 或不传表示返回全部，这里使用 top_n 作为上限
        if self.top_n > 0:
            payload["top_n"] = self.top_n

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            # 智谱使用 Bearer Token 认证
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data

    def _call_tei_style_rerank(self, query_str: str, texts: list[str]) -> Any:
        """
        调用 TEI 风格 / 自建的 /rerank 接口。

        假定接口：
        - URL:  POST {base_url}/rerank
        - Body: {"query": str, "texts": [str, ...], "model": str?}

        响应格式较为宽松，统一由后续解析逻辑负责兼容：
        - 纯数组: [{"index": int, "score": float}, ...]
        - 带包装: {"results": [...]} / {"data": [...]}
        """
        url = f"{self.base_url}/rerank"

        payload: dict[str, Any] = {"query": query_str, "texts": texts}
        # 对于 TEI / 自建服务：若 model != "default" 则附加到 payload 中
        if self.model and self.model != "default":
            payload["model"] = self.model

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        # 1. 空输入保护：没有候选节点，直接返回空列表
        if not nodes:
            return []

        # 2. query 为空时：无法做语义重排，直接按原顺序截断
        if query_bundle is None or not (query_bundle.query_str or "").strip():
            return nodes[: self.top_n]

        # 3. 从节点中提取纯文本，构造待重排的文本数组
        query_str = query_bundle.query_str or ""
        texts: list[str] = []
        for n in nodes:
            node = n.node
            content = getattr(node, "text", None)
            if content is None and hasattr(node, "get_content"):
                content = node.get_content(metadata_mode=0)
            texts.append(content if isinstance(content, str) else str(content or ""))

        # 4. 调用具体 Rerank 服务（根据 provider 类型分流）
        try:
            if self._is_zhipu:
                data = self._call_zhipu_rerank(query_str, texts)
            else:
                data = self._call_tei_style_rerank(query_str, texts)
        except Exception as e:
            # 任意异常（网络错误 / 鉴权失败 / 响应解析失败等）均降级为「原序截断」
            logger.warning("Rerank API 调用失败，降级为原序: base_url=%s error=%s", self.base_url, e)
            return nodes[: self.top_n]

        # 5. 统一解析响应结果，兼容不同服务的字段命名
        #    - 若 data 本身就是列表，则直接视为 results
        #    - 否则优先尝试 "results"，再退回 "data"
        results = data if isinstance(data, list) else data.get("results", data.get("data", []))
        if not results:
            logger.warning("Rerank API 返回空结果，降级为原序: base_url=%s", self.base_url)
            return nodes[: self.top_n]

        index_to_score: dict[int, float] = {}
        for item in results:
            if isinstance(item, dict):
                idx = item.get("index", item.get("idx", len(index_to_score)))
                # 各服务字段命名差异：score / relevance_score / 其它
                sc = item.get("score", item.get("relevance_score", 0.0))
                index_to_score[int(idx)] = float(sc)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                index_to_score[int(item[0])] = float(item[1])

        if not index_to_score:
            logger.warning(
                "Rerank API 响应无法解析，降级为原序: base_url=%s raw=%s",
                self.base_url,
                repr(results)[:200],
            )
            return nodes[: self.top_n]

        out: List[NodeWithScore] = []
        sorted_pairs = sorted(index_to_score.items(), key=lambda x: -x[1])[: self.top_n]
        for idx, score in sorted_pairs:
            if 0 <= idx < len(nodes):
                out.append(NodeWithScore(node=nodes[idx].node, score=score))

        if out:
            top_score = out[0].score
            logger.info(
                "Rerank API 成功: base_url=%s returned=%d top_score=%.4f",
                self.base_url,
                len(out),
                float(top_score),
            )
            return out

        logger.warning("Rerank API 结果越界/为空，降级为原序: base_url=%s", self.base_url)
        return nodes[: self.top_n]

    # 对外公开的方法名保持与 LlamaIndex 的 BaseNodePostprocessor 一致，方便复用调用代码
    # （RAG 侧调用的是 postprocess_nodes，这里做一个薄包装转调到内部实现 _postprocess_nodes）。
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        return self._postprocess_nodes(nodes, query_bundle=query_bundle)

