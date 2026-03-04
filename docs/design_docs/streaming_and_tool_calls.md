# 流式响应与工具调用展示

## 「边搜索边流式展示」vs「先展示调用、再展示结果」

- **边搜索边流式展示**：在搜索请求尚未返回的这段时间里，前端就持续收到并展示内容（例如「搜索中…」或部分结果）。当前架构**做不到**：LangGraph 只会在 **tools 节点整体执行完** 后才发出 updates，所以在这段时间内没有任何事件可发。
- **先展示调用、再展示结果**：前端先收到「模型请求了 WebSearch(关键词)」（来自 `tool_calls`），过一段时间再收到「搜索结果的 Markdown」（来自 ToolMessage）。**这个我们支持**：只要流式通道正常，就会先发 tool_calls，再发 tool 结果。你之前看到的「能流式展示搜索结果」的例子，多半是这种——先出现「正在调用 WebSearch」，再出现结果；而不是在搜索进行中逐字/逐条流出内容。

若之前在你这边**连「先调用、再结果」都没看到**，常见原因有：  
1）tools 节点耗时很长（如 DDGS 多引擎超时），结果和最终回答几乎同时到达，看起来像「一股脑出来」；  
2）前端或中间层对流的缓冲/聚合策略，导致要等一段才渲染。  
通过 `DDGS_TIMEOUT` 和工具轮次上限缩短等待后，一般能更明显看到「先 tool_calls，再 tool 结果，再最终回答」的两步式展示。

## 为什么「连网搜索」无法做到真正的边搜索边流式？

### 结论（先给结论）

- **服务层没有拦截**：`TOOLS_WITH_HIDDEN_OUTPUT` 仅包含 `search_knowledge`（RAG 检索），**不包含** `WebSearch`。工具调用与工具结果会正常下发。
- **根本原因是 LangGraph 的流式语义**：`astream(stream_mode=["updates", "messages", ...])` 里，**updates 是按「节点完成」触发的**，不是按「每条消息」或「每个 token」触发。

### 时间线（以 research_assistant 为例）

1. **guard_input 节点完成** → 无消息需下发（或仅内部状态）。
2. **model 节点完成** → 产生一条 `AIMessage`（含 `tool_calls`）→ 此时才向客户端下发「模型请求了 WebSearch」。
3. **tools 节点开始执行** → 调用 DDGS 等，可能耗时很久（例如多引擎超时 8s+5s）→ **这段时间内没有任何 updates 事件**，因为 tools 节点尚未完成。
4. **tools 节点完成** → 产生若干 `ToolMessage` → 此时才向客户端下发「搜索结果的 Markdown」。
5. **model 节点再次完成** → 产生最终 `AIMessage`（或通过 messages 通道流式 token）→ 下发最终回答。

因此，用户会看到：先收到「要搜什么」，然后**长时间无新数据**，最后在较短时间内收到「搜索结果 + 总结」。这不是前端或服务层故意缓冲，而是 **updates 只在节点边界产生**。

### 可选改进方向（不在此次实现）

- 若希望「搜索中…」等中间状态也能流式展示，需要：
  - 在 tools 节点内每完成一个工具就通过 **custom** 流发送一条事件，并在 handlers 中解析并转发；或
  - 依赖 LangGraph 未来提供更细粒度的事件（例如 per-tool 完成）。
- 当前实现选择：**不修改服务层**，仅通过 **DDGS 超时** 与 **工具轮次上限** 控制等待时间和调用次数，避免长时间无反馈与过多检索。

## 响应变慢（约 15 秒）的原因

从日志可见：

- `Error in engine duckduckgo: TimeoutException(...)`（约 8 秒）
- `Error in engine yandex: TimeoutException(...)`（再约 5 秒）

底层 DDGS 会依次尝试多个引擎；任一引擎超时都会拉长整次调用。因此：

- 在 **agents/tools.py** 中为 `FormattedDuckDuckGoSearchResults` 的 `_arun` 增加了 **DDGS_TIMEOUT**（默认 12 秒）：超时后直接返回友好提示，避免单次搜索阻塞过久。
- 同时通过 **research_assistant** 的 **工具轮次上限**（如 2 轮）限制每轮对话内的工具调用次数，减少「多次检索 + 多次超时」的累积时间。

## 相关配置

| 位置 | 说明 |
|------|------|
| `core/settings.py` | 所有 DDGS 相关配置（`DDGS_MAX_RESULTS`、`DDGS_TOP_K`、`DDGS_MIN_SCORE`、`DDGS_TIMEOUT`、`DDGS_BM25_K1`、`DDGS_BM25_B`），从 .env 读取，带默认值。 |
| `service/handlers.py` | `TOOLS_WITH_HIDDEN_OUTPUT`：仅过滤此处列出的工具输出（如 `search_knowledge`），其他工具（如 `WebSearch`）会正常下发。 |
| `service/openai_paradigm.py` | 将内部 SSE 转为 OpenAI 兼容流；收到 `message`（含 type=tool）会按文档转发，不做额外拦截。 |
| `agents/tools.py` | `FormattedDuckDuckGoSearchResults` 从 `core.settings` 读取上述 DDGS 配置，不再使用环境变量直读。 |
