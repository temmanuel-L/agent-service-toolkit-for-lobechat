from functools import cache
from typing import Any, TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings

from core.rerank_api import RerankAPIPostprocessor
from utils.log_utils import get_logger
from core.settings import settings, is_ollama_reachable
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    VertexAIModelName,
    ZhipuModelName,
    RerankModelName,
)

logger = get_logger(__name__)

# 仅对话 LLM 使用；Rerank 通过 get_rerank() 单独获取
_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in ZhipuModelName}
    | {m: m.value for m in FakeModelName}
)

# Rerank 模型映射表：服务商为 key，模型名为 value（来自 schema.models.RerankModelName）。改模型时只改 schema 与下表 value。
_RERANK_MODEL_TABLE = {
    "ollama": RerankModelName.OLLAMA_RERANK.value,
    "cohere": RerankModelName.COHERE_RERANK.value,
}

# Embedding 模型映射表
_EMBEDDING_MODEL_TABLE = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text:latest",
    "zhipu": "embedding-2",
}


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


ModelT: TypeAlias = (
    AzureChatOpenAI
    | ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatVertexAI
    | ChatGroq
    | ChatBedrock
    | ChatOllama
    | FakeToolModel
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    logger.info(f'api_model_name: {api_model_name}')
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, streaming=True)
    if model_name in OpenAICompatibleName:
        # Check for both explicit compatible settings and DMX proxy settings
        if not settings.COMPATIBLE_BASE_URL and not settings.DMX_CHAT_URL:
            logger.error("OpenAICompatible provider is active but missing required base_url configuration.")
            raise ValueError("OpenAICompatible provider is active but missing required base_url configuration.")
        
        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL or api_model_name,
            temperature=0.5,
            streaming=True,
            base_url=settings.COMPATIBLE_BASE_URL or settings.DMX_CHAT_URL,
            api_key=settings.COMPATIBLE_API_KEY or (settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None),
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.5,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in VertexAIModelName:
        return ChatVertexAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_4_12B:
            return ChatGroq(model=api_model_name, temperature=0.0)  # type: ignore[call-arg]
        return ChatGroq(model=api_model_name, temperature=0.5)  # type: ignore[call-arg]
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    if model_name in OllamaModelName:
        # Use provided OLLAMA_BASE_URL if available
        base_url = settings.OLLAMA_BASE_URL
        ollama_model = ChatOllama(
            model=api_model_name,
            temperature=0.5,
            base_url=base_url if base_url else None,
            # num_predict=4096,
        )
        
        # 如果配置了 DMX 代理，则为 Ollama 添加回退机制
        if settings.DMX_CHAT_URL or settings.COMPATIBLE_BASE_URL:
            # 创建一个用于回退的 DMX 模型
            fallback_model = ChatOpenAI(
                model=settings.COMPATIBLE_MODEL or OpenAICompatibleName.GPT_4O_MINI.value,
                temperature=0.5,
                streaming=True,
                base_url=settings.COMPATIBLE_BASE_URL or settings.DMX_CHAT_URL,
                api_key=settings.COMPATIBLE_API_KEY or (settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None),
            )
            # 使用 LangChain 的 with_fallbacks 实现运行时自动切换
            return ollama_model.with_fallbacks([fallback_model]) # type: ignore
            
        return ollama_model
    if model_name in OpenRouterModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )
    if model_name in ZhipuModelName:
        if not settings.ZHIPU_API_KEY:
            raise ValueError("Zhipu provider is active but ZHIPU_API_KEY is not set.")
        # 必须用同步对话补全: .../v4/chat/completions（支持流式）。异步接口 .../v4/async/chat/completions 只返回 task_id，需轮询取结果，与 ChatOpenAI 不兼容，会报 No generations found in stream
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=settings.ZHIPU_API_KEY.get_secret_value(),
        )
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")


@cache
def get_embedding_model(embedding_provider: str = "ollama") -> Any:
    """获取 embedding 模型，按优先级返回已缓存的实例。

    优先级：Ollama > 智谱 (Zhipu) > OpenAI 代理 (DMX) > 标准 OpenAI API。
    使用 @cache 进程级缓存，同一进程内多次调用返回同一实例，不重复建连。

    注意：不再在首次调用时执行 embed_query("test") 做连接测试，避免冷启动阻塞
    （约 100–300ms）。连接在首次实际 embedding 时验证；若需启动时预热，请在
    lifespan 中显式调用一次 embed_query 并 await。
    """
    # 1. Ollama embedding（最高优先级）：先做可达性检查，与 settings 中 Provider 判断一致
    if settings.OLLAMA_EMBEDDING_MODEL and settings.OLLAMA_BASE_URL:
        if is_ollama_reachable(settings.OLLAMA_BASE_URL):
            try:
                logger.info(f'embedding_model_name: {_EMBEDDING_MODEL_TABLE["ollama"]}')
                return OllamaEmbeddings(
                    model=_EMBEDDING_MODEL_TABLE["ollama"],
                    base_url=settings.OLLAMA_BASE_URL,
                )
            except Exception as e:
                logger.error("ollama 的 embedding 不可用: %s", e)
        else:
            logger.debug("Ollama 不可达 (%s)，跳过 embedding 优先使用", settings.OLLAMA_BASE_URL)

    # 2. 智谱 Embedding（Ollama 不可达时可选用，国内提速）
    # 官方文档: POST https://open.bigmodel.cn/api/paas/v4/embeddings
    if settings.ZHIPU_API_KEY:
        try:
            logger.info(f'embedding_model_name: {_EMBEDDING_MODEL_TABLE["zhipu"]}')
            return OpenAIEmbeddings(
                model=_EMBEDDING_MODEL_TABLE["zhipu"],
                base_url="https://open.bigmodel.cn/api/paas/v4",
                api_key=settings.ZHIPU_API_KEY.get_secret_value(),
            )
        except Exception as e:
            logger.debug("智谱 embedding 不可用，跳过: %s", e)

    # 3. OpenAI 代理服务 (DMX)
    if (settings.DMX_CHAT_URL or settings.COMPATIBLE_BASE_URL) and settings.OPENAI_API_KEY:
        try:
            logger.info(f'embedding_model_name: {_EMBEDDING_MODEL_TABLE["openai"]}')
            return OpenAIEmbeddings(
                model=_EMBEDDING_MODEL_TABLE["openai"],
                base_url=settings.COMPATIBLE_BASE_URL or settings.DMX_CHAT_URL,
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            )
        except Exception as e:
            logger.error(f"openai 代理服务的 embedding 不可用: {e}")
            pass

    # 4. 标准 OpenAI API
    if settings.OPENAI_API_KEY:
        try:
            return OpenAIEmbeddings(
                model=_EMBEDDING_MODEL_TABLE["openai"],
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            )
        except Exception as e:
            logger.error(f"openai 服务的 embedding 不可用: {e}")
            pass

    raise ValueError("所有 embedding 服务都不可用：Ollama、智谱、DMX 代理服务和 OpenAI API")


@cache
def get_rerank(model_name: RerankModelName | str | None = None) -> Any:
    """
    获取 Rerank 后处理器（通过外部 API），供 RAG 精排使用。
    效仿 get_model/get_embedding_model：base_url + api_key 兼容内网 TEI 或智谱等外网服务，不占宿主机算力。

    参数:
        model_name: Rerank 模型枚举或字符串。若为空则从 settings.RAG_RERANK_MODEL 读取。
    返回:
        若未启用或未配置 BASE_URL 返回 None，否则返回可 postprocess_nodes 的实例。
    """
    if not settings.RAG_RERANK_ENABLED:
        return None

    base_url = (settings.RAG_RERANK_BASE_URL or "").strip().strip("'").strip('"')
    if not base_url:
        logger.warning(
            "Rerank 已启用但未配置 RAG_RERANK_BASE_URL，跳过重排。"
            "请在 .env 中设置 RAG_RERANK_BASE_URL（例如指向 TEI 的根地址）。"
        )
        return None

    m_name = (model_name.value if isinstance(model_name, RerankModelName) else model_name) or settings.RAG_RERANK_MODEL
    m_name = (m_name or "").strip()
    if m_name == RerankModelName.OLLAMA_RERANK.value:
        m_name = _RERANK_MODEL_TABLE.get("ollama", m_name)
    elif m_name == RerankModelName.COHERE_RERANK.value:
        m_name = _RERANK_MODEL_TABLE.get("cohere", m_name)
    if not m_name:
        m_name = "default"

    api_key = None
    if settings.RAG_RERANK_API_KEY:
        # 优先使用专门为 Rerank 配置的 API Key
        api_key = settings.RAG_RERANK_API_KEY.get_secret_value()
    # 智谱 Rerank：若未单独配置 RAG_RERANK_API_KEY，则复用 ZHIPU_API_KEY
    elif "open.bigmodel.cn" in base_url and settings.ZHIPU_API_KEY:
        api_key = settings.ZHIPU_API_KEY.get_secret_value()

    # 常见误配置提示：Ollama 的 OpenAI-compatible base_url 通常是 /v1，但它不提供 /rerank。
    if "11434" in base_url and base_url.rstrip("/").endswith("/v1"):
        logger.warning(
            "RAG_RERANK_BASE_URL=%s 看起来像 Ollama 的 OpenAI-compatible /v1 地址；"
            "当前 Rerank 会调用 {base_url}/rerank，请确认你的服务确实提供 /rerank（建议使用 TEI 等重排服务）。",
            base_url,
        )

    logger.info(
        "Rerank 使用外部 API: base_url=%s, model=%s, top_n=%s",
        base_url,
        m_name,
        settings.RAG_RERANK_TOP_K,
    )
    # 统一的 Rerank 时间限制（秒）：用于控制一次外部 rerank 调用的最长等待时间
    # - >0：作为 httpx timeout 传入（更“硬”的限制，避免拖慢请求）
    # - <=0：不限制，使用 RerankAPIPostprocessor 默认 timeout
    rr_timeout = float(getattr(settings, "RAG_RERANK_TIME_LIMIT", 0.0) or 0.0)
    return RerankAPIPostprocessor(
        base_url=base_url,
        api_key=api_key,
        model=m_name,
        top_n=settings.RAG_RERANK_TOP_K,
        timeout=rr_timeout if rr_timeout > 0 else 30.0,
    )


def get_postprocessor(model_name: RerankModelName | str | None = None) -> Any:
    """兼容别名：与 get_rerank 相同。"""
    return get_rerank(model_name)
