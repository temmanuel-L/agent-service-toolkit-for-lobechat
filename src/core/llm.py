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
)

logger = get_logger(__name__)

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
