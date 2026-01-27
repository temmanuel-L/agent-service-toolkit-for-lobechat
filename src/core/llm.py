from functools import cache
from typing import Any, TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from utils.log_utils import get_logger
from core.settings import settings
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
    | {m: m.value for m in FakeModelName}
)

# Embedding 模型映射表
_EMBEDDING_MODEL_TABLE = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text:latest",
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
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")


@cache
def get_embedding_model(embedding_provider: str = "ollama") -> Any:
    """获取embedding模型，按优先级尝试连接可用的服务
    
    优先级顺序：
    1. Ollama embedding (最高优先级，但需要在公司网络内)
    2. OpenAI 代理服务 (当Ollama不可用时)
    3. 标准OpenAI API (最后备选方案)
    """
        # 按优先级尝试获取embedding模型
    # 1. 首先尝试Ollama embedding（优先级最高）
    if settings.OLLAMA_EMBEDDING_MODEL and settings.OLLAMA_BASE_URL:
        try:
            from langchain_ollama import OllamaEmbeddings
            # 尝试创建并返回Ollama embedding实例
            ollama_embeddings = OllamaEmbeddings(
                model=_EMBEDDING_MODEL_TABLE["ollama"],
                base_url=settings.OLLAMA_BASE_URL,
            )
            # 尝试进行一次简单的测试调用以确认连接可用
            ollama_embeddings.embed_query("test")
            return ollama_embeddings
        except Exception as e:
            # 如果Ollama不可用，则忽略错误并继续尝试下一个选项
            logger.error(f'ollama的embedding不可用, {e}')
            pass
    
    # 2. 尝试OpenAI代理服务 (DMX)
    if (settings.DMX_CHAT_URL or settings.COMPATIBLE_BASE_URL) and settings.OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            openai_embeddings = OpenAIEmbeddings(
                model=_EMBEDDING_MODEL_TABLE["openai"],
                base_url=settings.COMPATIBLE_BASE_URL or settings.DMX_CHAT_URL,
                api_key=settings.OPENAI_API_KEY.get_secret_value()
            )
            # 尝试进行一次简单的测试调用以确认连接可用
            openai_embeddings.embed_query("test")
            return openai_embeddings
        except Exception as e:
            # 如果OpenAI代理服务不可用，则忽略错误并继续尝试下一个选项
            logger.error(f'openai代理服务的embedding不可用, {e}')
            pass
    
    # 3. 最后尝试标准OpenAI API
    if settings.OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            openai_embeddings = OpenAIEmbeddings(
                model=_EMBEDDING_MODEL_TABLE["openai"],
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            )
            # 尝试进行一次简单的测试调用以确认连接可用
            openai_embeddings.embed_query("test")
            return openai_embeddings
        except Exception as e:
            # 所有服务都不可用
            logger.error(f'openai服务的embedding不可用, {e}')
            pass
    
    raise ValueError("所有embedding服务都不可用：Ollama、DMX代理服务和OpenAI API")
