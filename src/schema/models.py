from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    OPENAI_COMPATIBLE = auto()
    AZURE_OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    VERTEXAI = auto()
    GROQ = auto()
    AWS = auto()
    OLLAMA = auto()
    OPENROUTER = auto()
    ZHIPU = auto()
    FAKE = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_1 = "gpt-5.1"


class AzureOpenAIModelName(StrEnum):
    """Azure OpenAI model names"""

    AZURE_GPT_4O = "azure-gpt-4o"
    AZURE_GPT_4O_MINI = "azure-gpt-4o-mini"


class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_CHAT = "deepseek-chat"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_45 = "claude-haiku-4-5"
    SONNET_45 = "claude-sonnet-4-5"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_30_PRO = "gemini-3-pro-preview"


class VertexAIModelName(StrEnum):
    """https://cloud.google.com/vertex-ai/generative-ai/docs/models"""

    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "models/gemini-2.0-flash-lite"
    GEMINI_25_FLASH = "models/gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_30_PRO = "gemini-3-pro-preview"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "llama-3.1-8b"
    LLAMA_33_70B = "llama-3.3-70b"

    LLAMA_GUARD_4_12B = "meta-llama/llama-guard-4-12b"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    BEDROCK_HAIKU = "bedrock-3.5-haiku"
    BEDROCK_SONNET = "bedrock-3.5-sonnet"


class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    # OLLAMA_GENERIC = "alibayram/Qwen3-30B-A3B-Instruct-2507:latest"
    OLLAMA_GENERIC = "qwen3:30b-a3b-instruct-16k"
    # OLLAMA_GENERIC = "qwen3-coder-next:latest"
    # OLLAMA_GENERIC = "glm-4.7-flash"


class OpenRouterModelName(StrEnum):
    """https://openrouter.ai/models"""

    GEMINI_25_FLASH = "google/gemini-2.5-flash"


class OpenAICompatibleName(StrEnum):
    """https://platform.openai.com/docs/guides/text-generation"""

    GPT_4O_MINI = "gpt-4o-mini"


class ZhipuModelName(StrEnum):
    """智谱 GLM 系列，https://docs.bigmodel.cn/cn/guide/start/model-overview
    对话补全: https://open.bigmodel.cn/api/paas/v4/chat/completions
    """

    GLM_4_7 = "glm-4.7"  # 高智能旗舰，通用对话/推理/智能体
    GLM_4_6 = "glm-4.6"  # 超强性能，200K 上下文
    GLM_4_7_FLASH = "glm-4.7-flash"  # 免费普惠，速度与效果平衡
    GLM_4_7_FLASHX = "glm-4.7-flashx"  # 轻量高速，中文写作/翻译/长文本
    GLM_4_5_AIR = "glm-4.5-air"  # 高性价比，推理/编码/智能体


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


class RerankModelName(StrEnum):
    """Placeholder rerank models for local (Ollama) and external APIs."""

    OLLAMA_RERANK = "bge-reranker-v2-m3:latest"  # Placeholder for local rerank
    COHERE_RERANK = "rerank-english-v3.0"        # Placeholder for external rerank


# 仅用于对话的 LLM 枚举；Rerank 使用 RerankModelName，不参与 get_model 分发
AllModelEnum: TypeAlias = (
    OpenAIModelName
    | OpenAICompatibleName
    | AzureOpenAIModelName
    | DeepseekModelName
    | AnthropicModelName
    | GoogleModelName
    | VertexAIModelName
    | GroqModelName
    | AWSModelName
    | OllamaModelName
    | OpenRouterModelName
    | ZhipuModelName
    | FakeModelName
)
