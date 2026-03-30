from enum import StrEnum
from json import loads
from typing import Annotated, Any
from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    Provider,
    RerankModelName,
    VertexAIModelName,
    ZhipuModelName,
)


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"
    QDRANT = "qdrant"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """Convert to Python logging level constant."""
        import logging

        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


def is_ollama_reachable(base_url: str | None) -> bool:
    """Check if the Ollama server is reachable and functional."""
    if not base_url:
        return False
    try:
        import httpx
        # 尝试调用 Ollama 的 tags 接口，确认服务不仅端口通，而且能响应请求
        # 使用较短的超时时间，避免阻塞启动
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 8080
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30
    LOG_LEVEL: LogLevel = LogLevel.WARNING

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    USE_AWS_BEDROCK: bool = False
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False
    OPENROUTER_API_KEY: str | None = None
    ZHIPU_API_KEY: SecretStr | None = None

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str] = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # Set openai compatible api, mainly used for proof of concept
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    # Ollama embedding configuration
    OLLAMA_EMBEDDING_MODEL: str | None = None
    # DMX API configuration for OpenAI-compatible service
    DMX_CHAT_URL: str | None = None

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    # MCP Configuration
    GITHUB_PAT: SecretStr | None = None
    MCP_GITHUB_SERVER_URL: str = "https://api.githubcopilot.com/mcp/"

    # Database Configuration
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # Options: DatabaseType.SQLITE or DatabaseType.POSTGRES
    SQLITE_DB_PATH: str = "checkpoints.db"

    # PostgreSQL Configuration
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_APPLICATION_NAME: str = "agent-service-toolkit"
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = 1
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = 1

    # MongoDB Configuration
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None

    # Qdrant Configuration
    QDRANT_HOST: str | None = None
    QDRANT_PORT: int | None = 6333
    QDRANT_API_KEY: SecretStr | None = None

    # LangSimth
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    # LangFuse
    LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # Long-term memory configuration
    LONG_TERM_MEMORY_ENABLED: bool = True
    # Options: "pg_plus_qdrant", "postgres_only", "qdrant_only"
    LONG_TERM_MEMORY_BACKEND: str = "pg_plus_qdrant"
    LONG_TERM_MEMORY_TOP_K: int = 5
    LONG_TERM_MEMORY_MAX_ITEM_CHARS: int = 400
    LONG_TERM_MEMORY_SUMMARY_MAX_CHARS: int = 1500
    LONG_TERM_MEMORY_MODEL: str | None = None
    LONG_TERM_MEMORY_STORE_TIMEOUT_MS: int = 500
    
    # Memory enhancement configuration (new)
    LONG_TERM_MEMORY_DEDUP_THRESHOLD: float = 0.95  # Semantic similarity threshold for deduplication
    LONG_TERM_MEMORY_MAX_CONTEXT_TOKENS: int = 4000  # Max tokens for entire memory context injection
    LONG_TERM_MEMORY_COMPRESSION_INTERVAL: int = 10  # Compress memories every N turns
    LONG_TERM_MEMORY_MIN_RELEVANCE_SCORE: float = 0.3  # Filter snippets below this relevance score
    # 长期记忆片段检索最大等待时间(ms)。>0 时超时则仅用摘要，保证首包不因代理慢而拖死；0 表示不限制
    LONG_TERM_MEMORY_MAX_WAIT_MS: int = 1000

    # Embedding cache configuration
    EMBEDDING_CACHE_ENABLED: bool = True
    EMBEDDING_CACHE_PATH: str = "./data/embedding_cache.json"

    # RAG 混合检索配置
    # 说明：默认偏向「语义向量检索」，BM25 主要作为兜底补充标题/关键词命中。
    # 如果你的文档多为合同/规章这类结构化条款，建议适度减小分块尺寸，
    # 让每个 chunk 更接近「一条/几条条款」，有利于命中精确实体（如甲方/乙方名称）。
    RAG_CHUNK_SIZE: int = 512           # 文档分块的目标大小（token 级），适中偏小
    RAG_CHUNK_OVERLAP: int = 64         # 相邻块重叠 token 数，保持在 chunk_size 的约 10–15%
    RAG_SPLITTER_TYPE: str = "token"    # 分块器类型: "token" 或 "sentence"
    RAG_FATHER_SON_RATIO: int = 3       # parent_child 模式下父块/子块倍率（父块≈子块*倍率）
    # parent_child 检索时，每个返回父段落的最大 token（0=自动按 chunk_size*father_son_ratio 推导）
    # 目的：避免个别超长父块导致上下文预算被单段吞噬。
    RAG_PARENT_MAX_TOKENS_PER_SEGMENT: int = 0
    RAG_PARENT_STORE_SELF_HEAL_ENABLED: bool = True   # 启动时若 parent 映射缺失，是否自动尝试自愈重建
    RAG_PARENT_STORE_SELF_HEAL_MAX_POINTS: int = 50000  # 自愈扫描单个KB的最大 points 数（防止超大库启动过慢）
    RAG_DEFAULT_TOP_K: int = 8          # 单次检索返回的最相关文本块数量（向量/混合检索最终截断条数）
    RAG_HYBRID_SEARCH: bool = True      # 是否启用 BM25 + 向量的混合检索（关闭则仅向量检索）
    # BM25 权重适当上调至 0.35：关键词信号更强，对"春节值班"/"专利条款"等精确查询，
    # BM25 能有效将正确来源文档的得分与无关文档拉开差距，从而让 source 级过滤更准确。
    RAG_BM25_WEIGHT: float = 0.35       # BM25 在 RRF 融合中的权重（0.0-1.0，越大越偏关键词匹配）
    RAG_MIN_RELEVANCE_SCORE: float = 0.008  # 若最高相关分低于此值则本库不返回片段（0=不启用），默认略微抑制弱相关结果
    # 来源级别相对过滤：按来源文档分组，只保留最高分 >= 最佳来源分 * ratio 的整个来源。
    # 与 segment 级过滤的本质区别：要么保留该来源的全部段落，要么整体排除，避免随机截断相关片段。
    # 适用场景：一个查询针对某份合同，该合同段落得分整体高于其他文档时，
    #           其他文档会被整体过滤，LLM 不再产生答非所问；
    #           若多文档得分相近（合理的跨文档查询），则都保留。
    # 值 0.65 = 允许最高分 ±35% 以内的来源共存；0 = 关闭此过滤。
    RAG_SOURCE_SCORE_RATIO: float = 0.65
    RAG_RERANK_ENABLED: bool = False      # 是否启用 Rerank（通过外部 API，不占宿主机算力），默认关闭，需在 .env 中显式开启
    RAG_RERANK_BASE_URL: str = ""         # Rerank 服务地址（如 TEI 或智谱），与 api_key 配合使用
    RAG_RERANK_API_KEY: SecretStr | None = None  # 可选，智谱等需鉴权时配置
    RAG_RERANK_MODEL: str = ""            # Rerank 模型名称（供 API 使用，若为空则启用时用默认）
    RAG_RERANK_TOP_K: int = 6             # Rerank 后保留的最终片段数量
    RAG_RERANK_TIME_LIMIT: float = 1.0     # Rerank 最长允许耗时（秒）。<=0 表示不限制（使用内部默认超时）
    RAG_RERANK_MIN_SCORE: float = 0.0      # Rerank 分数下限（默认不启用）。低于此阈值的片段会被剔除

    # RAG Query 改写 / HyDE 配置
    # - RAG_HYDE_ENABLED: 是否启用 HyDE 风格的 Query 改写（默认关闭，保持兼容）
    # - RAG_HYDE_NUM_VARIANTS: 每次为同一个问题生成多少条改写/假想文档（建议 1–3）
    RAG_HYDE_ENABLED: bool = False
    RAG_HYDE_NUM_VARIANTS: int = 2

    # RAG 检索过滤推断配置
    # - RAG_QUERY_FILTER_INFERENCE_ENABLED: 是否尝试从自然语言问题中推断简单的过滤条件
    #   （例如“某某论文的作者是谁” → 过滤 doc_title 包含“某某论文”）。默认关闭，保持兼容。
    RAG_QUERY_FILTER_INFERENCE_ENABLED: bool = False

    # RAG 分块策略配置
    # - RAG_CHUNKING_STRATEGY: "simple" 或 "parent_child"
    #   默认使用 parent_child，并在检索阶段将叶子命中提升为父级上下文。
    # - RAG_CHUNKING_TITLE_AWARE: 是否在上述策略基础上启用「标题感知」预分段：
    #   - simple + True       → 先按标题切 section，再做 SentenceSplitter 分块；
    #   - parent_child + True → 先按标题切 section，再做 HierarchicalNodeParser 父子分块；
    #   - 设为 False 时则退化为纯长度/层级驱动的分块行为。
    RAG_CHUNKING_STRATEGY: str = "parent_child"
    RAG_CHUNKING_TITLE_AWARE: bool = False

    # DuckDuckGo 网页搜索（FormattedDuckDuckGoSearchResults）
    # 逻辑：每次调用 WebSearch(query) = 只发 1 次搜索请求；DDGS_TIMEOUT 限制这次调用的总耗时。
    # DDGS_TOP_K 是「这一次搜索」返回结果经 BM25/去重/过滤后最多保留几条，不是「搜几次」，不会 12*5 秒。
    DDGS_MAX_RESULTS: int = 5          # 向 API 请求的最大条数（单次请求）
    DDGS_TOP_K: int = 5                # 单次搜索经 BM25 筛选后最多返回条数
    DDGS_MIN_SCORE: float = 0.1       # BM25 相关性下限，低于此分数的结果丢弃
    DDGS_TIMEOUT: float = 12.0         # 单次工具调用的总超时（秒），即一次 WebSearch(query) 的最长等待
    DDGS_BM25_K1: float = 1.5          # BM25 参数 k1
    DDGS_BM25_B: float = 0.75          # BM25 参数 b

    # 数据清理配置
    CLEANUP_INTERVAL_HOURS: int = 24  # 清理间隔，单位：小时
    DATA_RETENTION_DAYS: int = 30     # 数据保留天数

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OLLAMA: self.OLLAMA_MODEL and is_ollama_reachable(self.OLLAMA_BASE_URL),
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_API_KEY,
            Provider.ZHIPU: self.ZHIPU_API_KEY,
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.VERTEXAI: self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.FAKE: self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI: self.AZURE_OPENAI_API_KEY,
            Provider.OPENROUTER: self.OPENROUTER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_NAME
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.ZHIPU:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = ZhipuModelName.GLM_4_6
                    self.AVAILABLE_MODELS.update(set(ZhipuModelName))
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_CHAT
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_45
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.VERTEXAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = VertexAIModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.AWS:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                    self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OPENROUTER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_25_FLASH
                    self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # Validate Azure OpenAI settings if Azure provider is available
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # Parse deployment map if it's a string
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # Validate required deployments exist
                    required_models = {"gpt-4o", "gpt-4o-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

        # Rerank 默认模型：
        # - TEI / 自建 /rerank 服务通常是“服务绑定模型”，请求体不需要传 model 字段 → 使用 "default"
        # - 智谱 Rerank 明确要求 model="rerank" → 若 BASE_URL 指向 open.bigmodel.cn 且未显式配置，则设为 "rerank"
        if self.RAG_RERANK_ENABLED and (self.RAG_RERANK_BASE_URL or "").strip():
            base = (self.RAG_RERANK_BASE_URL or "").strip()
            if not (self.RAG_RERANK_MODEL or "").strip():
                if "open.bigmodel.cn" in base:
                    self.RAG_RERANK_MODEL = "rerank"
                else:
                    self.RAG_RERANK_MODEL = "default"

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"
    
    @computed_field
    @property
    def STATIC_URL(self) -> str:
        return "/static"

    @computed_field
    @property
    def STATIC_DIR(self) -> Path:
        # Returns absolute path using current working directory
        # Docker: /app/static (WORKDIR is /app)
        # Local: <project_root>/static (assuming run from root)
        return Path("static").absolute()

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()