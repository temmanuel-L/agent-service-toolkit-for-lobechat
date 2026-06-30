import logging

from core.settings import settings

# Configure logging at import time - this runs very early
# Suppress noisy third-party libraries BEFORE they get imported
_noisy_loggers = [
    "httpcore",
    "httpcore.connection", 
    "httpcore.http11",
    "httpx",
    "primp",
    "rquest",
    "cookie_store",
    "urllib3",
    "urllib3.connectionpool",
    "asyncio",
    "hpack",
    "h2",
    "charset_normalizer",
    "duckduckgo_search",
    "openai",
    "qdrant_client",
    "grpc",
]

_log_level = settings.LOG_LEVEL.to_logging_level()
_min_level = max(_log_level, logging.WARNING)

for _logger_name in _noisy_loggers:
    logging.getLogger(_logger_name).setLevel(_min_level)

# Now import other modules
from core.llm import get_model, get_embedding_model, get_model_for_neo4j_graphrag

__all__ = ["settings", "get_model", "get_embedding_model", "get_model_for_neo4j_graphrag"]
