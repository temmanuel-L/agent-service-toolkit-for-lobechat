"""
HyDE / Query 改写实现模块。

当前策略（第一阶段实现，兼顾简单与可控开销）：
- 是否启用由环境变量控制：RAG_HYDE_ENABLED / RAG_HYDE_NUM_VARIANTS；
- 使用一个异步 LLM 调用（get_model + ainvoke）一次性生成多条改写/假想文档；
- 解析为结构化的 HyDEResult，供检索阶段选择使用。

后续可以在此处迭代：
- 区分“短查询改写”和“长假想文档 (HyDE-style)”两种模式；
- 支持按领域/知识库定制提示词；
- 针对多路改写做 A/B 测试与召回质量评估。
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_model
from core.settings import settings
from rag.schema.schema_hyde import HyDEResult, QueryVariant
from utils.log_utils import get_logger

logger = get_logger(__name__)


HYDE_SYSTEM_PROMPT = """你是一个检索查询改写助手，用于知识库搜索（RAG/HyDE 场景）。

给定用户的原始问题，你需要生成若干条适合用于检索的等价查询或简短“假想文档”。

要求：
1. 保持语义等价或高度相关，不能改变用户真正想问的内容。
2. 使用可能出现在文档中的关键词、专业术语、设备/规范的全称或别名。
3. 可以补充隐含前提，但不要发明新的事实。
4. 输出格式：每一行是一条改写，不要编号，不要解释，不要添加前后缀说明。
"""


async def generate_hyde_variants(
    query: str,
    num_variants: int,
) -> HyDEResult:
    """
    基于 HyDE 思想对用户查询进行多路改写/假想文档生成。

    当前实现：
    - 使用 settings.DEFAULT_MODEL 对应的对话模型；
    - 单次 ainvoke，根据 num_variants 控制期望条数；
    - 将模型输出按行拆分，去空行、去重，最多保留 num_variants 条。
    """
    num_variants = max(0, num_variants)
    if num_variants == 0:
        return HyDEResult(original_query=query, variants=[])

    try:
        model = get_model(settings.DEFAULT_MODEL)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"HyDE: 获取默认模型失败，跳过改写: {e}")
        return HyDEResult(original_query=query, variants=[])

    system_msg = SystemMessage(content=HYDE_SYSTEM_PROMPT)
    human_msg = HumanMessage(
        content=(
            "用户原始问题如下，请根据上面的要求生成多条检索查询或假想文档，每行一条：\n\n"
            f"{query}\n\n"
            f"期望条数：{num_variants}（不必严格相等，但不要明显少于 1 条）。"
        )
    )

    try:
        response = await model.ainvoke([system_msg, human_msg])
    except Exception as e:
        logger.warning(f"HyDE: LLM 改写调用失败，跳过改写，直接使用原始查询: {e}")
        return HyDEResult(original_query=query, variants=[])

    content = response.content if isinstance(response.content, str) else str(response.content)
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    # 去重并截断到 num_variants
    seen: set[str] = set()
    variants: List[QueryVariant] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        variants.append(QueryVariant(text=line, source="hyde"))
        if len(variants) >= num_variants:
            break

    if not variants:
        logger.info("HyDE: 模型未返回有效改写，继续使用原始查询。")
        return HyDEResult(original_query=query, variants=[])

    logger.info(
        "HyDE: 为查询生成了 %d 条改写，示例：%s",
        len(variants),
        variants[0].text[:80].replace("\n", " "),
    )
    return HyDEResult(original_query=query, variants=variants)


__all__ = [
    "generate_hyde_variants",
]

