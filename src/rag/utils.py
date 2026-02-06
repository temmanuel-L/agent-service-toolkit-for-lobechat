"""
RAG 通用工具模块

提供 RAG 相关的通用工具函数，可被任何需要 RAG 能力的智能体复用。

主要功能：
- 结果格式化与长度控制
- 空响应回退处理
- 结果解析与摘要
"""
from typing import Optional, List
from utils.log_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# 配置常量
# ============================================================================
RAG_MAX_RESULT_LENGTH = 6000  # 单次检索结果的最大字符数
RAG_MAX_SEGMENTS_DISPLAY = 5  # 最多显示的段落数
RAG_SEGMENT_SEPARATOR = "\n\n---\n\n"  # 段落分隔符


def truncate_rag_result(
    content: str,
    max_length: int = RAG_MAX_RESULT_LENGTH,
    max_segments: int = RAG_MAX_SEGMENTS_DISPLAY
) -> str:
    """
    截断 RAG 检索结果，确保长度可控
    
    Args:
        content: 原始检索结果
        max_length: 最大字符数
        max_segments: 最大段落数
        
    Returns:
        截断后的结果，如果被截断会添加提示
    """
    if not content:
        return content
    
    # 按段落分割
    segments = content.split(RAG_SEGMENT_SEPARATOR)
    
    # 限制段落数
    if len(segments) > max_segments:
        segments = segments[:max_segments]
        truncated_by_segments = True
    else:
        truncated_by_segments = False
    
    # 重新组合
    result = RAG_SEGMENT_SEPARATOR.join(segments)
    
    # 限制总长度
    if len(result) > max_length:
        result = result[:max_length]
        truncated_by_length = True
    else:
        truncated_by_length = False
    
    # 添加截断提示
    if truncated_by_segments or truncated_by_length:
        result += "\n\n[... 检索结果已截断，请提出更具体的问题以获取更精确的信息 ...]"
    
    return result


def format_rag_fallback_response(
    tool_result: str,
    max_preview_length: int = 2000,
    max_preview_segments: int = 3
) -> str:
    """
    当模型无法正确处理工具结果时，生成回退响应。

    设计原则：
    - **绝不**将原始检索 segment 暴露给前端用户
    - 仅告知用户"检索到了信息但无法生成摘要"
    - 引导用户重试或换个问法

    之所以不展示原始 segment：
    1. RAG segment 是内部上下文，包含大量无关噪声（公式、表格碎片等）
    2. 用户期望的是 LLM 的整合回答，而非检索原文
    3. 暴露原文会严重影响用户体验
    """
    if not tool_result:
        return (
            "抱歉，我尝试搜索了知识库，但未找到相关内容。"
            "请尝试用不同的方式提问，或确认知识库中包含相关信息。"
        )

    # 统计检索到多少段落（仅用于提示，不暴露内容）
    segments = tool_result.split(RAG_SEGMENT_SEPARATOR)
    segment_count = len([s for s in segments if s.strip()])

    return (
        f"我在知识库中检索到了 {segment_count} 条相关信息，"
        f"但模型未能成功生成摘要回答。\n\n"
        f"建议您：\n"
        f"1. 尝试用更具体的问题重新提问\n"
        f"2. 缩小问题范围，聚焦某一方面\n"
        f"3. 如果问题持续，可切换到其他智能体再试"
    )


def is_rag_tool_message(tool_name: str) -> bool:
    """
    判断是否为 RAG 相关的工具
    
    Args:
        tool_name: 工具名称
        
    Returns:
        是否为 RAG 工具
    """
    RAG_TOOLS = {"search_knowledge"}  # 可扩展
    return tool_name in RAG_TOOLS


def parse_rag_segments(content: str) -> List[dict]:
    """
    解析 RAG 检索结果为结构化的段落列表
    
    Args:
        content: 原始检索结果
        
    Returns:
        段落列表，每个元素包含 segment_id 和 text
    """
    if not content:
        return []
    
    segments = content.split(RAG_SEGMENT_SEPARATOR)
    parsed = []
    
    for i, segment in enumerate(segments):
        # 尝试解析 [Knowledge Segment N] 格式
        lines = segment.strip().split("\n", 1)
        if lines and lines[0].startswith("[Knowledge Segment"):
            header = lines[0]
            text = lines[1] if len(lines) > 1 else ""
        else:
            header = f"[Segment {i + 1}]"
            text = segment
        
        parsed.append({
            "segment_id": i + 1,
            "header": header,
            "text": text.strip()
        })
    
    return parsed
