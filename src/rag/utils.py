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
    当模型无法正确处理工具结果时，生成格式化的回退响应
    
    这个函数用于处理模型返回空内容的情况，将原始工具结果
    格式化为用户可读的形式。
    
    Args:
        tool_result: 工具返回的原始结果
        max_preview_length: 预览内容的最大长度
        max_preview_segments: 预览的最大段落数
        
    Returns:
        格式化的回退响应内容
    """
    if not tool_result:
        return (
            "抱歉，我尝试搜索了知识库，但未找到相关内容。"
            "请尝试用不同的方式提问，或确认知识库中包含相关信息。"
        )
    
    # 解析段落
    segments = tool_result.split(RAG_SEGMENT_SEPARATOR)
    
    if not segments:
        return (
            "抱歉，知识库检索结果无法正确解析。"
            "请尝试用不同的方式提问。"
        )
    
    # 只展示前几个段落
    preview_segments = segments[:max_preview_segments]
    formatted_preview = RAG_SEGMENT_SEPARATOR.join(preview_segments)
    
    # 限制总长度
    if len(formatted_preview) > max_preview_length:
        formatted_preview = formatted_preview[:max_preview_length] + "...\n\n(内容已截断)"
    
    # 构建响应
    remaining_count = len(segments) - max_preview_segments
    remaining_note = f"\n\n*还有 {remaining_count} 个相关段落未显示*" if remaining_count > 0 else ""
    
    fallback_content = (
        f"我在知识库中找到了以下相关信息：\n\n"
        f"{formatted_preview}"
        f"{remaining_note}\n\n"
        f"---\n"
        f"*注：以上为知识库原始检索结果。如需更详细的分析，请提出更具体的问题。*"
    )
    
    return fallback_content


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
