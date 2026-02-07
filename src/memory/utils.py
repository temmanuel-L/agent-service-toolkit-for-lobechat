"""
内存管理工具函数模块。
"""
import re
import zlib
from utils.log_utils import get_logger

logger = get_logger(__name__)

def is_low_quality_text(text: str) -> bool:
    """
    检测文本是否为低质量内容（垃圾数据、重复循环、幻觉噪音）。
    
    主要用于防止 LLM 产生的无限循环（如 "A. A. A..."）或无意义重复字符污染长期记忆。
    
    检测策略 (Heuristics):
    1. 压缩率检查 (Compression Ratio):
       高质量文本通常具有一定的熵值。简单的重复模式（如 "K. M. K. M."）压缩率极高。
       如果压缩后的体积远小于原始体积（比率 < 0.2），通常意味着内容高度重复。
       
    2. 字符密度检查 (Character Density):
       防止单一字符重复填充（如 "AAAAAAAA"）。
       如果去除空格后，某一个字符占据了超过 50% 的长度，则视为无效内容。
    
    Args:
        text: 需要检测的文本内容
        
    Returns:
        bool: 如果是低质量文本返回 True，否则返回 False
    """
    # 文本太短通常不适用统计规律，直接放行
    if not text or len(text.strip()) < 50:
        return False

    encoded = text.encode('utf-8')
    input_len = len(encoded)
    
    # 1. 压缩率检查 (检测循环模式)
    # 高质量文本的压缩比通常 > 0.3 (短文本) 或 > 0.4 (长文本)
    # 低质量的循环文本 (如 "K. M. K. M.") 压缩比通常 < 0.1 或 0.2
    compressed = zlib.compress(encoded)
    ratio = len(compressed) / input_len
    
    # 阈值设定：长度 > 200 且 压缩比 < 0.2 视为异常
    if input_len > 200 and ratio < 0.2:
        logger.warning(f"检测到低质量文本 (低熵/重复循环, 压缩比 {ratio:.2f}): {text[:50]}...")
        return True
        
    # 2. 字符密度检查 (检测单一字符灌水)
    # 移除所有空白字符
    clean_text = re.sub(r'\s', '', text)
    if not clean_text:
        return True
        
    from collections import Counter
    counts = Counter(clean_text)
    # 获取出现频率最高的字符及其次数
    most_common_char, count = counts.most_common(1)[0]
    
    # 如果特定字符占比超过 50%，视为垃圾数据
    if len(clean_text) > 50 and (count / len(clean_text)) > 0.5:
         logger.warning(f"检测到低质量文本 (单一字符 '{most_common_char}' 密度过高): {text[:50]}...")
         return True

    # 3. [NEW] 异常标点模式检查 (检测 A.B.C.D.E... 这种高熵但无意义的序列)
    # 统计“字母+点”的模式出现频率
    # 正常文本中，periods (点) 的密度通常较低 (< 5%)
    # 如果点号出现的频率极高（例如每两个字符就有一个点），且长度较长，视为异常
    dot_count = text.count('.')
    comma_count = text.count(',')
    if len(text) > 50:
        # 如果点号密度超过 15% (正常句子结束也就 2-3%)
        if (dot_count / len(text)) > 0.15:
             logger.warning(f"检测到低质量文本 (点号密度过高: {dot_count/len(text):.2%}): {text[:50]}...")
             return True
             
        # 检测连续的 X.Y.Z. 模式
        # 匹配 "字母+点" 重复出现的次数
        pattern = r'([A-Za-z0-9]\.){3,}'
        matches = re.findall(pattern, text)
        if matches and len(''.join(matches)) > len(text) * 0.3:
             logger.warning(f"检测到低质量文本 (疑似字母缩写灌水): {text[:50]}...")
             return True

    return False
