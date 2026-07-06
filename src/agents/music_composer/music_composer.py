# -*- coding: utf-8 -*-
"""
@Time ： 2026/1/28 18:27
@Auth ： luanxing
@File ：music_composer.py
@IDE ：PyCharm

AI 音乐作曲智能体 - 基于 LangGraph 实现
===================================

该智能体能够根据用户输入生成音乐作品，并提供可下载的 MIDI 文件链接。

工作流程:
1. 提取信息节点 (extract_info): 从用户消息中提取音乐风格、情绪、调式等信息
2. 询问缺失信息节点 (ask_missing): 如果关键信息缺失，通过中断机制向用户询问
3. 生成旋律节点 (generate_melody): 使用 LLM 生成旋律描述
4. 生成和声节点 (generate_harmony): 为旋律生成和声
5. 生成节奏节点 (generate_rhythm): 为作品生成节奏
6. 转换 MIDI 节点 (convert_to_midi): 将作品转换为 MIDI 文件并返回下载链接

注意事项:
- 由于 OpenAI /v1/chat/completions 接口只支持文本传输，无法直接播放音频
- 本实现采用「返回音频 URL」方案，用户可以点击链接下载/播放 MIDI 文件
"""

import os
import re
import asyncio
import random
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings
from memory.long_term_concat_for_agents import prepare_long_term_entry
from utils.log_utils import get_logger

# 尝试导入 music21 库，如果不存在则标记为不可用
try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

logger = get_logger(__name__)


# =============================================================================
# 状态定义 (State Definition)
# =============================================================================
class MusicState(MessagesState, total=False):
    """
    音乐作曲智能体的状态定义。
    
    继承自 MessagesState，添加音乐创作所需的字段。
    """
    # 用户输入的音乐描述
    musician_input: Optional[str]
    # 音乐风格 (如: 古典、爵士、流行、电子)
    style: Optional[str]
    # 音乐情绪 (如: 欢快、悲伤、激昂、平静)
    mood: Optional[str]
    # 调式 (如: C大调、A小调)
    key: Optional[str]
    # 音乐时长 (秒)
    duration: Optional[int]
    # 生成的旋律描述
    melody: Optional[str]
    # 生成的和声描述
    harmony: Optional[str]
    # 生成的节奏描述
    rhythm: Optional[str]
    # 完整的作曲描述
    composition: Optional[str]
    # 生成的 MIDI 文件路径/URL
    midi_file: Optional[str]
    # 剩余步数追踪
    remaining_steps: RemainingSteps


# =============================================================================
# 提取模式定义 (Extraction Schema)
# =============================================================================
class MusicInfoExtraction(BaseModel):
    """
    从用户消息中提取音乐创作信息的模式定义。
    
    此模式作为 LLM 工具调用的参数结构。
    """
    style: Optional[str] = Field(
        default=None,
        description="音乐风格，如：古典(classical)、爵士(jazz)、流行(pop)、"
                    "电子(electronic)、摇滚(rock)、乡村(country)、浪漫(romantic)等。"
    )
    mood: Optional[str] = Field(
        default=None,
        description="音乐情绪/氛围，如：欢快(happy)、悲伤(sad)、激昂(exciting)、"
                    "平静(calm)、神秘(mysterious)、史诗(epic)等。"
    )
    key: Optional[str] = Field(
        default=None,
        description="音乐调式，如：C大调(C major)、A小调(A minor)、G大调(G major)等。"
                    "如果用户没有明确指定，可以留空。"
    )
    duration: Optional[int] = Field(
        default=30,
        description="音乐时长（秒）。如果用户指定了分钟，请转换为秒。默认30秒。"
    )


# =============================================================================
# 必填字段配置 (Required Fields Configuration)
# =============================================================================
REQUIRED_FIELDS = ["style", "mood"]  # key 不是必填，可以自动选择

FIELD_PROMPTS = {
    "style": "请问您想要什么风格的音乐？（如：古典、爵士、流行、电子、摇滚等）",
    "mood": "请问您希望音乐表达什么情绪？（如：欢快、悲伤、激昂、平静等）",
    "key": "请问您有偏好的调式吗？（如：C大调、A小调，不确定可以跳过）",
}


def get_missing_fields(state: MusicState) -> List[str]:
    """
    检查哪些必填字段仍然缺失。
    
    Args:
        state: 当前的音乐创作状态
    
    Returns:
        缺失字段名称的列表
    """
    missing = []
    for required_field in REQUIRED_FIELDS:
        value = state.get(required_field)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(required_field)
    logger.info(f'缺失字段为：{missing}')
    return missing


# =============================================================================
# Few-shot 工具调用示例 (Few-shot Examples for Tool Calling)
# =============================================================================
def _build_tool_call_example(
        user_input: str,
        style: Optional[str],
        mood: Optional[str],
        key: Optional[str] = None,
        duration: Optional[int] = None) -> List:
    """
    构建单个 few-shot 示例，使用正确的 tool_calls 格式。
    
    Args:
        user_input: 用户输入消息
        style: 提取的音乐风格
        mood: 提取的音乐情绪
        key: 提取的调式（可选）
    
    Returns:
        包含 HumanMessage, AIMessage, ToolMessage 的列表
    """
    tool_call_id = str(uuid.uuid4())
    return [
        HumanMessage(content=user_input),
        AIMessage(content="", tool_calls=[{
            "id": tool_call_id,
            "name": "MusicInfoExtraction",
            "args": {"style": style, "mood": mood, "key": key, "duration": duration}
        }]),
        ToolMessage(content="已成功提取音乐信息", tool_call_id=tool_call_id),
    ]


def _get_extraction_examples() -> List:
    """
    构建所有 few-shot 示例。
    
    Returns:
        示例消息列表
    """
    examples = []
    # 示例 1: 完整信息
    examples.extend(_build_tool_call_example(
        "帮我创作一首欢快的爵士乐，用G大调，大概1分钟",
        "爵士", "欢快", "G大调", 60
    ))
    # 示例 2: 只有风格和情绪
    examples.extend(_build_tool_call_example(
        "我想要一首悲伤的古典钢琴曲",
        "古典", "悲伤", None, 30
    ))
    # 示例 3: 只有情绪
    examples.extend(_build_tool_call_example(
        "来点激昂的音乐",
        None, "激昂", None, 30
    ))
    # 示例 4: 没有可提取的信息
    examples.extend(_build_tool_call_example(
        "你好",
        None, None, None, None
    ))
    return examples


# 预构建示例（仅在模块加载时执行一次）
EXTRACTION_EXAMPLES = _get_extraction_examples()

# 系统提示词
EXTRACTION_SYSTEM_PROMPT = """你是一个音乐信息提取助手。你的任务是从用户消息中提取音乐创作相关信息。

规则：
1. style: 提取音乐风格，如古典、爵士、流行、电子、摇滚、乡村等。
2. mood: 提取音乐情绪/氛围，如欢快、悲伤、激昂、平静、神秘等。
3. key: 提取调式，如C大调、A小调等。如果用户没有明确指定，则为 null。
3. key: 提取调式，如C大调、A小调等。如果用户没有明确指定，则为 null。
4. duration: 提取音乐时长（秒）。"2分钟"->120。默认30。
5. 只提取用户明确提到的信息，不要猜测。
6. 必须调用 MusicInfoExtraction 工具返回结果。"""


# =============================================================================
# 节点: 提取信息 (Extract Information Node)
# =============================================================================
async def extract_info(state: MusicState, config: RunnableConfig) -> dict:
    """
    使用 LLM + 工具调用从用户消息中提取音乐创作信息。
    
    Args:
        state: 当前状态
        config: 运行配置
    
    Returns:
        状态更新字典
    """
    logger.info(f"--- [EXTRACT INFO] Remaining steps: {state.get('remaining_steps')} ---")
    messages = state.get("messages", [])
    if not messages:
        logger.info("没有消息可提取，跳过提取步骤")
        return {}
    
    # 获取模型并绑定提取工具
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_with_tools = llm.bind_tools([MusicInfoExtraction])
    
    # 构建带示例占位符的提示模板
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="examples"),
        ("human", "{user_message}"),
    ])
    
    # 获取最后一条用户消息用于提取
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break
    
    if not last_human_msg:
        logger.info("未找到用户消息，跳过提取")
        return {}
    
    logger.info(f"从以下内容提取信息: {last_human_msg}")
    
    try:
        # 调用 LLM 进行工具调用 - 包含 few-shot 示例
        formatted_messages = extraction_prompt.format_messages(
            user_message=last_human_msg,
            examples=EXTRACTION_EXAMPLES
        )
        response = await llm_with_tools.with_config(tags=["skip_stream"]).ainvoke(formatted_messages, config)
        
        # 解析工具调用结果
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = tool_call["args"]
            
            extracted = MusicInfoExtraction(**args)
            logger.info(f"提取结果: style={extracted.style}, mood={extracted.mood}, key={extracted.key}")
            
            # 构建更新字典
            updates = {"musician_input": last_human_msg}
            
            # 只有当字段为空时才更新（避免覆盖已有值）
            if extracted.style and not state.get("style"):
                updates["style"] = extracted.style
            if extracted.mood and not state.get("mood"):
                updates["mood"] = extracted.mood
            if extracted.key and not state.get("key"):
                updates["key"] = extracted.key
            if extracted.duration and not state.get("duration"):
                updates["duration"] = extracted.duration
            
            return updates
        else:
            logger.warning("LLM 未返回工具调用")
            return {}
            
    except Exception as e:
        logger.error(f"提取失败: {e}")
        return {}


# =============================================================================
# 节点: 询问缺失信息 (Ask Missing Info Node)
# =============================================================================
def ask_missing_info(state: MusicState, config: RunnableConfig) -> dict:
    """
    使用中断机制动态询问缺失的必填字段。
    
    Args:
        state: 当前状态
        config: 运行配置
    
    Returns:
        状态更新字典
    """
    logger.info(f"--- [ASK MISSING] Remaining steps: {state.get('remaining_steps')} ---")
    missing_fields = get_missing_fields(state)
    
    if not missing_fields:
        return {}
    
    # 构建询问提示 - 使用编号列表以便更好地展示
    prompts = [f"{i+1}. {FIELD_PROMPTS.get(f, f'请提供{f}')}" for i, f in enumerate(missing_fields)]
    combined_prompt = "🎵 为了帮您创作音乐，我需要了解以下信息：\n" + "\n".join(prompts)
    
    logger.info(f"询问缺失字段: {missing_fields}")
    
    # 中断以获取用户输入
    user_response = interrupt(combined_prompt)
    logger.info(f"用户响应: {user_response}")
    
    return {
        "messages": [HumanMessage(content=user_response)]
    }


# =============================================================================
# 节点: 生成旋律 (Generate Melody Node)
# =============================================================================
melody_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位专业的作曲家。请根据以下要求生成旋律。\n"
     "风格: {style}\n"
     "情绪: {mood}\n"
     "调式: {key}\n"
     "用户原始要求: {user_input}\n\n"
     "规则：\n"
     "1. 用 music21 格式描述旋律 (如 C4, D4, E4)。\n"
     "2. 严禁输出任何对话文本！严禁输出 '当然可以' 等废话。\n"
     "3. 只输出音符序列 CSV，用逗号分隔，不要换行。\n"
     "4. 生成长度限制：4小节 (约16-32个音符)，保持精简。"),
    ("human", "开始生成。"),
])


async def generate_melody(state: MusicState, config: RunnableConfig) -> dict:
    """
    使用 LLM 生成旋律描述。
    
    Args:
        state: 当前状态
        config: 运行配置
    
    Returns:
        包含旋律的状态更新字典
    """
    logger.info(f"--- [GENERATE MELODY] ---")
    
    style = state.get("style", "古典")
    mood = state.get("mood", "平静")
    key = state.get("key", "C大调")
    user_input = state.get("musician_input", "")
    
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    formatted_messages = melody_prompt.format_messages(
        style=style,
        mood=mood,
        key=key,
        user_input=user_input
    )
    
    try:
        # 增加超时控制 (30秒)
        response = await asyncio.wait_for(
            llm.with_config(tags=["skip_stream"]).ainvoke(formatted_messages, config),
            timeout=30.0
        )
        melody_content = response.content
        logger.info(f"生成的旋律: {melody_content[:100]}...")
        return {"melody": melody_content}
    except asyncio.TimeoutError:
        logger.error("旋律生成超时")
        return {"melody": "C4, D4, E4, F4, G4, A4, B4, C5"}
    except Exception as e:
        logger.error(f"旋律生成失败: {e}")
        return {"melody": "C4, D4, E4, F4, G4, A4, B4, C5"}  # 默认旋律


# =============================================================================
# 节点: 生成和声 (Generate Harmony Node)
# =============================================================================
harmony_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位专业的作曲家。请为以下旋律创建和声。\n"
     "旋律: {melody}\n"
     "风格: {style}\n\n"
     "规则：\n"
     "1. 用 music21 格式描述和弦 (如 C4-E4-G4)。\n"
     "2. 严禁输出任何对话文本！只输出 CSV。\n"
     "3. 和弦数量应与旋律小节数匹配 (约4-8个和弦即可)。"),
    ("human", "开始生成。"),
])


async def generate_harmony(state: MusicState, config: RunnableConfig) -> dict:
    """
    使用 LLM 为旋律生成和声。
    
    Args:
        state: 当前状态
        config: 运行配置
    
    Returns:
        包含和声的状态更新字典
    """
    logger.info(f"--- [GENERATE HARMONY] ---")
    
    melody = state.get("melody", "")
    style = state.get("style", "古典")
    
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    formatted_messages = harmony_prompt.format_messages(
        melody=melody,
        style=style
    )
    
    try:
        response = await asyncio.wait_for(
            llm.with_config(tags=["skip_stream"]).ainvoke(formatted_messages, config),
            timeout=30.0
        )
        harmony_content = response.content
        logger.info(f"生成的和声: {harmony_content[:100]}...")
        return {"harmony": harmony_content}
    except asyncio.TimeoutError:
         logger.error("和声生成超时")
         return {"harmony": "C4-E4-G4, F4-A4-C5, G4-B4-D5, C4-E4-G4"}
    except Exception as e:
        logger.error(f"和声生成失败: {e}")
        return {"harmony": "C4-E4-G4, F4-A4-C5, G4-B4-D5, C4-E4-G4"}  # 默认和声


# =============================================================================
# 节点: 生成节奏 (Generate Rhythm Node)
# =============================================================================
rhythm_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位专业的作曲家。请为以下旋律和和声建议节奏。\n"
     "旋律: {melody}\n"
     "和声: {harmony}\n"
     "情绪: {mood}\n\n"
     "规则：\n"
     "1. 用时值描述节奏 (如 quarter, half)。\n"
     "2. 严禁输出任何对话文本！只输出 CSV。\n"
     "3. 保持节奏简单明了。"),
    ("human", "开始生成。"),
])


async def generate_rhythm(state: MusicState, config: RunnableConfig) -> dict:
    """
    使用 LLM 为作品生成节奏。
    
    Args:
        state: 当前状态
        config: 运行配置
    
    Returns:
        包含节奏和完整作曲描述的状态更新字典
    """
    logger.info(f"--- [GENERATE RHYTHM] ---")
    
    melody = state.get("melody", "")
    harmony = state.get("harmony", "")
    mood = state.get("mood", "平静")
    
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    formatted_messages = rhythm_prompt.format_messages(
        melody=melody,
        harmony=harmony,
        mood=mood
    )
    
    try:
        response = await asyncio.wait_for(
            llm.with_config(tags=["skip_stream"]).ainvoke(formatted_messages, config),
            timeout=30.0
        )
        rhythm_content = response.content
        logger.info(f"生成的节奏: {rhythm_content[:100]}...")
        
        # 组合完整的作曲描述
        composition = f"旋律: {melody}\n和声: {harmony}\n节奏: {rhythm_content}"
        
        return {
            "rhythm": rhythm_content,
            "composition": composition
        }
    except asyncio.TimeoutError:
        logger.error("节奏生成超时")
        return {
            "rhythm": "quarter, quarter, quarter, quarter",
            "composition": f"旋律: {melody}\n和声: {harmony}\n节奏: quarter, quarter, quarter, quarter"
        }
    except Exception as e:
        logger.error(f"节奏生成失败: {e}")
        return {
            "rhythm": "quarter, quarter, quarter, quarter, half, half, whole",
            "composition": f"旋律: {melody}\n和声: {harmony}\n节奏: quarter, quarter, quarter, quarter"
        }


# =============================================================================
# 节点: 转换为 MIDI (Convert to MIDI Node)
# =============================================================================
# MIDI 文件存储目录（静态资源目录）
MIDI_OUTPUT_DIR = settings.STATIC_DIR / "music"

# 音阶定义
SCALES = {
    'C大调': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'C小调': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
    'G大调': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
    'A小调': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'D大调': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
    'E小调': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
    'F大调': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
}

# 和弦定义
CHORDS = {
    'C大调': ['C4', 'E4', 'G4'],
    'C小调': ['C4', 'Eb4', 'G4'],
    'G大调': ['G4', 'B4', 'D5'],
    'A小调': ['A3', 'C4', 'E4'],
    'D大调': ['D4', 'F#4', 'A4'],
    'E小调': ['E4', 'G4', 'B4'],
    'F大调': ['F4', 'A4', 'C5'],
}


def _parse_key(key_str: Optional[str]) -> str:
    """
    将用户输入的调式标准化。
    
    Args:
        key_str: 用户输入的调式字符串
    
    Returns:
        标准化的调式名称
    """
    if not key_str:
        return 'C大调'
    
    key_str = key_str.lower().replace(' ', '')
    
    # 常见映射
    mappings = {
        'c大调': 'C大调', 'cmajor': 'C大调', 'c': 'C大调',
        'c小调': 'C小调', 'cminor': 'C小调', 'cm': 'C小调',
        'g大调': 'G大调', 'gmajor': 'G大调', 'g': 'G大调',
        'a小调': 'A小调', 'aminor': 'A小调', 'am': 'A小调',
        'd大调': 'D大调', 'dmajor': 'D大调', 'd': 'D大调',
        'e小调': 'E小调', 'eminor': 'E小调', 'em': 'E小调',
        'f大调': 'F大调', 'fmajor': 'F大调', 'f': 'F大调',
    }
    
    return mappings.get(key_str, 'C大调')


async def convert_to_midi(state: MusicState, config: RunnableConfig) -> dict:
    """
    将作曲转换为 MIDI 文件并返回下载链接。
    
    功能升级：
    1. 使用正则解析 LLM 输出，过滤“话痨”文本。
    2. 根据用户要求的 duration 循环填充旋律。
    3. 增加容错逻辑。
    """
    logger.info(f"--- [CONVERT TO MIDI] ---")
    
    style = state.get("style", "古典")
    mood = state.get("mood", "平静")
    key = state.get("key", "C大调")
    duration = state.get("duration", 30) or 30
    
    parsed_key = _parse_key(key)
    
    # 获取生成的内容
    melody_str = state.get("melody", "")
    harmony_str = state.get("harmony", "")
    rhythm_str = state.get("rhythm", "")
    
    if not MUSIC21_AVAILABLE:
        logger.warning("music21 库未安装")
        return {
            "messages": [AIMessage(content="无法生成 MIDI (Missing music21)")],
            "midi_file": None
        }
    
    try:
        # 创建输出目录
        MIDI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 创建 music21 乐谱
        piece = music21.stream.Score()
        tempo_val = 120 if mood in ['欢快', '激昂'] else 60
        piece.insert(0, music21.tempo.MetronomeMark(number=tempo_val))
        
        # --- 解析旋律 ---
        # 提取形如 C4, D#5, E-3 的音符
        melody_notes_str = re.findall(r"([A-G][b#-]?[0-9])", melody_str)
        if not melody_notes_str:
            logger.warning("未解析到有效旋律，使用随机兜底")
            scale = SCALES.get(parsed_key, SCALES['C大调'])
            melody_notes_str = [random.choice(scale) + '4' for _ in range(8)]
            
        # --- 解析和声 ---
        # 提取形如 C4-E4-G4 的和弦
        harmony_chords_str = re.findall(r"([A-G][b#-]?[0-9](?:-[A-G][b#-]?[0-9])+)", harmony_str)
        if not harmony_chords_str:
            chord_notes = CHORDS.get(parsed_key, CHORDS['C大调'])
            harmony_chords_str = ['-'.join(chord_notes)] * 4

        # --- 计算循环次数 ---
        # 假设每个音符约 0.5 秒 (120BPM quarter=0.5s, 60BPM quarter=1s)
        # 简单估算：120 BPM 下，一拍 0.5s。
        seconds_per_beat = 60 / tempo_val
        total_beats_needed = duration / seconds_per_beat
        
        # 假设 LLM 生成的 motif 长度为 N 个音符，每个音符默认 1 拍
        motif_len = len(melody_notes_str)
        loops = int(total_beats_needed / max(motif_len, 1)) + 1
        
        logger.info(f"目标时长: {duration}s, 速度: {tempo_val} BPM, 需要拍数: {total_beats_needed}, 循环次数: {loops}")

        # --- 构建旋律声部 ---
        melody_part = music21.stream.Part()
        melody_part.id = 'melody'
        
        current_beat = 0
        for _ in range(loops):
            for note_name in melody_notes_str:
                if current_beat >= total_beats_needed:
                    break
                try:
                    n = music21.note.Note(note_name)
                    n.quarterLength = 1.0 # 默认一拍
                    melody_part.append(n)
                    current_beat += 1
                except:
                    pass
        
        # --- 构建和声声部 ---
        harmony_part = music21.stream.Part()
        harmony_part.id = 'harmony'
        
        current_beat = 0
        harmony_len = len(harmony_chords_str)
        for i in range(loops * 4): # 和声通常比旋律慢，稍微多循环一点以防万一
             if current_beat >= total_beats_needed:
                    break
             chord_str = harmony_chords_str[i % harmony_len]
             try:
                 notes = chord_str.split('-')
                 c = music21.chord.Chord(notes)
                 c.quarterLength = 2.0 # 和弦占2拍
                 harmony_part.append(c)
                 current_beat += 2
             except:
                 pass

        piece.append(melody_part)
        piece.append(harmony_part)
        
        # 保存文件
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"composition_{timestamp}_{file_id}.mid"
        filepath = MIDI_OUTPUT_DIR / filename
        piece.write('midi', fp=str(filepath))
        
        base_url = settings.BASE_URL.replace("0.0.0.0", "localhost")
        download_url = f"{base_url}{settings.STATIC_URL}/music/{filename}"
        
        return {
            "messages": [AIMessage(content=(
                f"🎵 **您的音乐作品已完成！**\n\n"
                f"**风格**: {style}\n"
                f"**情绪**: {mood}\n"
                f"**调式**: {parsed_key}\n"
                f"**时长**: 约 {duration} 秒\n\n"
                f"---\n"
                f"📥 **[点击此处下载 MIDI 文件]({download_url})**"
            ))],
            "midi_file": str(filepath),
            # 清理状态
            "style": None, "mood": None, "key": None, "duration": None,
            "melody": None, "harmony": None, "rhythm": None, "composition": None, 
            "musician_input": None
        }

    except Exception as e:
        logger.error(f"MIDI 转换失败: {e}", exc_info=True)
        return {
             "messages": [AIMessage(content=f"生成出错: {str(e)}")],
             "midi_file": None,
             "style": None, "mood": None, "key": None, "duration": None
        }


# =============================================================================
# 路由逻辑 (Routing Logic)
# =============================================================================
def route_by_completeness(state: MusicState) -> Literal["complete", "incomplete"]:
    """
    根据是否所有必填字段都已填充来进行路由。
    
    Args:
        state: 当前状态
    
    Returns:
        "complete" 或 "incomplete"
    """
    missing = get_missing_fields(state)
    if missing:
        logger.info(f"缺失字段: {missing} -> incomplete")
        return "incomplete"
    logger.info("所有字段完整 -> 开始生成音乐")
    return "complete"


# =============================================================================
# 图构建 (Graph Construction)
# =============================================================================
workflow = StateGraph(MusicState)

# 添加节点
workflow.add_node("prepare_long_term", prepare_long_term_entry)
workflow.add_node("extract_info", extract_info)
workflow.add_node("ask_missing", ask_missing_info)
workflow.add_node("generate_melody", generate_melody)
workflow.add_node("generate_harmony", generate_harmony)
workflow.add_node("generate_rhythm", generate_rhythm)
workflow.add_node("convert_to_midi", convert_to_midi)

workflow.add_edge(START, "prepare_long_term")
workflow.add_edge("prepare_long_term", "extract_info")

# 添加条件边：提取信息后根据完整性路由
workflow.add_conditional_edges(
    "extract_info",
    route_by_completeness,
    {
        "complete": "generate_melody",
        "incomplete": "ask_missing"
    }
)

# 询问后返回提取节点处理新输入
workflow.add_edge("ask_missing", "extract_info")

# 音乐生成流水线
workflow.add_edge("generate_melody", "generate_harmony")
workflow.add_edge("generate_harmony", "generate_rhythm")
workflow.add_edge("generate_rhythm", "convert_to_midi")

# 结束
workflow.add_edge("convert_to_midi", END)

# 编译图
music_composer_agent = workflow.compile().with_config({'recursion_limit': 15})

# 生成流程图（可选，用于调试）
# try:
#     graph_obj = music_composer_agent.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     graph_path = Path(__file__).parent / 'state_graph_music_composer.png'
#     with open(graph_path, 'wb') as f:
#         f.write(pic)
#     logger.info(f"流程图已保存: {graph_path}")
# except Exception as e:
#     logger.warning(f"无法生成流程图: {e}")
