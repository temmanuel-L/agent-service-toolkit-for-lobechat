"""
响应示例模块
"""
from typing import Any

from fastapi import status


def _sse_response_example() -> dict[int | str, Any]:
    """
    生成服务器发送事件(SSE)响应示例

    该函数用于为流式API端点生成标准的SSE响应示例，展示返回数据的格式
    主要用于OpenAPI文档中显示流式响应的示例

    Returns:
        dict[int | str, Any]: 包含HTTP状态码和对应响应示例的字典
            - 键为HTTP状态码(200)
            - 值为包含描述、内容类型和示例数据的嵌套字典
    """
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }