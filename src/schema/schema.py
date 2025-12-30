from datetime import datetime
from typing import Any, Literal, NotRequired

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from typing_extensions import TypedDict

from schema.models import AllModelEnum, AnthropicModelName, OpenAIModelName


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum = Field(
        description="Default model used when none is specified.",
    )


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM Model to use for the agent. Defaults to the default model set in the settings of the service.",
        default=None,
        examples=[OpenAIModelName.GPT_5_NANO, AnthropicModelName.HAIKU_45],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):  # type: ignore[no-redef]
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    messages: list[ChatMessage]

class ConversationInput(BaseModel):
    """
    查询用户对话列表的参数

    Args:
        BaseModel (_type_): BaseModel继承
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    user_id: str = Field(
        description="用户ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent: str = Field(
        description="智能体",
        examples=["simulate_agent"],
    )


class ConversationsList(BaseModel):
    """
    查询用户的对话列表结果

    Args:
        BaseModel (_type_): BaseModel继承
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    user_id: str = Field(
        description="用户ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    thread_id: str = Field(
        description="线程ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    title: str = Field(
        description="会话主题",
        examples=["哈喽"],
        alias="titile",
    )
    last_message_preview: str | None = Field(
        default=None,
        description="摘要最近一条消息，方便列表展示",
        examples=["刚才聊到巡检注意事项……"],
    )
    updated_at: datetime | None = Field(
        default=None,
        description="最近一次互动的时间戳，用于排序",
        examples=["2024-05-01T12:30:00Z"],
    )


class DeleteConversationInput(ConversationInput):
    """
    删除用户对话输入参数

    Args:
        ConversationInput (_type_): ConversationInput 继承
    """

    thread_id: str | None = Field(
        default=None,
        description="对话ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class DeleteConversationOutput(BaseModel):
    """
    删除对话输出
    """

    status: Literal["success"] = Field(
        default="success",
        description="删除结果状态",
    )
    deleted_thread_ids: list[str] = Field(
        default_factory=list,
        description="被删除的对话ID列表",
        examples=[["847c6285-8fc9-4560-a83f-4e6285809254"]],
    )


class StopTaskInput(BaseModel):
    """
    停止任务输入参数

    Args:
        BaseModel (_type_): BaseModel继承
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    user_id: str = Field(
        description="用户ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    thread_id: str = Field(
        description="线程ID",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class StopTaskOutput(BaseModel):
    """
    停止任务输出

    Args:
        BaseModel (_type_): BaseModel 继承
    """

    status: Literal["success"] = Field(
        default="success",
        description="停止成功",
    )
    deleted_thread_ids: list[str] = Field(
        default_factory=list,
        description="被删除的对话ID列表",
        examples=[["847c6285-8fc9-4560-a83f-4e6285809254"]],
    )


# OpenAI API 兼容的数据模型
class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: str = "stop"


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: dict | None = None


class OpenAIStreamChoice(BaseModel):
    index: int
    delta: OpenAIChatMessage
    finish_reason: str | None = None


class OpenAIChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIStreamChoice]


# Webhook 数据模型
class WebhookPayload(BaseModel):
    event: str  # 事件类型，如 "thread.deleted", "user.deleted" 等
    thread_id: str | None = None
    user_id: str | None = None