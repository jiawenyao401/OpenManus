from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    """消息角色枚举
    
    定义对话中不同参与者的角色类型。
    """

    SYSTEM = "system"  # 系统消息
    USER = "user"  # 用户消息
    ASSISTANT = "assistant"  # 助手消息
    TOOL = "tool"  # 工具消息


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """工具选择枚举
    
    控制模型如何使用工具的选项。
    """

    NONE = "none"  # 不使用工具
    AUTO = "auto"  # 自动选择是否使用工具
    REQUIRED = "required"  # 必须使用工具


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """代理执行状态枚举
    
    表示代理在执行过程中的各种状态。
    """

    IDLE = "IDLE"  # 空闲状态
    RUNNING = "RUNNING"  # 运行中
    FINISHED = "FINISHED"  # 执行完成
    ERROR = "ERROR"  # 错误状态


class Function(BaseModel):
    """函数调用信息
    
    表示一个具体的函数调用的名称和参数。
    """
    name: str  # 函数名称
    arguments: str  # 函数参数（通常为 JSON 字符串）


class ToolCall(BaseModel):
    """工具调用信息
    
    表示消息中的一个工具或函数调用。
    """

    id: str  # 工具调用的唯一标识符
    type: str = "function"  # 调用类型，默认为 "function"
    function: Function  # 函数调用详情


class Message(BaseModel):
    """对话消息
    
    表示对话中的一条消息，包含角色、内容、工具调用等信息。
    """

    role: ROLE_TYPE = Field(...)  # type: ignore  # 消息角色（system/user/assistant/tool）
    content: Optional[str] = Field(default=None)  # 消息内容文本
    tool_calls: Optional[List[ToolCall]] = Field(default=None)  # 工具调用列表
    name: Optional[str] = Field(default=None)  # 消息发送者名称
    tool_call_id: Optional[str] = Field(default=None)  # 工具调用的 ID
    base64_image: Optional[str] = Field(default=None)  # Base64 编码的图像数据

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作
        
        允许消息与列表或其他消息进行加法操作，返回消息列表。
        """
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作
        
        允许列表与消息进行加法操作（反向操作）。
        """
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """将消息转换为字典格式
        
        Returns:
            dict: 包含消息所有字段的字典
        """
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建用户消息
        
        Args:
            content: 消息内容
            base64_image: 可选的 Base64 编码图像
            
        Returns:
            Message: 用户消息对象
        """
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """创建系统消息
        
        Args:
            content: 系统消息内容
            
        Returns:
            Message: 系统消息对象
        """
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """创建助手消息
        
        Args:
            content: 消息内容
            base64_image: 可选的 Base64 编码图像
            
        Returns:
            Message: 助手消息对象
        """
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建工具消息
        
        Args:
            content: 工具执行结果内容
            name: 工具名称
            tool_call_id: 对应的工具调用 ID
            base64_image: 可选的 Base64 编码图像
            
        Returns:
            Message: 工具消息对象
        """
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """从原始工具调用创建消息

        Args:
            tool_calls: 来自 LLM 的原始工具调用列表
            content: 可选的消息内容
            base64_image: 可选的 Base64 编码图像
            
        Returns:
            Message: 包含工具调用的助手消息对象
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    """对话记忆存储
    
    管理对话历史消息，支持消息的添加、清除和检索操作。
    """
    messages: List[Message] = Field(default_factory=list)  # 消息列表
    max_messages: int = Field(default=100)  # 最大保存消息数

    def add_message(self, message: Message) -> None:
        """添加单条消息到记忆
        
        Args:
            message: 要添加的消息对象
        """
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """添加多条消息到记忆
        
        Args:
            messages: 消息对象列表
        """
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """清空所有消息"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """获取最近的 n 条消息
        
        Args:
            n: 要获取的消息数量
            
        Returns:
            List[Message]: 最近的 n 条消息列表
        """
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """将所有消息转换为字典列表
        
        Returns:
            List[dict]: 消息字典列表
        """
        return [msg.to_dict() for msg in self.messages]
