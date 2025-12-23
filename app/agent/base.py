from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        # 检查并初始化LLM
        if self.llm is None or not isinstance(self.llm, LLM):
            logger.info(f"Initializing LLM for agent '{self.name}'")
            self.llm = LLM(config_name=self.name.lower())
        # 检查并初始化内存
        if not isinstance(self.memory, Memory):
            logger.info(f"Initializing memory for agent '{self.name}'")
            self.memory = Memory()
        logger.debug(f"Agent '{self.name}' initialized successfully")
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            logger.error(f"Invalid state transition attempted: {new_state}")
            raise ValueError(f"Invalid state: {new_state}")

        # 记录状态转换
        previous_state = self.state
        logger.info(f"Agent '{self.name}' transitioning from {previous_state} to {new_state}")
        self.state = new_state
        try:
            yield
        except Exception as e:
            # 错误时转换到ERROR状态
            logger.error(f"Error in state {new_state}: {str(e)}", exc_info=True)
            self.state = AgentState.ERROR
            raise e
        finally:
            # 恢复到之前的状态
            logger.debug(f"Agent '{self.name}' reverting to state {previous_state}")
            self.state = previous_state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            logger.error(f"Unsupported message role: {role}")
            raise ValueError(f"Unsupported message role: {role}")

        content_preview = content[:100] + "..." if len(content) > 100 else content
        logger.debug(f"Adding {role} message to memory: {content_preview}")
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))
        logger.debug(f"Memory now contains {len(self.memory.messages)} messages")

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            logger.error(f"Cannot run agent from state: {self.state}")
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        logger.info(f"Starting agent '{self.name}' execution")
        if request:
            logger.debug(f"Initial request: {request}")
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                if self.is_stuck():
                    logger.warning(f"Agent '{self.name}' detected stuck state at step {self.current_step}")
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                logger.warning(f"Agent '{self.name}' reached max steps ({self.max_steps})")
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
        
        logger.info(f"Agent '{self.name}' execution completed")
        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent '{self.name}' detected stuck state. Added strategy change prompt")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # 计算重复内容的出现次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        # 记录检测结果
        if duplicate_count >= self.duplicate_threshold:
            logger.debug(f"Agent '{self.name}' detected {duplicate_count} duplicate responses (threshold: {self.duplicate_threshold})")
        
        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
