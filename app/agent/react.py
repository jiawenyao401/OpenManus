import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent, ABC):
    """ReAct（推理+行动）代理基类
    
    实现了思考-行动循环的抽象代理框架，支持多步骤任务执行。
    子类需要实现 think() 和 act() 方法来定义具体的推理和行动逻辑。
    """
    
    # 代理基本信息
    name: str  # 代理名称
    description: Optional[str] = None  # 代理描述

    # 提示词配置
    system_prompt: Optional[str] = None  # 系统级提示词
    next_step_prompt: Optional[str] = None  # 下一步行动的提示词

    # 核心组件
    llm: Optional[LLM] = Field(default_factory=LLM)  # 大语言模型实例
    memory: Memory = Field(default_factory=Memory)  # 代理记忆存储
    state: AgentState = AgentState.IDLE  # 代理当前状态 空闲

    # 执行控制
    max_steps: int = 10  # 最大执行步数
    current_step: int = 0  # 当前执行步数

    @abstractmethod
    async def think(self) -> bool:
        """思考阶段：分析当前状态并决定是否需要执行行动
        
        Returns:
            bool: True 表示需要执行行动，False 表示思考完成无需行动
        """

    @abstractmethod
    async def act(self) -> str:
        """行动阶段：执行思考阶段决定的具体操作
        
        Returns:
            str: 行动执行的结果或反馈信息
        """

    async def step(self) -> str:
        """执行单个步骤：完整的思考-行动循环
        
        该方法实现了 ReAct 框架的核心循环：
        1. 增加步数计数
        2. 调用 think() 进行推理
        3. 根据推理结果决定是否执行 act()
        4. 返回执行结果或完成信号
        
        Returns:
            str: 步骤执行的结果信息
            
        Raises:
            Exception: 执行过程中发生的任何异常
        """
        self.current_step += 1
        logger.info(f"[{self.name}] 开始执行第 {self.current_step}/{self.max_steps} 步")
        
        try:
            # 记录当前代理状态
            logger.debug(f"[{self.name}] 当前状态: {self.state.value}")
            
            # 执行思考阶段
            should_act = await self.think()
            
            # 如果思考结果表示无需行动，直接返回
            if not should_act:
                logger.info(f"[{self.name}] 思考完成 - 无需执行操作")
                return "Thinking complete - no action needed"
            
            # 执行行动阶段
            logger.info(f"[{self.name}] 开始执行操作")
            result = await self.act()
            logger.info(f"[{self.name}] 操作执行完成")
            return result
        except Exception as e:
            # 捕获并记录执行过程中的异常
            logger.error(f"[{self.name}] 执行步骤时出错: {str(e)}", exc_info=True)
            raise
