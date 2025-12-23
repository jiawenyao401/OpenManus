# 导入必要的标准库和第三方库
import json  # 用于JSON序列化和反序列化
import time  # 用于时间戳生成
from enum import Enum  # 用于定义枚举类型
from typing import Dict, List, Optional, Union  # 类型提示

from pydantic import Field, ConfigDict  # Pydantic字段定义和配置

# 导入应用内部模块
from app.agent.base import BaseAgent  # 基础代理类
from app.flow.base import BaseFlow  # 基础流程类
from app.llm import LLM  # 大语言模型接口
from app.logger import logger  # 日志记录器
from app.schema import AgentState, Message, ToolChoice  # 数据模型和枚举
from app.tool import PlanningTool  # 规划工具


class PlanStepStatus(str, Enum):
    """计划步骤状态枚举类
    
    定义了计划中每个步骤可能的状态，用于跟踪步骤的执行进度。
    """

    # 步骤状态常量
    NOT_STARTED = "not_started"  # 未开始
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"  # 已完成
    BLOCKED = "blocked"  # 被阻止

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """获取所有可能的步骤状态值列表
        
        Returns:
            list[str]: 包含所有状态值的列表
        """
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """获取活跃状态列表（未开始或进行中）
        
        活跃状态表示步骤还需要继续执行。
        
        Returns:
            list[str]: 包含活跃状态值的列表
        """
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """获取状态标记符号映射
        
        为每个状态提供对应的可视化符号，用于在文本中显示步骤状态。
        
        Returns:
            Dict[str, str]: 状态值到符号的映射字典
        """
        return {
            cls.COMPLETED.value: "[✓]",  # 已完成：勾号
            cls.IN_PROGRESS.value: "[→]",  # 进行中：箭头
            cls.BLOCKED.value: "[!]",  # 被阻止：感叹号
            cls.NOT_STARTED.value: "[ ]",  # 未开始：空方框
        }


class PlanningFlow(BaseFlow):
    """规划流程类
    
    管理任务的规划和执行。主要功能：
    1. 使用LLM根据用户输入创建计划
    2. 不断获取下一个步骤并执行
    3. 跟踪每个步骤的执行状态
    4. 最后总结计划的执行结果
    """

    # 大语言模型实例，用于与模型交互
    llm: LLM
    # 规划工具实例，管理计划的存储和操作
    planning_tool: PlanningTool
    # 执行器代理的键值列表，指定哪些代理可以执行步骤
    executor_keys: List[str]
    # 当前活跃计划的ID，使用时间戳作为默认值
    active_plan_id: str
    # 当前步骤的索引，None表示没有步骤正在执行
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        """初始化PlanningFlow
        
        处理下列参数：
        - executors: 执行器代理的键值列表
        - plan_id: 计划ID
        - planning_tool: 规划工具实例
        
        Args:
            agents: 代理对象，可以是单个、列表或字典
            **data: 其他配置参数
        """
        # 处理executors参数，将其转换为executor_keys
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 处理plan_id参数，将其转换为active_plan_id
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 如果没有提供规划工具，为其创建一个新实例
        if "planning_tool" not in data:
            data["planning_tool"] = PlanningTool()
        
        # 如果没有提供LLM，为其创建一个新实例
        if "llm" not in data:
            data["llm"] = LLM()
        
        # 如果没有提供active_plan_id，为其生成一个
        if "active_plan_id" not in data:
            data["active_plan_id"] = f"plan_{int(time.time())}"
        
        # 如果没有提供executor_keys，为其创建一个空列表
        if "executor_keys" not in data:
            data["executor_keys"] = []

        # 调用父类的初始化方法
        super().__init__(agents, **data)

        # 如果没有指定执行器，使用所有代理作为执行器
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """获取执行器代理
        
        根据步骤类型选择合适的代理。优先级：
        1. 如果步骤类型匹配代理键，使用该代理
        2. 使用执行器列表中的第一个代理
        3. 最后回退到主代理
        
        Args:
            step_type: 步骤类型，用于选择特定的代理
            
        Returns:
            BaseAgent: 选中的执行器代理
        """
        # 如果步骤类型提供了且匹配代理键，使用该代理
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 使用执行器列表中的第一个可用代理
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 最后回退到主代理
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """执行规划流程
        
        主要流程：
        1. 根据输入创建计划
        2. 循环获取下一个步骤
        3. 使用合适的代理执行步骤
        4. 更新步骤状态
        5. 最后总结计划
        
        Args:
            input_text: 用户输入的任务描述
            
        Returns:
            str: 执行结果
        """
        try:
            # 检查是否有主代理
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # 如果提供了输入，根据其创建计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 验证计划是否成功创建
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            # 累积执行结果
            result = ""
            while True:
                # 获取当前步骤信息
                self.current_step_index, step_info = await self._get_current_step_info()

                # 如果没有更多步骤或计划已完成，退出循环
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 使用合适的代理执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 检查代理是否要求终止
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """创建初始计划
        
        根据用户输入，使用LLM和规划工具创建计划。
        流程：
        1. 构建系统提示词
        2. 收集执行器代理的描述
        3. 调用LLM生成计划
        4. 执行规划工具或创建默认计划
        
        Args:
            request: 用户的任务输入
        """
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # 构建系统提示词，指寺LLM如何创建计划
        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )
        # 收集执行器代理的描述
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {
                        "name": key.upper(),
                        "description": self.agents[key].description,
                    }
                )
        # 如果有多个代理，将其信息添加到系统提示词
        if len(agents_description) > 1:
            system_message_content += (
                f"\nNow we have {agents_description} agents. "
                f"The infomation of them are below: {json.dumps(agents_description)}\n"
                "When creating steps in the planning tool, please specify the agent names using the format '[agent_name]'."
            )

        # 创建系统消息
        system_message = Message.system_message(system_message_content)

        # 创建用户消息，包含任务输入
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        # 调用LLM与规划工具交互
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 处理LLM返回的工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析工具参数
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {args}")
                            continue

                    # 确保计划ID正确且执行工具
                    args["plan_id"] = self.active_plan_id

                    # 执行规划工具
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"Plan creation result: {str(result)}")
                    return

        # 如果没有成功执行，创建默认计划
        logger.warning("Creating default plan")

        # 使用规划工具创建默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """获取当前步骤信息
        
        从计划中找到第一个未完成的步骤。
        流程：
        1. 验证计划是否存在
        2. 遍历步骤列表，找到第一个活跃步骤
        3. 提取步骤类型信息
        4. 标记步骤为进行中
        
        Returns:
            tuple[Optional[int], Optional[dict]]: 返回步骤索引和步骤信息，没有活跃步骤时返回(None, None)
        """
        # 检查计划ID是否有效且计划是否存在
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # 从规划工具存储中直接访问计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])  # 步骤文本列表
            step_statuses = plan_data.get("step_statuses", [])  # 步骤状态列表

            # 遍历步骤，找到第一个活跃步骤（未开始或进行中）
            for i, step in enumerate(steps):
                # 判断步骤是否有对应的状态
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                # 如果是活跃步骤，返回其信息
                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤信息
                    step_info = {"text": step}

                    # 尝试从步骤文本中提取类型（例如[SEARCH]或[CODE]）
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 标记当前步骤为进行中
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        # 如果工具执行失败，直接更新存储中的数据
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            # 没有找到活跃步骤
            return None, None

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """执行单个步骤
        
        使用指定的代理执行当前步骤。
        流程：
        1. 构建包含计划状态的提示词
        2. 调用代理执行步骤
        3. 标记步骤为已完成
        
        Args:
            executor: 执行步骤的代理
            step_info: 步骤信息字典
            
        Returns:
            str: 步骤执行结果
        """
        # 构建包含当前计划状态的上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 为agent构建执行当前步骤的提示词
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        Please only execute this current step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        # 使用agent.run()执行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 成功执行后，标记步骤为已完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """标记当前步骤为已完成
        
        更新计划中当前步骤的状态为已完成。
        如果工具执行失败，会直接更新存储中的数据。
        """
        # 如果没有当前步骤索引，直接返回
        if self.current_step_index is None:
            return

        try:
            # 调用规划工具标记步骤为已完成
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")
            # 如果工具执行失败，直接更新存储中的数据
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保步骤状态列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新步骤状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """获取计划的文本表示
        
        优先使用规划工具获取，失败时使用存储中的数据直接生成。
        
        Returns:
            str: 格式化的计划文本
        """
        try:
            # 调用规划工具获取计划
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            # 失败时使用存储中的数据直接生成
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """从存储中直接生成计划文本
        
        当规划工具失败时，从内部存储中读取计划数据并格式化输出。
        
        Returns:
            str: 格式化的计划文本，包含进度、步骤状态等信息
        """
        try:
            # 检查计划是否存在
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            # 从存储中获取计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")  # 计划标题
            steps = plan_data.get("steps", [])  # 步骤列表
            step_statuses = plan_data.get("step_statuses", [])  # 步骤状态列表
            step_notes = plan_data.get("step_notes", [])  # 步骤笔记列表

            # 确保步骤状态和笔记列表与步骤数量一致
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 统计每个状态的步骤数
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            # 计算完成进度
            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            # 构建计划文本头部
            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            # 添加进度信息
            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            # 获取步骤状态符号映射
            status_marks = PlanStepStatus.get_status_marks()

            # 添加每个步骤的信息
            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # 使用状态符号表示步骤状态
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                # 如果有笔记，也添加到文本中
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _finalize_plan(self) -> str:
        """最终化计划
        
        计划执行完成后，使用LLM或代理生成总结。
        流程：
        1. 获取计划文本
        2. 优先使用LLM生成总结
        3. 失败时回退使用代理生成总结
        
        Returns:
            str: 计划完成总结
        """
        # 获取计划文本
        plan_text = await self._get_plan_text()

        # 使用LLM直接生成总结
        try:
            # 构建系统提示词
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to summarize the completed plan."
            )

            # 构建用户消息，要求总结计划
            user_message = Message.user_message(
                f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide a summary of what was accomplished and any final thoughts."
            )

            # 调用LLM生成总结
            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"Plan completed:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # 回退使用代理生成总结
            try:
                # 使用主代理生成总结
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "Plan completed. Error generating summary."
