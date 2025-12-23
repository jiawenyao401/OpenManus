# 导入类型提示
from typing import Dict, List, Optional

# 导入Pydantic用于数据验证
from pydantic import Field, model_validator

# 导入浏览器上下文助手
from app.agent.browser import BrowserContextHelper
# 导入工具调用代理基类
from app.agent.toolcall import ToolCallAgent
# 导入全局配置
from app.config import config
# 导入日志记录器
from app.logger import logger
# 导入提示词模板
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
# 导入工具相关类
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


class Manus(ToolCallAgent):
    """多功能通用代理，支持本地工具和MCP（Model Context Protocol）工具。
    
    Manus是一个功能强大的AI代理，能够：
    - 执行Python代码
    - 进行浏览器自动化
    - 编辑文件
    - 与用户交互
    - 连接和使用MCP服务器提供的工具
    """

    # 代理名称
    name: str = "Manus"
    # 代理描述
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    # 系统提示词，包含工作目录信息
    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    # 下一步行动的提示词
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 最大观察长度（用于限制输出）
    max_observe: int = 10000
    # 最大执行步数
    max_steps: int = 20

    # MCP客户端，用于访问远程工具
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # 可用工具集合，包含所有本地工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),          # Python代码执行工具
            BrowserUseTool(),         # 浏览器自动化工具
            StrReplaceEditor(),       # 文件编辑工具
            AskHuman(),               # 人工交互工具
            Terminate(),              # 终止工具
        )
    )

    # 特殊工具名称列表（如终止工具）
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    # 浏览器上下文助手，用于管理浏览器状态
    browser_context_helper: Optional[BrowserContextHelper] = None

    # 已连接的MCP服务器映射表 {server_id -> url/command}
    connected_servers: Dict[str, str] = Field(default_factory=dict)
    # 初始化标志，用于跟踪MCP服务器是否已初始化
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """同步初始化基础组件（浏览器上下文助手）。
        
        这个方法在Pydantic模型验证完成后调用，用于初始化
        不能异步初始化的组件。
        """
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """工厂方法：创建并正确初始化Manus实例。
        
        这个方法确保所有异步初始化（如MCP服务器连接）
        都在实例创建后完成。
        
        Args:
            **kwargs: 传递给Manus构造函数的参数
            
        Returns:
            完全初始化的Manus实例
        """
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """初始化所有配置的MCP服务器连接。
        
        根据配置文件中的MCP服务器设置，连接到SSE或stdio类型的服务器。
        如果连接失败，会记录错误但不会中断其他服务器的连接。
        """
        # 遍历配置中的所有MCP服务器
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                # 处理SSE（Server-Sent Events）类型的服务器
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                # 处理stdio（标准输入输出）类型的服务器
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                # 记录连接失败但继续处理其他服务器
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """连接到MCP服务器并添加其提供的工具。
        
        Args:
            server_url: 服务器URL或命令路径
            server_id: 服务器唯一标识符
            use_stdio: 是否使用stdio连接方式（否则使用SSE）
            stdio_args: stdio连接的命令行参数
        """
        # 根据连接类型选择连接方式
        if use_stdio:
            # 使用标准输入输出连接
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            # 使用SSE连接
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # 从该服务器获取新工具并添加到可用工具集合
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """断开与MCP服务器的连接并移除其工具。
        
        Args:
            server_id: 要断开的服务器ID。如果为空，则断开所有服务器。
        """
        # 断开MCP客户端连接
        await self.mcp_clients.disconnect(server_id)
        
        # 更新已连接服务器映射表
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # 重建可用工具集合，移除来自断开服务器的工具
        # 保留所有非MCP工具（本地工具）
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        # 添加剩余MCP服务器的工具
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """清理Manus代理的所有资源。
        
        包括关闭浏览器连接和断开所有MCP服务器连接。
        """
        # 清理浏览器资源
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        
        # 仅在已初始化时断开MCP服务器连接
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    async def think(self) -> bool:
        """处理当前状态并决定下一步行动，使用适当的上下文。
        
        这个方法是代理的核心思考逻辑，它会：n        1. 确保MCP服务器已初始化
        2. 检查是否正在使用浏览器工具
        3. 如果使用浏览器，动态调整提示词以包含浏览器上下文
        4. 执行父类的思考逻辑
        5. 恢复原始提示词
        
        Returns:
            bool: 思考过程是否成功
        """
        # 延迟初始化MCP服务器（如果还未初始化）
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        # 保存原始提示词
        original_prompt = self.next_step_prompt
        
        # 获取最近的3条消息用于分析
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        
        # 检查最近是否使用了浏览器工具
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        # 如果浏览器正在使用，更新提示词以包含浏览器上下文信息
        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        # 执行父类的思考逻辑
        result = await super().think()

        # 恢复原始提示词
        self.next_step_prompt = original_prompt

        return result
