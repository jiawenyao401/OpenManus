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
# 导入沙箱相关函数
from app.daytona.sandbox import create_sandbox, delete_sandbox
from app.daytona.tool_base import SandboxToolsBase
# 导入日志记录器
from app.logger import logger
# 导入提示词模板
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
# 导入工具相关类
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.mcp import MCPClients, MCPClientTool
# 导入沙箱工具集
from app.tool.sandbox.sb_browser_tool import SandboxBrowserTool
from app.tool.sandbox.sb_files_tool import SandboxFilesTool
from app.tool.sandbox.sb_shell_tool import SandboxShellTool
from app.tool.sandbox.sb_vision_tool import SandboxVisionTool


class SandboxManus(ToolCallAgent):
    """沙箱环境中的多功能通用代理，支持沙箱工具和MCP工具。
    
    SandboxManus在隔离的沙箱环境中运行，提供以下功能：
    - 在沙箱中执行浏览器自动化
    - 在沙箱中进行文件操作
    - 在沙箱中执行Shell命令
    - 在沙箱中进行视觉识别
    - 支持MCP（Model Context Protocol）工具集成
    """

    # 代理名称
    name: str = "SandboxManus"
    # 代理描述
    description: str = "A versatile agent that can solve various tasks using multiple sandbox-tools including MCP-based tools"

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

    # 可用工具集合，包含基础工具
    # 注：沙箱工具（浏览器、文件、Shell、视觉）在initialize_sandbox_tools中动态添加
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # 注释掉的工具是本地工具，在沙箱环境中由沙箱工具替代
            # PythonExecute(),
            # BrowserUseTool(),
            # StrReplaceEditor(),
            AskHuman(),           # 人工交互工具
            Terminate(),          # 终止工具
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
    # 沙箱链接映射表 {sandbox_id -> {"vnc": url, "website": url}}
    sandbox_link: Optional[dict[str, dict[str, str]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def initialize_helper(self) -> "SandboxManus":
        """同步初始化基础组件（浏览器上下文助手）。
        
        这个方法在Pydantic模型验证完成后调用，用于初始化
        不能异步初始化的组件。
        """
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "SandboxManus":
        """工厂方法：创建并正确初始化SandboxManus实例。
        
        这个方法确保所有异步初始化（MCP服务器和沙箱工具）
        都在实例创建后完成。
        
        Args:
            **kwargs: 传递给SandboxManus构造函数的参数
            
        Returns:
            完全初始化的SandboxManus实例
        """
        instance = cls(**kwargs)
        # 初始化MCP服务器连接
        await instance.initialize_mcp_servers()
        # 初始化沙箱工具
        await instance.initialize_sandbox_tools()
        # 标记为已初始化
        instance._initialized = True
        return instance

    async def initialize_sandbox_tools(
        self,
        password: str = config.daytona.VNC_password,
    ) -> None:
        """初始化沙箱工具并创建沙箱环境。
        
        这个方法会：
        1. 创建一个新的沙箱实例
        2. 获取VNC和网站预览链接
        3. 创建沙箱工具（浏览器、文件、Shell、视觉）
        4. 将这些工具添加到可用工具集合中
        
        Args:
            password: VNC连接密码，从配置中获取
            
        Raises:
            ValueError: 如果未提供密码
            Exception: 沙箱创建或工具初始化失败
        """
        try:
            # 验证密码是否提供
            if not password:
                raise ValueError("password must be provided")
            
            # 创建新沙箱实例
            sandbox = create_sandbox(password=password)
            self.sandbox = sandbox
            
            # 获取VNC预览链接（端口6080用于VNC访问）
            vnc_link = sandbox.get_preview_link(6080)
            # 获取网站预览链接（端口8080用于Web访问）
            website_link = sandbox.get_preview_link(8080)
            
            # 提取URL字符串
            vnc_url = vnc_link.url if hasattr(vnc_link, "url") else str(vnc_link)
            website_url = (
                website_link.url if hasattr(website_link, "url") else str(website_link)
            )

            # 获取沙箱ID
            actual_sandbox_id = sandbox.id if hasattr(sandbox, "id") else "new_sandbox"
            
            # 初始化沙箱链接映射表
            if not self.sandbox_link:
                self.sandbox_link = {}
            
            # 存储沙箱的访问链接
            self.sandbox_link[actual_sandbox_id] = {
                "vnc": vnc_url,
                "website": website_url,
            }
            
            # 记录访问链接
            logger.info(f"VNC URL: {vnc_url}")
            logger.info(f"Website URL: {website_url}")
            SandboxToolsBase._urls_printed = True
            
            # 创建沙箱工具集合
            sb_tools = [
                SandboxBrowserTool(sandbox),    # 沙箱浏览器工具
                SandboxFilesTool(sandbox),      # 沙箱文件操作工具
                SandboxShellTool(sandbox),      # 沙箱Shell命令工具
                SandboxVisionTool(sandbox),     # 沙箱视觉识别工具
            ]
            
            # 将沙箱工具添加到可用工具集合
            self.available_tools.add_tools(*sb_tools)

        except Exception as e:
            logger.error(f"Error initializing sandbox tools: {e}")
            raise

    async def initialize_mcp_servers(self) -> None:
        """初始化所有配置的MCP服务器连接。
        
        根据配置文件中的MCP服务器设置，连接到SSE或stdio类型的服务器。
        如果连接失败，会记录错误但继续处理其他服务器。
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
        # 根据连接方式选择连接方法
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
        # 保留所有非MCP工具（本地工具和沙箱工具）
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        # 添加剩余MCP服务器的工具
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def delete_sandbox(self, sandbox_id: str) -> None:
        """删除指定ID的沙箱。
        
        Args:
            sandbox_id: 要删除的沙箱ID
            
        Raises:
            Exception: 删除沙箱失败时抛出异常
        """
        try:
            # 调用删除沙箱函数
            await delete_sandbox(sandbox_id)
            logger.info(f"Sandbox {sandbox_id} deleted successfully")
            # 从沙箱链接映射表中移除
            if sandbox_id in self.sandbox_link:
                del self.sandbox_link[sandbox_id]
        except Exception as e:
            logger.error(f"Error deleting sandbox {sandbox_id}: {e}")
            raise e

    async def cleanup(self):
        """清理SandboxManus代理的所有资源。
        
        包括：
        - 关闭浏览器连接
        - 断开所有MCP服务器连接
        - 删除沙箱环境
        """
        # 清理浏览器资源
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        
        # 仅在已初始化时进行清理
        if self._initialized:
            # 断开所有MCP服务器
            await self.disconnect_mcp_server()
            # 删除沙箱
            await self.delete_sandbox(self.sandbox.id if self.sandbox else "unknown")
            # 标记为未初始化
            self._initialized = False

    async def think(self) -> bool:
        """处理当前状态并决定下一步行动，使用适当的上下文。
        
        这是代理的核心思考逻辑，会：
        1. 确保MCP服务器已初始化
        2. 检查最近是否使用了沙箱浏览器工具
        3. 如果使用了浏览器，动态调整提示词以包含浏览器上下文
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
        
        # 获取最近3条消息用于分析
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        
        # 检查最近是否使用了沙箱浏览器工具
        browser_in_use = any(
            tc.function.name == SandboxBrowserTool().name
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
