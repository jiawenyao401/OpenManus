# 配置日志系统
import logging  # 日志模块
import sys  # 系统模块

# 配置基础日志处理器，输出到标准错误流
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])

# 导入其他必要模块
import argparse  # 命令行参数解析
import asyncio  # 异步编程
import atexit  # 程序退出时执行清理
import json  # JSON序列化
from inspect import Parameter, Signature  # 用于构建函数签名
from typing import Any, Dict, Optional  # 类型提示

# 导入MCP服务器框架
from mcp.server.fastmcp import FastMCP

# 导入应用内部模块
from app.logger import logger  # 应用日志记录器
from app.tool.base import BaseTool  # 工具基类
from app.tool.bash import Bash  # Bash命令执行工具
from app.tool.browser_use_tool import BrowserUseTool  # 浏览器自动化工具
from app.tool.str_replace_editor import StrReplaceEditor  # 文本编辑工具
from app.tool.terminate import Terminate  # 程序终止工具


class MCPServer:
    """模型上下文协议(MCP)服务器实现
    
    主要功能：
    1. 管理工具的注册和执行
    2. 提供工具参数验证和文档生成
    3. 处理工具执行结果的序列化
    4. 管理服务器资源的清理
    """

    def __init__(self, name: str = "openmanus"):
        """初始化MCP服务器
        
        Args:
            name: 服务器名称，默认为"openmanus"
        """
        # 创建快速与MCP服务器实例
        self.server = FastMCP(name)
        # 存储已注册的工具
        self.tools: Dict[str, BaseTool] = {}

        # 初始化标准工具
        self.tools["bash"] = Bash()  # Bash命令执行工具
        self.tools["browser"] = BrowserUseTool()  # 浏览器自动化工具
        self.tools["editor"] = StrReplaceEditor()  # 文本编辑工具
        self.tools["terminate"] = Terminate()  # 程序终止工具

    def register_tool(self, tool: BaseTool, method_name: Optional[str] = None) -> None:
        """注册工具并进行参数验证和文档生成
        
        流程：
        1. 提取工具的参数信息
        2. 构建异步执行函数
        3. 构建函数签名和文档
        4. 注册到服务器
        
        Args:
            tool: 要注册的工具对象
            method_name: 方法名称，默认为工具名称
        """
        # 确定方法名称
        tool_name = method_name or tool.name
        # 获取工具的参数定义
        tool_param = tool.to_param()
        tool_function = tool_param["function"]

        # 定义异步执行函数
        async def tool_method(**kwargs):
            # 记录工具执行信息
            logger.info(f"Executing {tool_name}: {kwargs}")
            # 执行工具
            result = await tool.execute(**kwargs)

            # 记录执行结果
            logger.info(f"Result of {tool_name}: {result}")

            # 处理不同类型的结果
            if hasattr(result, "model_dump"):
                # 如果是Pydantic模型，转换为JSON
                return json.dumps(result.model_dump())
            elif isinstance(result, dict):
                # 如果是字典，直接序列化
                return json.dumps(result)
            # 其他类型直接返回
            return result

        # 设置方法元数据
        tool_method.__name__ = tool_name
        tool_method.__doc__ = self._build_docstring(tool_function)
        tool_method.__signature__ = self._build_signature(tool_function)

        # 存储参数模式（供工具程序化访问）
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        tool_method._parameter_schema = {
            param_name: {
                "description": param_details.get("description", ""),
                "type": param_details.get("type", "any"),
                "required": param_name in required_params,
            }
            for param_name, param_details in param_props.items()
        }

        # 注册到服务器
        self.server.tool()(tool_method)
        logger.info(f"Registered tool: {tool_name}")

    def _build_docstring(self, tool_function: dict) -> str:
        """从工具函数元数据构建格式化文档字符串
        
        Args:
            tool_function: 工具函数元数据字典
            
        Returns:
            str: 格式化的文档字符串
        """
        # 获取工具描述
        description = tool_function.get("description", "")
        # 获取参数属性
        param_props = tool_function.get("parameters", {}).get("properties", {})
        # 获取必填参数
        required_params = tool_function.get("parameters", {}).get("required", [])

        # 构建文档字符串
        docstring = description
        if param_props:
            docstring += "\n\nParameters:\n"
            for param_name, param_details in param_props.items():
                # 标记是否是必填参数
                required_str = (
                    "(required)" if param_name in required_params else "(optional)"
                )
                # 获取参数类型
                param_type = param_details.get("type", "any")
                # 获取参数描述
                param_desc = param_details.get("description", "")
                docstring += (
                    f"    {param_name} ({param_type}) {required_str}: {param_desc}\n"
                )

        return docstring

    def _build_signature(self, tool_function: dict) -> Signature:
        """从工具函数元数据构建函数签名
        
        Args:
            tool_function: 工具函数元数据字典
            
        Returns:
            Signature: Python函数签名对象
        """
        # 获取参数属性
        param_props = tool_function.get("parameters", {}).get("properties", {})
        # 获取必填参数
        required_params = tool_function.get("parameters", {}).get("required", [])

        # 参数列表
        parameters = []

        # 遍历每个参数并构建签名
        for param_name, param_details in param_props.items():
            # 获取参数类型
            param_type = param_details.get("type", "")
            # 判断是否是必填参数
            default = Parameter.empty if param_name in required_params else None

            # 将JSON Schema类型映射到Python类型
            annotation = Any
            if param_type == "string":
                annotation = str
            elif param_type == "integer":
                annotation = int
            elif param_type == "number":
                annotation = float
            elif param_type == "boolean":
                annotation = bool
            elif param_type == "object":
                annotation = dict
            elif param_type == "array":
                annotation = list

            # 构建参数对象
            param = Parameter(
                name=param_name,
                kind=Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
            parameters.append(param)

        return Signature(parameters=parameters)

    async def cleanup(self) -> None:
        """清理服务器资源
        
        主要清理浏览器工具的资源。
        """
        logger.info("Cleaning up resources")
        # 仅清理浏览器工具的资源
        if "browser" in self.tools and hasattr(self.tools["browser"], "cleanup"):
            await self.tools["browser"].cleanup()

    def register_all_tools(self) -> None:
        """注册所有工具到服务器
        
        遍历工具字典并注册每个工具。
        """
        for tool in self.tools.values():
            self.register_tool(tool)

    def run(self, transport: str = "stdio") -> None:
        """运行MCP服务器
        
        流程：
        1. 注册所有工具
        2. 注册程序退出时的清理函数
        3. 启动服务器
        
        Args:
            transport: 传输方式，默认为"stdio"
        """
        # 注册所有工具
        self.register_all_tools()

        # 注册程序退出时的清理函数
        atexit.register(lambda: asyncio.run(self.cleanup()))

        # 启动服务器
        logger.info(f"Starting OpenManus server ({transport} mode)")
        self.server.run(transport=transport)


def parse_args() -> argparse.Namespace:
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="OpenManus MCP Server")
    # 添加传输方式参数
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Communication method: stdio or http (default: stdio)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 创建并运行MCP服务器
    server = MCPServer()
    server.run(transport=args.transport)
