"""MCP 客户端：从独立 MCP server 按协议发现并调用工具。

用 langchain-mcp-adapters 的 MultiServerMCPClient 启动一个 stdio 子进程 server，
发现它暴露的工具，返回 LangChain BaseTool 列表——可直接和原生工具一起 bind_tools。

三个关键点（面试可讲）：
1. MCP 是 async，LangGraph 的节点是 sync → 用「常驻事件循环（后台线程）」桥接，
   把所有 MCP 协程派到同一个 loop 上执行，避免 session 挂在已关闭的 loop 上而失效。
2. MCP 工具是纯异步的（StructuredTool 不支持同步 invoke），必须走 ainvoke。
3. client 必须保持在模块级存活（不能用 `async with` 在函数里用完就关），否则工具调不动。
"""
import asyncio
import sys
import threading
from functools import lru_cache

from langchain_mcp_adapters.client import MultiServerMCPClient


# Windows 坑：stdio 子进程需要 SelectorEventLoop。默认的 ProactorEventLoop 不支持
# 带 stdin 管道的 subprocess，会抛 NotImplementedError。必须在任何 loop 创建之前设好。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# stdio 传输：客户端把 server 当子进程拉起，通过标准输入输出按协议通信
# 坑：command 必须用 sys.executable（当前解释器的完整路径），不能写裸 "python"。
#     裸 "python" 走系统 PATH，可能指到没装 mcp 的环境，server 一起就 ModuleNotFoundError。
_MCP_SERVERS = {
    "time": {
        "command": sys.executable,
        "args": ["-m", "app.mcp_server"],
        "transport": "stdio",
    }
}

# 模块级 client：保持引用，底层 session 才能存活到后续工具调用
_client = MultiServerMCPClient(_MCP_SERVERS)

# 常驻事件循环：MCP 是 async，但 LangGraph 节点是 sync。
# 用一个后台线程跑一个永不退出的 loop，把「发现工具」「调用工具」都派到它上面，
# 保证 MCP session 始终挂在同一个 loop 上，跨节点调用不失效。
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def _run(coro):
    """把协程派到常驻 loop 上执行，同步等待结果。"""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=60)


@lru_cache(maxsize=1)
def get_mcp_tools() -> list:
    """启动 MCP server、发现工具。结果缓存，只启动一次。

    返回的是 LangChain BaseTool 列表，可直接塞进 bind_tools，
    和 app/tools.py 里的原生工具完全同一套接口。
    """
    return _run(_client.get_tools())


def call_tool(tool, args) -> str:
    """调用任意工具（原生 sync 或 MCP async 都行），统一返回纯文本结果。

    MCP 工具只支持 ainvoke（StructuredTool 不支持同步 invoke），
    所以统一走 ainvoke 并派到常驻 loop——原生工具也会被 LangChain
    包装成能在同一 loop 上跑的协程，两边接口一致。
    """
    return _to_text(_run(tool.ainvoke(args)))


def _to_text(result) -> str:
    """把工具结果统一成纯文本。

    原生工具返回 str；MCP 工具返回「内容块列表」
    （[{'type':'text','text':...}, ...]），这里抽成纯文本再喂给生成模型。
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            parts.append(item["text"] if isinstance(item, dict) and "text" in item else str(item))
        return "\n".join(parts)
    return str(result)
