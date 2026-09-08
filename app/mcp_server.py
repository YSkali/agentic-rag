"""一个极简的 MCP Server：用 FastMCP 暴露一个「查当前时间」工具。

它和 agent 是独立的进程，按 MCP 协议暴露工具，被 app/mcp_client.py 发现并调用。
这里刻意写得自包含（不 import agent 的任何代码）——MCP 的意义就是「工具提供方和调用方解耦」：
这个 server 理论上可以换成任何语言、部署在任何机器上，agent 这边的接线一行都不用改。

启动方式：python -m app.mcp_server（stdio 传输，由 MCP 客户端作为子进程拉起）
"""
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# 给 server 起个名字，客户端按它识别
mcp = FastMCP("time-server")


@mcp.tool()
def get_current_time() -> str:
    """返回当前日期和时间（精确到秒），例如 '2026-09-08 14:30:05'。无需参数。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    mcp.run()
