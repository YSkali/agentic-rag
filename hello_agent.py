"""最小验证：确认 DeepSeek API 能通。

跑通后本文件可删除或保留作参考。真正项目代码会放到 app/ 目录下。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # 读取 .env 里的密钥和配置

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

resp = llm.invoke("用一句话说明什么是 RAG。")
print(resp.content)
