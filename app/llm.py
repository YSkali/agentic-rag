"""DeepSeek 主模型封装。"""
from langchain_openai import ChatOpenAI
#导入 LangChain 中专门对接 OpenAI API 格式（DeepSeek 兼容）的聊天模型类。
from app.config import settings
#从项目的配置中心（通常是 Pydantic BaseSettings）拉取全局配置对象

def get_llm() -> ChatOpenAI:        #定义一个工厂函数，每次调用它都会“生产”一个 LLM 实例。
    """返回配置好的 DeepSeek 大模型实例。"""
    return ChatOpenAI(
        #实例化模型对象，传入的三个参数分别指定：
        # 用哪个模型、密钥是什么、请求发往哪个地址（DeepSeek 的网关）。
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )
