"""统一读取项目配置（.env 中的密钥与参数）。

原则：所有可配置的值都从这里取，其它模块不直接 os.getenv、不硬编码。
好处是配置集中、好维护，测试时也容易替换。
"""
import os

from dotenv import load_dotenv

# 在导入本模块时立即加载 .env，保证后续 os.getenv 能读到
load_dotenv()


class Settings:
    """项目配置。只读，集中管理所有可调参数。"""

    # ---- DeepSeek 主模型 ----
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ---- 本地 Embedding（BGE）----
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

     # ---- 重排（cross-encoder）----
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

    # ---- 文档切块 ----
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ---- 检索 ----
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    TOP_K_RECALL: int = int(os.getenv("TOP_K_RECALL", "8"))   # 召回多、精排少

    # ---- 向量库 ----
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", ".chroma")


settings = Settings()
