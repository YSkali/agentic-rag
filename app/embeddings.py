"""本地 Embedding（BGE）封装。

直接用 sentence-transformers 加载 BGE 模型（不经过 LangChain 包装），
这样你能看清模型加载与向量化的底层细节，面试时也能讲透原理。
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """加载 BGE 模型。

    用 lru_cache 缓存：模型加载很慢、很占内存，只需加载一次，之后复用。
    """
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_query(text: str) -> list[float]:
    """把单个问题转成向量（归一化）。"""
    return get_model().encode(text, normalize_embeddings=True).tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """把一批文档块转成向量（归一化）。"""
    return get_model().encode(texts, normalize_embeddings=True).tolist()


if __name__ == "__main__":
    vec = embed_query("什么是 RAG")
    print(f"向量维度：{len(vec)}")
    print(f"前 5 维数值：{[round(x, 4) for x in vec[:5]]}")
