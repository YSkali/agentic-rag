"""本地 Embedding（BGE）封装。

直接用 sentence-transformers 加载 BGE 模型（不经过 LangChain 包装），
这样你能看清模型加载与向量化的底层细节，面试时也能讲透原理。
"""
from functools import lru_cache

# 必须先导入 config：它内部 load_dotenv() 会把 .env 里的 HF_HUB_OFFLINE 写进环境变量，
# 而 sentence_transformers 导入时会立即读这个变量。顺序反了 HF_HUB_OFFLINE 就失效（离线加载变成联网超时）。
from app.config import settings

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
#@lru_cache：模型加载很慢、很占内存，这个装饰器让模型只加载一次，之后复用缓存。面试官问"你代码里模型加载几次"，你能答"一次，用 lru_cache 缓存"——这就是工程意识。
def get_model() -> SentenceTransformer:
    """加载 BGE 模型。

    用 lru_cache 缓存：模型加载很慢、很占内存，只需加载一次，之后复用。
    """
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_query(text: str) -> list[float]:
    """把单个问题转成向量（归一化）。"""
    return get_model().encode(text, normalize_embeddings=True).tolist()
    #normalize_embeddings=True：把向量归一化成单位长度。归一化后，余弦相似度 = 两个向量点积，检索时算得快、数值也稳定。这是向量检索的标准操作，原理面试必考。

def embed_documents(texts: list[str]) -> list[list[float]]:
    """把一批文档块转成向量（归一化）。"""
    return get_model().encode(texts, normalize_embeddings=True).tolist()


if __name__ == "__main__":
    vec = embed_query("什么是 RAG")
    print(f"向量维度：{len(vec)}")
    #维度：bge-small-zh-v1.5 输出的向量是 512 维——你运行后会看到，记住这个数字。
    print(f"前 5 维数值：{[round(x, 4) for x in vec[:5]]}")
