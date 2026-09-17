"""
vectorstore.py 的核心：显式向量化
看 add_chunks 和 search 里，都是我自己先调 embeddings.embed_documents/embed_query 算出向量，再交给 Chroma 存/查——而不是把 embedding 函数塞给 Chroma 让它内部处理。

为什么要这样？
因为这样"文档和查询用的是同一个模型、映射到同一个向量空间"这个 RAG 的本质，在代码里一眼就能看到。面试官问你"向量怎么存的"，你指着代码就能讲清，而不是说"Chroma 内部帮我处理的"。



Chroma 向量库封装：存储文档块向量 + 相似度检索。

这里显式地"先算向量、再存/查"，而不是把 embedding 交给 chromadb 内部处理，
目的是让"文档与查询用同一个模型向量化"这个 RAG 核心一目了然。
"""
from functools import lru_cache

import chromadb

from app import embeddings
from app.config import settings


@lru_cache(maxsize=1)
def _client() -> chromadb.PersistentClient:
    """持久化客户端：数据存在本地 .chroma/ 目录（已 gitignore）。"""
    return chromadb.PersistentClient(path=settings.CHROMA_DIR)


def get_collection(name: str = "rag_docs"):
    """获取（或创建）集合，使用余弦距离度量。"""
    return _client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        #做归一化，用余弦距离度量最合理
    )


def add_chunks(chunks: list[str], metadatas: list[dict] | None = None) -> int:
    """把文档块向量化后写入向量库。upsert 保证重复执行不报错。

    metadatas: 与 chunks 一一对应的元数据列表，如 [{"source": "rag_notes.txt"}, ...]。
               用于追溯每个块来自哪个原始文档（前端展示来源文件名）。

    注意：ID 基于内容哈希生成，相同内容重复添加会覆盖（幂等），
          不同内容不会因 ID 冲突互相覆盖。
    """
    import hashlib
    collection = get_collection()
    vectors = embeddings.embed_documents(chunks)
    # 用「内容+来源」生成稳定 ID，避免简单序号导致的增量添加冲突
    if metadatas is None:
        metadatas = [{} for _ in chunks]
    ids = [
        hashlib.md5(f"{metadatas[i].get('source', '')}:{chunks[i]}".encode("utf-8")).hexdigest()
        for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)
    return len(chunks)


def search(query: str, k: int | None = None) -> list[dict]:
    """检索与问题最相似的 top-k 个文档块。

    返回 [{"text": ..., "source": "xxx.txt"}, ...] 列表，
    source 为原始文件名（若建索引时未记录元数据则为空字符串）。
    """
    collection = get_collection()
    k = k or settings.TOP_K
    q_vec = embeddings.embed_query(query)
    result = collection.query(query_embeddings=[q_vec], n_results=k)
    docs = result["documents"][0]
    metas = (result.get("metadatas") or [[]])[0] if result.get("metadatas") else [{}] * len(docs)
    return [
        {"text": doc, "source": (metas[i] or {}).get("source", "")}
        for i, doc in enumerate(docs)
    ]


def reset_collection(name: str = "rag_docs") -> None:
     #reset_collection()	重建前先清空，保证每次从零开始，不留旧数据
    """删除集合（重建索引前调用，保证从零开始）。"""
    try:
        _client().delete_collection(name)
    except Exception:
        pass  # 集合不存在则忽略
