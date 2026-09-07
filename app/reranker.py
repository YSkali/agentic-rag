"""重排器：cross-encoder 对召回的候选精排。"""
from functools import lru_cache

# 先 config（load_dotenv 先跑），再 sentence_transformers——和 embeddings.py 同一个顺序坑
from app.config import settings

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.RERANK_MODEL)


def rerank(query: str, candidates: list[str]) -> list[str]:
    pairs = [[query, c] for c in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:settings.TOP_K]]
