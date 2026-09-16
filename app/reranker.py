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


def rerank_with_index(query: str, indexed_candidates: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """带原始下标的重排，返回 [(orig_idx, text), ...]，按分数降序。

    用于 agent.py 的 rerank 节点——重排后仍能追溯每个块对应的元数据（来源文件名）。
    """
    pairs = [[query, text] for _, text in indexed_candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(
        zip(indexed_candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [item for item, _ in ranked[:settings.TOP_K]]
