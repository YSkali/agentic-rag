"""文档切块。

把长文本切成小块（chunk），是 RAG 的必需步骤：
大模型上下文有限，且小块更容易被精确检索。
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def split_text(text: str) -> list[str]:
    """把一段文本切成若干 chunk。

    separators 是"递归切分"的优先级顺序：先按段落、再按换行、
    再按中文标点（。！？；，）切，最后才按空格/逐字符，
    这样能尽量保证每块语义完整，而不是粗暴地从中间截断。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    from pathlib import Path

    text = Path("data/rag_notes.txt").read_text(encoding="utf-8")
    chunks = split_text(text)
    print(f"共切成 {len(chunks)} 块")
    print(f"第一块长度：{len(chunks[0])} 字")
    print(f"第一块内容预览：{chunks[0][:80]}...")
