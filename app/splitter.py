"""文档切块。

把长文本切成小块（chunk），是 RAG 的必需步骤：
大模型上下文有限，且小块更容易被精确检索。

1，为什么用"递归切分"：separators 是一个优先级列表——先按段落 \n\n、再按换行 \n、再按中文标点 。！？；，、最后才逐字符。它先尝试"最安全"的切法，切不动再降级，尽量不把一句话从中间截断。这比粗暴地每 500 字硬切高明得多。

2，为什么 separators 里要加中文标点：默认的切分器是为英文设计的（按空格、句号），切中文会把一句话切得支离破碎。加了 。！？；， 才切得干净——这是"中文场景"的细节，面试官看到会点头。

3，overlap（重叠）干嘛的：切块时相邻两块保留 50 字重叠。防止一句话刚好卡在两块边界，被割断后语义丢失。

"""
# 必须先导入 config：它内部 load_dotenv() 会把 .env 里的 HF_HUB_OFFLINE 写进环境变量，
# 而 langchain_text_splitters 会间接导入 huggingface_hub——它会在 import 时立刻读取该变量并固化。
# 顺序反了，HF_HUB_OFFLINE 就失效（离线加载变成联网 HEAD 超时、重试 5 次假死）。
from app.config import settings

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
