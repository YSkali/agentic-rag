"""建索引脚本：把 data/ 下的文档切块、向量化、存入 Chroma。

用法：python -m scripts.build_index
"""
from pathlib import Path

from app import splitter, vectorstore


def load_docs(data_dir: str = "data") -> list[str]:
    """读取 data/ 下所有 .txt 文件的内容。"""
    texts = []
    for p in Path(data_dir).glob("*.txt"):
        texts.append(p.read_text(encoding="utf-8"))
    return texts


def main() -> None:
    docs = load_docs()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(splitter.split_text(doc))

    print(f"读取 {len(docs)} 个文档，切成 {len(all_chunks)} 块")

    vectorstore.reset_collection()
    n = vectorstore.add_chunks(all_chunks)
    print(f"已写入 {n} 块到向量库")

    # 自测：检索一个相关问题
    results = vectorstore.search("什么是向量检索")
    print("\n检索「什么是向量检索」的 top 结果：")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r[:60]}...")


if __name__ == "__main__":
    main()
