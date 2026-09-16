"""建索引脚本：把 data/ 下的文档切块、向量化、存入 Chroma。

用法：python -m scripts.build_index
"""
from pathlib import Path

from app import splitter, vectorstore


def load_docs(data_dir: str = "data") -> list[tuple[str, str]]:
    """读取 data/ 下所有 .txt 文件，返回 [(文件名, 内容), ...]。"""
    docs = []
    for p in sorted(Path(data_dir).glob("*.txt")):
        docs.append((p.name, p.read_text(encoding="utf-8")))
    return docs


def main() -> None:
    docs = load_docs()
    all_chunks = []
    all_metadatas = []
    for filename, content in docs:
        chunks = splitter.split_text(content)
        all_chunks.extend(chunks)
        all_metadatas.extend([{"source": filename}] * len(chunks))

    print(f"读取 {len(docs)} 个文档，切成 {len(all_chunks)} 块")

    vectorstore.reset_collection()
    n = vectorstore.add_chunks(all_chunks, metadatas=all_metadatas)
    print(f"已写入 {n} 块到向量库（含来源元数据）")

    # 自测：检索一个相关问题
    results = vectorstore.search("什么是向量检索")
    print("\n检索「什么是向量检索」的 top 结果：")
    for i, r in enumerate(results, 1):
        src = r.get("source", "")
        tag = f" [{src}]" if src else ""
        print(f"[{i}]{tag} {r['text'][:50]}...")


if __name__ == "__main__":
    main()
