"""RAG 问答核心：检索 + 生成。"""
from app import vectorstore
from app.llm import get_llm

PROMPT_TEMPLATE = """你是一个知识库问答助手，只根据下面的「资料」回答问题。

资料：
{context}

问题：{question}

要求：
- 如果资料中没有相关信息，直接回答「我不知道」，不要编造。
- 回答要简洁、准确，基于资料原文。"""


def answer(question: str) -> dict:
    # 1. 检索：拿 top-k 片段（返回含 text/source 的 dict 列表）
    hits = vectorstore.search(question)
    context = "\n\n".join(h["text"] for h in hits)   # 多块拼成一段

    # 2. 生成：填模板 → 调模型
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = get_llm().invoke(prompt)

    # 3. 返回答案 + 来源（保留完整 dict 供前端展示文件名）
    return {"answer": result.content, "sources": hits}


if __name__ == "__main__":
    print("=== 测试1：资料内问题 ===")
    res1 = answer("什么是向量检索")
    print(f"答案：{res1['answer']}")
    print(f"来源数量：{len(res1['sources'])}")

    print("\n=== 测试2：资料外问题 ===")
    res2 = answer("如何做红烧肉")
    print(f"答案：{res2['answer']}")