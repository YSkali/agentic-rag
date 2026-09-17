"""评测脚本：用 RAGAS 给 Agentic RAG 打分。

三个指标（对应 RAG 三段流水线）：
- ContextPrecision（上下文精度）：检索到的上下文是否相关、且相关项是否排得靠前 → 评「检索」的排序质量
- Faithfulness（忠实度）：答案是否忠于上下文、有没有瞎编 → 评「生成」的防幻觉
- AnswerRelevancy（答案相关性）：答案是否答非所问 → 评「生成」的相关性

用法：python -u -m scripts.evaluate
"""
# 先加载项目配置（load_dotenv 会设置 HF_HUB_OFFLINE），再导入 ragas——顺序坑
from app.agent import ask
from app.config import settings

from openai import AsyncOpenAI

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

from scripts.test_set import TEST_SET


def main():
    # 1. 评委 LLM：ragas 0.4 要求 InstructorLLM，用 llm_factory 拿 DeepSeek 的 OpenAI 客户端包一层
    # 用 AsyncOpenAI（不是 OpenAI）：ragas 指标内部走 agenerate()，需要异步客户端
    client = AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )
    llm = llm_factory(settings.DEEPSEEK_MODEL, client=client)

    # 2. 评委 Embeddings：本地 BGE（ragas 自带的 HuggingfaceEmbeddings 包装，离线加载）
    embeddings = HuggingFaceEmbeddings(model=settings.EMBEDDING_MODEL)

    # 3. 三个指标（ragas 0.4 的 collections 指标在构造时就绑定评委）
    # 用 ContextPrecision 而非 ContextRelevance：前者对「相关项排第几」敏感（正好被 rerank 影响），
    # 后者只看「相关不相关」、对排序无感——所以 rerank 的收益只有前者才测得到。
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)

    # 有 rerank / 无 rerank 各跑一遍，最后算平均对比
    summary = {
        "faithfulness": {"有rerank": [], "无rerank": []},
        "answer_relevancy": {"有rerank": [], "无rerank": []},
        "context_precision": {"有rerank": [], "无rerank": []},
    }

    for item in TEST_SET:
        print(f"\n问题：{item['question']}")
        for use_rerank in (True, False):
            label = "有rerank" if use_rerank else "无rerank"
            try:
                result = ask(item["question"], use_rerank=use_rerank)
            except Exception as e:
                print(f"  [{label}] ⚠️ 提问失败，跳过：{e}")
                continue

            try:
                r1 = faithfulness.score(
                    user_input=item["question"],
                    response=result["answer"],
                    retrieved_contexts=result["context"],
                )
                r2 = answer_relevancy.score(
                    user_input=item["question"],
                    response=result["answer"],
                    retrieved_contexts=result["context"],  # 补充上下文，让评测更准确
                )
                r3 = context_precision.score(
                    user_input=item["question"],
                    reference=item["reference"],
                    retrieved_contexts=result["context"],
                )
                print(f"  [{label}] 忠实度={r1.value:.3f}  答案相关={r2.value:.3f}  上下文精度={r3.value:.3f}")

                summary["faithfulness"][label].append(r1.value)
                summary["answer_relevancy"][label].append(r2.value)
                summary["context_precision"][label].append(r3.value)
            except Exception as e:
                print(f"  [{label}] ⚠️ 评分失败，跳过：{e}")
                continue

    print("\n===== 平均对比（有 rerank vs 无 rerank）=====")
    for name, cols in summary.items():
        on_vals = cols["有rerank"]
        off_vals = cols["无rerank"]
        if not on_vals or not off_vals:
            print(f"{name:18s}  数据不足，跳过")
            continue
        on = sum(on_vals) / len(on_vals)
        off = sum(off_vals) / len(off_vals)
        print(f"{name:18s}  有rerank={on:.3f}  无rerank={off:.3f}  差={on - off:+.3f}")


if __name__ == "__main__":
    main()
