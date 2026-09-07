"""评测脚本：用 RAGAS 给 Agentic RAG 打分。

三个指标（对应 RAG 三段流水线）：
- ContextRelevance（上下文相关性）：检索到的上下文是否相关 → 评「检索」
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
from ragas.metrics.collections import AnswerRelevancy, ContextRelevance, Faithfulness

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
    # 注意：这三个都是「无参考答案」指标，各自要的入参不一样，reference 暂时用不上
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_relevance = ContextRelevance(llm=llm)

    # 4. 逐个问题：跑 RAG → 让评委打分
    for item in TEST_SET:
        result = ask(item["question"])
        print(f"\n问题：{item['question']}")
        print(f"答案：{result['answer']}")

        r1 = faithfulness.score(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=result["context"],
        )
        r2 = answer_relevancy.score(
            user_input=item["question"],
            response=result["answer"],
        )
        r3 = context_relevance.score(
            user_input=item["question"],
            retrieved_contexts=result["context"],
        )
        print(f"  faithfulness:      {r1.value}")
        print(f"  answer_relevancy:  {r2.value}")
        print(f"  context_relevance: {r3.value}")


if __name__ == "__main__":
    main()
