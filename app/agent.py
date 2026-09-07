"""
Agentic RAG:LangGraph版。索引质量不达标的时候，用来改写问题，重新检索，
最多MAX_ATTEMPTS次
"""

from typing import TypedDict

from langgraph.graph import StateGraph,START,END

from app import vectorstore
from app.llm import get_llm

#接入新的导入
from app import reranker
from app.config import settings



MAX_ATTEMPTS = 2
#这种永远不改变的变量就用全部大写，方便一眼看出来

class RAGState(TypedDict):
    """图的状态：所有节点共享"""
    question: str
    chat_history: list[str]   
    # 历史对话，格式 ["问：...\n答：...", ...]
    context: list[str]
    answer: str
    retrieval_ok: bool
    attempts: int


GENERATE_TEMPLATE = """你是一个知识库问答助手，只根据下面的「资料」回答问题。

资料：
{context}

问题：{question}

要求：
- 如果资料中没有相关信息，直接回答「我不知道」，不要编造。
- 回答要简洁、准确，基于资料原文。"""

GRADE_TEMPLATE = """判断下面的「资料」是否足以回答「问题」。

资料：
{context}

问题：{question}

只回答一个字：「够」或「不够」。资料与问题相关、且包含能回答的信息，答「够」；否则答「不够」。"""

REWRITE_TEMPLATE = """下面是一个可能检索不到相关资料的问题。请把它改写得更具体、更利于检索（提取核心关键词、去掉口语化），只输出改写后的问题，不要解释。

原问题：{question}"""

CONTEXTUALIZE_TEMPLATE = """根据下面的「对话历史」，把「追问」里的指代词（如「它」「这」「那」）替换成具体内容，改写成一句可以独立理解、独立检索的问题。
如果「追问」里本来就没有指代词、也不依赖上文，就直接原样输出。
只输出改写后的问题，不要任何解释。

对话历史：
{history}

追问：{question}"""


#--------5个重要主要节点（“图”的操作员）函数-------------

#1，retrieve(state) —— 执行者（去向量库捞资料）
def retrieve(state:RAGState)->dict:
    """检索：向量库查 top-k 相关文档块，更新 state"""
#    chunks = vectorstore.search(state["question"])  重排修改
    chunks = vectorstore.search(state["question"], k=settings.TOP_K_RECALL)
    return {"context": chunks, "attempts": state.get("attempts", 0) + 1}

#7，节点名 rerank 是给图用的，真正干活的是 reranker.rerank()
def rerank(state: RAGState) -> dict:
    """重排：对召回的候选块精排，只留最相关的 top-k（向量召回多、精排少）。"""
    return {"context": reranker.rerank(state["question"], state["context"])}


#2，grade(state) —— 裁判员（LLM自我反思）
def grade(state:RAGState)->dict:
    prompt = GRADE_TEMPLATE.format(
        context="\n\n".join(state["context"]),
        question=state["question"],
    )
    result = get_llm().invoke(prompt)
    return {"retrieval_ok": "够" in result.content}


#3，generate(state) —— 发言人（最终输出）
def generate(state: RAGState) -> dict:
    prompt = GENERATE_TEMPLATE.format(
        context="\n\n".join(state["context"]),
        question=state["question"],
    )
    result = get_llm().invoke(prompt)
    return {"answer": result.content}


#4，rewrite(state) —— 智囊团（查询改写）
def rewrite(state: RAGState) -> dict:
    prompt = REWRITE_TEMPLATE.format(question=state["question"])
    result = get_llm().invoke(prompt)
    return {"question": result.content}

#6，加个门卫（多轮记忆）
def contextualize(state: RAGState) -> dict:
    """第一轮无历史 → 原样返回；有历史 → 把代词追问改写成独立问题。"""
    history = state.get("chat_history", [])
    if not history:
        return {"question": state["question"]}

    prompt = CONTEXTUALIZE_TEMPLATE.format(
        history="\n\n".join(history),
        question=state["question"],
    )
    result = get_llm().invoke(prompt)
    return {"question": result.content}


#5，should_rewrite(state) —— 最终回答（路由决策）
def should_rewrite(state: RAGState) -> str:
    """路由：检索达标 或 次数用完 → 生成；否则 → 改写重试。"""
    if state.get("retrieval_ok") or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "generate"
    return "rewrite"



#--------------组装图----------------------
#1，build_graph() —— 流水线的总设计师
def build_graph():
    g = StateGraph(RAGState)

    g.add_node("retrieve", retrieve)
    #插入build_graph 接线
    g.add_node("rerank", rerank)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("rewrite", rewrite)

    g.add_node("contextualize", contextualize)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "retrieve")

    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    #rewrite → retrieve 那条边不用动——改写后回到 retrieve，仍会走 retrieve → rerank → grade，循环照旧
    # 条件边：grade 之后由 should_rewrite 决定走哪条
    g.add_conditional_edges("grade", should_rewrite, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")   # 改写后回到检索 → 形成循环
    g.add_edge("generate", END)

    return g.compile()

#2，graph = build_graph() —— 开工投产
graph = build_graph()
#通过这个单例对象，只需要这一个编译好的图即可防止每次问答重复编译


#3，ask(question: str) -> dict —— 给用户的“启动按钮”
#def ask(question: str) -> dict:
#    """对外接口，对齐 rag.answer()，返回完整最终状态。"""
#    return graph.invoke({
#         "question": question,        # 用户问的问题
#    "context": [],              # 初始资料为空
#    "answer": "",               # 初始答案为空
#    "retrieval_ok": False,      # 默认质检不通过
#    "attempts": 0,              # 从第 0 次开始计数
#    })
#改写对外接口 ask()
def ask(question: str, chat_history: list[str] | None = None) -> dict:
    return graph.invoke({
        "question": question,
        "chat_history": chat_history or [],
        "context": [],
        "answer": "",
        "retrieval_ok": False,
        "attempts": 0,
    })




#if __name__ == "__main__":
#    for step in graph.stream({"question": "什么是向量检索", "context": [], "answer": "", "retrieval_ok": False, "attempts": 0}):
#        print(step)

if __name__ == "__main__":
    # 第一轮
    r1 = ask("什么是向量检索")
    print("Q1:", r1["question"], "| 尝试:", r1["attempts"])

    # 第二轮：代词追问，带着上一轮历史
    history = [f"问：什么是向量检索\n答：{r1['answer']}"]
    for step in graph.stream({
        "question": "那它有什么缺点呢？",
        "chat_history": history,
        "context": [], "answer": "", "retrieval_ok": False, "attempts": 0,
    }):
        print(step)   # 看 contextualize 节点输出的改写后问题
