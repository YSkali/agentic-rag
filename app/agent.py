"""
Agentic RAG:LangGraph版。索引质量不达标的时候，用来改写问题，重新检索，
最多MAX_ATTEMPTS次
"""

from typing import TypedDict
from functools import lru_cache

from langgraph.graph import StateGraph,START,END

from app import vectorstore
from app.llm import get_llm

#接入新的导入
from app import reranker
from app.config import settings
from app.tools import TOOLS
from app.mcp_client import get_mcp_tools, call_tool



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
    tool_result: str   # 工具调用的结果文本（没调工具则为空）
    used_tool: bool    # 本轮是否调用了工具（路由判断用，对齐 retrieval_ok 的显式 flag 风格）
    trace: list[str]   # 记录每个节点的执行轨迹（用于前端可视化）
    context_meta: list[dict]  # 与 context 一一对应的元数据（含 source 文件名）


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

GENERATE_FROM_TOOL_TEMPLATE = """你是一个问答助手，根据下面的「工具返回结果」回答问题。

工具返回结果：
{tool_result}

问题：{question}

要求：
- 直接基于工具结果作答，简洁、准确。
- 如果工具结果出错或不足以回答，就如实说明。"""

TOOL_SYSTEM_PROMPT = """你是一个知识库问答助手，可以调用工具来回答问题。
只有当问题明确需要工具时才调用——
- 计算器：只要问题中包含算术表达式（含 +-*/%** 运算符，如「1+1」「123*456」「2的10次方」），就必须调用计算器，不要凭自己知识口算。注意「算」字不一定代表算术（如「算一下明天是几号」是日期题，不是算术），关键看有没有算术运算符。
- 查时间：仅当问题明确询问当前时间/日期（如「现在几点」「今天几号」）时才调用。
如果是普通知识问答（问概念、原理、流程等），不要调用任何工具，直接输出普通文本即可。"""


#--------5个重要主要节点（“图”的操作员）函数-------------

#1，retrieve(state) —— 执行者（去向量库捞资料）
#def retrieve(state:RAGState)->dict:
#    """检索：向量库查 top-k 相关文档块，更新 state"""
##    chunks = vectorstore.search(state["question"])  重排修改
#    chunks = vectorstore.search(state["question"], k=settings.TOP_K_RECALL)
#    return {"context": chunks, "attempts": state.get("attempts", 0) + 1}
# retrieve 改成「按参数生成」，因为有无 rerank 时检索的 k 不一样
def make_retrieve(k: int):
    """生成一个「召回 k 个」的检索节点。k 由有无 rerank 决定。"""
    def retrieve(state: RAGState) -> dict:
        hits = vectorstore.search(state["question"], k=k)
        context = [h["text"] for h in hits]
        meta = [{"source": h.get("source", "")} for h in hits]
        trace = state.get("trace", []) + [f"📚 语义检索：召回 {len(context)} 个片段（top-{k}）"]
        return {"context": context, "context_meta": meta, "attempts": state.get("attempts", 0) + 1, "trace": trace}
    return retrieve


#7，节点名 rerank 是给图用的，真正干活的是 reranker.rerank()
def rerank(state: RAGState) -> dict:
    """重排：对召回的候选块精排，只留最相关的 top-k（向量召回多、精排少）。"""
    before = len(state["context"])
    # 带原始下标一起重排，避免重复文本导致 meta 错位
    indexed = list(enumerate(state["context"]))
    reranked_indexed = reranker.rerank_with_index(state["question"], indexed)
    new_context = [t for _, t in reranked_indexed]
    new_meta = []
    for orig_idx, _ in reranked_indexed:
        if orig_idx < len(state.get("context_meta", [])):
            new_meta.append(state["context_meta"][orig_idx])
        else:
            new_meta.append({})
    trace = state.get("trace", []) + [f"📊 重排精排：{before} 个 → {len(new_context)} 个（速度换精度）"]
    return {"context": new_context, "context_meta": new_meta, "trace": trace}


#2，grade(state) —— 裁判员（LLM自我反思）
def grade(state:RAGState)->dict:
    prompt = GRADE_TEMPLATE.format(
        context="\n\n".join(state["context"]),
        question=state["question"],
    )
    result = get_llm().invoke(prompt)
    ok = "够" in result.content
    trace = state.get("trace", []) + [f"⚖️ 质检：{'✅ 资料足够' if ok else '❌ 资料不够'}"]
    return {"retrieval_ok": ok, "trace": trace}


#3，generate(state) —— 发言人（最终输出）
def generate(state: RAGState) -> dict:
    # 走了工具分支 → 答案来自工具结果；否则来自检索到的资料
    if state.get("used_tool"):
        prompt = GENERATE_FROM_TOOL_TEMPLATE.format(
            tool_result=state["tool_result"],
            question=state["question"],
        )
        result = get_llm().invoke(prompt)
        trace = state.get("trace", []) + ["💡 生成答案（来自工具）"]
        return {"answer": result.content, "trace": trace}

    # 检索分支
    prompt = GENERATE_TEMPLATE.format(
        context="\n\n".join(state["context"]),
        question=state["question"],
    )
    result = get_llm().invoke(prompt)

    # 模型答「我不知道」→ 清空引用，避免「答不知道却挂着引用」的矛盾
    if "我不知道" in result.content:
        trace = state.get("trace", []) + ["💡 生成答案：模型判断资料不足，答「我知道」（不展示引用）"]
        return {"answer": result.content, "context": [], "context_meta": [], "trace": trace}

    trace = state.get("trace", []) + ["💡 生成答案（来自检索）"]
    return {"answer": result.content, "trace": trace}


#4，rewrite(state) —— 智囊团（查询改写）
def rewrite(state: RAGState) -> dict:
    prompt = REWRITE_TEMPLATE.format(question=state["question"])
    result = get_llm().invoke(prompt)
    new_q = result.content
    trace = state.get("trace", []) + [f"🔄 改写重试：「{state['question']}」→「{new_q}」"]
    return {"question": new_q, "trace": trace}

#6，加个门卫（多轮记忆）
def contextualize(state: RAGState) -> dict:
    """第一轮无历史 → 原样返回；有历史 → 把代词追问改写成独立问题。"""
    history = state.get("chat_history", [])
    trace = state.get("trace", []) + ["🔍 多轮记忆：无历史，原样通过"]
    if not history:
        return {"question": state["question"], "trace": trace}

    prompt = CONTEXTUALIZE_TEMPLATE.format(
        history="\n\n".join(history),
        question=state["question"],
    )
    result = get_llm().invoke(prompt)
    new_q = result.content
    if new_q.strip() != state["question"].strip():
        trace[-1] = f"🔍 查询改写：「{state['question']}」→「{new_q}」"
    return {"question": new_q, "trace": trace}


#5，should_rewrite(state) —— 最终回答（路由决策）
def should_rewrite(state: RAGState) -> str:
    """路由：检索达标 或 次数用完 → 生成；否则 → 改写重试。"""
    if state.get("retrieval_ok") or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "generate"
    return "rewrite"


#8，call_tools(state) —— 工具箱（让 LLM 自己决定要不要调工具）
@lru_cache(maxsize=1)
def _all_tools() -> tuple:
    """合并「原生工具 + MCP 发现的工具」，缓存一次。

    MCP 可能因为环境问题加载失败，失败就退回只用原生工具——不阻塞 RAG 主流程。
    """
    tools = list(TOOLS)
    try:
        tools.extend(get_mcp_tools())
    except Exception as e:
        print(f"[警告] MCP 工具加载失败，仅用原生工具：{e}")
    return tuple(tools)


def call_tools(state: RAGState) -> dict:
    """用 bind_tools 把工具挂到模型上，由模型自行判断是否调用。

    - 模型判断需要工具 → 回复里带 tool_calls（工具名 + 参数），逐个执行、拼成文本写回 state；
    - 模型判断不需要（普通知识问答）→ tool_calls 为空，走检索分支。
    """
    tools = _all_tools()
    llm = get_llm().bind_tools(tools)
    msg = llm.invoke([
        ("system", TOOL_SYSTEM_PROMPT),
        ("human", state["question"]),
    ])

    trace = state.get("trace", [])
    if not msg.tool_calls:
        trace.append("🚫 工具判断：无需调用工具，走检索")
        return {"used_tool": False, "tool_result": "", "trace": trace}

    results = []
    tool_names = []
    for tc in msg.tool_calls:
        tool = next(t for t in tools if t.name == tc["name"])
        result = call_tool(tool, tc["args"])
        results.append(f"{tc['name']} 结果：{result}")
        tool_names.append(tc["name"])
    trace.append(f"🔧 调用工具：{', '.join(tool_names)}")
    return {"used_tool": True, "tool_result": "\n".join(results), "trace": trace}


#9，route_after_tools(state) —— 工具箱之后往哪走
def route_after_tools(state: RAGState) -> str:
    """用了工具 → 直接生成答案；没用 → 走检索（进入 RAG 环）。"""
    return "generate" if state.get("used_tool") else "retrieve"



#--------------组装图----------------------
#1，build_graph() —— 流水线的总设计师
#def build_graph():
#    g = StateGraph(RAGState)
#
#    g.add_node("retrieve", retrieve)
#    #插入build_graph 接线
#    g.add_node("rerank", rerank)
#    g.add_node("grade", grade)
#    g.add_node("generate", generate)
#    g.add_node("rewrite", rewrite)

#    g.add_node("contextualize", contextualize)
#    g.add_edge(START, "contextualize")
#    g.add_edge("contextualize", "retrieve")

#    g.add_edge("retrieve", "rerank")
#    g.add_edge("rerank", "grade")
#    #rewrite → retrieve 那条边不用动——改写后回到 retrieve，仍会走 retrieve → rerank → grade，循环照旧
#    # 条件边：grade 之后由 should_rewrite 决定走哪条
#    g.add_conditional_edges("grade", should_rewrite, {"generate": "generate", "rewrite": "rewrite"})
#    g.add_edge("rewrite", "retrieve")   # 改写后回到检索 → 形成循环
#    g.add_edge("generate", END)
#
#    return g.compile()
#添加 rerank 节点，build_graph 改写如下：
def build_graph(use_rerank: bool = True):
    g = StateGraph(RAGState)

    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("rewrite", rewrite)
    g.add_node("contextualize", contextualize)
    g.add_node("call_tools", call_tools)

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "call_tools")
    # 工具判断：要调工具 → 直接生成；否则 → 进入检索
    g.add_conditional_edges("call_tools", route_after_tools, {"generate": "generate", "retrieve": "retrieve"})

    if use_rerank:
        g.add_node("retrieve", make_retrieve(settings.TOP_K_RECALL))   # 召回 8
        g.add_node("rerank", rerank)
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "grade")
    else:
        g.add_node("retrieve", make_retrieve(settings.TOP_K))          # 直接取 4
        g.add_edge("retrieve", "grade")

    g.add_conditional_edges("grade", should_rewrite, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")   # 改写后回到检索 → 两个版本都会回到 retrieve
    g.add_edge("generate", END)

    return g.compile()




#2，graph = build_graph() —— 开工投产
#graph = build_graph()
#通过这个单例对象，只需要这一个编译好的图即可防止每次问答重复编译
#添加 rerank 节点，graph 改写如下：
graph = build_graph()                            # 默认：带 rerank（线上/命令行用）
graph_no_rerank = build_graph(use_rerank=False)  # 对比用（评测 A/B）


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
#def ask(question: str, chat_history: list[str] | None = None) -> dict:
#    return graph.invoke({
#        "question": question,
#        "chat_history": chat_history or [],
#        "context": [],
#        "answer": "",
#        "retrieval_ok": False,
#        "attempts": 0,
#    })
#添加 rerank 节点，ask 改写如下：
def ask(question: str, chat_history: list[str] | None = None, use_rerank: bool = True) -> dict:
    g = graph if use_rerank else graph_no_rerank
    return g.invoke({
        "question": question,
        "chat_history": chat_history or [],
        "context": [],
        "answer": "",
        "retrieval_ok": False,
        "attempts": 0,
        "tool_result": "",
        "used_tool": False,
        "trace": [],
        "context_meta": [],
    })


def stream_ask(question: str, chat_history: list[str] | None = None, use_rerank: bool = True):
    """流式执行 Agent，逐节点 yield 事件。

    graph.stream() 每完成一个节点就 yield 一次：{节点名: 部分状态更新}。
    我们把它转成前端友好的格式：
      - {"type": "node", "node": 节点名, "trace": [...]}  —— 节点完成
      - {"type": "done", "result": 最终状态字典}          —— 全部完成

    前端可以实时显示 Agent 正在走哪个节点，体验远优于一次性返回。
    """
    g = graph if use_rerank else graph_no_rerank
    initial_state = {
        "question": question,
        "chat_history": chat_history or [],
        "context": [],
        "answer": "",
        "retrieval_ok": False,
        "attempts": 0,
        "tool_result": "",
        "used_tool": False,
        "trace": [],
        "context_meta": [],
    }

    final_state = dict(initial_state)
    for node_event in g.stream(initial_state):
        # node_event 格式：{"节点名": {该节点的部分状态更新}}
        node_name = next(iter(node_event))
        node_output = node_event[node_name]
        final_state.update(node_output)
        trace = final_state.get("trace", [])
        yield {"type": "node", "node": node_name, "trace": trace}

    yield {"type": "done", "result": final_state}



#if __name__ == "__main__":
#    for step in graph.stream({"question": "什么是向量检索", "context": [], "answer": "", "retrieval_ok": False, "attempts": 0}):
#        print(step)

if __name__ == "__main__":
    # 第一轮：普通知识问答（走检索）
    r1 = ask("什么是向量检索")
    print("Q1:", r1["question"], "| 尝试:", r1["attempts"], "| 用了工具:", r1["used_tool"])

    # 第二轮：代词追问，带着上一轮历史
    history = [f"问：什么是向量检索\n答：{r1['answer']}"]
    r2 = ask("那它有什么缺点呢？", chat_history=history)
    print("Q2 改写后:", r2["question"], "| 用了工具:", r2["used_tool"])

    # 第三轮：需要工具的问题（走工具分支，不检索）
    r3 = ask("算一下 123*456 等于多少")
    print("Q3:", r3["question"], "| 用了工具:", r3["used_tool"])
    print("  工具结果:", r3["tool_result"])
    print("  答案:", r3["answer"])
