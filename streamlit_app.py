"""Streamlit 前端：把 Agentic RAG 包装成多轮对话网页。

用法：streamlit run streamlit_app.py
"""
import streamlit as st

from app.agent import stream_ask

st.set_page_config(page_title="Agentic RAG 问答", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG 知识库问答")
st.caption("LangGraph + DeepSeek + 本地 BGE —— 自我纠错 + 多轮对话 + 工具调用")

# ── 0. 会话状态：跨 rerun 保存对话 ──
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"/"assistant", "content": str}]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── 侧边栏 ──
with st.sidebar:
    st.header("⚙️ 控制")

    # #3 示例问题按钮（一键演示）
    st.subheader("💡 示例问题（点击发送）")
    DEMO_QUESTIONS = [
        ("📖 什么是 RAG？", "什么是 RAG？"),
        ("🔄 那它有什么缺点？", "那它有什么缺点？"),
        ("🚫 怎么做红烧肉？", "怎么做红烧肉？"),
        ("🔧 算一下 123*456", "算一下 123*456"),
        ("🕐 现在几点？", "现在几点？"),
    ]
    for label, q in DEMO_QUESTIONS:
        if st.button(label, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()

    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # #5 评测结果面板
    st.divider()
    st.subheader("📊 评测指标")
    st.caption("RAGAS 三指标（有 rerank vs 无 rerank）")
    # 硬编码最近一次评测结果（跑 evaluate.py 后更新这里）
    with st.expander("查看 A/B 对比", expanded=False):
        st.markdown("""
| 指标 | 有 rerank | 无 rerank | 差值 |
|---|---|---|---|
| 忠实度 | 1.000 | 1.000 | +0.000 |
| 答案相关 | 0.872 | 0.863 | +0.009 |
| 上下文精度 | **0.931** | 0.819 | **+0.111** |

> 完整评测：`python -m scripts.evaluate`
> 结论：rerank 在 ContextPrecision（评排位）上收益 +0.111。
        """)

# ── 1. 渲染历史对话 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── 2. 输入框（回车即发送）──
question = st.chat_input("输入问题，例如：什么是向量检索？") or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    # 2.1 从已有消息里拼出「一问一答」的历史（此时都是完整轮次）
    history = []
    for i in range(0, len(st.session_state.messages), 2):
        if i + 1 < len(st.session_state.messages):
            history.append(
                f"问：{st.session_state.messages[i]['content']}\n"
                f"答：{st.session_state.messages[i+1]['content']}"
            )

    # 2.2 追加并显示用户问题
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 2.3 流式生成答案（A：流式输出）
    with st.chat_message("assistant"):
        # 用 placeholder 容器，流式更新
        status_container = st.empty()      # 显示当前节点
        path_container = st.empty()        # 显示执行路径
        answer_container = st.empty()      # 显示答案
        detail_container = st.container()  # 详情（工具/引用）

        result = None
        for event in stream_ask(question, chat_history=history):
            if event["type"] == "node":
                node = event["node"]
                trace = event["trace"]
                # 实时显示当前节点
                status_container.info(f"🔄 Agent 执行中：**{node}** 节点")
                # 实时更新路径
                path_str = "  →  ".join(t.split("：")[0] for t in trace)
                path_container.caption(f"**Agent 执行路径**：{path_str}")

            elif event["type"] == "done":
                result = event["result"]

        # ── 流式结束，渲染最终结果 ──
        status_container.empty()

        answer_container.write(result["answer"])

        trace = result.get("trace", [])
        if trace:
            path_str = "  →  ".join(t.split("：")[0] for t in trace)
            path_container.caption(f"**Agent 执行路径**：{path_str}")

            with st.expander("🔎 查看详细执行过程", expanded=False):
                for t in trace:
                    st.markdown(f"- {t}")

        # ── 工具 / 纠错提示 ──
        if result.get("used_tool"):
            st.info(f"🔧 本次调用了工具：{result['tool_result']}")
        elif result["attempts"] > 1:
            st.info(f"本次尝试 {result['attempts']} 次 —— 触发「改写重试」，自我纠错生效")
        else:
            st.success("本次一次命中，未触发改写")

        # ── 引用来源 + 文件名（#4）──
        with st.expander("📚 查看引用来源"):
            if result["context"]:
                meta = result.get("context_meta", [])
                for i, src in enumerate(result["context"], 1):
                    # 显示来源文件名（从 context_meta 读）
                    source_file = (meta[i - 1].get("source", "")) if (i - 1) < len(meta) else ""
                    label = f"来源 {i}" + (f"（`{source_file}`）" if source_file else "")
                    st.markdown(f"**{label}**")
                    st.text(src)
            else:
                st.caption("本次未检索知识库（走了工具分支）。")

    # 2.4 追加答案
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
