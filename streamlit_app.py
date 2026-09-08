"""Streamlit 前端：把 Agentic RAG 包装成多轮对话网页。

用法：streamlit run streamlit_app.py
"""
import streamlit as st

from app.agent import ask

st.set_page_config(page_title="Agentic RAG 问答", page_icon="🤖")
st.title("🤖 Agentic RAG 知识库问答")
st.caption("LangGraph + DeepSeek + 本地 BGE —— 自我纠错 + 多轮对话")

# ── 1. 会话状态：跨 rerun 保存对话 ──
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"/"assistant", "content": str}]

# 侧边栏：清空对话
with st.sidebar:
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# ── 2. 渲染历史对话 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── 3. 输入框（回车即发送）──
question = st.chat_input("输入问题，例如：什么是向量检索？")

if question:
    # 3.1 从已有消息里拼出「一问一答」的历史（此时都是完整轮次）
    history = []
    for i in range(0, len(st.session_state.messages), 2):
        if i + 1 < len(st.session_state.messages):
            history.append(
                f"问：{st.session_state.messages[i]['content']}\n"
                f"答：{st.session_state.messages[i+1]['content']}"
            )

    # 3.2 追加并显示用户问题
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 3.3 生成答案
    with st.chat_message("assistant"):
        with st.spinner("检索中（首次加载本地模型，约 20-40 秒）..."):
            result = ask(question, chat_history=history)

        st.write(result["answer"])

        if result.get("used_tool"):
            st.info(f"🔧 本次调用了工具：{result['tool_result']}")
        elif result["attempts"] > 1:
            st.info(f"本次尝试 {result['attempts']} 次 —— 触发「改写重试」，自我纠错生效")
        else:
            st.success("本次一次命中，未触发改写")

        with st.expander("📚 查看引用来源"):
            if result["context"]:
                for i, src in enumerate(result["context"], 1):
                    st.markdown(f"**来源 {i}**")
                    st.text(src)
            else:
                st.caption("本次未检索知识库（走了工具分支）。")

    # 3.4 追加答案
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
