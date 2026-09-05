"""Streamlit 前端：把 Agentic RAG 包装成网页问答界面。

用法：streamlit run streamlit_app.py
"""
import streamlit as st

from app.agent import ask

# 页面配置
st.set_page_config(page_title="Agentic RAG 问答", page_icon="🤖")

st.title("🤖 Agentic RAG 知识库问答")
st.caption("LangGraph + DeepSeek + 本地 BGE —— 自我纠错 RAG")

# 输入框
question = st.text_input("输入问题：", placeholder="例如：什么是向量检索？")

# 点按钮才执行（避免每敲一个字就调一次模型）
if st.button("提问", type="primary") and question:
    with st.spinner("检索中（首次会加载本地 BGE 模型，约 20-40 秒）..."):
        result = ask(question)

    st.divider()

    # 1. agent 执行信息 —— 面试点：自我纠错「看得见」
    if result["attempts"] > 1:
        st.info(f"本次共尝试 {result['attempts']} 次 —— 触发过「改写重试」，自我纠错生效")
    else:
        st.success("本次检索一次命中，未触发改写")

    # 2. 答案
    st.markdown("### 💡 答案")
    st.write(result["answer"])

    # 3. 引用来源 —— RAG 的卖点：答案来自哪几段原文
    with st.expander("📚 查看引用来源（检索到的原文片段）"):
        for i, src in enumerate(result["context"], 1):
            st.markdown(f"**来源 {i}**")
            st.text(src)
            st.divider()
