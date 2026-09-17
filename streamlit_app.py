"""Streamlit 前端：把 Agentic RAG 包装成多轮对话网页。

用法：streamlit run streamlit_app.py
"""
import time

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

    # C：rerank 开关（放在示例按钮之前，避免 st.rerun() 重置 toggle）
    st.subheader("🔀 Rerank 设置")
    use_rerank = st.toggle(
        "启用 Rerank 精排",
        value=True,
        key="use_rerank",
        help="开启时：先召回 8 个候选，再用 cross-encoder 精排到 4 个（精度更高）。\n关闭时：直接取向量检索 top-4（速度更快）。",
    )
    if use_rerank:
        st.caption("✅ 当前：召回 8 → 精排 4（高精度）")
    else:
        st.caption("⚡ 当前：直接 top-4（更快速）")

    st.divider()

    # E：主题切换（手动切换亮色/暗色）
    st.subheader("🎨 主题设置")
    dark_mode = st.toggle(
        "暗色模式",
        value=False,  # 默认亮色（Streamlit 原生外观）
        key="dark_mode",
        help="切换界面主题：开启为暗色，关闭为亮色（默认）。",
    )
    # 只有开启暗色时才注入 CSS；关闭时用 Streamlit 默认亮色
    if dark_mode:
        st.markdown("""
        <style>
            /* ── 全局基础 ── */
            html, body, #root, .main, .stApp,
            [data-testid="stAppViewContainer"] {
                background-color: #0E1117 !important;
                color: #E1E4E8 !important;
            }

            /* ── 侧边栏 ── */
            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"] {
                background-color: #0E1117 !important;
                color: #E1E4E8 !important;
            }

            /* ── 顶部 header ── */
            [data-testid="stHeader"] {
                background-color: #0E1117 !important;
            }

            /* ── 文字 ── */
            body, p, h1, h2, h3, h4, h5, h6, li, span, div,
            [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] li {
                color: #E1E4E8 !important;
            }

            /* ── 次要文字（caption、说明） ── */
            .stCaption, [data-testid="stCaption"],
            .stHelp, [data-testid="stHelp"] {
                color: #8B949E !important;
            }

            /* ── 链接 ── */
            a { color: #58A6FF !important; }

            /* ── 卡片/容器 ── */
            [data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stExpander"] {
                background-color: #161B22 !important;
                border-color: #30363D !important;
            }
            [data-testid="stExpander"] details {
                background-color: #161B22 !important;
                border-color: #30363D !important;
            }
            [data-testid="stExpander"] summary {
                color: #E1E4E8 !important;
            }

            /* ── 输入框 ── */
            [data-testid="stChatInput"],
            [data-testid="stChatInput"] > div,
            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInput"] > div > div,
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            input, textarea {
                background-color: #161B22 !important;
                color: #E1E4E8 !important;
                border-color: #30363D !important;
            }
            [data-testid="stChatInput"] textarea,
            textarea {
                background-color: #161B22 !important;
                color: #E1E4E8 !important;
            }
            input::placeholder, textarea::placeholder {
                color: #6E7681 !important;
            }

            /* ── 按钮 ── */
            [data-testid="stButton"] > button {
                background-color: #21262D !important;
                color: #E1E4E8 !important;
                border-color: #30363D !important;
            }
            [data-testid="stButton"] > button:hover {
                background-color: #30363D !important;
                border-color: #58A6FF !important;
                color: #FFFFFF !important;
            }
            /* 主按钮（primary） */
            [data-testid="stButton"] > button[kind="primary"] {
                background-color: #238636 !important;
                color: #FFFFFF !important;
                border-color: #2EA043 !important;
            }

            /* ── Toggle 开关 ── */
            [data-testid="stToggle"] label div {
                background-color: #30363D !important;
            }
            [data-testid="stToggle"] label div[aria-checked="true"] {
                background-color: #238636 !important;
            }

            /* ── 信息框（info/success） ── */
            [data-testid="stAlert"] {
                background-color: #0D1117 !important;
                border-color: #30363D !important;
                color: #E1E4E8 !important;
            }
            [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
                color: #E1E4E8 !important;
            }

            /* ── 分割线 ── */
            hr { border-color: #30363D !important; }

            /* ── 表格 ── */
            table, th, td {
                border-color: #30363D !important;
                color: #E1E4E8 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    st.divider()

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

    st.divider()

    # D：导出对话
    st.subheader("💾 导出对话")
    if st.session_state.messages:
        md_text = "# Agentic RAG 对话记录\n\n"
        for msg in st.session_state.messages:
            role = "**用户**" if msg["role"] == "user" else "**Agent**"
            md_text += f"{role}：\n\n{msg['content']}\n\n---\n\n"
        st.download_button(
            "📥 下载 Markdown",
            data=md_text.encode("utf-8"),
            file_name="conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.caption("暂无对话可导出")

    st.divider()

    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []

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

# 2.0 输入验证：拦截空输入
if question:
    question = question.strip()
    if not question:
        st.warning("⚠️ 请输入有效问题，不能为空")
        question = None

if question:
    # 2.1 从已有消息里拼出「一问一答」的历史（此时都是完整轮次）
    # 限制历史轮次，避免上下文过长（最多保留 10 轮）
    MAX_HISTORY = 10
    history = []
    messages = st.session_state.messages
    # 从旧到新取最近的完整轮次
    start_idx = max(0, len(messages) - MAX_HISTORY * 2)
    for i in range(start_idx, len(messages), 2):
        if i + 1 < len(messages):
            history.append(
                f"问：{messages[i]['content']}\n"
                f"答：{messages[i+1]['content']}"
            )

    # 2.2 追加并显示用户问题
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 2.3 流式生成答案（A：流式输出）
    with st.chat_message("assistant"):
        # ── 第一遍：实时显示节点执行进度 ──
        progress = st.empty()  # 进度提示（执行完清空）
        result = None

        try:
            for event in stream_ask(question, chat_history=history, use_rerank=use_rerank):
                if event["type"] == "node":
                    node = event["node"]
                    trace = event["trace"]
                    path_str = " → ".join(t.split("：")[0] for t in trace)
                    progress.info(f"🔄 Agent 执行中：{path_str}")
                elif event["type"] == "done":
                    result = event["result"]
        except Exception as e:
            progress.empty()
            error_msg = str(e)
            # 根据错误类型给出友好提示
            if "API_KEY" in error_msg.upper() or "AUTH" in error_msg.upper():
                st.error("🔑 API 密钥无效或已过期，请检查 .env 文件中的 DEEPSEEK_API_KEY")
            elif "CONNECTION" in error_msg.upper() or "NETWORK" in error_msg.upper():
                st.error("🌐 网络连接失败，请检查网络设置")
            elif "TIMEOUT" in error_msg.upper():
                st.error("⏱️ 请求超时，请稍后重试")
            else:
                st.error(f"❌ 生成答案时出错：{error_msg}")

            # 添加重试按钮
            if st.button("🔄 重新生成"):
                st.session_state.pending_question = question
                st.rerun()

        # ── 仅在成功获取结果时展示答案 ──
        if result is not None:
            # ── 第二遍：答案逐字揭示（模拟流式） ──
            answer = result.get("answer", "")
            if answer:
                answer_placeholder = st.empty()
                displayed = ""
                for char in answer:
                    displayed += char
                    answer_placeholder.markdown(displayed + "▌")  # 光标效果
                    time.sleep(0.005)  # 控制打字速度（优化：从 0.015 改为 0.005）
                answer_placeholder.markdown(answer)  # 去掉光标

            # ── Agent 执行路径 ──
            trace = result.get("trace", [])
            if trace:
                path_str = "  →  ".join(t.split("：")[0] for t in trace)
                st.caption(f"**Agent 执行路径**：{path_str}")

                with st.expander("🔎 查看详细执行过程", expanded=False):
                    for t in trace:
                        st.markdown(f"- {t}")

            # ── 工具 / 纠错提示 ──
            if result.get("used_tool") and result.get("tool_result"):
                # 友好展示工具调用结果
                st.info(f"🔧 本次调用了工具：{result['tool_result']}")
            elif result.get("attempts", 1) > 1:
                st.info(f"本次尝试 {result['attempts']} 次 —— 触发「改写重试」，自我纠错生效")
            else:
                st.success("本次一次命中，未触发改写")

            # ── 引用来源 + 文件名（#4）──
            with st.expander("📚 查看引用来源"):
                if result.get("used_tool"):
                    # 真走了工具分支（计算器/查时间）→ 无知识库引用
                    st.caption("本次调用工具获取答案，未检索知识库。")
                elif result.get("context"):
                    # 检索到资料且模型正常作答 → 展示引用
                    meta = result.get("context_meta", [])
                    for i, src in enumerate(result["context"], 1):
                        # 显示来源文件名（从 context_meta 读）
                        source_file = (meta[i - 1].get("source", "")) if (i - 1) < len(meta) else ""
                        label = f"来源 {i}" + (f"（`{source_file}`）" if source_file else "")
                        st.markdown(f"**{label}**")
                        st.text(src)
                elif "我不知道" in result.get("answer", ""):
                    # 走了检索，但模型判断资料不足 → 清空引用，答「不知道」
                    st.caption("模型判断资料不足以回答，未展示引用。")
                else:
                    st.caption("本次未检索知识库。")

            # 2.4 追加答案（保留完整信息，便于多轮对话和调试）
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                # 以下为扩展字段，供后续多轮对话/调试使用
                "_meta": {
                    "context": result.get("context", []),
                    "context_meta": result.get("context_meta", []),
                    "used_tool": result.get("used_tool", False),
                    "tool_result": result.get("tool_result", ""),
                    "attempts": result.get("attempts", 1),
                    "retrieval_ok": result.get("retrieval_ok", False),
                }
            })
        else:
            # 失败时添加错误占位，保持消息列表对齐
            st.session_state.messages.append({
                "role": "assistant",
                "content": "生成失败，请重试",
            })
