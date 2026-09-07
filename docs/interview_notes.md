# Agentic RAG 开发与面试笔记

> 作用：沉淀设计决策、踩坑、面试追问点。**每次开发后更新**，面试前对着「代码位置」回看源码。
> 本项目 = 基于 LangGraph + DeepSeek + 本地 BGE 的知识库问答智能体（Agentic RAG）。

---

## 0. 项目一句话

从本地文档检索相关知识，结合 DeepSeek 生成答案，支持多轮对话与工具调用。

---

## 1. 技术选型（为什么这么选）

| 选型 | 为什么 | 代码 |
|---|---|---|
| 主模型 = DeepSeek API | 大模型大/贵/更新快，API 便宜且质量高 | [app/llm.py](../app/llm.py) |
| Embedding = 本地 BGE | 小/固定/可控，能学底层原理，省调用费 | [app/embeddings.py](../app/embeddings.py) |
| 编排 = LangGraph | 显式状态图，能讲清控制流 | [app/agent.py](../app/agent.py) |
| 向量库 = Chroma | 嵌入式，无需 Docker | [app/vectorstore.py](../app/vectorstore.py) |

**面试官问**：为什么主模型用 API，embedding 却本地跑？
**答**：重模型上云（大、贵、更新快），轻模型本地（小、固定、可控、省钱）——业界标准分工。

---

## 2. 工程规范与安全

| 面试官问 | 答 | 位置 |
|---|---|---|
| 密钥怎么管？ | `.env` 存真实密钥，`.gitignore` 排除，仓库只留 `.env.example` 模板 | .env / .gitignore |
| 模型要不要进 git？ | **绝不**。模型是「依赖」不是「代码」，用模型名声明，别人跑代码自动下载 | embeddings.py |
| commit 规范？ | feat/fix/docs/chore 前缀，小步提交 | git log |

---

## 3. 配置管理 config.py

**面试官问**：配置怎么管理？为什么集中？
**答**：所有可调参数集中在 `Settings` 一处，其他模块只 `import settings`，不散落 `os.getenv`、不硬编码。改参数只动一处、测试易替换。
**代码**：[app/config.py](../app/config.py)

---

## 4. LLM 封装 llm.py

**面试官问**：为什么用 `ChatOpenAI` + `base_url` 指向 DeepSeek？
**答**：DeepSeek 接口是 OpenAI 兼容格式，复用 OpenAI 客户端只改 `base_url` 即可切换模型——「多模型可切换」的标准做法。
**代码**：[app/llm.py](../app/llm.py)

---

## 5. Embedding embeddings.py

| 面试官问 | 答 | 代码 |
|---|---|---|
| 为什么本地 BGE？ | 见选型；bge-small-zh 轻量，**512 维** | embeddings.py |
| `lru_cache` 干嘛？ | 模型加载慢/占内存，只加载一次并缓存 | `get_model()` |
| `normalize_embeddings` 干嘛？ | 归一化成单位向量后，余弦相似度 = 点积，检索快且稳 | `embed_query()` |
| 向量维度多少？ | 512（bge-small-zh-v1.5） | — |

---

## 6. 切块 splitter.py

| 面试官问 | 答 | 代码 |
|---|---|---|
| 为什么递归切分？ | 按 separators 优先级「段落→换行→中文标点→逐字符」，尽量不截断语义 | `split_text()` |
| 为什么加中文标点？ | 默认切分器为英文设计，加 `。！？；，` 才切得干净 | separators 列表 |
| overlap 干嘛？ | 相邻块留 50 字重叠，防止一句话被切在边界 | config.CHUNK_OVERLAP |

---

## 7. 向量库 vectorstore.py

| 面试官问 | 答 | 代码 |
|---|---|---|
| 为什么「显式向量化」？ | 自己先 embed 再存/查，让「文档和查询同一模型」一目了然 | `add_chunks`/`search` |
| cosine vs L2？ | 归一化向量用余弦距离最合理（`hnsw:space=cosine`） | `get_collection()` |
| upsert vs add？ | upsert 幂等，重复建索引不报 ID 重复错 | `add_chunks()` |

---

## 8. 建索引 build_index.py

**面试官问**：建索引和问答为什么分离？
**答**：建索引是离线一次性（文档变化才重建），问答是在线高频，分离可独立优化、独立部署。
**代码**：[scripts/build_index.py](../scripts/build_index.py)


---
## 9. RAG 问答核心 rag.py

**面试官问**：RAG 怎么防止模型胡编？
**答**：提示词约束「资料中没有就直接回答不知道」，实测资料外问题（如何做红烧肉）→ 模型如实答「我不知道」。
**代码**：[app/rag.py](../app/rag.py)

**面试官问**：`answer()` 为什么返回 dict 而不是 str？
**答**：同时返回 `answer`（答案）+ `sources`（来源块），前端能展示「引用了哪几段原文」——这是 RAG 相对裸大模型的核心卖点。

**面试官问**：检索和生成为什么分成两个函数？
**答**：`search()` 检索、`answer()` 生成，解耦后可独立替换模型/向量库、独立测试。

## 10. Agentic RAG：LangGraph 编排 agent.py

**面试官问**：为什么从 rag.py 直线版升级成 LangGraph 图？
**答**：直线 RAG 检索不相关也会硬着头皮答；LangGraph 用「条件边」让 agent 先自评检索质量，不够就改写问题、重新检索，形成自我纠错闭环。
**代码**：[app/agent.py](../app/agent.py)

**面试官问**：图里有环（retrieve→grade→rewrite→retrieve），为什么不会死循环？
**答**：`MAX_ATTEMPTS` 兜底——`should_rewrite` 里「检索达标 或 次数用完」就强制走 generate，最多重试 2 次。

**面试官问**：条件边怎么决定「下一步走哪个节点」？
**答**：`should_rewrite` 只返回字符串；`add_conditional_edges` 的第三个参数 `{"generate": ..., "rewrite": ...}` 是映射表，把字符串翻译成节点名。

**面试官问**：节点为什么返回 dict（部分字段）而不是完整 state？
**答**：节点返回「部分更新」，LangGraph 只覆盖对应字段、其余保持不动——每个节点只改自己负责的部分。

**面试官问**：多轮对话怎么记住上下文？
**答**：两层记忆——前端 st.session_state 存消息列表（UI 层）；传给 agent 的 chat_history 进图状态（逻辑层）。agent 用它做「查询改写」，把「那它呢」改写成独立问题再检索。

**面试官问**：为什么前端和 agent 的历史格式不一样？
**答**：前端存标准的 {role, content}（方便渲染），agent 只要扁平字符串列表（拼进提示词）。两边各自用最顺手的格式，交界处做一次转换——接口隔离。

**面试官问**：多轮对话里「那它呢」这种代词追问，怎么处理？
**答**：向量检索无状态、不认识代词，不能拿原话去检索。图最前面加了一个 `contextualize` 节点：用对话历史把追问改写成「不含指代词的独立问题」再往下检索。这叫**查询改写 / standalone question**。
**代码**：app/agent.py 的 `contextualize` + `CONTEXTUALIZE_TEMPLATE`

**面试官问**：为什么「改写」放检索前，而不是生成时再结合历史？
**答**：检索质量决定答案质量——拿「那它呢」去检索必然召回垃圾，生成再好也没用。所以要在源头把问题改写成可检索的形式，这是 RAG 多轮的标准做法。

**面试官问**：为什么向量检索之外还要加 rerank（重排）？
**答**：向量检索是 bi-encoder（双塔），query 和 doc 各自编码、没见过面，快但不准；rerank 是 cross-encoder（交叉编码），query 和 doc 拼一起喂模型，准但慢。所以「多召回（top-8）→ 精排（top-4）」，用速度换精度。
**代码**：app/reranker.py + agent.py 的 rerank 节点

**面试官问**：加了 rerank 效果变化大吗？
**答**：实测某问题从「我不知道」变成正确回答——rerank 把含答案的块排到了 context 最前面，模型更容易找到答案。


---

## 11. 前端 Streamlit

**面试官问**：为什么用 Streamlit 而不是自己写前端？
**答**：纯 Python、几十行出网页，适合快速验证/演示（MVP）；生产高并发会换 FastAPI + 前端框架。这里目标是「演示能力」。
**代码**：[streamlit_app.py](../streamlit_app.py)

**面试官问**：前端展示了什么？为什么展示这些？
**答**：答案 + 引用来源（RAG 卖点：可追溯）+ agent 尝试次数（自我纠错「看得见」）——让面试官一眼看到 Agentic 的核心价值。

**面试官问**：为什么 Streamlit 会「失忆」？
**答**：Streamlit 有一个反直觉的行为：每交互一次（点按钮/发消息），它就把整个脚本从上到下重跑一遍。所以你上一轮存进普通变量 history = [...] 的东西，下一次交互就没了——因为脚本重跑时变量被重新初始化。

解法就是 st.session_state：这是 Streamlit 提供的一个「跨重跑存活」的字典。凡是需要记住的东西（对话历史），都塞进去。这也是 Streamlit 面试必考的一个点。


---

## 12. 踩坑记录

| 坑 | 现象 | 解法 |
|---|---|---|
| Windows GBK 打印 emoji | `UnicodeEncodeError: gbk codec` | 终端加 `PYTHONUTF8=1`（cmd 用 `set`） |
| cmd `set` 尾空格 | 连不上 `hf-mirror.com `（带尾空格） | `set "VAR=值"` 加引号，或分两行 |
| HuggingFace 下载慢/被墙 | 连接失败 | `HF_ENDPOINT=https://hf-mirror.com` |
| Python 环境错 | 依赖装到 base(3.14) | 先 `conda activate agent_rag` |
| 已缓存模型仍联网检查 | `WinError 10060` 连 huggingface.co 超时 | `.env` 加 `HF_HUB_OFFLINE=1` 离线加载 |
| 新函数追加到文件末尾 | `NameError: name 'contextualize' is not defined` | Python 顺序执行：被调用的函数必须先定义，`build_graph()` 之前必须先有 `contextualize` |
| 重复定义 `RAGState` | 图用旧 schema、`ask` 传新字段，两边不一致 | 一个类只定义一次，改字段就改原定义，别再写一个 |
| Streamlit 用了缓存旧模块 | `TypeError: ask() got an unexpected keyword argument` | 改完代码重启 Streamlit（或删 `__pycache__`） |
| `HF_HUB_OFFLINE` 在 .env 里不生效 | 还是连 huggingface.co 超时（WinError 10060） | 导入顺序：`sentence_transformers` 导入时会立刻读该变量，必须让 `load_dotenv()`（import config）跑在它**前面** |
| hf 下载大文件 401 | `CAS Client Error ... 401 Unauthorized ... xethub.hf.co` | 新版默认走 Xet 存储，国内被墙。加 `HF_HUB_DISABLE_XET=1` 禁用，走传统 HTTP |
| 改了图但节点没执行 | stream 输出里少了该节点 | 没入边的节点是「死节点」，LangGraph 不报错但永不执行；改图要「加新边 + 删旧边」同时做 |
| hf 下载大文件 401 | CAS Client Error ... xethub.hf.co | 新版默认走 Xet，国内被墙；加 HF_HUB_DISABLE_XET=1 禁用 |




---

## 13. 待补（后续开发继续记）

- [ ] 检索精度 vs chunk_size 的关系（已观察到：大块合并多个话题）
- [x] rag.py：检索 + 生成拼接
- [x] LangGraph 自我纠错闭环
- [x] Streamlit 前端
- [x] 多轮记忆（查询改写）
- [ ] rerank / 工具调用 / 评测（RAGAS）
