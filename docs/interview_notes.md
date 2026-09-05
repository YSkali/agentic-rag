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


---

## 11. 前端 Streamlit

**面试官问**：为什么用 Streamlit 而不是自己写前端？
**答**：纯 Python、几十行出网页，适合快速验证/演示（MVP）；生产高并发会换 FastAPI + 前端框架。这里目标是「演示能力」。
**代码**：[streamlit_app.py](../streamlit_app.py)

**面试官问**：前端展示了什么？为什么展示这些？
**答**：答案 + 引用来源（RAG 卖点：可追溯）+ agent 尝试次数（自我纠错「看得见」）——让面试官一眼看到 Agentic 的核心价值。


---

## 12. 踩坑记录

| 坑 | 现象 | 解法 |
|---|---|---|
| Windows GBK 打印 emoji | `UnicodeEncodeError: gbk codec` | 终端加 `PYTHONUTF8=1`（cmd 用 `set`） |
| cmd `set` 尾空格 | 连不上 `hf-mirror.com `（带尾空格） | `set "VAR=值"` 加引号，或分两行 |
| HuggingFace 下载慢/被墙 | 连接失败 | `HF_ENDPOINT=https://hf-mirror.com` |
| Python 环境错 | 依赖装到 base(3.14) | 先 `conda activate agent_rag` |
| 已缓存模型仍联网检查 | `WinError 10060` 连 huggingface.co 超时 | `.env` 加 `HF_HUB_OFFLINE=1` 离线加载 |



---

## 13. 待补（后续开发继续记）

- [ ] 检索精度 vs chunk_size 的关系（已观察到：大块合并多个话题）
- [x] rag.py：检索 + 生成拼接
- [x] LangGraph 自我纠错闭环
- [x] Streamlit 前端
- [ ] 多轮记忆 / rerank / 工具调用 / 评测（RAGAS）
