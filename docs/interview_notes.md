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
| 编排 = LangGraph | 显式状态图，能讲清控制流 | （后续） |
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

## 9. 踩坑记录

| 坑 | 现象 | 解法 |
|---|---|---|
| Windows GBK 打印 emoji | `UnicodeEncodeError: gbk codec` | 终端加 `PYTHONUTF8=1`（cmd 用 `set`） |
| cmd `set` 尾空格 | 连不上 `hf-mirror.com `（带尾空格） | `set "VAR=值"` 加引号，或分两行 |
| HuggingFace 下载慢/被墙 | 连接失败 | `HF_ENDPOINT=https://hf-mirror.com` |
| Python 环境错 | 依赖装到 base(3.14) | 先 `conda activate agent_rag` |

---

## 10. 待补（后续开发继续记）

- [ ] 检索精度 vs chunk_size 的关系（已观察到：大块合并多个话题）
- [ ] rag.py：检索 + 生成拼接
- [ ] 多轮记忆 / query 改写 / rerank / 工具调用 / 评测
