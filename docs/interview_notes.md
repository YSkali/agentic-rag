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

**面试官问**：rerank 的 A/B 对比，代码上怎么在「有 rerank / 无 rerank」两张图之间切换？
**答**：`build_graph(use_rerank)` 加开关参数，编译出两张图——`graph`（带 rerank）和 `graph_no_rerank`（不带），`ask()` 按 `use_rerank` 选。关键是把 `retrieve` 挪进 `if` 分支：两个版本检索的 k 不同——有 rerank 用 `make_retrieve(TOP_K_RECALL=8)`（多召回），无 rerank 用 `make_retrieve(TOP_K=4)`（直接取少），保证两边喂给生成的上下文块数一致，唯一变量就是「有没有精排」。有 rerank 时 `retrieve → rerank → grade`，无 rerank 时 `retrieve → grade` 直连。
**代码**：app/agent.py 的 `build_graph` + `make_retrieve`

**面试官问**：为什么 `contextualize → retrieve` 这条边也得跟着挪进 `if`？
**答**：LangGraph 要求「先 add_node 再 add_edge」——`retrieve` 现在是分支里才 add 的，边留在 if 外面会在 add_node 之前引用它，直接 `NameError`。同理 `rewrite → retrieve` 不用动，因为两个分支里都有叫 `retrieve` 的节点，这条回环边对两张图都成立。


---

## 11. 评测（RAGAS）

**面试官问**：你怎么证明你的 RAG 真的有用？
**答**：用 RAGAS 量化评测——让另一个 LLM（DeepSeek）当评委，对每个问题打三个分：忠实度 / 答案相关性 / 上下文相关性。
**代码**：[scripts/evaluate.py](../scripts/evaluate.py) + [scripts/test_set.py](../scripts/test_set.py)

**面试官问**：三个指标分别评什么？
**答**：忠实度（Faithfulness）评「答案有没有瞎编」；答案相关性（AnswerRelevancy）评「有没有答非所问」；上下文相关性（ContextRelevance）评「检索到的上下文相不相关」。三者正好对应 RAG 的「检索→生成」两段。

**面试官问**：实测结果如何？
**答**：忠实度 1.0、上下文相关性 1.0（全满分），答案相关性平均约 0.77——说明防幻觉和检索都在正常工作。

**面试官问**：单题答案相关性只有 0.47，是不是答案错了？
**答**：不是。LLM 当评委本身有噪声——answer_relevancy 靠「用答案反推问题、再比相似度」打分，对「为什么…」类题容易误判。正确用法是看平均/趋势、用来做「改动前后对比」（如有 rerank vs 无 rerank），而不是看单题绝对值。

**面试官问**：你做 rerank 的 A/B 对比时指标全打平了，怎么解释？
**答**：深挖后发现是语料只有 2 个块——任何问题召回都「全中」，rerank 没有可重排的候选，三个指标自然都饱和。这让我意识到「评测结论受数据规模限制」：要先扩语料到几十块、让检索真的变难，对比才有意义。评测的瓶颈常在数据，不在模型或指标。

**面试官问**：扩语料到 18 块后，rerank 的 A/B 还是没差异（甚至略负），是不是 rerank 没用？
**答**：这是合法的负结果，原因有三：① 测试集 5 道题都是「某文档小节的近义词标题」，bi-encoder 靠关键词就命中，向量 top-4 已含答案——而 rerank 只在「正确块被干扰项挤出 top-4、但还在 top-8」时才发挥价值，当前不存在这种情况；② `context_relevance` 只评「相关不相关」、不评「排位」，即使 rerank 改了顺序它也看不见（应换 `Context Precision`）；③ 负差 -0.02 远小于评委噪声（同一题多次打分能从 0.47 跳到 1.0）。结论：小而干净的语料上 rerank 收益趋近于零——这本身就是发现。要展示 rerank 的价值，需要「难题 + 评排位的指标」。

**面试官问**：那 rerank 的价值最后是怎么展示出来的？
**答**：按上面的思路补了两件事。① **换指标**：把 `ContextRelevance`（只看「相关不相关」）换成 `ContextPrecision`（看「相关项排第几」、对排序敏感，需传 `reference`）——这才是 rerank 能影响的维度。② **造干扰**：往语料加一篇「推荐系统的召回与精排」，它和 RAG 重排共用行话（召回/精排/打分/速度换精度），让 bi-encoder 只看词面时把推荐系统的块排第一、把 RAG 重排笔记挤下去。重跑 A/B：ContextPrecision 从无 rerank 的 0.819 升到有 rerank 的 0.931（**+0.111**）。其中「RAG 相比微调」+0.500（只靠换指标就暴露出来的收益）、干扰题「检索里为什么先召回再精排」+0.167、「什么是 RAG」+0.167；两题 -0.083 落在评委噪声内。这证明 rerank 的价值一直存在，只是之前「不评排位」的指标测不到。
**代码**：[scripts/evaluate.py](../scripts/evaluate.py)（换 ContextPrecision）+ [scripts/test_set.py](../scripts/test_set.py)（干扰题）+ 本地语料 `data/recommender_ranking_notes.txt`（干扰文档，data/ 已 gitignore，不入库）

---

## 12. 前端 Streamlit

**面试官问**：为什么用 Streamlit 而不是自己写前端？
**答**：纯 Python、几十行出网页，适合快速验证/演示（MVP）；生产高并发会换 FastAPI + 前端框架。这里目标是「演示能力」。
**代码**：[streamlit_app.py](../streamlit_app.py)

**面试官问**：前端展示了什么？为什么展示这些？
**答**：答案 + 引用来源（RAG 卖点：可追溯）+ agent 尝试次数（自我纠错「看得见」）——让面试官一眼看到 Agentic 的核心价值。

**面试官问**：前端有哪些增强功能？
**答**：5 个增强——① 流式输出（答案逐字出现，模拟真实 LLM 体验）；② 节点计时（每个 Agent 节点显示耗时 ⏱️ 1.2s，性能可视化）；③ Rerank 开关（侧边栏切换有/无 rerank，实时对比效果）；④ 导出对话（下载 Markdown 格式对话记录）；⑤ 暗色主题（手动切换亮色/暗色）。

**面试官问**：流式输出怎么实现的？
**答**：用 LangGraph 的 `graph.stream()` 逐节点 yield 事件，前端用 `st.empty()` 占位符实时更新节点进度；答案部分用字符级循环 + `time.sleep(0.015)` 模拟打字机效果。

**面试官问**：为什么 Streamlit 会「失忆」？
**答**：Streamlit 有一个反直觉的行为：每交互一次（点按钮/发消息），它就把整个脚本从上到下重跑一遍。所以你上一轮存进普通变量 history = [...] 的东西，下一次交互就没了——因为脚本重跑时变量被重新初始化。

解法就是 st.session_state：这是 Streamlit 提供的一个「跨重跑存活」的字典。凡是需要记住的东西（对话历史），都塞进去。这也是 Streamlit 面试必考的一个点。


---

## 13. 踩坑记录

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
| ragas 装完 import 报错 | `No module named 'langchain_community.chat_models.vertexai'` | ragas 与新版 langchain-community 不兼容；降级 `pip install "langchain-community<0.4.2"` |
| ragas 0.4 API 大改 | 旧教程代码跑不通（`evaluate()` 报错、指标找不到） | 指标移到 `ragas.metrics.collections`；用 `metric.score()` 而非 `evaluate()`；评委用 `llm_factory` + `AsyncOpenAI`；查版本用 `inspect.signature`，别照抄旧教程 |
| 语料太小导致评测饱和 | rerank A/B 三指标全打平、`context_relevance` 恒 1.0 | 先扩语料到几十块再测；评测结论受数据规模限制，数据不足时任何指标都会饱和 |
| 测试集太简单 + 指标不敏感排序 | 扩到 18 块后 A/B 仍打平，`context_relevance` 恒 1.0、负差 -0.02 | rerank 只在「正确块被挤到 top-8 却不在 top-4」时显价值；要造干扰题 + 换评排位的 `Context Precision` |
| MCP server 子进程起不来 | 客户端能 `import mcp`，但拉子进程报 `ModuleNotFoundError: No module named 'mcp'` | 客户端用 `sys.executable` 拉子进程，别写裸 `python`（PATH 可能指到没装 mcp 的环境） |
| MCP 工具同步调用报错 | `NotImplementedError: StructuredTool does not support sync invocation` | MCP 工具只支持 `ainvoke`；用常驻事件循环跑 `tool.ainvoke()` 统一接口 |
| Windows stdio MCP 子进程 | ProactorEventLoop 不支持带 stdin 管道的 subprocess | `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`，必须在建 loop 前设 |
| pip 升级依赖后 build_index 假死 | `WinError 10060` 反复请求 `adapter_config.json`、重试 5 次像卡死 | 新 sentence-transformers 6.x 会检查 adapter 等文件；`langchain_text_splitters` 间接 import huggingface_hub，导致 `HF_HUB_OFFLINE` 在 `load_dotenv` 前被固化 False。splitter.py 把 `from app.config import settings` 提到最前（和 embeddings/reranker 同规矩）。**教训：升级依赖后要重跑建索引验证离线加载** |
| bind_tools 裸调、模型乱调无参工具 | 问「什么是 MCP」却调了查时间工具、走错分支 | 给 `call_tools` 加系统提示，约束「只有明确需要工具（计算/查时间）才调，知识问答不要调」。LLM 路由是概率性的，要靠提示词把边界讲清 |
| 工具路由的语义歧义（「算」字误触发） | 「帮我算一下明天是几号」误触发计算器（答案错误），因为系统提示词写「算一下→计算器」，模型看到「算」字就调 | 把触发条件从「含『算』字」改成「含 +-*/%** 运算符」——看题型本质而非关键字。日期题走 MCP 时间工具，无运算符的「算一算我有多少存款」答「我不知道」。**教训：LLM 路由靠关键词匹配不可靠，要描述「特征」而非「关键字」** |
| 质检与生成的语义偏差（答不知道却挂引用） | 质检节点判「够」（retrieval_ok=True），但 generate 按模板答「我不知道」，context 仍非空 → 前端展示引用，造成矛盾 | generate 节点检测答案含「我不知道」→ 清空 context/context_meta，不展示引用。**教训：质检（grade）和生成（generate）是两个独立 LLM 调用，可能产生语义偏差，要在下游做兜底** |
| LLM 路由概率性（简单算术跳步） | 「1+1」约 30% 概率不触发计算器，模型觉得太简单就口算 | 提示词加固「只要含运算符就必须调用计算器，不要口算」，稳定到 100%。**教训：LLM 函数调用是概率性的，提示词要写「必须」而非「可以」** |
| 质检误判导致答非所问 | 问「怎么做红烧肉」答了 RAG 技术内容——检索到不相关资料，质检判「够」，模型基于无关文本硬答 | 质检模板加两条明确判断标准（主题相关 + 包含具体信息），避免宽松误判。**教训：质检（grade）是独立 LLM 调用，提示词要写清「相关且具体」而非模糊的「够不够」** |
| 质检匹配 bug | `grade` 用 `"够" in result.content` 判断，但「不够」也含「够」字，导致该判「不够」时误判为「够」 | 改为 `result.content.strip() == "够"` 精确匹配。**教训：LLM 输出做关键词匹配要小心子串误判** |
| 多轮追问答非所问（代词消解失败） | 问「怎么做红烧肉」答「我不知道」，追问「那它有什么缺点？」却答了 RAG 缺点——历史里答「我不知道」无主题信息，模型无法推断「它」指什么 | contextualize 模板加第三条规则：答句无主题时，从【问句】提取核心名词替换代词。**教训：多轮改写要处理「答不知道」的退化情况** |
| Streamlit toggle 状态被 st.rerun() 重置 | 侧边栏 rerank 开关关闭后，点示例按钮发送问题，开关自动变回开启——示例按钮里的 `st.rerun()` 打断了 toggle 状态同步 | 去掉所有 `st.rerun()`，让 Streamlit 自然 rerun；toggle 放在示例按钮之前渲染。**教训：st.rerun() 会打断 widget 状态同步，应尽量避免** |
| 暗色主题 CSS 覆盖问题 | 用 CSS 变量覆盖 Streamlit 主题，但部分元素仍保持亮色（输入框白底黑字） | 用 `!important` 显式覆盖所有关键元素（stApp、侧边栏、卡片、输入框、按钮、toggle、信息框）。**教训：Streamlit 主题覆盖需要全面选择器 + !important** |




---

## 14. 待补（后续开发继续记）

- [ ] 检索精度 vs chunk_size 的关系（已扩语料到 9 篇；chunk=500 下无重复话题问题，待用难题进一步验证）
- [x] 评测的「改动前后对比」——A/B 已跑出**负结果**：小而干净的语料上 rerank 收益≈0（原因见第 11 节）
- [x] 展示 rerank 价值：换评排位的 `ContextPrecision` 指标 + 造干扰文档（推荐系统召回精排）——实测有 rerank 0.931 vs 无 rerank 0.819（+0.111），见第 11 节
- [x] rag.py：检索 + 生成拼接
- [x] LangGraph 自我纠错闭环
- [x] Streamlit 前端（+ 流式输出、节点计时、rerank 开关、导出、暗色主题）
- [x] 多轮记忆（查询改写）
- [x] 重排（rerank）
- [x] 评测（RAGAS）
- [x] 工具调用（MCP）——原生 function calling（计算器）+ MCP（独立 server 提供「查时间」工具），见第 15 节

---

## 15. 工具调用（MCP）

**面试官问**：为什么 RAG 还要加工具调用？
**答**：RAG 只能答「知识库里有的」问题，遇到「算一下 123×456」「现在几点」这类问题只能硬答「我不知道」。工具调用让 agent 多了一条「执行动作」的路径——查库之外还能调外部工具，这才是从「RAG 问答」到「真 Agent」的关键一跃。
**代码**：[app/agent.py](../app/agent.py) 的 `call_tools` 节点

**面试官问**：工具调用加在图里哪个位置、怎么接线？
**答**：在图最前面 `contextualize` 之后插一个 `call_tools` 节点 + 条件边。`call_tools` 用 `bind_tools` 把工具挂到模型上，模型若在回复里带 `tool_calls` 就执行、写进 `tool_result`；`route_after_tools` 按 `used_tool` 决定走 `generate`（用了工具）还是 `retrieve`（普通知识问答）。工具和检索是「并行的两条路」，原有的 RAG 环（retrieve→grade→rewrite）原封不动。
**代码**：[app/agent.py](../app/agent.py) 的 `call_tools` + `route_after_tools` + `build_graph`

**面试官问**：模型怎么知道什么时候该调工具、传什么参数？
**答**：不用我写 if/else 规则。`bind_tools` 会把每个工具的「名称 + 参数说明」作为 schema 传给模型，模型自己决定要不要调、传什么参数。它想调就返回 `tool_calls`，不想调就返回普通文本——这正是 function calling 的机制，不是规则路由。工具 docstring 就是模型判断的依据。但 LLM 路由是概率性的：裸调会让模型对「无参工具」过度触发（问「什么是 MCP」却去查时间），所以我在 `call_tools` 里加了一句系统提示，把边界讲清——「只有明确需要工具（计算/查时间）才调，知识问答不要调」。

**面试官问**：函数调用（function calling）和 MCP 有什么区别？
**答**：两个层次。函数调用是「机制」——模型输出 `{name, arguments}`，宿主执行后把结果喂回；MCP 是「协议/传输」——把工具怎么被发现、怎么被调用标准化，让工具从「硬编码在代码里」变成「独立进程按协议提供」，避免 M×N 集成地狱。项目里我先用原生 function calling 写了个计算器（搞懂机制），再用 `langchain-mcp-adapters` 接一个独立 MCP server 提供「查当前时间」工具（搞懂协议）——agent 的 `bind_tools` 接线一行没改，只是工具来源变了。这就是「MCP 只是换了个工具来源」的实证。
**代码**：[app/tools.py](../app/tools.py)（原生）+ [app/mcp_server.py](../app/mcp_server.py) + [app/mcp_client.py](../app/mcp_client.py)（MCP）

**面试官问**：计算器工具为什么不用 eval？
**答**：`eval("__import__('os').system(...)")` 会执行任意代码，有注入风险。我用 `ast` 把字符串解析成表达式树，白名单只放行数字、括号和 `+ - * / % **`，其余一律拒绝。这是「安全求值」的常见做法。
**代码**：[app/tools.py](../app/tools.py) 的 `_safe_eval`

**面试官问**：加了工具调用，每次问答都要多一次 LLM 调用，划算吗？
**答**：这是 agentic 的代价。`call_tools` 每次先问一次模型「要不要调工具」，多一次往返和延迟。换来的是「能处理知识库之外的问题」。可以优化（先做轻量关键词判断再决定要不要调 LLM），但这里先保证正确性和演示价值，延迟权衡可以讲清楚即可。

**面试官问**：MCP 工具在 Windows 上踩了什么坑？
**答**：三个。① stdio 子进程需要 `SelectorEventLoop`，默认 `ProactorEventLoop` 不支持带 stdin 管道的 subprocess；② 拉起 server 要用 `sys.executable` 不能用裸 `python`，否则 PATH 可能指到没装 mcp 的环境；③ MCP 是异步的、LangGraph 节点是同步的，MCP 工具还是「只支持 ainvoke 的 StructuredTool」，必须用常驻事件循环（后台线程）把 session 钉在同一个 loop 上，否则跨 loop 调用会失效。
**代码**：[app/mcp_client.py](../app/mcp_client.py)
