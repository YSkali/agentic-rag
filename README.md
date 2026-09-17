# Agentic RAG 知识库问答智能体

> 基于 **LangGraph + DeepSeek + 本地 BGE** 的知识库问答智能体。
> 核心亮点：**自我纠错的 Agentic RAG** —— 检索质量不达标时，Agent 会自己改写问题、重新检索，而不是硬着头皮用无关资料作答。

## 它能做什么

- 从本地文档（`.txt`）构建向量索引，语义检索
- 基于检索到的原文生成答案，**拒绝编造**（资料没有就答「我不知道」）
- **自我纠错闭环**：`检索 → 自评 →（不够则）改写重试 → 生成`
- 每个答案附带「引用来源」，可追溯
- **RAGAS 评测 + A/B 对比**：LLM 当评委打分，同一测试集对比「有/无 rerank」
- **工具调用（MCP）**：知识库查不到就调外部工具——计算器（原生 function calling）+ 查当前时间（MCP 独立 server 按协议提供）
- Streamlit 网页界面，开箱即用

## 架构

```
问题 ──► contextualize ──► call_tools ──(要工具)──► generate ──► 答案
                                    │
                                 (要检索)
                                    │
                                    ▼
                                retrieve ──►rerank──► grade ──(资料够/次数用完)──► generate ──► 答案 + 来源
                                                     ▲            │
                                                     │         (资料不够)
                                                     └────── rewrite

```

- **contextualize**：多轮记忆——把代词追问改写成独立问题（第一轮原样通过）
- **call_tools**：用 `bind_tools` 挂上工具，让 LLM 自行判断「要不要调工具」（工具 or 检索的分叉点）
- **retrieve**：Chroma 语义检索 top-k 片段（多召回 top-8）
- **rerank**：cross-encoder 精排，把召回候选按相关性重排、只留 top-4
- **grade**：LLM 自评「检索结果够不够回答」（Agent 的决策点）
- **rewrite**：LLM 改写问题，更利于检索
- **generate**：基于达标后的原文（或工具结果）生成答案，防幻觉约束

## 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 编排 | LangGraph | 显式状态图 + 条件边，实现自我纠错 |
| 主模型 | DeepSeek API | OpenAI 兼容接口，**重模型上云** |
| Embedding | BGE-small-zh-v1.5（本地） | 512 维，**轻模型本地** |
| 向量库 | Chroma | 嵌入式持久化，无需 Docker |
| 前端 | Streamlit | 纯 Python，快速演示 |

## 快速开始

### 1. 环境与依赖

```bash
conda create -n agent_rag python=3.11 -y
conda activate agent_rag
pip install -r requirements.txt
```

### 2. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入真实的 DEEPSEEK_API_KEY
```

### 3. 下载 embedding 模型（国内走镜像）

首次需要下载 BGE 模型。国内直连 HuggingFace 会失败，用镜像：

```bash
# Windows cmd
set "HF_ENDPOINT=https://hf-mirror.com"
python -m app.embeddings
```

下载完成后，在 `.env` 里加一行 `HF_HUB_OFFLINE=1`，之后运行时走本地缓存、不再联网。

### 4. 建索引

把文档（`.txt`）放进 `data/` 目录，然后：

```bash
python -m scripts.build_index
```

### 5. 启动问答

命令行：

```bash
python -m app.agent
```

网页界面：

```bash
streamlit run streamlit_app.py
```

浏览器自动打开 `localhost:8501`，输入问题即可（首次提问会加载本地模型，约 20-40 秒）。

## 演示

> 图片通过 jsDelivr CDN 加速，确保 GitHub 上加载稳定。

### 主界面

![主界面](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/1.png)

### 侧边栏：控制 + 主题 + 示例问题

![侧边栏控制](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/2.png)

### 导出对话 + 清空 + 评测指标

![导出与评测](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/3.png)

### 知识问答：问「什么是 RAG？」

![RAG问答](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/4.png)

### 详细执行过程（节点计时）

![执行过程](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/5.png)

### 引用来源（可追溯）

![引用来源](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/6.png)

### 多轮对话：追问「那它有什么缺点？」

![多轮对话](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/7.png)

### 防幻觉：问「怎么做红烧肉？」答「我不知道」

![防幻觉](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/8.png)

### 工具调用：算术题（计算器）

![计算器](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/9.png)

### 工具调用：查当前时间

![查时间](https://cdn.jsdelivr.net/gh/YSkali/agentic-rag@main/docs/images/10.png)

## 评测

- 用 RAGAS 让 LLM 当评委，给每个回答打三个分：忠实度 / 答案相关性 / 上下文精度（ContextPrecision，`scripts/evaluate.py`）
- 支持 **A/B 对比**：同一测试集跑「有 rerank / 无 rerank」两张图，输出平均差值（`app/agent.py` 的 `build_graph(use_rerank)` 开关切换）
- 实测：忠实度接近满分；**换用评排位的 ContextPrecision 后 rerank 的收益显现**——有 rerank 0.931 vs 无 rerank 0.819（+0.111）。之前「收益≈0」是因为旧指标 ContextRelevance 不评排位、测不到。完整结论见 [docs/interview_notes.md](docs/interview_notes.md)

## 项目结构

```
Agentlearnopen/
├── app/                    # 核心包
│   ├── config.py           # 集中配置（读 .env）
│   ├── llm.py              # DeepSeek 客户端
│   ├── embeddings.py       # 本地 BGE 向量化
│   ├── splitter.py         # 文档切块
│   ├── vectorstore.py      # Chroma 存储 + 检索
│   ├── reranker.py         # cross-encoder 重排（bi-encoder 召回 + 精排）
│   ├── rag.py              # 直线 RAG（检索 + 生成）
│   ├── tools.py            # 原生工具（计算器，ast 安全求值）
│   ├── mcp_server.py       # MCP server（独立进程，按协议暴露「查时间」工具）
│   ├── mcp_client.py       # MCP 客户端（发现并调用 server 工具）
│   └── agent.py            # Agentic RAG（LangGraph 自我纠错 + 工具调用）
├── scripts/
│   ├── build_index.py      # 建索引流水线
│   ├── evaluate.py         # RAGAS 评测 + rerank A/B 对比（有/无 rerank）
│   └── test_set.py         # 评测测试集（问题 + 参考答案）
├── data/                   # 你的文档（gitignore）
├── docs/
│   └── interview_notes.md  # 开发笔记 + 面试问答
├── streamlit_app.py        # 前端入口
└── hello_agent.py          # DeepSeek 连通性测试
```

## Roadmap

- [x] 建索引流水线（切块 + 向量化 + Chroma）
- [x] 直线 RAG（检索 + 生成 + 防幻觉）
- [x] Agentic RAG（LangGraph 自我纠错闭环）
- [x] Streamlit 前端
- [x] 多轮对话记忆
- [x] 重排（rerank）
- [x] 评测（RAGAS）
- [x] rerank A/B 对比（有/无 rerank）
- [x] 工具调用（MCP）
- [x] ContextPrecision 评测 + 干扰题（证明 rerank 排序收益）
