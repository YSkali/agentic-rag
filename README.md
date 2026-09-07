# Agentic RAG 知识库问答智能体

> 基于 **LangGraph + DeepSeek + 本地 BGE** 的知识库问答智能体。
> 核心亮点：**自我纠错的 Agentic RAG** —— 检索质量不达标时，Agent 会自己改写问题、重新检索，而不是硬着头皮用无关资料作答。

## 它能做什么

- 从本地文档（`.txt`）构建向量索引，语义检索
- 基于检索到的原文生成答案，**拒绝编造**（资料没有就答「我不知道」）
- **自我纠错闭环**：`检索 → 自评 →（不够则）改写重试 → 生成`
- 每个答案附带「引用来源」，可追溯
- Streamlit 网页界面，开箱即用

## 架构

```
问题 ──► contextualize ──► retrieve ──►rerank──► grade ──(资料够/次数用完)──► generate ──► 答案 + 来源
                                   ▲            │
                                   │         (资料不够)
                                   └────── rewrite

```

- **retrieve**：Chroma 语义检索 top-k 片段
- **grade**：LLM 自评「检索结果够不够回答」（Agent 的决策点）
- **rewrite**：LLM 改写问题，更利于检索
- **generate**：基于达标后的原文生成答案，防幻觉约束
- **contextualize**：多轮记忆——把代词追问改写成独立问题（第一轮原样通过）

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

## 项目结构

```
Agentlearnopen/
├── app/                    # 核心包
│   ├── config.py           # 集中配置（读 .env）
│   ├── llm.py              # DeepSeek 客户端
│   ├── embeddings.py       # 本地 BGE 向量化
│   ├── splitter.py         # 文档切块
│   ├── vectorstore.py      # Chroma 存储 + 检索
│   ├── rag.py              # 直线 RAG（检索 + 生成）
│   └── agent.py            # Agentic RAG（LangGraph 自我纠错）
├── scripts/
│   └── build_index.py      # 建索引流水线
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
- [×] 多轮对话记忆
- [ ] 重排（rerank）
- [ ] 工具调用（MCP）
- [ ] 评测（RAGAS）
