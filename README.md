# 智能清洁客服系统 (Smart Cleaning CS)

基于 LangGraph 的多智能体客服系统，面向扫地机器人 / 扫拖一体机器人领域。

**LLM 语义路由 + 父子文档 RAG 检索 + 证据裁判硬约束 + Streamlit 流式 UI**

## 技术架构

```
用户 → Streamlit UI → LangGraph StateGraph
                         ├── front_desk (前台客服)
                         ├── multi_agent_router (LLM 语义路由)
                         └── claims_expert (定损专家)
                              └── ParentDocRetriever (父子文档检索)
                                   ├── ChromaDB (子块向量匹配)
                                   └── parent_docs.json (父块完整上下文)
```

| 层级 | 技术 |
|------|------|
| LLM | 阿里云 Qwen3-Max（ChatTongyi） |
| Embedding | DashScope text-embedding-v4 |
| Agent 框架 | LangGraph StateGraph |
| 向量数据库 | ChromaDB |
| 状态持久化 | PostgreSQL（LangGraph Checkpoint） |
| 会话缓存 | Redis（7 天 TTL） |
| 前端 | Streamlit（流式输出） |
| 配置 | YAML |

## 项目结构

```
├── app.py                          # Streamlit Web 入口
├── agent/
│   ├── multi_agent.py              # 多智能体编排 + 路由 + 定损
│   └── tools/                      # 工具子包（预留）
├── model/
│   └── factory.py                  # LLM / Embedding 模型工厂
├── rag/
│   ├── rag_service.py              # RAG 检索摘要服务
│   └── vector_store.py             # 父子文档分割、向量入库、MD5 去重
├── utils/
│   ├── config_handler.py           # YAML 配置加载
│   ├── logger_handler.py           # 日志（控制台 + 按日轮转）
│   ├── prompt_loader.py            # 提示词模板加载
│   ├── file_handler.py             # 文件 I/O、MD5 计算
│   └── path_tool.py                # 项目路径解析
├── config/
│   ├── chroma.yml                  # 向量库 / 分割参数
│   ├── rag.yml                     # 模型名称
│   ├── prompts.yml                 # 提示词路径
│   └── router.yml                  # 路由关键词（旧版，已废弃）
├── prompts/
│   ├── front_desk.txt              # 前台客服系统提示词
│   ├── router.txt                  # LLM 路由分发提示词
│   ├── expert_judge.txt            # 定损裁判提示词（证据硬约束）
│   └── rag_summarize.txt           # RAG 摘要提示词
└── data/                           # 知识库文档（TXT / PDF）
```

## 多智能体协作流程

```
START → front_desk → router (LLM 语义分类)
                        ├── claims → claims_expert → front_desk → END
                        ├── chat ──────────────────→ END
                        └── other ─────────────────→ END
```

### 1. 前台客服 (front_desk)

接待用户、解答日常选购/保养/使用方法问题。收到定损专家结论后，以温柔口吻转述给用户。

### 2. LLM 语义路由 (multi_agent_router)

使用 LLM 结构化输出（`RouterResult` Pydantic 模型）将用户意图分为三类：
- `claims` — 售后/定损：故障、报错、退换货、隐晦损坏表述
- `chat` — 日常问答：选购、保养、使用方法
- `other` — 无关问题：礼貌拒绝

异常时自动降级为 `chat`。

### 3. 定损专家 (claims_expert)

- 通过父子文档检索器获取知识库证据
- 使用 `ExpertVerdict` 结构化输出进行裁判：
  - `analysis` — 推理过程
  - `is_supported` — 证据是否足以支撑结论（**代码级硬约束**）
  - `conclusion` — 定损结论
  - `cited_source` — 引用的原文条款
- `is_supported == False` 时强制拒绝输出，防止 LLM 幻觉

## RAG 父子文档检索

| 层级 | 大小 | 存储 | 用途 |
|------|------|------|------|
| 父块 | 1000 字 | `data/parent_docs.json` | 返回完整上下文 |
| 子块 | 200 字 | ChromaDB | 精准向量匹配 |

检索流程：查询 → Chroma 子块向量匹配 → 提取 `parent_id` → 去重 → 返回完整父块。

## 快速开始

### 环境要求

- Python 3.10+
- PostgreSQL 16+（LangGraph 状态持久化）
- Redis 7+（会话缓存）
- 阿里云 [DashScope](https://dashscope.console.aliyun.com/) API Key

### 安装

```bash
git clone <repo-url>
cd Agent-Multi
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

确保 PostgreSQL 和 Redis 已启动，库表会自动创建。

### 构建知识库

```bash
python -m rag.vector_store
```

### 启动

```bash
streamlit run app.py
```

访问 `http://localhost:8501`。

## License

MIT
