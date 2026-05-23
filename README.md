# 智能清洁客服系统 (Smart Cleaning CS)

基于 LangGraph 的多智能体客服系统，面向扫地机器人 / 扫拖一体机器人领域，集成 RAG 知识检索、多智能体协作（前台接待 + 定损专家），提供 Streamlit Web 交互界面。

> 流式 Web UI · Token 级输出 · 中文知识库 · 阿里云通义千问 Qwen3-Max

| 层级 | 技术 |
|------|------|
| LLM | 阿里云通义千问 Qwen3-Max（ChatTongyi） |
| Embedding | DashScope text-embedding-v4 |
| Agent 框架 | LangGraph（多智能体 StateGraph） |
| 向量数据库 | ChromaDB |
| 前端 | Streamlit |
| 文档解析 | PyPDF + LangChain Document Loaders |
| 配置 | YAML |

## 技术架构

```
├── app.py                       # Streamlit Web 入口
├── agent/
│   └── multi_agent.py           # 多智能体编排（前台客服 + 定损专家）
├── model/
│   └── factory.py               # LLM / Embedding 模型工厂
├── rag/
│   ├── rag_service.py           # RAG 检索摘要服务
│   └── vector_store.py          # 文档加载、分块、向量化及 MD5 去重
├── utils/
│   ├── config_handler.py        # YAML 配置加载
│   ├── logger_handler.py        # 日志（控制台 + 按日轮转文件）
│   ├── prompt_loader.py         # 提示词模板加载
│   ├── file_handler.py          # 文件 I/O、MD5 计算
│   └── path_tool.py             # 项目路径解析
├── config/
│   ├── chroma.yml               # 向量库配置
│   ├── rag.yml                  # 模型名称配置
│   ├── prompts.yml              # 提示词路径配置
│   └── router.yml               # 路由关键词配置
├── prompts/
│   ├── front_desk.txt           # 前台客服系统提示词
│   └── rag_summarize.txt        # RAG 摘要提示词
└── data/                        # 知识库（TXT / PDF）
```

### 多智能体协作

系统由两个 Agent 节点组成，通过 LangGraph StateGraph 编排：

- **前台客服**：接待用户、解答日常问答。对无关问题礼貌拒绝并建议转人工。
- **定损专家**：当路由识别到售后 / 理赔意图时自动唤醒，通过 RAG 检索知识库给出技术判定。

路由逻辑：领域相关性预检 → 关键词匹配 → 派单给专家或直接结束。

### RAG 检索增强生成

- 文档分块：`RecursiveCharacterTextSplitter`（chunk=200, overlap=20，中文优化）
- 向量检索 Top-K=3，基于 ChromaDB 语义搜索
- MD5 去重，避免重复加载已处理的文档

### Web UI

Streamlit 聊天界面，支持流式输出。

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云 [DashScope](https://dashscope.console.aliyun.com/) API Key

### 安装

```bash
git clone <your-repo-url>
cd Agent-Multi
pip install -r requirements.txt
```

### 配置

1. 复制环境变量模板：`cp .env.example .env`
2. 编辑 `.env`，填入 DashScope API Key
3. 确保 `data/` 目录下有知识库文档

### 启动

```bash
streamlit run app.py
```

访问 `http://localhost:8501` 即可使用。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | 阿里云通义千问 Qwen3-Max |
| Embedding | DashScope text-embedding-v4 |
| Agent 框架 | LangChain + LangGraph（ReAct） |
| 向量数据库 | ChromaDB |
| 前端 | Streamlit |
| 文档解析 | PyPDF + LangChain Document Loaders |
| 配置管理 | YAML |

## License

MIT
