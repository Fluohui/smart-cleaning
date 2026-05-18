# 智能清洁客服系统（smart-cleaning-cs）

基于 LangChain + LangGraph 的 ReAct 智能客服系统，面向扫地机器人/扫拖一体机器人领域，集成 RAG 知识检索、工具调用、动态提示词切换和个性化报告生成。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | 阿里云通义千问 Qwen3-Max（ChatTongyi） |
| Embedding | DashScope text-embedding-v4 |
| Agent 框架 | LangChain + LangGraph（ReAct 模式） |
| 向量数据库 | ChromaDB |
| 前端 | Streamlit |
| 文档解析 | PyPDF + LangChain Document Loaders |
| 配置 | YAML |

## 项目架构

```
Agent-pj/
├── app.py                     # Streamlit Web 入口
├── agent/                     # Agent 模块
│   ├── react_agent.py         # ReAct Agent（LangGraph）
│   ├── tools/
│   │   ├── agent_tools.py     # 7 个工具定义（RAG、天气、用户信息、外部数据等）
│   │   └── middleware.py       # 中间件：工具监控、动态提示词切换
│   └── chroma_db/             # Agent 向量库持久化
├── model/                     # 模型工厂
│   └── factory.py             # 抽象工厂（ChatModel / EmbeddingModel）
├── rag/                       # RAG 检索增强生成
│   ├── rag_service.py         # 检索 + 摘要生成链
│   └── vector_store.py        # 文档加载、分块、向量化索引
├── utils/                     # 工具模块
│   ├── config_handler.py      # YAML 配置加载
│   ├── logger_handler.py      # 日志（控制台 + 按日轮转文件）
│   ├── prompt_loader.py       # 提示词模板加载
│   ├── file_handler.py        # 文件 I/O、MD5 去重
│   └── path_tool.py           # 项目路径解析
├── config/                    # 配置文件
│   ├── agent.yml / chroma.yml / rag.yml / prompts.yml
├── prompts/                   # 提示词模板
│   ├── main_prompt.txt        # 通用客服提示词
│   ├── rag_summarize.txt       # RAG 摘要提示词
│   └── report_prompt.txt      # 报告生成提示词
└── data/                      # 知识库（PDF + TXT，含选购/故障/维护指南）
```

## 核心功能

### ReAct Agent 工具调用
Agent 自主推理并调用 7 个工具完成复杂任务：
- `rag_summarize` — 向量检索知识库并生成摘要回答
- `get_weather` / `get_user_location` — 上下文感知
- `get_user_id` / `get_current_month` — 用户画像
- `fetch_external_data` — 从外部 CSV 拉取使用记录
- `fill_context_for_report` — 触发报告模式

### 动态提示词切换
通过 LangGraph 中间件（`@before_model` / `@dynamic_prompt`）实现运行时提示词切换：检测到报告生成意图后，自动从通用客服提示词切换为报告写作专家提示词。

### RAG 检索增强生成
- 文档分块策略：`RecursiveCharacterTextSplitter`（chunk=200, overlap=20，中文优化）
- 向量检索 Top-K=3，基于 ChromaDB 语义搜索
- MD5 去重避免重复索引

### 个性化月度报告
自动获取用户 ID、月份、外部使用数据，结合知识库生成 Markdown 格式的个性化使用报告与建议。

### 流式 Web UI
基于 Streamlit 的聊天界面，支持 token 级流式输出。

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云 DashScope API Key

### 安装

```bash
git clone https://github.com/Fluohui/smart-cleaning-cs.git
cd smart-cleaning-cs
pip install -r requirements.txt
```

### 配置

1. 设置 API Key：`export DASHSCOPE_API_KEY="your-api-key"`
2. 首次运行前确保 `data/` 目录下有知识库文档

### 启动

```bash
streamlit run app.py
```

访问 `http://localhost:8501` 即可与智能客服对话。

## 示例对话

```
用户: 扫地机器人显示故障代码E002怎么办？

Agent: [调用 rag_summarize 检索故障排除知识库]
根据知识库，E002 表示主刷缠绕。请按以下步骤处理：
1. 关闭电源，翻转机器
2. 使用清理工具剪断缠绕的毛发/线头
3. 重新安装主刷并开机测试
如问题仍存在，请联系售后。

用户: 帮我生成这个月的使用报告

Agent: [自动切换报告模式 → 获取用户信息 → 拉取外部数据 → 生成报告]
## 2025年1月使用报告
- 本月清扫时长：28 小时
- 清扫面积：1,200 平方米
- 建议：滤网已使用 3 个月，建议近期更换...
```

## License

MIT
