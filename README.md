# 智能清洁客服系统

基于 **LangChain + LangGraph** 的 ReAct 智能客服系统，面向扫地机器人 / 扫拖一体机器人领域，集成 RAG 知识检索、7 工具自主调用、动态提示词切换和个性化报告生成。

> 流式 Web UI · Token 级输出 · 中文知识库 · 阿里云通义千问 Qwen3-Max

## 项目亮点

- **ReAct 自主推理** — Agent 遵循「思考→行动→观察→再思考」循环，自主判断何时调用何种工具
- **RAG 检索增强** — ChromaDB 向量库 + DashScope text-embedding-v4，中文优化分块策略（chunk=200, overlap=20）
- **动态提示词切换** — LangGraph 中间件（`@before_model` / `@dynamic_prompt`）运行时检测意图，自动切换系统提示词
- **个性化报告生成** — 自动获取用户数据 + 外部 CSV 记录 → 生成 Markdown 使用报告与保养建议
- **流式输出** — Streamlit 聊天界面，Token 级实时流式响应
- **MD5 去重索引** — 增量加载知识库文件，避免重复向量化

## 技术架构

```
┌──────────────────────────────────────────────────┐
│                  Streamlit UI                     │
│                 (app.py)                          │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              ReAct Agent                          │
│           (agent/react_agent.py)                  │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │  7 Tools     │  │  Middleware  │              │
│  │  · RAG检索   │  │  · 工具监控  │              │
│  │  · 天气查询   │  │  · 模型日志  │              │
│  │  · 用户定位   │  │  · 提示词切换 │              │
│  │  · ID/月份    │  │              │              │
│  │  · 外部数据   │  │              │              │
│  │  · 报告触发   │  │              │              │
│  └──────────────┘  └──────────────┘              │
└──────┬────────────────────┬──────────────────────┘
       │                    │
┌──────▼──────┐    ┌────────▼──────────┐
│  Model      │    │  RAG Service      │
│  Factory    │    │  ┌──────────────┐ │
│  · Chat     │    │  │Vector Store  │ │
│  · Embed    │    │  │(ChromaDB)    │ │
│  (Tongyi)   │    │  └──────────────┘ │
└─────────────┘    └───────────────────┘
```

## 项目结构

```
Agent-Multi/
├── app.py                        # Streamlit 入口
├── agent/
│   ├── react_agent.py            # ReAct Agent（LangGraph create_agent）
│   └── tools/
│       ├── agent_tools.py        # 7 个工具定义
│       └── middleware.py          # 中间件：工具监控、动态提示词切换
├── model/
│   └── factory.py                # 抽象工厂：ChatTongyi + DashScopeEmbeddings
├── rag/
│   ├── rag_service.py            # 检索 + 摘要生成链
│   └── vector_store.py           # 文档加载、分块、向量化、MD5 去重
├── utils/
│   ├── config_handler.py         # YAML 配置加载
│   ├── logger_handler.py         # 日志（控制台 + 按日轮转）
│   ├── prompt_loader.py          # 提示词模板加载
│   ├── file_handler.py           # 文件 I/O、MD5 计算
│   └── path_tool.py              # 项目路径解析
├── config/
│   ├── agent.yml                 # Agent 配置（外部数据路径等）
│   ├── chroma.yml                # ChromaDB 配置
│   ├── rag.yml                   # 模型名称配置
│   └── prompts.yml               # 提示词路径配置
├── prompts/
│   ├── main_prompt.txt           # 通用客服系统提示词
│   ├── rag_summarize.txt         # RAG 摘要提示词模板
│   └── report_prompt.txt         # 报告生成专家提示词
├── data/
│   ├── 扫地机器人100问.pdf        # 知识库 PDF
│   ├── 扫地机器人100问2.txt       # 知识库文本
│   ├── 扫拖一体机器人100问.txt     # 知识库文本
│   ├── 故障排除.txt               # 故障排查指南
│   ├── 维护保养.txt               # 维护保养指南
│   ├── 选购指南.txt               # 选购指南
│   └── external/
│       └── records.csv           # 模拟用户使用记录（10 用户 × 12 月）
└── requirements.txt
```

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

```bash
# 设置 DashScope API Key
export DASHSCOPE_API_KEY="sk-your-api-key"
```

或创建 `.env` 文件（参考 `.env.example`）。

首次运行时，程序会自动将 `data/` 目录下的 PDF 和 TXT 文件向量化存入 ChromaDB，后续运行自动跳过已索引文件（MD5 去重）。

### 启动

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可与智能客服对话。

## 工具能力

| 工具 | 功能 | 入参 | 出参 |
|------|------|------|------|
| `rag_summarize` | RAG 知识库检索 + 摘要 | query: str | 专业资料内容 |
| `get_weather` | 获取城市天气 | city: str | 天气信息字符串 |
| `get_user_location` | 获取用户所在城市 | 无 | 城市名称 |
| `get_user_id` | 获取用户 ID | 无 | 用户 ID |
| `get_current_month` | 获取当前月份 | 无 | YYYY-MM 格式月份 |
| `fetch_external_data` | 拉取用户使用记录 | user_id, month | 结构化使用数据 |
| `fill_context_for_report` | 触发报告生成模式 | 无 | 触发中间件 |

## 使用示例

```
用户：扫地机器人显示故障代码 E002 怎么办？

Agent：[调用 rag_summarize 检索故障排除知识库]
根据知识库，E002 表示主刷缠绕。请按以下步骤处理：
1. 关闭电源，翻转机器
2. 使用清理工具剪断缠绕的毛发/线头
3. 重新安装主刷并开机测试
如问题仍存在，请联系售后。

用户：帮我生成这个月的使用报告

Agent：[自动切换报告模式 → 获取用户 ID → 拉取外部数据 → 生成报告]

# AI 扫地机器人使用情况报告与保养建议

## 本月概况
- 清扫面积：65㎡ 公寓
- 覆盖率：89%
- 日均清扫：49㎡

## 耗材状态
- 主刷寿命剩余 40 天
- HEPA 滤网剩余 20%，建议近期更换

## 使用建议
1. 滤网已接近更换周期，建议本月内更换
2. 当前清洁效率优于 80% 同面积用户，表现优秀
```

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
