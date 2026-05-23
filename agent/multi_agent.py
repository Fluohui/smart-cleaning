import operator
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from model.factory import chat_model
from rag.rag_service import RagSummaryService
from utils.logger_handler import get_logger
from utils.path_tool import get_abs_path

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

logger = get_logger("multi_agent")


def _load_prompt(filename: str) -> str:
    path = get_abs_path(f"prompts/{filename}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Prompt 文件不存在: %s", path)
        return "你是一个智能客服助手，请礼貌地回复用户。"


class RouterResult(BaseModel):
    category: str = Field(description="分类结果: claims/chat/other")


router_system_prompt = _load_prompt("router.txt")
router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("human", "{user_input}")
])

router_chain = router_prompt | chat_model.with_structured_output(RouterResult)


class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    expert_verdict: str


class ExpertVerdict(BaseModel):
    analysis: str = Field(description="基于提供的证据进行逻辑推理的过程，不要超过50字")
    is_supported: bool = Field(description="核心判断：提供的知识库证据是否足以明确解答用户的理赔/故障诉求？")
    conclusion: str = Field(description="如果证据充足，给出最终定损结论；如果不足，回复'证据不足'")
    cited_source: list[str] = Field(description="引用的具体依据（比如：提取原文中的某句关键条款），必须是原文真实存在的！")

expert_system_prompt = _load_prompt("expert_judge.txt")
expert_prompt = ChatPromptTemplate.from_messages([
    ("system", expert_system_prompt),
    ("human", "用户诉求：{user_input}")
])
expert_chain = expert_prompt | chat_model.with_structured_output(ExpertVerdict)


rag_service = RagSummaryService()


def front_desk_node(state: MultiAgentState):
    logger.info("[前台客服] 正在接待用户...")

    base_prompt = _load_prompt("front_desk.txt")

    if state.get("expert_verdict"):
        system_instruction = (
            f"{base_prompt}\n\n"
            f"【最新情报】：后台技术专家已经给出了定损结论：\"{state['expert_verdict']}\"。\n"
            f"请用极其温柔、有人情味的客服语言，把这个好消息或坏消息转述给用户，并给出抚慰。"
        )
    else:
        system_instruction = (
            f"{base_prompt}\n\n"
            f"请根据用户的输入做出得体的回复或引导。"
        )

    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    response = chat_model.invoke(messages)
    return {"messages": [response]}


def claims_expert_node(state: MultiAgentState):
    logger.info("[定损专家] 正在翻阅规章手册进行技术核验...")
    last_user_message = state["messages"][-1].content

    # 1. 强制获取原始证据（不直接让 RAG 生成文本，而是把文档拿过来自己审）
    # 注意：确保你的 rag_service 有 retriever_docs 这个返回文档列表的方法
    docs = rag_service.retriever_docs(last_user_message)

    if not docs:
        logger.warning("[定损专家] 未检索到任何相关保修证据，触发硬拦截！")
        final_verdict = "经核查，系统知识库中未找到与您描述相关的保修政策。为避免误判，建议为您转接人工客服做进一步核实。"
        return {
            "expert_verdict": final_verdict,
            "messages": [HumanMessage(content="[内部通知] 缺乏检索证据，专家已拒绝下发结论。", name="claims_expert")]
        }

    # 2. 拼接强约束上下文
    context = "\n\n".join([f"【证据 {i+1}】:\n{doc.page_content}" for i, doc in enumerate(docs)])

    # 3. 让大模型进行证据裁判
    try:
        logger.debug("[定损专家] 正在进行 Evidence Judge 裁判...")
        result = expert_chain.invoke({
            "context": context,
            "user_input": last_user_message
        })

        logger.debug(f"[定损专家] 裁判思考过程: {result.analysis}")

        # 4. 面试杀手锏：代码级后置校验（Engineering Hard Constraint）
        if not result.is_supported:
            logger.info("[定损专家] 模型判定现有证据不足以支撑结论，拒绝自由发挥。")
            final_verdict = "知识库现有的保修政策未能完全覆盖您的特殊故障情况，为保障您的权益，已为您记录，稍后请人工客服协助复核。"
        else:
            logger.info(f"[定损专家] 证据充足，允许输出结论。引用条款: {result.cited_source}")
            # 只有在 is_supported 为 True 时，才采纳模型的 conclusion
            final_verdict = f"{result.conclusion} (内部核查引用: {', '.join(result.cited_source)})"

    except Exception as e:
        logger.error(f"[定损专家] 判定过程发生异常，触发降级保护: {e}")
        final_verdict = "系统定损模块当前繁忙，请稍后由人工为您定损。"

    return {
        "expert_verdict": final_verdict,
        "messages": [HumanMessage(
            content="[内部通知] 售后定损专家已完成保修政策和技术故障的审计核验，请查收最终结论。",
            name="claims_expert",
        )],
    }


def multi_agent_router(state: MultiAgentState):
    # 【保留原逻辑】第一优先级：如果专家已经判定过了，直接结束，防止死循环
    if state.get("expert_verdict"):
        logger.debug("[路由] 专家已给出结论，引导前台客服组装回复。")
        return "end"

    # 获取用户最后一句话
    last_user_message = state["messages"][-1].content

    # 【核心升级】调用大模型进行语义推断
    try:
        logger.debug(f"[路由] 正在进行大模型语义分析: {last_user_message}")
        result = router_chain.invoke({"user_input": last_user_message})
        intent = result.category
    except Exception as e:
        # 面试高分点：防崩溃兜底机制
        logger.error(f"[路由] LLM 解析失败，触发降级保护。错误: {e}")
        intent = "chat" # 降级为普通问答

    # 【结果分发】根据 LLM 的判断返回对应的图节点名称
    if intent == "claims":
        logger.info(f"[路由] LLM 识别到售后/理赔意图 -> 派单给 [定损专家]")
        return "claims_expert"
    elif intent == "chat":
        logger.info(f"[路由] LLM 识别为普通问答 -> 直接退出图流转")
        return "end"
    else: # other
        logger.info(f"[路由] LLM 识别为无关问题 -> 直接退出图流转")
        return "end"


workflow = StateGraph(MultiAgentState)
workflow.add_node("front_desk", front_desk_node)
workflow.add_node("claims_expert", claims_expert_node)

workflow.add_edge(START, "front_desk")
workflow.add_conditional_edges(
    "front_desk",
    multi_agent_router,
    {
        "claims_expert": "claims_expert",
        "front_desk": "front_desk",
        "end": END,
    },
)
workflow.add_edge("claims_expert", "front_desk")

checkpointer = MemorySaver()
# checkpointer = SqliteSaver.from_conn_string(get_abs_path("data/checkpoints.db"))
multi_agent_app = workflow.compile(checkpointer=checkpointer)


# ==========================================
# PostgreSQL 状态持久化
# ==========================================
DB_URI = "postgresql://admin:password@localhost:5432/agent_db"
# 使用连接池 (Connection Pool) 应对高并发
pool = ConnectionPool(
    conninfo=DB_URI,
    kwargs={"autocommit": True}  # 核心修复：开启自动提交，解决并发建表死锁
)
checkpointer = PostgresSaver(pool)
checkpointer.setup() # 自动在 PG 库中创建所需的 JSONB 状态表

multi_agent_app = workflow.compile(checkpointer=checkpointer)
