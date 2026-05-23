import operator
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
# TODO: 升级 langgraph>=1.3 后切换为 SqliteSaver 持久化
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from model.factory import chat_model
from rag.rag_service import RagSummaryService
from utils.config_handler import router_conf
from utils.logger_handler import get_logger
from utils.path_tool import get_abs_path

logger = get_logger("multi_agent")


class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    expert_verdict: str


rag_service = RagSummaryService()


def _load_prompt(filename: str) -> str:
    path = get_abs_path(f"prompts/{filename}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Prompt 文件不存在: %s", path)
        return "你是一个智能客服助手，请礼貌地回复用户。"


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

    verdict_result = rag_service.rag_summary(last_user_message)
    logger.info("[专家鉴定完毕] 结论: %s...", verdict_result[:50])

    return {
        "expert_verdict": verdict_result,
        "messages": [HumanMessage(
            content="[内部通知] 售后定损专家已完成保修政策和技术故障的审计核验，请查收最终结论。",
            name="claims_expert",
        )],
    }


def multi_agent_router(state: MultiAgentState):
    if state.get("expert_verdict"):
        logger.debug("[路由] 专家已给出结论，引导前台客服组装回复。")
        return "end"

    last_user_message = state["messages"][-1].content

    if not any(kw in last_user_message for kw in router_conf["domain_keywords"]):
        logger.debug("[路由] 非清洁机器人领域问题，直接退出。")
        return "end"

    if any(kw in last_user_message for kw in router_conf["claim_keywords"]):
        logger.info("[路由] 识别到售后/理赔意图，派单给定损专家。")
        return "claims_expert"

    if "换新" in last_user_message and "变换" not in last_user_message:
        logger.info("[路由] 识别到换新诉求，派单给定损专家。")
        return "claims_expert"

    logger.debug("[路由] 普通问答，直接退出。")
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
