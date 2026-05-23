import time
import json
import redis
import streamlit as st
from langchain_core.messages import HumanMessage

from agent.multi_agent import multi_agent_app
from utils.logger_handler import get_logger

logger = get_logger("app")

# ==========================================
# 初始化 Redis 客户端
# ==========================================
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
TTL_7_DAYS = 7 * 24 * 3600  # 7天的秒数

st.set_page_config(page_title="智能清洁客服", page_icon="🤖")
st.title("智能清洁多智能体客服")
st.divider()

# 初始化 Thread ID
if "thread_id" not in st.session_state:
    # st.session_state["thread_id"] = f"user_session_{int(time.time())}"
    st.session_state["thread_id"] = "user_ou_haochen_888"

thread_id = st.session_state["thread_id"]


# ==========================================
# 从 Redis 高速缓存加载短期记忆
# ==========================================
def load_chat_history(tid):
    cached_history = redis_client.get(f"chat_history:{tid}")
    if cached_history:
        return json.loads(cached_history)
    return []


def save_chat_history(tid, messages):
    # 写入 Redis 并设置 7 天过期时间 (TTL)
    redis_client.setex(f"chat_history:{tid}", TTL_7_DAYS, json.dumps(messages))


# 加载历史记录
chat_messages = load_chat_history(thread_id)

for msg in chat_messages:
    st.chat_message(msg["role"]).write(msg["content"])

st.sidebar.subheader("运行状态")
st.sidebar.info(f"会话线程 ID:\n`{thread_id}`")
st.sidebar.success("🟢 PostgreSQL 长期记忆已连接")
st.sidebar.success("🟢 Redis 短期高速缓存已连接")

prompt = st.chat_input("您好！遇到什么清洁难题了？")

if prompt:
    st.chat_message("user").write(prompt)
    chat_messages.append({"role": "user", "content": prompt})

    with st.spinner("正在处理，请稍候..."):
        agent_config = {"configurable": {"thread_id": thread_id}}

        try:
            # LangGraph 内部会自动将复杂的状态机快照写入 PostgreSQL
            final_state = multi_agent_app.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=agent_config,
            )

            final_reply = final_state["messages"][-1].content
            expert_verdict = final_state.get("expert_verdict", "")


            def stream_generator(text):
                for char in text:
                    time.sleep(0.01)
                    yield char


            st.chat_message("assistant").write_stream(stream_generator(final_reply))
            chat_messages.append({"role": "assistant", "content": final_reply})

            # 及时将更新后的对话写入 Redis 缓存
            save_chat_history(thread_id, chat_messages)

            if expert_verdict:
                with st.expander("查看定损专家鉴定结论"):
                    st.info(expert_verdict)

        except Exception as e:
            logger.exception("Agent invoke failed")
            st.error(f"抱歉，系统暂时无法处理您的请求: {e}")