import time

import streamlit as st
from langchain_core.messages import HumanMessage

from agent.multi_agent import multi_agent_app
from utils.logger_handler import get_logger

logger = get_logger("app")

st.set_page_config(page_title="智能清洁客服", page_icon="🤖")
st.title("智能清洁多智能体客服")
st.divider()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"user_session_{int(time.time())}"

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

st.sidebar.subheader("运行状态")
st.sidebar.info(f"会话线程 ID:\n`{st.session_state['thread_id']}`")

prompt = st.chat_input("您好！")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.spinner("正在处理，请稍候..."):
        agent_config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        try:
            final_state = multi_agent_app.invoke(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "expert_verdict": "",
                },
                config=agent_config,
            )

            final_reply = final_state["messages"][-1].content
            expert_verdict = final_state.get("expert_verdict", "")

            def stream_generator(text):
                for char in text:
                    time.sleep(0.01)
                    yield char

            st.chat_message("assistant").write_stream(stream_generator(final_reply))
            st.session_state["message"].append({"role": "assistant", "content": final_reply})

            if expert_verdict:
                with st.expander("查看定损专家鉴定结论"):
                    st.info(expert_verdict)

        except Exception:
            logger.exception("Agent invoke failed")
            st.error("抱歉，系统暂时无法处理您的请求，请稍后重试或联系人工客服。")
