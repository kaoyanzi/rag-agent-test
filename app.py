import time
import uuid

import streamlit as st
from agent.react_agent import ReactAgent

st.title("智扫通机器人客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

#每个浏览器会话一个 thread_id：同一个会话里的追问共享记忆，不同会话互不干扰
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    response_message = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            prompt,
            thread_id=st.session_state["thread_id"]
        )

        def capture(generator,cache_list):

            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char
        st.chat_message("assistant").write_stream(capture(res_stream,response_message))
        st.session_state["message"].append({"role":"assistant","content":response_message[-1]})
        st.rerun()