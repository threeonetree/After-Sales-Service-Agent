import streamlit as st
import time
import uuid

from utils.model_errors import user_facing_model_error


try:
    from agent.react_agent import ReactAgent
    from agent.tools import agent_tools
except Exception as error:
    st.error(user_facing_model_error(error))
    st.stop()

user_ids = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]

#标题
st.title("扫地机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = ReactAgent()
    except Exception as error:
        st.error(user_facing_model_error(error))
        st.stop()
if "message" not in st.session_state:
    st.session_state["message"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

#侧边栏：用户选择
with st.sidebar:
    st.subheader("用户设置")
    selected_user = st.selectbox("选择用户ID", user_ids)
    if st.button("新对话"):
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["message"] = []
        st.rerun()

for message in st.session_state["message"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])
#用户输入
prompt = st.chat_input("请输入问题")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    response_messages = []
    try:
        with st.spinner("智能客服思考中..."):
            agent_tools.set_current_user_id(selected_user)
            res_stream = st.session_state["agent"].execute_stream(
                prompt,
                thread_id=st.session_state["thread_id"],
                context={"report":False,"user_id":selected_user}
            )

            def capture(generator,cache_list):
                for chunk in generator:
                    cache_list.append(chunk)

                    #输出更加“流式”
                    for char in chunk:
                        time.sleep(0.01)
                        yield char

            st.chat_message("assistant").write_stream(capture(res_stream,response_messages))
            st.session_state["message"].append(
                {"role":"assistant","content":"".join(response_messages)}
            )
            st.rerun() #刷新页面。不保留思考过程
    except Exception as error:
        st.error(user_facing_model_error(error))
