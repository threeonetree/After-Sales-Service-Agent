from dataclasses import asdict
import uuid

import streamlit as st

from services.chat_input import limit_image_history, read_submission
from services.image_input import ImageInputError, MAX_QUESTION_CHARS
from utils.model_errors import user_facing_model_error


try:
    from agent.react_agent import ReactAgent
except Exception as error:
    st.error(user_facing_model_error(error))
    st.stop()


def render_message(message):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        st.write(message["content"])
        for index, data in enumerate(message.get("images", []), 1):
            st.image(data, caption=f"图 {index}", width=320)
        if message.get("images_released"):
            st.caption("较早的图片预览已释放。若要重新看图，请再次上传。")
        observation = message.get("observation")
        if observation:
            with st.expander("图片识别结果"):
                for label, key in (("可见现象", "findings"), ("识别文字", "visible_text"),
                                   ("仍不确定", "uncertainties")):
                    if observation.get(key):
                        st.write(f"**{label}**")
                        for text in observation[key]:
                            st.text(text)
        if message.get("sources"):
            with st.expander("查看知识库依据"):
                for source in message["sources"]:
                    st.text(f"[{source['number']}] {source['title']}")
                    st.text(source["excerpt"])


def start_new_conversation():
    previous_user = st.session_state.get("selected_user")
    if previous_user:
        st.session_state["agent"].reset_conversation(
            st.session_state["thread_id"], previous_user
        )
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["message"] = []


st.title("扫地机器人智能客服")
st.caption("可发送文字，或点击输入框附件按钮上传故障照片、配件照片、App 报错截图。")
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

with st.sidebar:
    st.subheader("用户设置")
    selected_user = st.selectbox("选择用户ID", [str(i) for i in range(1001, 1011)])
    if st.session_state.get("selected_user") != selected_user:
        start_new_conversation()
        st.session_state["selected_user"] = selected_user
    if st.button("新对话"):
        start_new_conversation()
        st.rerun()
    st.caption("每次最多 3 张，单张不超过 5 MB，支持 JPG / PNG / WebP。")
    st.caption("可继续追问当前图片；讨论另一台设备时建议开启新对话。")
    st.caption("图片会发送到已配置的百炼模型进行分析。请先遮挡无关的个人信息。")

for message in st.session_state["message"]:
    render_message(message)

value = st.chat_input(
    "请输入问题或上传图片",
    key=f"chat_{st.session_state['thread_id']}",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=5,
    max_chars=MAX_QUESTION_CHARS,
)

if value:
    try:
        submission = read_submission(value)
    except ImageInputError as error:
        st.error(str(error))
        st.stop()

    user_message = {
        "role": "user", "content": submission.question,
        "images": [image.data for image in submission.images],
    }
    st.session_state["message"].append(user_message)
    limit_image_history(st.session_state["message"])
    render_message(user_message)
    try:
        with st.spinner("智能客服正在分析问题、查询资料…"):
            execution = st.session_state["agent"].execute_with_trace(
                submission.question,
                thread_id=st.session_state["thread_id"],
                context={"user_id": selected_user},
                images=submission.images,
            )
        answer = {
            "role": "assistant", "content": execution.response,
            "sources": [asdict(source) for source in execution.sources],
            "observation": execution.observation,
        }
    except Exception as error:
        answer = {"role": "assistant", "content": user_facing_model_error(error), "error": True}
    st.session_state["message"].append(answer)
    render_message(answer)
