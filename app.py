import uuid

import streamlit as st

from services.chat_input import limit_image_history, read_submission
from services.image_input import ImageInputError, MAX_QUESTION_CHARS
from utils.app_errors import record_app_error
from utils.chat_style import CHAT_CSS
from utils.customer_text import customer_text

st.set_page_config(page_title="扫地机器人售后助手", page_icon="💬", layout="wide")
st.html(CHAT_CSS)

try:
    from agent.react_agent import ReactAgent
except Exception as error:
    st.error(record_app_error(error))
    st.stop()


@st.dialog("查看图片", width="large")
def view_picture(data, caption):
    st.image(data, caption=caption, width="stretch")


def render_message(message, message_index):
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return
        content = message["content"]
        st.write(customer_text(content) if message["role"] == "assistant" else content)
        pictures = message.get("images", [])
        if pictures:
            for index, (column, data) in enumerate(zip(st.columns(len(pictures)), pictures), 1):
                with column:
                    st.image(data, caption=f"图 {index}", width="stretch")
                    if st.button("查看大图", key=f"image_{message_index}_{index}", type="tertiary"):
                        view_picture(data, f"图 {index}")
        if message.get("images_released"):
            st.caption("如需再次查看较早的图片，请重新上传。")


def start_new_conversation():
    previous_user = st.session_state.get("selected_user")
    if previous_user:
        st.session_state["agent"].reset_conversation(
            st.session_state["thread_id"], previous_user
        )
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["message"] = []


def prefill_question(question):
    st.session_state[f"chat_{st.session_state['thread_id']}"] = question


if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = ReactAgent()
    except Exception as error:
        st.error(record_app_error(error))
        st.stop()
st.session_state.setdefault("message", [])
st.session_state.setdefault("thread_id", str(uuid.uuid4()))

with st.container(key="support_header"):
    title, user, action = st.columns([5, 2, 1.5], vertical_alignment="center")
    with title:
        st.subheader("扫地机器人售后助手")
    with user:
        selected_user = st.selectbox(
            "选择用户", [str(i) for i in range(1001, 1011)],
            key="user_selector", format_func=lambda value: f"用户 {value}",
            label_visibility="collapsed",
        )
    if st.session_state.get("selected_user") != selected_user:
        start_new_conversation()
        st.session_state["selected_user"] = selected_user
    with action:
        st.button("新对话", key="new_chat", on_click=start_new_conversation, width="stretch")

welcome = st.empty()
if not st.session_state["message"]:
    with welcome.container():
        with st.container(key="welcome"):
            st.markdown("## 有什么可以帮你？")
            st.write("描述使用中遇到的问题，也可以附上设备照片或报错截图。")
            with st.container(horizontal=True):
                for index, (label, question) in enumerate([
                    ("清扫效果不好", "扫地机器人清扫时经常漏扫怎么办？"),
                    ("查询使用记录", "帮我查询本月的使用记录"),
                    ("图片怎么提问", "请帮我看看图片里的问题，应该怎么处理？"),
                ]):
                    st.button(label, key=f"example_{index}", on_click=prefill_question, args=(question,))
            st.caption("图片支持 JPG / PNG / WebP，每次最多 3 张，单张不超过 5 MB。")

for index, message in enumerate(st.session_state["message"]):
    render_message(message, index)

value = st.chat_input(
    "输入问题，或点 + 上传图片（最多 3 张）",
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
    welcome.empty()
    user_message = {
        "role": "user", "content": submission.question,
        "images": [image.data for image in submission.images],
    }
    st.session_state["message"].append(user_message)
    limit_image_history(st.session_state["message"])
    render_message(user_message, len(st.session_state["message"]) - 1)
    try:
        with st.spinner("正在整理建议…"):
            execution = st.session_state["agent"].execute_with_trace(
                submission.question,
                thread_id=st.session_state["thread_id"],
                context={"user_id": selected_user},
                images=submission.images,
            )
        # Sources and observations remain in AgentExecution for internal checks.
        answer = {"role": "assistant", "content": customer_text(execution.response)}
    except Exception as error:
        answer = {"role": "assistant", "content": record_app_error(error), "error": True}
    st.session_state["message"].append(answer)
    render_message(answer, len(st.session_state["message"]) - 1)
