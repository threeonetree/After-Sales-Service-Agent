"""Run minimal live checks against the configured free-tier Bailian models."""

from __future__ import annotations

import argparse
from io import BytesIO
import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from PIL import Image, ImageDraw

from model.multimodal_options import json_model
from services.image_input import prepare_images
from utils.config_handler import rag_conf
from utils.model_errors import user_facing_model_error


@tool(description="Return the supplied text unchanged. Used only by the API probe.")
def probe_echo(text: str) -> str:
    return text


def run_chat_probe(chat_model) -> None:
    response = chat_model.invoke("这是连通性测试。请只回复：CHAT_OK")
    if not getattr(response, "content", None):
        raise RuntimeError("Chat model returned no content")


def run_embedding_probe(embed_model) -> None:
    vector = embed_model.embed_query("扫地机器人售后服务")
    if not vector or not all(isinstance(value, (int, float)) for value in vector):
        raise RuntimeError("Embedding model returned an invalid vector")


def run_tool_calling_probe(chat_model) -> None:
    model_with_tool = chat_model.bind_tools([probe_echo])
    response = model_with_tool.invoke(
        "必须调用 probe_echo 工具，并把 text 参数设为 TOOL_OK；不要直接回答。"
    )
    tool_calls = getattr(response, "tool_calls", None) or []
    if not any(call.get("name") == "probe_echo" for call in tool_calls):
        raise RuntimeError("Chat model did not return the expected tool call")


def run_vision_probe(chat_model) -> None:
    """One opt-in call checks real pixels and JSON, not just non-empty text."""
    picture = Image.new("RGB", (480, 240), "white")
    draw = ImageDraw.Draw(picture)
    draw.rectangle((20, 60, 220, 220), fill="red")
    draw.rectangle((260, 60, 460, 220), fill="blue")
    draw.text((190, 5), "E42", fill="black", font_size=36)
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    image = prepare_images([buffer.getvalue()])[0]
    response = json_model(chat_model, 120).invoke([HumanMessage(content=[
        {"type": "text", "text": (
            '请识别图片：左侧方块颜色、右侧方块颜色、顶部文字。'
            '只输出 JSON，字段 left_color、right_color 用小写英文颜色名，code 为顶部文字。'
        )}, image.content_block(),
    ])])
    try:
        result = json.loads(response.content)
    except (ValueError, TypeError):
        raise RuntimeError("图片接口未返回有效 JSON。") from None
    if result != {"left_color": "red", "right_color": "blue", "code": "E42"}:
        raise RuntimeError("图片接口已响应，但颜色或文字识别未通过，请检查模型的视觉支持。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--vision", action="store_true", help="在原有三项检查后增加一次图片检查")
    options.add_argument("--vision-only", action="store_true", help="只检查图片输入；消耗一次聊天模型调用")
    args = parser.parse_args()
    try:
        from model.factory import chat_model, embed_model
    except Exception as error:
        print(f"[FAIL] setup: {user_facing_model_error(error)}")
        return 1

    probes = [
        ("chat", lambda: run_chat_probe(chat_model)),
        ("embedding", lambda: run_embedding_probe(embed_model)),
        ("tool_calling", lambda: run_tool_calling_probe(chat_model)),
    ]
    if args.vision_only:
        probes = [("vision", lambda: run_vision_probe(chat_model))]
    elif args.vision:
        probes.append(("vision", lambda: run_vision_probe(chat_model)))
    print(f"Chat model: {rag_conf['chat_model_name']}")
    print(f"Embedding model: {rag_conf['embedding_model_name']}")
    for name, probe in probes:
        try:
            probe()
        except Exception as error:
            print(f"[FAIL] {name}: {user_facing_model_error(error)}")
            return 1
        print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
