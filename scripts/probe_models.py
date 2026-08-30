"""Run minimal live checks against the configured free-tier Bailian models."""

from __future__ import annotations

from langchain_core.tools import tool

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


def main() -> int:
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
