"""The image/RAG chain with scripted model replies and real message objects."""
import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from services.image_input import PreparedImage
from services.visual_support import VisualResponseError, VisualSupportService
from utils.model_errors import user_facing_model_error


OBSERVATION = {
    "status": "ready", "findings": ["图1：滚刷缠绕毛发"],
    "visible_text": ["图1：E42"], "uncertainties": ["无法确认内部电机状况"], "question": "",
}
ANSWER = {"supported": True, "answer": "可见滚刷有缠绕，建议按说明书清理滚刷。", "source_ids": [1]}
IMAGES = [PreparedImage(b"test-pixels", 100, 50)]
DOCUMENTS = [Document(page_content="停机后按说明书取下滚刷清理毛发。", metadata={
    "source": r"D:\private\data\机器人手册.pdf", "page": 2,
})]


def make_service(*responses, documents=None):
    model = Mock()
    model.bind.return_value = model
    model.invoke.side_effect = [
        AIMessage(content=json.dumps(value, ensure_ascii=False))
        if isinstance(value, dict) else value for value in responses
    ]
    retrieve = Mock(return_value=DOCUMENTS if documents is None else documents)
    return VisualSupportService(model, retrieve), model, retrieve


def test_image_to_observation_to_text_retrieval_and_cited_answer():
    # Arrange: known picture observations and a known knowledge-base passage.
    service, model, retrieve = make_service(OBSERVATION, ANSWER)
    # Act: run the same visual service used by the Agent.
    result = service.analyze("为什么扫不干净？", IMAGES)
    # Assert: images reach only the vision call; retrieval uses their meaning.
    assert result.response == ANSWER["answer"]
    assert result.sources[0].title == "机器人手册.pdf · 第 3 页"
    assert result.sources[0].excerpt == DOCUMENTS[0].page_content
    assert "滚刷缠绕毛发" in retrieve.call_args.args[0]
    assert "E42" in retrieve.call_args.args[0]
    assert "base64" not in retrieve.call_args.args[0]
    first, second = [call.args[0] for call in model.invoke.call_args_list]
    assert first[1].content[1]["type"] == "image_url"
    assert "base64" not in str(second)
    assert "private" not in str(second)
    assert json.loads(second[1].content)["knowledge"][0]["text"] == DOCUMENTS[0].page_content
    assert model.invoke.call_count == 2


@pytest.mark.parametrize("status", ["clarify", "unrelated"])
def test_unclear_or_irrelevant_image_asks_instead_of_diagnosing(status):
    service, model, retrieve = make_service({**OBSERVATION, "status": status, "question": "请补拍滚刷。"})
    result = service.analyze("什么问题？", IMAGES)
    assert "请补拍滚刷" in result.response
    assert not result.sources
    retrieve.assert_not_called()
    assert model.invoke.call_count == 1


def test_empty_ready_observation_is_treated_as_insufficient():
    service, _, retrieve = make_service({**OBSERVATION, "findings": [], "visible_text": []})
    assert "信息不足" in service.analyze("看看", IMAGES).response
    retrieve.assert_not_called()


def test_no_documents_does_not_make_up_repair_steps_or_call_model_again():
    service, model, _ = make_service(OBSERVATION, documents=[])
    result = service.analyze("看看", IMAGES)
    assert "没有返回可用资料" in result.response
    assert not result.sources
    assert model.invoke.call_count == 1


def test_retrieved_but_irrelevant_documents_are_not_cited():
    service, _, _ = make_service(OBSERVATION, {
        "supported": False, "answer": "请补充型号以核实 E42 的含义。", "source_ids": [],
    })
    result = service.analyze("看看", IMAGES)
    assert "暂未找到足够相关" in result.response
    assert not result.sources


@pytest.mark.parametrize("bad", [
    {**ANSWER, "source_ids": [99]}, {**ANSWER, "source_ids": []},
    {**ANSWER, "source_ids": [True]}, {**ANSWER, "supported": False},
    {**ANSWER, "answer": "  "},
])
def test_invalid_citations_or_empty_answer_are_rejected(bad):
    service, model, _ = make_service(OBSERVATION, bad)
    with pytest.raises(VisualResponseError):
        service.analyze("看看", IMAGES)
    assert model.invoke.call_count == 2  # No automatic paid/repair fallback.


@pytest.mark.parametrize("bad", ["not JSON", '{"status":"ready"}', '[]'])
def test_malformed_observation_never_reaches_retrieval(bad):
    service, _, retrieve = make_service(AIMessage(content=bad))
    with pytest.raises(VisualResponseError):
        service.analyze("看看", IMAGES)
    retrieve.assert_not_called()


def test_followups_reuse_text_and_bound_recent_history():
    service, model, retrieve = make_service(OBSERVATION, ANSWER, ANSWER, ANSWER, ANSWER)
    result = service.analyze("扫不干净", IMAGES)
    for question in ["型号是X1", "已经清理了", "下一步怎么检查？"]:
        result = service.follow_up(question, result.context)
    assert len(result.context.turns) == 2
    assert "型号是X1" in retrieve.call_args.args[0]
    assert "已经清理了" in retrieve.call_args.args[0]
    assert all("base64" not in str(call) for call in model.invoke.call_args_list[1:])


@pytest.mark.parametrize("stage", ["vision", "retrieval", "answer"])
def test_quota_failure_stops_the_chain_at_each_stage(stage):
    quota_error = RuntimeError("403 AllocationQuota.FreeTierOnly")
    responses = (quota_error,) if stage == "vision" else (OBSERVATION, quota_error)
    service, model, retrieve = make_service(*responses)
    if stage == "retrieval":
        retrieve.side_effect = quota_error
    with pytest.raises(RuntimeError) as caught:
        service.analyze("看看", IMAGES)
    assert "免费额度已用完" in user_facing_model_error(caught.value)
    assert model.invoke.call_count == (2 if stage == "answer" else 1)


def test_error_display_redacts_payload_and_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-test-key")
    message = user_facing_model_error(RuntimeError(
        "bad image data:image/jpeg;base64,YWJjZGVm key=secret-test-key sk-abc123"
    ))
    assert "YWJjZGVm" not in message
    assert "secret-test-key" not in message
    assert "sk-abc123" not in message


def test_real_chat_client_serializes_bailian_vision_options():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from model.multimodal_options import json_model
    with patch.dict("os.environ", {}, clear=True):
        model = ChatOpenAI(model="qwen3.7-flash-2026-07-15", api_key="test-only",
                           base_url="https://example.invalid/compatible-mode/v1")
    bound = json_model(model, 1400)
    payload = model._get_request_payload([HumanMessage(content=[IMAGES[0].content_block()])], **bound.kwargs)
    assert payload["model"] == "qwen3.7-flash-2026-07-15"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["extra_body"] == {"enable_thinking": False}
    assert payload["max_completion_tokens"] == 1400
    assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;")


def test_truncated_json_has_clear_error_and_no_repair_call():
    response = AIMessage(content='{"status":', response_metadata={"finish_reason": "length"})
    service, model, retrieve = make_service(response)
    with pytest.raises(VisualResponseError, match="输出上限"):
        service.analyze("看看", IMAGES)
    retrieve.assert_not_called()
    assert model.invoke.call_count == 1


def test_failed_followup_does_not_overwrite_prior_context():
    service, _, retrieve = make_service(OBSERVATION, ANSWER)
    first = service.analyze("看看", IMAGES)
    previous = list(first.context.turns)
    retrieve.side_effect = RuntimeError("403 AllocationQuota.FreeTierOnly")
    with pytest.raises(RuntimeError):
        service.follow_up("然后呢？", first.context)
    assert first.context.turns == previous


def test_long_followup_keeps_original_model_and_visual_evidence_in_retrieval():
    service, model, retrieve = make_service(OBSERVATION, ANSWER, ANSWER, ANSWER, ANSWER)
    result = service.analyze("型号X1，扫不干净", IMAGES)
    for question in ["清理过了", "还有声音", "请详细解释" * 300]:
        result = service.follow_up(question, result.context)
    query = retrieve.call_args.args[0]
    assert len(query) < 3000
    assert "型号X1" in query
    assert "滚刷缠绕毛发" in query
    assert "E42" in query
    assert json.loads(model.invoke.call_args.args[0][1].content)["initial_question"] == "型号X1，扫不干净"


def test_multiple_images_are_sent_in_order_in_one_observation_call():
    service, model, _ = make_service(OBSERVATION, ANSWER)
    images = [PreparedImage(b"first", 100, 100), PreparedImage(b"second", 100, 100)]
    service.analyze("比较两张图片", images)
    content = model.invoke.call_args_list[0].args[0][1].content
    assert content[1:] == [image.content_block() for image in images]
    assert model.invoke.call_count == 2
