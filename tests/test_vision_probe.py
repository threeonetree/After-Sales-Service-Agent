import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from scripts.probe_models import run_vision_probe


def test_vision_probe_checks_actual_pixel_answer():
    model = Mock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(content=json.dumps({"left_color": "red", "right_color": "blue", "code": "E42"}))
    run_vision_probe(model)
    content = model.invoke.call_args.args[0][0].content
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "E42" not in content[0]["text"]  # Answer isn't leaked through the prompt.


@pytest.mark.parametrize("bad", ["VISION_OK", '{"left_color":"blue","right_color":"red","code":"E42"}'])
def test_nonempty_but_wrong_vision_response_fails(bad):
    model = Mock()
    model.bind.return_value = model
    model.invoke.return_value = AIMessage(content=bad)
    with pytest.raises(RuntimeError):
        run_vision_probe(model)
