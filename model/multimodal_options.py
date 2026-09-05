"""Bailian JSON-mode options shared by vision service and live probe."""


def json_model(model, max_tokens: int):
    extra = getattr(model, "extra_body", None)
    return model.bind(
        response_format={"type": "json_object"},
        temperature=0,
        # Qwen3.7 Flash supports this documented completion budget.
        max_completion_tokens=max_tokens,
        extra_body={**(extra if isinstance(extra, dict) else {}),
                    "enable_thinking": False},
    )
