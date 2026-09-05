"""Translate Streamlit's text/file submission without importing Streamlit."""

from dataclasses import dataclass, field

from services.image_input import (
    ImageInputError, MAX_IMAGES, MAX_IMAGE_BYTES, PreparedImage,
    prepare_images, prepare_question,
)


@dataclass
class ChatSubmission:
    question: str
    images: list[PreparedImage] = field(default_factory=list, repr=False)


def read_submission(value) -> ChatSubmission:
    text = value if isinstance(value, str) else (value.text or "")
    files = [] if isinstance(value, str) else list(value.files or [])
    if len(files) > MAX_IMAGES:
        raise ImageInputError(f"每次最多上传 {MAX_IMAGES} 张图片，请减少后重新发送。")
    if any(getattr(file, "size", 0) > MAX_IMAGE_BYTES for file in files):
        raise ImageInputError("每张图片不能超过 5 MB，请压缩后上传。")
    question = prepare_question(text, bool(files))
    images = prepare_images([file.getvalue() for file in files])
    return ChatSubmission(question, images)


def limit_image_history(messages: list[dict], keep: int = 3) -> None:
    """Retain previews for the latest three image turns, not an unbounded gallery."""
    with_images = [message for message in messages if message.get("images")]
    for message in with_images[:-keep]:
        message.pop("images", None)
        message["images_released"] = True
