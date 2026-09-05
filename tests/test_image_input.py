"""Image fixtures are generated in memory; no API or personal photos are used."""
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from services.chat_input import limit_image_history, read_submission
from services.image_input import (
    DEFAULT_IMAGE_QUESTION, ImageInputError, MAX_IMAGE_BYTES, prepare_images,
)


def picture(fmt="PNG", size=(100, 50), **kwargs):
    buffer = BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_supported_formats_are_normalized_and_resize(fmt):
    result = prepare_images([picture(fmt, (2400, 1200))])[0]
    with Image.open(BytesIO(result.data)) as normalized:
        assert normalized.format == "JPEG"
        assert normalized.mode == "RGB"
        assert normalized.size == (1600, 800)
    assert result.content_block()["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "base64" not in repr(result)


def test_exif_rotation_is_applied_and_metadata_removed():
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    exif[270] = "private metadata"
    result = prepare_images([picture("JPEG", exif=exif)])[0]
    with Image.open(BytesIO(result.data)) as normalized:
        assert normalized.size == (50, 100)
        assert not normalized.getexif()


@pytest.mark.parametrize("data,expected", [
    (b"", "为空"), (b"not an image", "损坏"),
    (picture("GIF"), "仅支持"), (picture()[:40], "损坏"),
    (b"x" * (MAX_IMAGE_BYTES + 1), "5 MB"),
])
def test_invalid_files_fail_before_any_model_call(data, expected):
    with pytest.raises(ImageInputError, match=expected):
        prepare_images([data])


def test_count_and_pixel_limits(monkeypatch):
    with pytest.raises(ImageInputError, match="最多"):
        prepare_images([picture()] * 4)
    monkeypatch.setattr("services.image_input.MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ImageInputError, match="像素"):
        prepare_images([picture()])


def test_animation_rejected():
    data = BytesIO()
    first = Image.new("RGB", (50, 50), "red")
    first.save(data, format="PNG", save_all=True,
               append_images=[Image.new("RGB", (50, 50), "blue")])
    with pytest.raises(ImageInputError, match="动图"):
        prepare_images([data.getvalue()])


def test_transparency_is_flattened_on_white():
    data = BytesIO()
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(data, "PNG")
    normalized = prepare_images([data.getvalue()])[0]
    with Image.open(BytesIO(normalized.data)) as image:
        assert image.getpixel((10, 10)) == (255, 255, 255)


def test_text_and_image_only_chat_submissions():
    assert read_submission("  你好  ").question == "你好"
    assert not read_submission("你好").images
    value = SimpleNamespace(text="", files=[BytesIO(picture()), BytesIO(picture())])
    request = read_submission(value)
    assert request.question == DEFAULT_IMAGE_QUESTION
    assert len(request.images) == 2


def test_upload_count_checked_before_reading_bytes():
    files = [Mock()] * 4
    with pytest.raises(ImageInputError):
        read_submission(SimpleNamespace(text="问题", files=files))
    files[0].getvalue.assert_not_called()


@pytest.mark.parametrize("text", ["", "x" * 2001])
def test_empty_or_overlong_question_rejected(text):
    with pytest.raises(ImageInputError):
        read_submission(text)


def test_preview_memory_is_bounded_without_losing_dialogue():
    messages = [{"content": str(i), "images": [picture()]} for i in range(6)]
    limit_image_history(messages)
    assert sum(bool(message.get("images")) for message in messages) == 3
    assert messages[0]["content"] == "0"
    assert messages[0]["images_released"] is True
