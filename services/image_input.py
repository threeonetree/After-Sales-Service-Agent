"""Validate and normalize in-memory uploads before sending them to Bailian."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGES = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_EDGE = 1600
MAX_QUESTION_CHARS = 2000
DEFAULT_IMAGE_QUESTION = "请查看这些扫地机器人照片或截图，帮我排查问题。"


class ImageInputError(ValueError):
    """A validation failure that can be shown directly to the user."""


@dataclass(frozen=True)
class PreparedImage:
    # repr must never include image bytes or base64.
    data: bytes = field(repr=False)
    width: int
    height: int

    def content_block(self) -> dict:
        return {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(self.data).decode("ascii")
        }}


def prepare_images(uploads: list[bytes]) -> list[PreparedImage]:
    """All-or-nothing validation: no partial upload is sent after a bad file."""
    if len(uploads) > MAX_IMAGES:
        raise ImageInputError(f"每次最多上传 {MAX_IMAGES} 张图片，请减少后重新发送。")
    result = []
    for index, data in enumerate(uploads, 1):
        prefix = f"第 {index} 张图片："
        if not isinstance(data, bytes) or not data:
            raise ImageInputError(prefix + "文件为空，请重新选择图片。")
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageInputError(prefix + "超过 5 MB，请压缩后上传。")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as source:
                    if source.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ImageInputError(prefix + "仅支持 JPG、PNG、WebP 静态图片。")
                    if source.width * source.height > MAX_IMAGE_PIXELS:
                        raise ImageInputError(prefix + "超过 2000 万像素，请缩小后上传。")
                    if getattr(source, "n_frames", 1) != 1:
                        raise ImageInputError(prefix + "暂不支持动图，请上传一张静态截图。")
                    source.verify()
                with Image.open(BytesIO(data)) as source:
                    oriented = ImageOps.exif_transpose(source)
                    oriented.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
                    rgba = oriented.convert("RGBA")
                    # Fresh pixels remove EXIF/GPS and flatten transparency on white.
                    clean = Image.new("RGB", rgba.size, "white")
                    clean.paste(rgba, mask=rgba.getchannel("A"))
                    output = BytesIO()
                    clean.save(output, format="JPEG", quality=85, optimize=True)
                    result.append(PreparedImage(output.getvalue(), *clean.size))
        except ImageInputError:
            raise
        except (Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise ImageInputError(prefix + "图片尺寸过大，请缩小后上传。") from None
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise ImageInputError(prefix + "无法读取或文件已损坏，请重新导出图片。") from None
    return result


def prepare_question(text: str, has_images: bool) -> str:
    text = text.strip()
    if len(text) > MAX_QUESTION_CHARS:
        raise ImageInputError(f"问题请控制在 {MAX_QUESTION_CHARS} 字以内。")
    if not text and has_images:
        return DEFAULT_IMAGE_QUESTION
    if not text:
        raise ImageInputError("请输入问题或上传图片。")
    return text
