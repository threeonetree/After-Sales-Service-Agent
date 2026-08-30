from abc import ABC, abstractmethod
import os
from typing import Optional, Union

from dotenv import load_dotenv
from utils.config_handler import rag_conf
from utils.model_errors import require_dashscope_api_key
from utils.path_tool import get_abs_path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


# Local .env is optional and gitignored. Existing OS environment variables win.
load_dotenv(get_abs_path(".env"), override=False)

DEFAULT_DASHSCOPE_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def get_dashscope_base_url() -> str:
    """Return the OpenAI-compatible Bailian endpoint for the selected region."""
    return (
        os.getenv("DASHSCOPE_BASE_URL", "").strip()
        or DEFAULT_DASHSCOPE_BASE_URL
    )


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        api_key = require_dashscope_api_key()
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=api_key,
            base_url=get_dashscope_base_url(),
            extra_body={"enable_thinking": False},
            max_retries=2,
        )


class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        require_dashscope_api_key()
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingFactory().generator()
