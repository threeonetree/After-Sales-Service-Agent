from abc import ABC, abstractmethod
from typing import Optional, Union

from dotenv import load_dotenv
from utils.config_handler import rag_conf
from utils.model_errors import require_dashscope_api_key
from utils.path_tool import get_abs_path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel


# Local .env is optional and gitignored. Existing OS environment variables win.
load_dotenv(get_abs_path(".env"), override=False)


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        require_dashscope_api_key()
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            model_kwargs={"enable_thinking": False},
        )


class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        require_dashscope_api_key()
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingFactory().generator()
