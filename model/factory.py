from abc import ABC,abstractmethod
from typing import Optional, Union
from utils.config_handler import rag_conf

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingFactory().generator()