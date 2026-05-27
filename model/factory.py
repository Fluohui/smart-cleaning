import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from utils.config_handler import rag_conf
from utils.logger_handler import get_logger

logger = get_logger("factory")


def _validate_api_key():
    missing = []
    if not os.getenv("DASHSCOPE_API_KEY"):
        missing.append("DASHSCOPE_API_KEY")
    if not os.getenv("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")
    if missing:
        logger.error("缺少 API Key: %s", ", ".join(missing))
        raise RuntimeError(
            f"缺少 API Key: {', '.join(missing)}，请通过环境变量或 .env 文件配置。"
        )


_validate_api_key()


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )


class EmbeddingModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingModelFactory().generator()
