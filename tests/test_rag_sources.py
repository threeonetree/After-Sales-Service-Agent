"""Keyword-only retrieval must preserve the original document citation."""
import importlib
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest


@pytest.mark.parametrize("metadata", [{"source": "手册.txt"}, None])
def test_bm25_retains_source_metadata(metadata):
    factory = ModuleType("model.factory")
    factory.chat_model = Mock()
    store = ModuleType("rag.vector_store")
    store.VectorStoreService = Mock()
    with patch.dict(sys.modules, {"model.factory": factory, "rag.vector_store": store}):
        sys.modules.pop("rag.rag_service", None)
        module = importlib.import_module("rag.rag_service")
        service = module.RagSummerizeService.__new__(module.RagSummerizeService)
        service.vector_store = Mock()
        service.vector_store.vector_store.get.return_value = {
            "documents": ["滚刷 清理 毛发"], "metadatas": [metadata],
        }
        service._init_bm25()
        doc = service.bm25_retriever.invoke("滚刷")[0]
        assert doc.metadata == (metadata or {})
