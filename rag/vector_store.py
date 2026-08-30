"""Persistent Chroma index management for the local knowledge base."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf, rag_conf
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path, get_project_root


MANIFEST_VERSION = 1
EMBEDDING_BATCH_SIZE = 20


class VectorIndexError(RuntimeError):
    """Raised when a persisted index cannot be safely used."""


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
            embedding_function=embed_model,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self.manifest_path = Path(get_abs_path(chroma_conf["index_manifest"]))

    def get_retriever(self):
        self.assert_index_compatible()
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def collection_count(self) -> int:
        data = self.vector_store.get(include=[])
        return len(data.get("ids", []))

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "embedding_model": rag_conf["embedding_model_name"],
            "files": {},
        }

    def _load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            with self.manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise VectorIndexError(
                "向量索引清单损坏，请运行 python -m rag.rebuild_index --yes 重建。"
            ) from error
        if not isinstance(manifest, dict):
            raise VectorIndexError(
                "向量索引清单格式无效，请运行 python -m rag.rebuild_index --yes 重建。"
            )
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.manifest_path)

    def assert_index_compatible(self) -> None:
        """Refuse to mix vectors created by different embedding models."""
        count = self.collection_count()
        manifest = self._load_manifest()
        if count == 0:
            return
        if manifest is None:
            raise VectorIndexError(
                "检测到旧版向量库，但缺少模型清单。请运行 "
                "python -m rag.rebuild_index --yes 使用当前 Embedding 模型重建。"
            )
        indexed_model = manifest.get("embedding_model")
        current_model = rag_conf["embedding_model_name"]
        if indexed_model != current_model:
            raise VectorIndexError(
                f"向量库由 {indexed_model or '未知模型'} 创建，当前配置为 {current_model}。"
                "不同 Embedding 模型不能混用，请运行 python -m rag.rebuild_index --yes。"
            )

    @staticmethod
    def _get_file_documents(read_path: str) -> list[Document]:
        if read_path.endswith(".pdf"):
            return pdf_loader(read_path)
        if read_path.endswith(".txt"):
            return txt_loader(read_path)
        return []

    @staticmethod
    def _source_key(file_path: str) -> str:
        return os.path.relpath(file_path, get_project_root()).replace(os.sep, "/")

    def _add_documents_in_batches(self, documents: list[Document]) -> list[str]:
        """Embed documents within Bailian's per-request batch limit."""
        document_ids: list[str] = []
        try:
            for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
                batch = documents[start : start + EMBEDDING_BATCH_SIZE]
                document_ids.extend(self.vector_store.add_documents(batch))
        except Exception:
            # Prevent an interrupted file from leaving untracked vectors behind.
            if document_ids:
                self.vector_store.delete(ids=document_ids)
            raise
        return document_ids

    def load_document(self) -> dict[str, int]:
        """Synchronize knowledge files into Chroma and return operation counts."""
        self.assert_index_compatible()
        manifest = self._load_manifest() or self._empty_manifest()
        if self.collection_count() == 0:
            # A stale manifest without a database must never suppress ingestion.
            manifest = self._empty_manifest()

        manifest["version"] = MANIFEST_VERSION
        manifest["embedding_model"] = rag_conf["embedding_model_name"]
        indexed_files = manifest.setdefault("files", {})
        allowed_files = sorted(
            listdir_with_allowed_type(
                get_abs_path(chroma_conf["data_path"]),
                allowed_types=tuple(chroma_conf["allow_knowledge_file_type"]),
            )
        )
        current_sources = {self._source_key(path) for path in allowed_files}
        stats = {"added_files": 0, "updated_files": 0, "removed_files": 0, "chunks": 0}

        for source_key in list(indexed_files):
            if source_key in current_sources:
                continue
            old_ids = indexed_files[source_key].get("document_ids", [])
            if old_ids:
                self.vector_store.delete(ids=old_ids)
            del indexed_files[source_key]
            stats["removed_files"] += 1

        for file_path in allowed_files:
            source_key = self._source_key(file_path)
            md5_hex = get_file_md5_hex(file_path)
            old_entry = indexed_files.get(source_key, {})
            if md5_hex and old_entry.get("md5") == md5_hex:
                logger.info("[load_document]文件%s已存在，跳过", file_path)
                continue

            try:
                documents = self._get_file_documents(file_path)
                split_documents = self.splitter.split_documents(documents) if documents else []
                if not split_documents:
                    logger.info("[load_document]文件%s没有可入库内容，跳过", file_path)
                    continue

                old_ids = old_entry.get("document_ids", [])
                document_ids = self._add_documents_in_batches(split_documents)
                if old_ids:
                    self.vector_store.delete(ids=old_ids)
                indexed_files[source_key] = {
                    "md5": md5_hex,
                    "document_ids": document_ids,
                }
                stats["updated_files" if old_entry else "added_files"] += 1
                stats["chunks"] += len(document_ids)
                self._save_manifest(manifest)
                logger.info("[load_document]文件%s入库完成", file_path)
            except Exception as error:
                logger.error(
                    "[load_document]文件%s入库失败：%s", file_path, str(error), exc_info=True
                )
                raise

        self._save_manifest(manifest)
        return stats

    def reset_index(self) -> None:
        """Delete only the generated Chroma collection and its local manifest."""
        self.vector_store.reset_collection()
        self.manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
    service = VectorStoreService()
    print(service.load_document())
