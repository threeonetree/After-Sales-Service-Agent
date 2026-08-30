"""Explicitly rebuild the local vector index with the configured embedding model."""

from __future__ import annotations

import argparse

from utils.config_handler import rag_conf
from utils.model_errors import user_facing_model_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the local Chroma knowledge index")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of the generated local vector index before rebuilding",
    )
    args = parser.parse_args()
    if not args.yes:
        print(
            "This replaces only the generated chroma_db index; source files in data/ are kept.\n"
            "Run again with --yes to continue."
        )
        return 2

    try:
        from rag.vector_store import VectorStoreService

        service = VectorStoreService()
        service.reset_index()
        stats = service.load_document()
    except Exception as error:
        print(user_facing_model_error(error))
        return 1

    print(f"Embedding model: {rag_conf['embedding_model_name']}")
    print(
        "Index rebuilt: "
        f"added_files={stats['added_files']}, chunks={stats['chunks']}, "
        f"total_chunks={service.collection_count()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
