import importlib
import os
import sys
import unittest
from unittest.mock import Mock, patch


class VectorStoreBatchingTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("rag.vector_store", None)
        sys.modules.pop("model.factory", None)

    @staticmethod
    def _load_service_class():
        module = importlib.import_module("rag.vector_store")
        return module.VectorStoreService

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}, clear=True)
    def test_documents_are_sent_in_batches_of_at_most_twenty(self):
        service_class = self._load_service_class()
        service = service_class.__new__(service_class)
        service.vector_store = Mock()
        next_id = 0

        def add_documents(batch):
            nonlocal next_id
            ids = [f"doc-{index}" for index in range(next_id, next_id + len(batch))]
            next_id += len(batch)
            return ids

        service.vector_store.add_documents.side_effect = add_documents

        document_ids = service._add_documents_in_batches([object()] * 45)

        self.assertEqual(
            [len(call.args[0]) for call in service.vector_store.add_documents.call_args_list],
            [20, 20, 5],
        )
        self.assertEqual(len(document_ids), 45)

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}, clear=True)
    def test_partial_file_is_rolled_back_when_a_later_batch_fails(self):
        service_class = self._load_service_class()
        service = service_class.__new__(service_class)
        service.vector_store = Mock()
        service.vector_store.add_documents.side_effect = [
            [f"doc-{index}" for index in range(20)],
            RuntimeError("embedding failed"),
        ]

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            service._add_documents_in_batches([object()] * 25)

        service.vector_store.delete.assert_called_once_with(
            ids=[f"doc-{index}" for index in range(20)]
        )


if __name__ == "__main__":
    unittest.main()
