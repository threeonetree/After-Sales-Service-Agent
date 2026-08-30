'''
总结服务类：用户提问，搜索参考资料，将提问和参考资料交给模型，让模型总结回复
'''
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import BM25Retriever

from .vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


class RagSummerizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_txt = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_txt)
        self.model = chat_model
        self.chain = self.__init__chain()
        self._init_bm25()

    def _init_bm25(self):
        """从Chroma加载全部文档构建BM25关键词索引"""
        data = self.vector_store.vector_store.get()
        texts = data.get('documents', [])
        if texts:
            self.bm25_retriever = BM25Retriever.from_texts(texts)
            self.bm25_retriever.k = 5
        else:
            self.bm25_retriever = None


    def __init__chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def _rrf_fusion(self, vector_docs, bm25_docs, k=60):
        """RRF融合排序：对向量检索和关键词检索结果去重重排"""
        scores = {}
        meta_map = {}
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            meta_map[key] = doc.metadata
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in meta_map:
                meta_map[key] = doc.metadata
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [Document(page_content=text, metadata=meta_map.get(text, {})) for text, _ in sorted_items[:3]]

    def retriever_docs(self, query: str)-> list[Document]:   #query：用户提问
        vector_docs = self.vector_store.vector_store.similarity_search(query, k=10)
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            return self._rrf_fusion(vector_docs, bm25_docs)
        return vector_docs[:3]

    def rag_summarize(self, query: str)-> str:
        context_docs = self.retriever_docs(query)

        if not context_docs:
            return (
                "本地知识库尚未初始化。请先运行 "
                "python -m rag.rebuild_index --yes，再重试知识库问题。"
            )

        context = ""
        counter = 1
        for doc in context_docs:
            context += f"【参考资料{counter}】:参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            counter += 1

        return self.chain.invoke(
            {
                "input":query,
                "context":context
            }
        )


if __name__ == '__main__':
    rag_service = RagSummerizeService()
    print(rag_service.rag_summarize("小户型适合什么机器？"))
