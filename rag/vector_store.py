import os.path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import pdf_loader,txt_loader,listdir_with_allowed_type,get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
            embedding_function=embed_model
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})



    def load_document(self):

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False


        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith(".txt"):
                return txt_loader(read_path)
            else:
                return []

        allowed_files_type:list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            allowed_types=tuple(chroma_conf["allow_knowledge_file_type"])
        )
        for file_path in allowed_files_type:
            md5_hex = get_file_md5_hex(file_path)
            if check_md5_hex(md5_hex):
                logger.info(f"[load_document]文件{file_path}已存在，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(file_path)
                if not documents:
                    logger.info(f"[load_document]文件{file_path}没有内容，跳过")
                    continue

                split_document: list[Document] = self.splitter.split_documents(documents)

                if not split_document:
                    logger.info(f"[load_document]文件{file_path}分片后没有内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)  # 入库
                save_md5_hex(md5_hex)  # 保存md5_hex

                logger.info(f"[load_document]文件{file_path}入库完成")
            except Exception as e:
                #exc_info=True,会详细输出报错堆栈，false只输出报错信息
                logger.error(f"[load_document]文件{file_path}入库失败：{str(e)}", exc_info=True)


if __name__ == '__main__':
    vs=VectorStoreService()
    vs.load_document()

    retriever=vs.get_retriever()

    res = retriever.invoke("如何使用langchain?")
    for r in res:
        print(r.page_content)
        print("="*20)

