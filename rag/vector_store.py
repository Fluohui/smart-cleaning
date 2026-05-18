import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger

class VectorStoreService:
    def __init__(self):
        self.vector_story = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"]
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )

    def get_retrieve(self):
        return self.vector_story.as_retriever(search_kwargs={"k": chroma_conf["k"]})


    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量纯如向量库
        计算文件md5去重
        :return: None
        """
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            #未处理md5

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5已处理

                return False            # 未处理md5

        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}已存在知识库内，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}为空文件，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效内容，跳过")
                    continue

                self.vector_story.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载成功")

            except Exception as e:
                # exc_info 为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}内容加载失败，{str(e)}", exc_info=True)
                continue



if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retrieve()
    res = retriever.invoke("迷路")
    for i in res:
        print(i.page_content)
        print("-"*20)
