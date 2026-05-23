import json
import os
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_types, get_file_md5_hex
from utils.logger_handler import logger

PARENT_DOCS_PATH = get_abs_path("data/parent_docs.json")


def _load_parent_docstore() -> dict[str, Document]:
    if not os.path.exists(PARENT_DOCS_PATH):
        return {}
    with open(PARENT_DOCS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        pid: Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
        for pid, d in raw.items()
    }


def _save_parent_docstore(docstore: dict[str, Document]):
    os.makedirs(os.path.dirname(PARENT_DOCS_PATH), exist_ok=True)
    raw = {
        pid: {"page_content": doc.page_content, "metadata": doc.metadata}
        for pid, doc in docstore.items()
    }
    with open(PARENT_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


class ParentDocRetriever:
    """父子文档检索器：子块向量匹配 → 返回完整父块上下文"""

    def __init__(self, store_service: "VectorStoreService"):
        self.vs = store_service

    def invoke(self, query: str) -> list[Document]:
        child_docs = self.vs.vector_store.similarity_search(
            query, k=chroma_conf["k"]
        )

        parent_ids = list(dict.fromkeys([
            doc.metadata.get("parent_id") for doc in child_docs
            if doc.metadata.get("parent_id")
        ]))

        if not parent_ids:
            return []

        result = [
            self.vs._parent_docs[pid]
            for pid in parent_ids
            if pid in self.vs._parent_docs
        ]

        logger.info(
            "[父子检索] %d 个子块命中 → %d 个父文档",
            len(child_docs), len(result)
        )
        return result


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_parent_size"],
            chunk_overlap=chroma_conf["chunk_parent_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        self._parent_docs: dict[str, Document] = _load_parent_docstore()

    def get_retrieve(self):
        return ParentDocRetriever(self)

    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_types(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info("[加载知识库] %s 已存在，跳过", path)
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning("[加载知识库] %s 为空文件，跳过", path)
                    continue

                # 1. 父块切割（大块，保留完整上下文）
                parent_chunks = self.parent_splitter.split_documents(documents)

                child_chunks_to_add: list[Document] = []
                for parent in parent_chunks:
                    parent_id = str(uuid.uuid4())
                    parent.metadata["parent_id"] = parent_id

                    # 2. 子块切割（小块，用于精准向量匹配）
                    children = self.child_splitter.split_documents([parent])
                    for child in children:
                        child.metadata["parent_id"] = parent_id

                    child_chunks_to_add.extend(children)
                    self._parent_docs[parent_id] = parent

                # 3. 子块入 Chroma（向量检索用）
                self.vector_store.add_documents(child_chunks_to_add)

                # 4. 父块持久化到 JSON（重启不丢失）
                _save_parent_docstore(self._parent_docs)

                save_md5_hex(md5_hex)
                logger.info(
                    "[加载知识库] %s → %d 个父块 / %d 个子块，入库完成",
                    path, len(parent_chunks), len(child_chunks_to_add),
                )

            except Exception as e:
                logger.error("[加载知识库] %s 处理失败: %s", path, e, exc_info=True)


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retrieve()
    res = retriever.invoke("迷路")
    for doc in res:
        print(f"【父块，{len(doc.page_content)} 字符】:\n{doc.page_content}")
        print("-" * 40)
