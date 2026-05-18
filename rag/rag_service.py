"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给大模型
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate


def print_prompt(prompt):
    logger.debug(f"RAG Prompt:\n{'='*20}\n{prompt.to_string()}\n{'='*20}")
    return prompt

class RagSummaryService(object):
    def __init__(self):
        #向量存储
        self.vector_store = VectorStoreService()
        # 初始化时加载文档到向量库
        self.vector_store.load_document()
        self.retriever = self.vector_store.get_retrieve()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.__init_chain()

    def __init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summary(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 如果没有检索到任何资料，返回明确标识
        if not context_docs:
            return "无相关资料"

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )


if __name__ == '__main__':
    rag = RagSummaryService()

    print(rag.rag_summary("小户型适合那些机器人"))