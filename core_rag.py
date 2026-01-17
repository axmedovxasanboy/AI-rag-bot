import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGBot:
    
    _rag_chain = None
    
    def __init__(self):
        self.api_key = "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"
        self.data_path = "/home/xasanboy-axmedov/Linux Ubuntu/Python/AI RAG/data/"
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)

    def get_chain(self):
        
        if RAGBot._rag_chain is not None:
            logger.info("Returning cached RAG chain.")
            return RAGBot._rag_chain
        
        logger.info("Indexing new documents...")
        logger.info(f"API: {self.api_key[:30]}, data path: {self.data_path}, FAISS: {self.index_path}")
        
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.doc",
            loader_cls=UnstructuredFileLoader,
            loader_kwargs={"mode": "single"} 
        )
        
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key, temperature=0)
        
        system_prompt = (
            "Siz o'zbek tilidagi hujjatlar bo'yicha mutaxassis yordamchisiz. "
            "Faqat taqdim etilgan kontekstdan foydalanib javob bering. "
            "Javobni o'zbek tilida yozing.\n\n"
            "Kontekst: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        RAGBot._rag_chain = rag_chain
        return rag_chain