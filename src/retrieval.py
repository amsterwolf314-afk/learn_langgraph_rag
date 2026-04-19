from functools import lru_cache

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

from .config import BLOG_URLS, build_embeddings


@lru_cache(maxsize=1)
def load_documents():
    docs = [WebBaseLoader(url).load() for url in BLOG_URLS]
    return [item for sublist in docs for item in sublist]


@lru_cache(maxsize=1)
def get_doc_splits():
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    return text_splitter.split_documents(load_documents())


@lru_cache(maxsize=1)
def get_retriever():
    vectorstore = InMemoryVectorStore.from_documents(
        documents=get_doc_splits(), embedding=build_embeddings()
    )
    return vectorstore.as_retriever()


@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    docs = get_retriever().invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
