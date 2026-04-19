import json
import os
from functools import lru_cache

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

from .config import (
    BLOG_URLS,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    VECTORSTORE_META_PATH,
    VECTORSTORE_PATH,
    build_embeddings,
    get_vectorstore_cache_manifest,
)


@lru_cache(maxsize=1)
def load_documents():
    docs = [WebBaseLoader(url).load() for url in BLOG_URLS]
    return [item for sublist in docs for item in sublist]


@lru_cache(maxsize=1)
def get_doc_splits():
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP
    )
    return text_splitter.split_documents(load_documents())


def _load_cache_metadata() -> dict | None:
    if not VECTORSTORE_META_PATH.exists():
        return None
    try:
        return json.loads(VECTORSTORE_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _has_fresh_vectorstore_cache() -> bool:
    if not VECTORSTORE_PATH.exists():
        return False
    metadata = _load_cache_metadata()
    if metadata is None:
        return False
    return metadata.get("cache_manifest") == get_vectorstore_cache_manifest()


def _write_cache_metadata(document_count: int) -> None:
    VECTORSTORE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_manifest": get_vectorstore_cache_manifest(),
        "document_count": document_count,
    }
    VECTORSTORE_META_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_and_cache_vectorstore(embedding) -> InMemoryVectorStore:
    docs = get_doc_splits()
    vectorstore = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embedding,
    )
    vectorstore.dump(str(VECTORSTORE_PATH))
    _write_cache_metadata(document_count=len(docs))
    return vectorstore


def ensure_local_vectorstore(force_rebuild: bool = False) -> tuple[InMemoryVectorStore, bool]:
    embedding = build_embeddings()
    if not force_rebuild and _has_fresh_vectorstore_cache():
        return InMemoryVectorStore.load(str(VECTORSTORE_PATH), embedding=embedding), False
    return _build_and_cache_vectorstore(embedding), True


@lru_cache(maxsize=1)
def get_vectorstore():
    vectorstore, _ = ensure_local_vectorstore()
    return vectorstore


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever()


@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    docs = get_retriever().invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    force_rebuild = os.getenv("FORCE_REBUILD_VECTORSTORE") == "1"
    vectorstore, rebuilt = ensure_local_vectorstore(force_rebuild=force_rebuild)
    status = "rebuilt" if rebuilt else "loaded from cache"
    print(
        f"Vector store {status}: {VECTORSTORE_PATH} "
        f"({len(vectorstore.store)} chunks)"
    )
