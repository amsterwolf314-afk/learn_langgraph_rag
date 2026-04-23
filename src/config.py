import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
VECTORSTORE_PATH = CACHE_DIR / "blog_vectorstore.json"
VECTORSTORE_META_PATH = CACHE_DIR / "blog_vectorstore.meta.json"

BLOG_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "100"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Please set it in the environment, ~/.zshrc, or .env before running the project."
        )
    return value


@lru_cache(maxsize=1)
def build_embeddings() -> OpenAIEmbeddings:
    params = {
        "model": os.getenv("EMBED_MODEL_NAME", "text-embedding-v3"),
        "api_key": _get_required_env("EMBED_API_KEY"),
        "check_embedding_ctx_length": False,
        "chunk_size": 10,
    }
    base_url = os.getenv("EMBED_BASE_URL")
    if base_url:
        params["base_url"] = base_url

    return OpenAIEmbeddings(**params)


def build_chat_model():
    return init_chat_model(
        model=_get_required_env("LLM_MODEL_ID"),
        model_provider="openai",
        api_key=_get_required_env("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
    )


def get_vectorstore_cache_manifest() -> dict[str, object]:
    return {
        "blog_urls": BLOG_URLS,
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "embed_model_name": os.getenv("EMBED_MODEL_NAME", "text-embedding-v3"),
        "embed_base_url": os.getenv("EMBED_BASE_URL"),
    }


response_model = build_chat_model()
