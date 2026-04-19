import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BLOG_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


@lru_cache(maxsize=1)
def build_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("EMBED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing EMBED_API_KEY. Please set it in the environment, ~/.zshrc, or .env before running rag.py."
        )

    params = {
        "model": os.getenv("EMBED_MODEL_NAME", "text-embedding-v3"),
        "api_key": api_key,
        "check_embedding_ctx_length": False,
        "chunk_size": 10,
    }
    base_url = os.getenv("EMBED_BASE_URL")
    if base_url:
        params["base_url"] = base_url

    return OpenAIEmbeddings(**params)


def build_chat_model():
    return init_chat_model(
        model=os.getenv("LLM_MODEL_ID"),
        model_provider="openai",
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
    )


response_model = build_chat_model()
