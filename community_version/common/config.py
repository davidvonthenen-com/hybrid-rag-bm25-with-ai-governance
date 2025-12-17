"""Configuration loading utilities for the RAG service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _get_str(name: str, default_val: str) -> str:
    """Return an environment variable or ``default_val`` if unset."""

    value = os.getenv(name)
    return default_val if value is None else value


def _get_int(name: str, default_val: int) -> int:
    """Safely coerce an environment variable to ``int`` with fallback."""

    value = os.getenv(name)
    if value is None or value == "":
        return default_val
    try:
        return int(value)
    except ValueError:
        return default_val


def _get_bool(name: str, default_val: bool) -> bool:
    """Return a boolean flag parsed from common truthy strings."""

    value = os.getenv(name)
    if value is None:
        return default_val
    return value.lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default_val: float) -> float:
    """Safely coerce an environment variable to ``float`` with fallback."""

    value = os.getenv(name)
    if value is None or value == "":
        return default_val
    try:
        return float(value)
    except ValueError:
        return default_val


@dataclass(frozen=True)
class Settings:
    """Runtime configuration parameters for the application."""

    # OpenSearch (vector store)
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9201
    opensearch_vector_index: str = "bbc-vector-chunks"

    # full index for long-term storage and BM25
    opensearch_full_index: str = "bbc-bm25-full"

    # OpenSearch (lexical / ranking stores)
    opensearch_long_host: str = "127.0.0.1"
    opensearch_long_port: int = 9201
    opensearch_long_index: str = "bbc-bm25-chunks"
    opensearch_long_user: str = ""
    opensearch_long_password: str = ""
    opensearch_long_ssl: bool = False

    opensearch_hot_host: str = "127.0.0.1"
    opensearch_hot_port: int = 9202
    opensearch_hot_index: str = "bbc-bm25-chunks"
    opensearch_hot_user: str = ""
    opensearch_hot_password: str = ""
    opensearch_hot_ssl: bool = False

    # Retrieval / ranking parameters
    search_size: int = 10                      # candidate docs per store
    ranking_alpha: float = 0.5                 # per-store OpenSearch score threshold
    search_preference: str = "governance-audit-v1"
    os_explain: bool = False
    os_profile: bool = False

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # use thenlper/gte-small or Qwen/Qwen3-Embedding-0.6B

    # Llama.cpp
    llama_model_path: str = "neural-chat-7b-v3-3.Q4_K_M.gguf"
    llama_ctx: int = 65536                    # keep conservative by default on laptops 32768 or 65536
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llama_n_gpu_layers: int = 20             # modest offload; fallback logic drops to CPU if needed
    llama_n_batch: int = 256                 # prompt processing batch
    llama_n_ubatch: Optional[int] = 256      # physical micro-batch; None to let llama.cpp choose
    llama_low_vram: bool = True              # reduce Metal VRAM usage

    # Named entity recognition service
    ner_url: str = "http://127.0.0.1:8000/ner"
    ner_timeout_secs: float = 5.0

    # RAG
    rag_top_k: int = 3
    rag_num_candidates: int = 50

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000


def load_settings() -> Settings:
    """Load settings from environment variables (with .env support)."""
    load_dotenv()
    return Settings(
        # Vector OpenSearch
        opensearch_host=os.getenv("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_vector_index=os.getenv("OPENSEARCH_VECTOR_INDEX", Settings.opensearch_vector_index),

        # Lexical / ranking OpenSearch stores
        opensearch_full_index=os.getenv(
            "OPENSEARCH_FULL_INDEX", Settings.opensearch_full_index
        ),
        opensearch_long_host=os.getenv(
            "OPENSEARCH_LONG_HOST", Settings.opensearch_long_host
        ),
        opensearch_long_port=_get_int(
            "OPENSEARCH_LONG_PORT", Settings.opensearch_long_port
        ),
        opensearch_long_index=os.getenv(
            "OPENSEARCH_LONG_INDEX", Settings.opensearch_long_index
        ),
        opensearch_long_user=os.getenv(
            "OPENSEARCH_LONG_USER", Settings.opensearch_long_user
        ),
        opensearch_long_password=os.getenv(
            "OPENSEARCH_LONG_PASS", Settings.opensearch_long_password
        ),
        opensearch_long_ssl=_get_bool(
            "OPENSEARCH_LONG_SSL", Settings.opensearch_long_ssl
        ),
        opensearch_hot_host=os.getenv(
            "OPENSEARCH_HOT_HOST", Settings.opensearch_hot_host
        ),
        opensearch_hot_port=_get_int(
            "OPENSEARCH_HOT_PORT", Settings.opensearch_hot_port
        ),
        opensearch_hot_index=os.getenv(
            "OPENSEARCH_HOT_INDEX", Settings.opensearch_hot_index
        ),
        opensearch_hot_user=os.getenv(
            "OPENSEARCH_HOT_USER", Settings.opensearch_hot_user
        ),
        opensearch_hot_password=os.getenv(
            "OPENSEARCH_HOT_PASS", Settings.opensearch_hot_password
        ),
        opensearch_hot_ssl=_get_bool(
            "OPENSEARCH_HOT_SSL", Settings.opensearch_hot_ssl
        ),

        # Embeddings
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),

        # LLaMA
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=_get_int("LLAMA_CTX", Settings.llama_ctx),
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", Settings.llama_n_gpu_layers),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        llama_low_vram=_get_bool("LLAMA_LOW_VRAM", Settings.llama_low_vram),

        # Retrieval / ranking
        search_size=_get_int("SEARCH_SIZE", Settings.search_size),
        ranking_alpha=_get_float("RANKING_ALPHA", Settings.ranking_alpha),
        search_preference=os.getenv(
            "PREFERENCE_TOKEN", Settings.search_preference
        ),
        os_explain=_get_bool("OS_EXPLAIN", Settings.os_explain),
        os_profile=_get_bool("OS_PROFILE", Settings.os_profile),

        # RAG
        rag_top_k=_get_int("RAG_TOP_K", Settings.rag_top_k),
        rag_num_candidates=_get_int("RAG_NUM_CANDIDATES", Settings.rag_num_candidates),

        # NER
        ner_url=os.getenv("NER_URL", Settings.ner_url),
        ner_timeout_secs=_get_float("NER_TIMEOUT_SECS", Settings.ner_timeout_secs),

        # Server
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),
    )


__all__ = ["Settings", "load_settings"]
