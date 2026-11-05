from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_optional(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_raw(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    value = _env(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数") from exc


def _env_optional_int(name: str) -> int | None:
    value = _env_optional(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数") from exc


def _env_float(name: str, default: float) -> float:
    value = _env(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字") from exc


def _env_optional_float(name: str) -> float | None:
    value = _env_optional(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "true" if default else "false").lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是布尔值")


def _env_list(name: str) -> tuple[str, ...]:
    value = _env_optional(name)
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class RagSettings:
    """在线 RAG 服务配置，全部可通过环境变量覆盖。"""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "enterprise_knowledge_bge_m3"
    qdrant_text_index_field: str = "content"
    qdrant_enable_text_index: bool = True

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = "auto"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_api_base_url: str = "http://localhost:8000/v1"
    embedding_api_key: str | None = None
    embedding_dimension: int | None = None
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""

    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout: float = 120.0

    rag_top_k: int = 5
    rag_score_threshold: float | None = None
    rag_max_context_chars: int = 12000
    rag_retrieval_mode: str = "hybrid"
    rag_candidate_multiplier: int = 4
    rag_max_candidates: int = 100
    rag_rrf_k: int = 60
    rag_dense_weight: float = 1.0
    rag_keyword_weight: float = 0.8

    api_title: str = "Enterprise RAG API"
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        provider = self.embedding_provider.strip().lower().replace("-", "_")
        if provider not in {
            "sentence_transformers",
            "sentence_transformer",
            "local",
            "bge",
            "openai",
            "openai_compatible",
            "api",
        }:
            raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {self.embedding_provider}")
        if not self.qdrant_collection.strip():
            raise ValueError("QDRANT_COLLECTION 不能为空")
        if not self.qdrant_text_index_field.strip():
            raise ValueError("QDRANT_TEXT_INDEX_FIELD 不能为空")
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE 必须为正整数")
        if self.embedding_dimension is not None and self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION 必须为正整数")
        if not self.llm_model.strip():
            raise ValueError("LLM_MODEL 不能为空")
        if self.llm_max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS 必须为正整数")
        if self.llm_timeout <= 0:
            raise ValueError("LLM_TIMEOUT 必须为正数")
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM_TEMPERATURE 必须在 0 到 2 之间")
        if not 1 <= self.rag_top_k <= 50:
            raise ValueError("RAG_TOP_K 必须在 1 到 50 之间")
        if self.rag_score_threshold is not None and not (
            -1 <= self.rag_score_threshold <= 1
        ):
            raise ValueError("RAG_SCORE_THRESHOLD 必须在 -1 到 1 之间")
        if self.rag_max_context_chars <= 0:
            raise ValueError("RAG_MAX_CONTEXT_CHARS 必须为正整数")
        retrieval_mode = (
            self.rag_retrieval_mode.strip().lower().replace("-", "_")
        )
        if retrieval_mode not in {"dense", "hybrid", "multi_route"}:
            raise ValueError("RAG_RETRIEVAL_MODE 必须是 dense 或 hybrid")
        if self.rag_candidate_multiplier <= 0:
            raise ValueError("RAG_CANDIDATE_MULTIPLIER 必须为正整数")
        if self.rag_max_candidates <= 0:
            raise ValueError("RAG_MAX_CANDIDATES 必须为正整数")
        if self.rag_rrf_k <= 0:
            raise ValueError("RAG_RRF_K 必须为正整数")
        if self.rag_dense_weight < 0 or self.rag_keyword_weight < 0:
            raise ValueError("检索路权重不能为负数")
        if self.rag_dense_weight == 0 and self.rag_keyword_weight == 0:
            raise ValueError("dense 和 keyword 权重不能同时为 0")

    @classmethod
    def from_env(cls) -> "RagSettings":
        return cls(
            qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=_env_optional("QDRANT_API_KEY"),
            qdrant_collection=_env(
                "QDRANT_COLLECTION", "enterprise_knowledge_bge_m3"
            ),
            qdrant_text_index_field=_env("QDRANT_TEXT_INDEX_FIELD", "content"),
            qdrant_enable_text_index=_env_bool("QDRANT_ENABLE_TEXT_INDEX", True),
            embedding_provider=_env(
                "EMBEDDING_PROVIDER", "sentence_transformers"
            ),
            embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_device=_env_optional("EMBEDDING_DEVICE") or "auto",
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
            embedding_normalize=_env_bool("EMBEDDING_NORMALIZE", True),
            embedding_api_base_url=_env(
                "EMBEDDING_API_BASE_URL", "http://localhost:8000/v1"
            ),
            embedding_api_key=_env_optional("EMBEDDING_API_KEY"),
            embedding_dimension=_env_optional_int("EMBEDDING_DIMENSION"),
            embedding_query_prefix=_env_raw("EMBEDDING_QUERY_PREFIX", ""),
            embedding_document_prefix=_env_raw("EMBEDDING_DOCUMENT_PREFIX", ""),
            llm_base_url=_env("LLM_BASE_URL", "http://localhost:8000/v1"),
            llm_api_key=_env_optional("LLM_API_KEY"),
            llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
            llm_timeout=_env_float("LLM_TIMEOUT", 120.0),
            rag_top_k=_env_int("RAG_TOP_K", 5),
            rag_score_threshold=_env_optional_float("RAG_SCORE_THRESHOLD"),
            rag_max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 12000),
            rag_retrieval_mode=_env("RAG_RETRIEVAL_MODE", "hybrid"),
            rag_candidate_multiplier=_env_int("RAG_CANDIDATE_MULTIPLIER", 4),
            rag_max_candidates=_env_int("RAG_MAX_CANDIDATES", 100),
            rag_rrf_k=_env_int("RAG_RRF_K", 60),
            rag_dense_weight=_env_float("RAG_DENSE_WEIGHT", 1.0),
            rag_keyword_weight=_env_float("RAG_KEYWORD_WEIGHT", 0.8),
            api_title=_env("API_TITLE", "Enterprise RAG API"),
            cors_origins=_env_list("API_CORS_ORIGINS"),
        )


__all__ = ["RagSettings"]
