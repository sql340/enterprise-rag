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
    qdrant_create_payload_indexes: bool = True
    rag_bootstrap_search_indexes: bool = True

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
    rag_retrieval_mode: str = "hybrid_graph"
    rag_candidate_multiplier: int = 4
    rag_max_candidates: int = 100
    rag_rrf_k: int = 60
    rag_dense_weight: float = 1.0
    rag_keyword_weight: float = 0.8
    rag_graph_enabled: bool = True
    rag_graph_store_path: str = ".cache/knowledge_graph.sqlite3"
    rag_graph_weight: float = 0.8
    rag_graph_max_hops: int = 1
    rag_graph_entity_limit: int = 20
    rag_acl_enabled: bool = True
    rag_acl_proxy_token: str | None = None

    rag_governance_db_path: str = ".cache/knowledge_governance.sqlite3"
    rag_upload_dir: str = ".cache/knowledge_uploads"
    rag_admin_max_file_bytes: int = 50 * 1024 * 1024
    rag_ingestion_worker_enabled: bool = True
    rag_ingestion_poll_seconds: float = 1.0
    rag_ingestion_lease_seconds: int = 1800
    rag_ingestion_max_attempts: int = 3

    rag_rerank_enabled: bool = False
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rag_rerank_device: str | None = None
    rag_rerank_candidates: int = 30
    rag_expand_parent_context: bool = True

    rag_log_level: str = "INFO"
    rag_log_format: str = "json"
    rag_access_log_enabled: bool = True
    rag_metrics_enabled: bool = True
    rag_warmup_embedding: bool = True
    rag_dependency_health_cache_seconds: float = 15.0
    rag_dependency_health_timeout_seconds: float = 2.0
    rag_dependency_health_external_enabled: bool = True
    rag_dependency_health_fail_readiness: bool = False
    rag_dependency_health_llm_url: str | None = None
    rag_dependency_health_embedding_url: str | None = None

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
        if retrieval_mode not in {
            "dense",
            "hybrid",
            "graph",
            "hybrid_graph",
            "multi_route",
        }:
            raise ValueError(
                "RAG_RETRIEVAL_MODE 必须是 dense、hybrid、graph 或 hybrid_graph"
            )
        if self.rag_candidate_multiplier <= 0:
            raise ValueError("RAG_CANDIDATE_MULTIPLIER 必须为正整数")
        if self.rag_max_candidates <= 0:
            raise ValueError("RAG_MAX_CANDIDATES 必须为正整数")
        if self.rag_rrf_k <= 0:
            raise ValueError("RAG_RRF_K 必须为正整数")
        if (
            self.rag_dense_weight < 0
            or self.rag_keyword_weight < 0
            or self.rag_graph_weight < 0
        ):
            raise ValueError("检索路权重不能为负数")
        if retrieval_mode == "hybrid" and (
            self.rag_dense_weight == 0 and self.rag_keyword_weight == 0
        ):
            raise ValueError("hybrid 模式下 dense 和 keyword 权重不能同时为 0")
        if retrieval_mode == "hybrid_graph" and (
            self.rag_dense_weight == 0
            and self.rag_keyword_weight == 0
            and self.rag_graph_weight == 0
        ):
            raise ValueError("hybrid_graph 模式下三路检索权重不能同时为 0")
        if retrieval_mode == "graph" and not self.rag_graph_enabled:
            raise ValueError("graph 检索模式要求 RAG_GRAPH_ENABLED=true")
        if self.rag_graph_enabled and not self.rag_graph_store_path.strip():
            raise ValueError("RAG_GRAPH_STORE_PATH 不能为空")
        if not 1 <= self.rag_graph_max_hops <= 3:
            raise ValueError("RAG_GRAPH_MAX_HOPS 必须在 1 到 3 之间")
        if self.rag_graph_entity_limit <= 0:
            raise ValueError("RAG_GRAPH_ENTITY_LIMIT 必须为正整数")
        if not self.rag_governance_db_path.strip():
            raise ValueError("RAG_GOVERNANCE_DB_PATH 不能为空")
        if not self.rag_upload_dir.strip():
            raise ValueError("RAG_UPLOAD_DIR 不能为空")
        if self.rag_admin_max_file_bytes <= 0:
            raise ValueError("RAG_ADMIN_MAX_FILE_BYTES 必须为正整数")
        if self.rag_ingestion_poll_seconds <= 0:
            raise ValueError("RAG_INGESTION_POLL_SECONDS 必须为正数")
        if self.rag_ingestion_lease_seconds <= 0:
            raise ValueError("RAG_INGESTION_LEASE_SECONDS 必须为正整数")
        if self.rag_ingestion_max_attempts <= 0:
            raise ValueError("RAG_INGESTION_MAX_ATTEMPTS 必须为正整数")
        if not self.rag_rerank_model.strip():
            raise ValueError("RAG_RERANK_MODEL 不能为空")
        if self.rag_rerank_candidates <= 0:
            raise ValueError("RAG_RERANK_CANDIDATES 必须为正整数")
        if self.rag_log_format.strip().lower() not in {"json", "text"}:
            raise ValueError("RAG_LOG_FORMAT 必须是 json 或 text")
        if self.rag_log_level.strip().upper() not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ValueError(
                "RAG_LOG_LEVEL 必须是 CRITICAL/ERROR/WARNING/INFO/DEBUG"
            )
        if self.rag_dependency_health_cache_seconds <= 0:
            raise ValueError("RAG_DEPENDENCY_HEALTH_CACHE_SECONDS 必须为正数")
        if self.rag_dependency_health_timeout_seconds <= 0:
            raise ValueError("RAG_DEPENDENCY_HEALTH_TIMEOUT_SECONDS 必须为正数")
        if (
            self.rag_dependency_health_llm_url is not None
            and not self.rag_dependency_health_llm_url.strip()
        ):
            raise ValueError("RAG_DEPENDENCY_HEALTH_LLM_URL 不能为空")
        if (
            self.rag_dependency_health_embedding_url is not None
            and not self.rag_dependency_health_embedding_url.strip()
        ):
            raise ValueError("RAG_DEPENDENCY_HEALTH_EMBEDDING_URL 不能为空")

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
            qdrant_create_payload_indexes=_env_bool(
                "QDRANT_CREATE_PAYLOAD_INDEXES", True
            ),
            rag_bootstrap_search_indexes=_env_bool(
                "RAG_BOOTSTRAP_SEARCH_INDEXES", True
            ),
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
            rag_retrieval_mode=_env("RAG_RETRIEVAL_MODE", "hybrid_graph"),
            rag_candidate_multiplier=_env_int("RAG_CANDIDATE_MULTIPLIER", 4),
            rag_max_candidates=_env_int("RAG_MAX_CANDIDATES", 100),
            rag_rrf_k=_env_int("RAG_RRF_K", 60),
            rag_dense_weight=_env_float("RAG_DENSE_WEIGHT", 1.0),
            rag_keyword_weight=_env_float("RAG_KEYWORD_WEIGHT", 0.8),
            rag_graph_enabled=_env_bool("RAG_GRAPH_ENABLED", True),
            rag_graph_store_path=_env(
                "RAG_GRAPH_STORE_PATH", ".cache/knowledge_graph.sqlite3"
            ),
            rag_graph_weight=_env_float("RAG_GRAPH_WEIGHT", 0.8),
            rag_graph_max_hops=_env_int("RAG_GRAPH_MAX_HOPS", 1),
            rag_graph_entity_limit=_env_int("RAG_GRAPH_ENTITY_LIMIT", 20),
            rag_acl_enabled=_env_bool("RAG_ACL_ENABLED", True),
            rag_acl_proxy_token=_env_optional("RAG_ACL_PROXY_TOKEN"),
            rag_governance_db_path=_env(
                "RAG_GOVERNANCE_DB_PATH", ".cache/knowledge_governance.sqlite3"
            ),
            rag_upload_dir=_env("RAG_UPLOAD_DIR", ".cache/knowledge_uploads"),
            rag_admin_max_file_bytes=_env_int(
                "RAG_ADMIN_MAX_FILE_BYTES", 50 * 1024 * 1024
            ),
            rag_ingestion_worker_enabled=_env_bool(
                "RAG_INGESTION_WORKER_ENABLED", True
            ),
            rag_ingestion_poll_seconds=_env_float(
                "RAG_INGESTION_POLL_SECONDS", 1.0
            ),
            rag_ingestion_lease_seconds=_env_int(
                "RAG_INGESTION_LEASE_SECONDS", 1800
            ),
            rag_ingestion_max_attempts=_env_int(
                "RAG_INGESTION_MAX_ATTEMPTS", 3
            ),
            rag_rerank_enabled=_env_bool("RAG_RERANK_ENABLED", False),
            rag_rerank_model=_env(
                "RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            rag_rerank_device=_env_optional("RAG_RERANK_DEVICE"),
            rag_rerank_candidates=_env_int("RAG_RERANK_CANDIDATES", 30),
            rag_expand_parent_context=_env_bool(
                "RAG_EXPAND_PARENT_CONTEXT", True
            ),
            rag_log_level=_env("RAG_LOG_LEVEL", "INFO"),
            rag_log_format=_env("RAG_LOG_FORMAT", "json"),
            rag_access_log_enabled=_env_bool("RAG_ACCESS_LOG_ENABLED", True),
            rag_metrics_enabled=_env_bool("RAG_METRICS_ENABLED", True),
            rag_warmup_embedding=_env_bool("RAG_WARMUP_EMBEDDING", True),
            rag_dependency_health_cache_seconds=_env_float(
                "RAG_DEPENDENCY_HEALTH_CACHE_SECONDS", 15.0
            ),
            rag_dependency_health_timeout_seconds=_env_float(
                "RAG_DEPENDENCY_HEALTH_TIMEOUT_SECONDS", 2.0
            ),
            rag_dependency_health_external_enabled=_env_bool(
                "RAG_DEPENDENCY_HEALTH_EXTERNAL_ENABLED", True
            ),
            rag_dependency_health_fail_readiness=_env_bool(
                "RAG_DEPENDENCY_HEALTH_FAIL_READINESS", False
            ),
            rag_dependency_health_llm_url=_env_optional(
                "RAG_DEPENDENCY_HEALTH_LLM_URL"
            ),
            rag_dependency_health_embedding_url=_env_optional(
                "RAG_DEPENDENCY_HEALTH_EMBEDDING_URL"
            ),
            api_title=_env("API_TITLE", "Enterprise RAG API"),
            cors_origins=_env_list("API_CORS_ORIGINS"),
        )


__all__ = ["RagSettings"]
