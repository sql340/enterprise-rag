from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from access_control import PrincipalResolver
from configs import RagSettings
from data_processing.knowledge_indexing import (
    BaseEmbedding,
    KnowledgeIndexer,
    OpenAICompatibleEmbedding,
    SentenceTransformerEmbedding,
)
from generation import OpenAICompatibleLLM
from graph_rag import (
    GraphIndexer,
    GraphRetriever,
    OpenAICompatibleGraphExtractor,
    SqliteGraphStore,
)
from knowledge_management import (
    IngestionJobWorker,
    KnowledgeAdminService,
    KnowledgeGovernanceStore,
    KnowledgeIngestionService,
)
from retrieval import (
    KnowledgeRetriever,
    PromptBuilder,
    QdrantVectorStore,
    RagQueryService,
)
from retrieval.rerank import CrossEncoderReranker

from .routes.admin import router as admin_router
from .routes.health import router as health_router
from .routes.query import router as query_router


logger = logging.getLogger(__name__)


def build_query_service(settings: RagSettings) -> RagQueryService:
    """根据环境配置组装在线查询服务。"""

    embedding = _build_embedding(settings)
    vector_store = _build_vector_store(settings)
    llm = OpenAICompatibleLLM(
        model_name=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )
    graph_retriever = None
    if settings.rag_graph_enabled:
        graph_retriever = GraphRetriever(
            SqliteGraphStore(settings.rag_graph_store_path),
            max_hops=settings.rag_graph_max_hops,
            entity_limit=settings.rag_graph_entity_limit,
        )
    reranker = None
    if settings.rag_rerank_enabled:
        reranker = CrossEncoderReranker(
            settings.rag_rerank_model,
            device=settings.rag_rerank_device,
            max_candidates=settings.rag_rerank_candidates,
        )
    return RagQueryService(
        KnowledgeRetriever(
            embedding,
            vector_store,
            retrieval_mode=settings.rag_retrieval_mode,
            candidate_multiplier=settings.rag_candidate_multiplier,
            max_candidates=settings.rag_max_candidates,
            rrf_k=settings.rag_rrf_k,
            dense_weight=settings.rag_dense_weight,
            keyword_weight=settings.rag_keyword_weight,
            graph_retriever=graph_retriever,
            graph_weight=settings.rag_graph_weight,
            acl_enabled=settings.rag_acl_enabled,
            reranker=reranker,
            rerank_candidates=settings.rag_rerank_candidates,
            expand_parent_context=settings.rag_expand_parent_context,
        ),
        llm,
        PromptBuilder(max_context_chars=settings.rag_max_context_chars),
        default_top_k=settings.rag_top_k,
        default_score_threshold=settings.rag_score_threshold,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
    )


def build_admin_service(
    settings: RagSettings,
    query_service: RagQueryService | None = None,
) -> KnowledgeAdminService:
    """组装知识上传、构建、质量与审计服务。"""

    retriever = getattr(query_service, "retriever", None)
    embedding = getattr(retriever, "embedding", None)
    vector_store = getattr(retriever, "vector_store", None)
    if embedding is None:
        embedding = _build_embedding(settings)
    if vector_store is None:
        vector_store = _build_vector_store(settings)

    graph_indexer = None
    if settings.rag_graph_enabled:
        graph_indexer = GraphIndexer(
            SqliteGraphStore(settings.rag_graph_store_path),
            OpenAICompatibleGraphExtractor(
                settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout,
            ),
        )
    indexer = KnowledgeIndexer(embedding, vector_store, sync_scope="source")
    ingestion = KnowledgeIngestionService(
        governance_store=KnowledgeGovernanceStore(settings.rag_governance_db_path),
        data_dir=settings.rag_upload_dir,
        indexer=indexer,
        graph_indexer=graph_indexer,
        max_file_bytes=settings.rag_admin_max_file_bytes,
    )
    return KnowledgeAdminService(ingestion)


def _build_vector_store(settings: RagSettings) -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        create_payload_indexes=settings.qdrant_create_payload_indexes,
        text_index_field=settings.qdrant_text_index_field,
        enable_text_index=settings.qdrant_enable_text_index,
    )


def _build_embedding(settings: RagSettings) -> BaseEmbedding:
    provider = settings.embedding_provider.strip().lower().replace("-", "_")
    if provider in {"sentence_transformers", "sentence_transformer", "local", "bge"}:
        return SentenceTransformerEmbedding(
            settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=settings.embedding_normalize,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
    if provider in {"openai", "openai_compatible", "api"}:
        return OpenAICompatibleEmbedding(
            settings.embedding_model,
            base_url=settings.embedding_api_base_url,
            api_key=settings.embedding_api_key,
            dimension=settings.embedding_dimension,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
    raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {settings.embedding_provider}")


def create_app(
    *,
    service: RagQueryService | None = None,
    admin_service: KnowledgeAdminService | None = None,
    settings: RagSettings | None = None,
    job_worker_enabled: bool | None = None,
) -> FastAPI:
    """创建 FastAPI 应用；测试时可直接注入查询服务。"""

    resolved_settings = settings or RagSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = service or build_query_service(resolved_settings)
        resolved_admin = admin_service
        app.state.admin_service_error = None
        worker_enabled = (
            resolved_settings.rag_ingestion_worker_enabled
            if job_worker_enabled is None
            else job_worker_enabled
        )
        if resolved_admin is None and worker_enabled:
            try:
                resolved_admin = await asyncio.to_thread(
                    build_admin_service, resolved_settings, runtime
                )
            except Exception as exc:
                logger.exception("知识管理服务启动初始化失败")
                app.state.admin_service_error = exc
        app.state.settings = resolved_settings
        app.state.query_service = runtime
        app.state.admin_service = resolved_admin
        app.state.admin_owned = admin_service is None and resolved_admin is not None
        app.state.ingestion_worker_enabled = worker_enabled
        app.state.job_worker = None
        await _bootstrap_search_indexes(runtime, app, resolved_settings)
        worker: IngestionJobWorker | None = None
        if worker_enabled and resolved_admin is not None:
            worker = IngestionJobWorker(
                resolved_admin.ingestion,
                poll_seconds=resolved_settings.rag_ingestion_poll_seconds,
                lease_seconds=resolved_settings.rag_ingestion_lease_seconds,
                max_attempts=resolved_settings.rag_ingestion_max_attempts,
            )
            app.state.job_worker = worker
            worker.start()
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()
            owned_admin = getattr(app.state, "admin_owned", False)
            runtime_admin = getattr(app.state, "admin_service", None)
            if owned_admin and runtime_admin is not None:
                _close_admin_service(runtime_admin)
            await runtime.aclose()

    app = FastAPI(
        title=resolved_settings.api_title,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.principal_resolver = PrincipalResolver(
        proxy_token=resolved_settings.rag_acl_proxy_token
    )
    app.state.admin_service = admin_service
    app.state.admin_owned = admin_service is None
    app.state.job_worker = None
    app.state.ingestion_worker_enabled = resolved_settings.rag_ingestion_worker_enabled
    app.state.search_index_status = {
        "ready": False,
        "error": "not_initialized",
    }
    app.state.admin_service_factory = lambda: build_admin_service(
        resolved_settings,
        getattr(app.state, "query_service", None),
    )
    if resolved_settings.cors_origins:
        allow_all = resolved_settings.cors_origins == ("*",)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=not allow_all,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(admin_router)

    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")

        @app.get("/admin", include_in_schema=False)
        async def admin_console() -> FileResponse:
            return FileResponse(ui_dir / "index.html")

    return app


async def _bootstrap_search_indexes(
    runtime: RagQueryService,
    app: FastAPI,
    settings: RagSettings,
) -> None:
    vector_store = getattr(getattr(runtime, "retriever", None), "vector_store", None)
    ensure = getattr(vector_store, "ensure_search_indexes", None)
    status_check = getattr(vector_store, "search_index_status", None)
    if ensure is None and status_check is None:
        app.state.search_index_status = {
            "ready": True,
            "skipped": True,
            "reason": "vector_store_not_configured",
        }
        return

    def bootstrap() -> dict:
        status = status_check() if callable(status_check) else None
        if (
            status is not None
            and not status.get("collection_exists")
            and settings.rag_bootstrap_search_indexes
        ):
            embedding = getattr(
                getattr(runtime, "retriever", None),
                "embedding",
                None,
            )
            if embedding is None:
                raise RuntimeError("无法初始化 Qdrant collection：Embedding 未配置")
            dimension = int(embedding.dimension)
            if dimension <= 0:
                raise RuntimeError("无法初始化 Qdrant collection：向量维度无效")
            vector_store.ensure_collection(dimension)

        if callable(ensure):
            return ensure()
        if callable(status_check):
            return status_check()
        return status or {"ready": False, "error": "search_index_status_missing"}

    try:
        status = await asyncio.to_thread(bootstrap)
        app.state.search_index_status = status
        if not status.get("collection_exists"):
            logger.warning("Qdrant collection 尚未创建，检索索引未就绪: %s", status)
        elif not status.get("ready"):
            logger.warning("Qdrant 检索索引未就绪: %s", status)
    except Exception as exc:
        logger.exception("Qdrant 检索索引初始化失败")
        app.state.search_index_status = {
            "ready": False,
            "collection_exists": False,
            "error": "bootstrap_failed",
            "detail": str(exc)[:500],
        }


def _close_admin_service(service: KnowledgeAdminService) -> None:
    try:
        service.ingestion.governance_store.close()
    finally:
        graph_indexer = service.ingestion.graph_indexer
        graph_store = getattr(graph_indexer, "graph_store", None)
        close = getattr(graph_store, "close", None)
        if close is not None:
            close()


app = create_app()


__all__ = ["app", "build_admin_service", "build_query_service", "create_app"]
