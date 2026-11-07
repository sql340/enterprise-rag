from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from access_control import PrincipalResolver
from configs import RagSettings
from data_processing.knowledge_indexing import (
    BaseEmbedding,
    OpenAICompatibleEmbedding,
    SentenceTransformerEmbedding,
)
from generation import OpenAICompatibleLLM
from retrieval import (
    KnowledgeRetriever,
    PromptBuilder,
    QdrantVectorStore,
    RagQueryService,
)

from .routes.health import router as health_router
from .routes.query import router as query_router


def build_query_service(settings: RagSettings) -> RagQueryService:
    """根据环境配置组装在线查询服务。"""

    embedding = _build_embedding(settings)
    vector_store = QdrantVectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        create_payload_indexes=False,
        text_index_field=settings.qdrant_text_index_field,
        enable_text_index=settings.qdrant_enable_text_index,
    )
    llm = OpenAICompatibleLLM(
        model_name=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
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
            acl_enabled=settings.rag_acl_enabled,
        ),
        llm,
        PromptBuilder(max_context_chars=settings.rag_max_context_chars),
        default_top_k=settings.rag_top_k,
        default_score_threshold=settings.rag_score_threshold,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
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
    settings: RagSettings | None = None,
) -> FastAPI:
    """创建 FastAPI 应用；测试时可直接注入查询服务。"""

    resolved_settings = settings or RagSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = service or build_query_service(resolved_settings)
        app.state.settings = resolved_settings
        app.state.query_service = runtime
        try:
            yield
        finally:
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
    return app


app = create_app()


__all__ = ["app", "build_query_service", "create_app"]
