from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from time import perf_counter
from typing import Any

from generation import BaseLLM

from .KnowledgeRetriever import KnowledgeRetriever
from .PromptBuilder import PromptBuilder
from .RagStreamEvent import RagStreamEvent
from .vector_store import VectorSearchResult


class RagQueryService:
    """编排查询向量化、知识召回、Prompt 构造和流式生成。"""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        llm: BaseLLM,
        prompt_builder: PromptBuilder,
        *,
        default_top_k: int = 5,
        default_score_threshold: float | None = None,
        default_temperature: float = 0.2,
        default_max_tokens: int | None = 1024,
        no_context_answer: str = "未在知识库中检索到相关内容，无法回答该问题。",
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k 必须为正整数")
        if default_score_threshold is not None and not -1 <= default_score_threshold <= 1:
            raise ValueError("default_score_threshold 必须在 -1 到 1 之间")
        if not 0 <= default_temperature <= 2:
            raise ValueError("default_temperature 必须在 0 到 2 之间")
        if default_max_tokens is not None and default_max_tokens <= 0:
            raise ValueError("default_max_tokens 必须为正整数或 None")
        if not no_context_answer.strip():
            raise ValueError("no_context_answer 不能为空")

        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.default_top_k = default_top_k
        self.default_score_threshold = default_score_threshold
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.no_context_answer = no_context_answer

    async def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: Mapping[str, Any] | None = None,
        include_sources: bool = True,
        retrieval_mode: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[RagStreamEvent]:
        started_at = perf_counter()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 不能为空")
        question = question.strip()
        resolved_top_k = self.default_top_k if top_k is None else top_k
        resolved_score_threshold = (
            self.default_score_threshold
            if score_threshold is None
            else score_threshold
        )
        resolved_temperature = (
            self.default_temperature if temperature is None else temperature
        )
        resolved_max_tokens = (
            self.default_max_tokens if max_tokens is None else max_tokens
        )
        if resolved_top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if resolved_score_threshold is not None and not (
            -1 <= resolved_score_threshold <= 1
        ):
            raise ValueError("score_threshold 必须在 -1 到 1 之间")
        if not 0 <= resolved_temperature <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        if resolved_max_tokens is not None and resolved_max_tokens <= 0:
            raise ValueError("max_tokens 必须为正整数或 None")

        yield RagStreamEvent(
            "start",
            {"query": question, "top_k": resolved_top_k},
        )

        sources = await asyncio.to_thread(
            self.retriever.retrieve,
            question,
            top_k=resolved_top_k,
            score_threshold=resolved_score_threshold,
            filters=filters,
            retrieval_mode=retrieval_mode,
        )
        if include_sources:
            yield RagStreamEvent(
                "sources",
                {
                    "items": [
                        self._source_payload(source, index)
                        for index, source in enumerate(sources, start=1)
                    ]
                },
            )

        if not sources:
            yield RagStreamEvent("delta", {"content": self.no_context_answer})
            yield RagStreamEvent(
                "done",
                {
                    "finish_reason": "no_context",
                    "elapsed_ms": self._elapsed_ms(started_at),
                },
            )
            return

        messages = self.prompt_builder.build(question, sources)
        async for content in self.llm.stream_chat(
            messages,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
        ):
            if content:
                yield RagStreamEvent("delta", {"content": content})

        yield RagStreamEvent(
            "done",
            {
                "finish_reason": "stop",
                "elapsed_ms": self._elapsed_ms(started_at),
            },
        )

    async def aclose(self) -> None:
        try:
            await self.llm.aclose()
        finally:
            self.retriever.vector_store.close()

    @staticmethod
    def _source_payload(source: VectorSearchResult, index: int) -> dict[str, Any]:
        payload = source.payload
        return {
            "index": index,
            "score": round(source.score, 6),
            "chunk_id": source.chunk_id,
            "document_id": source.document_id,
            "source": payload.get("source"),
            "file_type": payload.get("file_type"),
            "page_number": payload.get("page_number"),
            "slide_number": payload.get("slide_number"),
            "sheet_name": payload.get("sheet_name"),
            "heading_path": payload.get("heading_path"),
            "retrieval_mode": source.retrieval_mode,
            "route_scores": source.route_scores,
            "content": source.content,
        }

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000, 2)


__all__ = ["RagQueryService"]
