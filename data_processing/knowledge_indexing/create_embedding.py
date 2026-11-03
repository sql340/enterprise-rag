from __future__ import annotations

from typing import Any

from .BaseEmbedding import BaseEmbedding
from .OpenAICompatibleEmbedding import OpenAICompatibleEmbedding
from .SentenceTransformerEmbedding import SentenceTransformerEmbedding


def create_embedding(
    provider: str = "sentence_transformers",
    **kwargs: Any,
) -> BaseEmbedding:
    """按名称创建 Embedding 提供者。"""

    name = provider.strip().lower().replace("-", "_")
    if name in {"sentence_transformers", "sentence_transformer", "local", "bge"}:
        return SentenceTransformerEmbedding(**kwargs)
    if name in {"openai", "openai_compatible", "api"}:
        return OpenAICompatibleEmbedding(**kwargs)
    raise ValueError(
        f"不支持的 Embedding 提供者: {provider}; "
        "可选 sentence_transformers、openai_compatible"
    )


__all__ = ["create_embedding"]
