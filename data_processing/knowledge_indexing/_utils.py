from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

from .EmbeddingError import EmbeddingError


def _batches(values: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _normalise_vectors(
    raw_vectors: Any,
    *,
    expected_count: int,
    dimension: int | None = None,
) -> list[list[float]]:
    if hasattr(raw_vectors, "tolist"):
        raw_vectors = raw_vectors.tolist()
    if not isinstance(raw_vectors, (list, tuple)):
        raise EmbeddingError("Embedding 模型返回了无法识别的向量格式")

    vectors: list[list[float]] = []
    for raw_vector in raw_vectors:
        if hasattr(raw_vector, "tolist"):
            raw_vector = raw_vector.tolist()
        if not isinstance(raw_vector, (list, tuple)):
            raise EmbeddingError("Embedding 模型返回了非向量元素")
        vector = [float(value) for value in raw_vector]
        if not vector:
            raise EmbeddingError("Embedding 模型返回了空向量")
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding 模型返回了 NaN 或无穷大")
        if dimension is not None and len(vector) != dimension:
            raise EmbeddingError(
                f"Embedding 维度不一致: 期望 {dimension}, 实际 {len(vector)}"
            )
        vectors.append(vector)

    if len(vectors) != expected_count:
        raise EmbeddingError(
            f"Embedding 返回数量不一致: 期望 {expected_count}, 实际 {len(vectors)}"
        )
    return vectors


def _validate_texts(texts: Sequence[str]) -> list[str]:
    values = list(texts)
    if any(not isinstance(text, str) for text in values):
        raise TypeError("Embedding 输入必须是字符串序列")
    return values
