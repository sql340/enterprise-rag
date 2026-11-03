from __future__ import annotations

from .EmbeddingError import EmbeddingError


class EmbeddingDependencyError(EmbeddingError, ImportError):
    """缺少 Embedding 所需的第三方依赖。"""


__all__ = ["EmbeddingDependencyError"]
