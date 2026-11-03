from __future__ import annotations


class EmbeddingError(RuntimeError):
    """Embedding 生成或模型加载失败。"""


__all__ = ["EmbeddingError"]
