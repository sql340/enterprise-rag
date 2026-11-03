from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ._utils import _validate_texts


class BaseEmbedding(ABC):
    """Embedding 提供者接口。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称或本地模型路径。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """为知识块生成向量，保持与输入一一对应。"""

        values = _validate_texts(texts)
        if not values:
            return []
        return self._embed_texts(values, query=False)

    def embed_query(self, text: str) -> list[float]:
        """为检索查询生成向量。"""

        if not isinstance(text, str):
            raise TypeError("查询文本必须是字符串")
        if not text.strip():
            raise ValueError("查询文本不能为空")
        vectors = self._embed_texts([text], query=True)
        return vectors[0]

    @abstractmethod
    def _embed_texts(
        self,
        texts: Sequence[str],
        *,
        query: bool,
    ) -> list[list[float]]:
        """由具体提供者实现批量向量化。"""


__all__ = ["BaseEmbedding"]
