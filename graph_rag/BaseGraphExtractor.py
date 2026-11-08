from __future__ import annotations

from abc import ABC, abstractmethod

from data_processing.chunker import Chunk

from .GraphExtraction import GraphExtraction


class BaseGraphExtractor(ABC):
    """从知识块抽取实体和关系的接口。"""

    @property
    @abstractmethod
    def extractor_version(self) -> str:
        """抽取器与 Prompt 的版本标识，用于增量跳过。"""

    @abstractmethod
    def extract(self, chunk: Chunk) -> GraphExtraction:
        """从一个知识块抽取实体关系。"""


__all__ = ["BaseGraphExtractor"]
