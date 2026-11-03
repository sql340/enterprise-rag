"""知识向量化与索引入库模块。

Embedding 提供者、异常、结果对象和索引器均按类拆分到同名文件，统一通过
包级入口使用::

    from data_processing.knowledge_indexing import (
        KnowledgeIndexer,
        SentenceTransformerEmbedding,
    )
"""

from .BaseEmbedding import BaseEmbedding
from .EmbeddingDependencyError import EmbeddingDependencyError
from .EmbeddingError import EmbeddingError
from .IndexingResult import IndexingResult
from .KnowledgeIndexError import KnowledgeIndexError
from .KnowledgeIndexer import KnowledgeIndexer
from .OpenAICompatibleEmbedding import OpenAICompatibleEmbedding
from .SentenceTransformerEmbedding import SentenceTransformerEmbedding
from .create_embedding import create_embedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingDependencyError",
    "EmbeddingError",
    "IndexingResult",
    "KnowledgeIndexError",
    "KnowledgeIndexer",
    "OpenAICompatibleEmbedding",
    "SentenceTransformerEmbedding",
    "create_embedding",
]
