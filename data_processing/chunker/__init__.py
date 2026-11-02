"""
企业 RAG 文本分块模块。

分块策略按类拆分为独立模块，保留统一入口::

    from data_processing.chunker import Chunker

    chunks = Chunker("recursive", chunk_size=600, chunk_overlap=80).split(
        documents
    )
"""

from .BaseChunker import BaseChunker
from .Chunk import Chunk
from .ChunkStrategy import ChunkStrategy
from .FixedSizeChunker import FixedSizeChunker
from .MarkdownHeadingChunker import MarkdownHeadingChunker
from .ParagraphChunker import ParagraphChunker
from .ParentChildChunker import ParentChildChunker
from .RecursiveChunker import RecursiveChunker
from .SentenceWindowChunker import SentenceWindowChunker
from .Chunker import Chunker, chunk_documents, create_chunker
from ._utils import DEFAULT_SEPARATORS, DocumentLike, LengthFunction

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkStrategy",
    "Chunker",
    "DEFAULT_SEPARATORS",
    "DocumentLike",
    "FixedSizeChunker",
    "LengthFunction",
    "MarkdownHeadingChunker",
    "ParagraphChunker",
    "ParentChildChunker",
    "RecursiveChunker",
    "SentenceWindowChunker",
    "chunk_documents",
    "create_chunker",
]
