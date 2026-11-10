from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from data_processing.chunker import Chunk


class ChunkQualityAnalyzer:
    """对分块结果做轻量质量检查，并写入可检索的质量标记。"""

    def __init__(
        self,
        *,
        min_chunk_chars: int = 30,
        max_chunk_chars: int = 8000,
        low_quality_ratio_threshold: float = 0.3,
    ) -> None:
        if min_chunk_chars < 0:
            raise ValueError("min_chunk_chars 不能为负数")
        if max_chunk_chars <= min_chunk_chars:
            raise ValueError("max_chunk_chars 必须大于 min_chunk_chars")
        if not 0 <= low_quality_ratio_threshold <= 1:
            raise ValueError("low_quality_ratio_threshold 必须在 0 到 1 之间")
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.low_quality_ratio_threshold = low_quality_ratio_threshold

    def analyze(self, chunks: Sequence[Chunk]) -> tuple[list[Chunk], dict[str, Any]]:
        analysed: list[Chunk] = []
        low_quality: list[dict[str, Any]] = []
        scores: list[float] = []

        for chunk in chunks:
            flags = self._flags(chunk.content)
            score = max(0.0, 1.0 - 0.25 * len(flags))
            scores.append(score)
            metadata = {
                **chunk.metadata,
                "quality_flags": flags,
                "quality_score": round(score, 4),
                "quality_review_required": bool(flags),
            }
            analysed.append(replace(chunk, metadata=metadata))
            if flags:
                low_quality.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "length": len(chunk.content),
                        "flags": flags,
                        "score": round(score, 4),
                    }
                )

        ratio = len(low_quality) / len(analysed) if analysed else 0.0
        report = {
            "chunk_count": len(analysed),
            "low_quality_count": len(low_quality),
            "low_quality_ratio": round(ratio, 4),
            "average_quality_score": round(sum(scores) / len(scores), 4)
            if scores
            else 0.0,
            "needs_rework": ratio > self.low_quality_ratio_threshold,
            "low_quality_chunks": low_quality[:20],
        }
        return analysed, report

    def _flags(self, content: str) -> list[str]:
        text = content.strip()
        flags: list[str] = []
        if len(text) < self.min_chunk_chars:
            flags.append("too_short")
        if len(text) > self.max_chunk_chars:
            flags.append("too_long")

        compact = "".join(text.split())
        if len(compact) >= 20:
            informative = sum(char.isalnum() for char in compact)
            if informative / len(compact) < 0.45:
                flags.append("low_information_density")
            if len(set(compact)) / len(compact) < 0.08:
                flags.append("repetitive")
        return flags


__all__ = ["ChunkQualityAnalyzer"]
