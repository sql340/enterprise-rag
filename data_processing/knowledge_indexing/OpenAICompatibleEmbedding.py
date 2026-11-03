from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from .BaseEmbedding import BaseEmbedding
from .EmbeddingError import EmbeddingError
from ._utils import _batches, _normalise_vectors


class OpenAICompatibleEmbedding(BaseEmbedding):
    """调用 OpenAI 兼容的 ``/embeddings`` HTTP 接口。

    适用于 OpenAI、vLLM、Ollama（OpenAI 兼容模式）和多数企业 Embedding
    网关。只使用标准库，不额外依赖 openai SDK。

    Args:
        model_name: 服务端模型名称。
        base_url: API 根地址，例如 ``http://localhost:8000/v1``。
        api_key: API Key；本地服务可以省略。
        dimension: 预期向量维度。留空时使用第一次响应中的维度。
        batch_size: 单次请求的文本数量。
        timeout: 单次 HTTP 请求超时秒数。
        query_prefix: 查询文本前缀。
        document_prefix: 文档文本前缀。
        extra_headers: 额外请求头。
        extra_body: 额外请求体字段。
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        query_prefix: str = "",
        document_prefix: str = "",
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name 不能为空")
        if not base_url.strip():
            raise ValueError("base_url 不能为空")
        if dimension is not None and dimension <= 0:
            raise ValueError("dimension 必须为正整数")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if timeout <= 0:
            raise ValueError("timeout 必须为正数")

        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise EmbeddingError(
                "OpenAI 兼容 Embedding 的维度尚未确定；"
                "请传入 dimension，或先调用 embed_documents()"
            )
        return self._dimension

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def _embed_texts(
        self,
        texts: Sequence[str],
        *,
        query: bool,
    ) -> list[list[float]]:
        prefix = self.query_prefix if query else self.document_prefix
        vectors: list[list[float]] = []
        for batch in _batches(texts, self.batch_size):
            values = [f"{prefix}{text}" for text in batch]
            vectors.extend(self._request_embeddings(values))
        return vectors

    def _request_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {
            **self.extra_body,
            "model": self.model_name,
            "input": list(texts),
            "encoding_format": "float",
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise EmbeddingError(
                f"Embedding API 请求失败 ({exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(
                f"无法连接 Embedding API {self.endpoint}: {exc}"
            ) from exc

        try:
            data = json.loads(body)
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            raw_vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EmbeddingError("Embedding API 返回格式不正确") from exc

        vectors = _normalise_vectors(
            raw_vectors,
            expected_count=len(texts),
            dimension=self._dimension,
        )
        if self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors


__all__ = ["OpenAICompatibleEmbedding"]
