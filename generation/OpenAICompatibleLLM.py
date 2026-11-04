from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from .BaseLLM import BaseLLM
from .LLMError import LLMError


logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """调用 OpenAI 兼容的流式 Chat Completions 接口。

    适用于 OpenAI、vLLM、Ollama（OpenAI 兼容模式）和多数企业 LLM 网关。

    Args:
        model_name: 服务端模型名称。
        base_url: API 根地址，例如 ``http://localhost:8000/v1``。
        api_key: API Key；本地服务可以省略。
        timeout: 单次流式请求超时秒数。
        extra_headers: 额外请求头。
        extra_body: 额外请求体字段。
        client: 可注入的 ``httpx.AsyncClient``，便于测试或复用连接池。
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name 不能为空")
        if not base_url.strip():
            raise ValueError("base_url 不能为空")
        if timeout <= 0:
            raise ValueError("timeout 必须为正数")

        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self._client = client
        self._owns_client = client is None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        payload = {
            **self.extra_body,
            "model": self.model_name,
            "messages": [dict(message) for message in messages],
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **self.extra_headers,
        }
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                self.endpoint,
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )[:500]
                    raise LLMError(
                        f"LLM API 请求失败 ({response.status_code}): {body}"
                    )

                async for line in response.aiter_lines():
                    data = line.strip()
                    if not data or data.startswith(":") or data.startswith("event:"):
                        continue
                    if data.startswith("data:"):
                        data = data[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        item = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("忽略无法解析的 LLM 流式数据: %s", data[:200])
                        continue
                    if item.get("error"):
                        raise LLMError(f"LLM 返回错误: {item['error']}")
                    choices = item.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if isinstance(content, str) and content:
                        yield content
        except LLMError:
            raise
        except httpx.RequestError as exc:
            raise LLMError(f"无法连接 LLM API {self.endpoint}: {exc}") from exc

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OpenAICompatibleLLM":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


__all__ = ["OpenAICompatibleLLM"]
