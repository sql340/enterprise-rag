from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence


class BaseLLM(ABC):
    """流式对话模型接口。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称。"""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        """流式返回模型生成内容。"""

    async def aclose(self) -> None:
        """释放连接等资源。"""


__all__ = ["BaseLLM"]
