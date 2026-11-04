from __future__ import annotations


class LLMError(RuntimeError):
    """LLM 请求或流式响应解析失败。"""


__all__ = ["LLMError"]
