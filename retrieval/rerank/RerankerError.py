from __future__ import annotations


class RerankerError(RuntimeError):
    """重排模型不可用或执行失败。"""


__all__ = ["RerankerError"]
