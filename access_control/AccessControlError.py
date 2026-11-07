from __future__ import annotations


class AccessControlError(RuntimeError):
    """身份或知识访问权限不满足要求。"""


__all__ = ["AccessControlError"]
