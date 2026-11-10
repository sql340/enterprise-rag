from __future__ import annotations


class IngestionJobLeaseError(RuntimeError):
    """当前 worker 已失去任务租约，不应继续写入任务状态。"""


__all__ = ["IngestionJobLeaseError"]
