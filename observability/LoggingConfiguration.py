from __future__ import annotations

import logging
import threading

from .JsonLogFormatter import JsonLogFormatter


class LoggingConfiguration:
    """配置项目根日志输出，重复调用不会叠加 handler。"""

    _lock = threading.Lock()

    @classmethod
    def configure(cls, *, level: str = "INFO", log_format: str = "json") -> None:
        resolved_level = level.strip().upper() or "INFO"
        resolved_format = log_format.strip().lower() or "json"
        with cls._lock:
            root = logging.getLogger()
            for handler in list(root.handlers):
                if getattr(handler, "_enterprise_rag_handler", False):
                    root.removeHandler(handler)

            handler = logging.StreamHandler()
            handler._enterprise_rag_handler = True  # type: ignore[attr-defined]
            if resolved_format == "json":
                handler.setFormatter(JsonLogFormatter())
            else:
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s %(levelname)s %(name)s %(message)s"
                    )
                )
            root.addHandler(handler)
            root.setLevel(getattr(logging, resolved_level, logging.INFO))
            if root.level > logging.DEBUG:
                for name in ("httpx", "httpx2", "httpcore", "httpcore2", "qdrant_client"):
                    logging.getLogger(name).setLevel(logging.WARNING)


__all__ = ["LoggingConfiguration"]
