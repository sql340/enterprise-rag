from __future__ import annotations

from enum import IntEnum


class Classification(IntEnum):
    """知识密级，数值越大访问要求越高。"""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3

    @classmethod
    def parse(cls, value: "Classification | int | str | None") -> "Classification":
        if value is None or value == "":
            return cls.PUBLIC
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            aliases = {
                "public": cls.PUBLIC,
                "internal": cls.INTERNAL,
                "confidential": cls.CONFIDENTIAL,
                "secret": cls.SECRET,
            }
            if normalised in aliases:
                return aliases[normalised]
            try:
                numeric = int(normalised)
            except ValueError as exc:
                raise ValueError(f"未知密级: {value}") from exc
        elif isinstance(value, int) and not isinstance(value, bool):
            numeric = value
        else:
            raise ValueError(f"未知密级: {value}")

        try:
            return cls(numeric)
        except ValueError as exc:
            raise ValueError(f"密级必须是 0-3: {value}") from exc


__all__ = ["Classification"]
