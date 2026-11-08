from __future__ import annotations

import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")
_ASCII_ENTITY = re.compile(r"^[a-z0-9][a-z0-9 ._-]*$")


def normalise_text(value: str) -> str:
    """统一大小写、兼容字符和空白，便于图谱实体匹配。"""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold().strip()
    return _WHITESPACE.sub(" ", text)


def normalise_relation_type(value: str) -> str:
    text = normalise_text(value).replace(" ", "_").replace("-", "_")
    return text.strip("_") or "related_to"


def contains_entity(question: str, entity_name: str) -> bool:
    """判断实体是否出现在问题中，并避免英文短实体误匹配单词片段。"""

    if len(entity_name) < 2 or entity_name not in question:
        return False
    if entity_name.isascii() and _ASCII_ENTITY.fullmatch(entity_name):
        pattern = rf"(?<![a-z0-9]){re.escape(entity_name)}(?![a-z0-9])"
        return re.search(pattern, question) is not None
    return True


__all__ = ["contains_entity", "normalise_relation_type", "normalise_text"]
