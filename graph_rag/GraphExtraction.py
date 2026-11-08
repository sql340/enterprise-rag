from __future__ import annotations

from dataclasses import dataclass, field

from .GraphEntity import GraphEntity
from .GraphRelation import GraphRelation


@dataclass(frozen=True, slots=True)
class GraphExtraction:
    """一个知识块中抽取出的实体和关系。"""

    entities: tuple[GraphEntity, ...] = field(default_factory=tuple)
    relations: tuple[GraphRelation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "relations", tuple(self.relations))


__all__ = ["GraphExtraction"]
