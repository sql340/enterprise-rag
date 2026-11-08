from __future__ import annotations

from dataclasses import dataclass

from ._utils import normalise_relation_type, normalise_text


@dataclass(frozen=True, slots=True)
class GraphRelation:
    """实体之间的有向关系；检索时按无向边扩展。"""

    source: str
    relation_type: str
    target: str
    description: str = ""

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        target = str(self.target).strip()
        if not source or not target:
            raise ValueError("relation.source 和 relation.target 不能为空")
        if normalise_text(source) == normalise_text(target):
            raise ValueError("relation 的 source 和 target 不能相同")
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "relation_type",
            normalise_relation_type(self.relation_type),
        )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "description", str(self.description).strip())


__all__ = ["GraphRelation"]
