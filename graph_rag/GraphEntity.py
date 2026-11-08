from __future__ import annotations

from dataclasses import dataclass

from ._utils import normalise_text


@dataclass(frozen=True, slots=True)
class GraphEntity:
    """知识图谱实体。"""

    name: str
    entity_type: str = "concept"
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("entity.name 不能为空")
        entity_type = normalise_text(self.entity_type) or "concept"
        aliases: list[str] = []
        seen = {normalise_text(name)}
        for alias in self.aliases:
            value = str(alias).strip()
            normalised = normalise_text(value)
            if value and normalised not in seen:
                seen.add(normalised)
                aliases.append(value)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "entity_type", entity_type)
        object.__setattr__(self, "aliases", tuple(aliases))

    @property
    def normalised_name(self) -> str:
        return normalise_text(self.name)


__all__ = ["GraphEntity"]
