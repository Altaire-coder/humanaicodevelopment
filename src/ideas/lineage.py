from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4

LineageEventType = Literal['birth','reproduction','recombination','correction','distortion','extinction','reintroduction']

@dataclass(frozen=True)
class LineageEvent:
    event_id: str
    run_id: str
    scenario_id: str
    time: int
    generation: int
    event_type: LineageEventType
    lineage_id: str
    idea_id: str
    parent_idea_id: str | None
    secondary_parent_id: str | None
    source_id: str
    target_id: str
    mutation_type: str
    validity: float
    error_severity: float
    alive: bool
    is_intervention: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, **kwargs):
        return cls(event_id=f'LE_{uuid4().hex[:12]}', **kwargs)

    def to_record(self):
        return asdict(self)

@dataclass
class Lineage:
    lineage_id: str
    root_idea_id: str
    birth_time: int
    idea_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    alive: bool = True
    extinction_time: int | None = None
    reintroduction_count: int = 0
