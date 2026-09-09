from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SourceType = Literal['human_original','ai_generated','hybrid','external_verified']
MutationType = Literal['none','neutral','beneficial','corrective','deleterious','catastrophic']

@dataclass
class Idea:
    idea_id: str
    lineage_id: str
    root_idea_id: str
    generation: int
    source_type: SourceType
    validity: float
    novelty: float
    alignment: float
    error_severity: float
    parent_id: str | None = None
    secondary_parent_id: str | None = None
    mutation_type: MutationType = 'none'
    mutation_magnitude: float = 0.0
    utility: float = 0.0
    confidence: float = 0.0
    creativity: float = 0.0
    technical_rigor: float = 0.0
    repetition_score: float = 0.0
    distortion_score: float = 0.0
    human_origin_share: float = 0.0
    ai_origin_share: float = 0.0
    external_origin_share: float = 0.0
    source_independence: float = 0.0
    lineage_depth: int = 0
    alive: bool = True
    birth_time: int = 0
    extinction_time: int | None = None
    survival_duration: int = 1
    descendant_count: int = 0
    cumulative_descendants: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ['validity','novelty','alignment','error_severity','mutation_magnitude','utility','confidence','creativity','technical_rigor','repetition_score','distortion_score','human_origin_share','ai_origin_share','external_origin_share','source_independence']:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0,1], got {value}')
        if self.human_origin_share + self.ai_origin_share + self.external_origin_share > 1.000001:
            raise ValueError('Origin shares cannot exceed 1.')

    @property
    def is_valid(self) -> bool:
        return self.validity >= 0.60 and self.error_severity < 0.40

    @property
    def is_erroneous(self) -> bool:
        return not self.is_valid

    def mark_extinct(self, time: int) -> None:
        self.alive = False
        self.extinction_time = time
        self.survival_duration = max(1, time - self.birth_time + 1)

    def reintroduce(self, time: int) -> None:
        self.alive = True
        self.extinction_time = None
        self.birth_time = time
        self.survival_duration = 1

    def update_survival(self, current_time: int) -> None:
        end = self.extinction_time if self.extinction_time is not None else current_time
        self.survival_duration = max(1, end - self.birth_time + 1)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
