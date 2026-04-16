from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Entity:
    name: str
    type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(eq=True, frozen=True)
class Triple:
    head: str
    relation: str
    tail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
        }


@dataclass
class KnowledgeGraph:
    entities: List[Entity] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "triples": [t.to_dict() for t in self.triples],
        }


@dataclass
class QAPair:
    question: str
    answer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Sample:
    sample_id: str
    claim: str
    reports: List[str]
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    sample_id: str
    claim: str
    reports: List[str]
    label: Optional[str]
    prediction: str
    mode: str
    claim_kg: Dict[str, Any] = field(default_factory=dict)
    report_kgs: List[Dict[str, Any]] = field(default_factory=list)
    merged_kg: Dict[str, Any] = field(default_factory=dict)
    claim_triples: List[Dict[str, Any]] = field(default_factory=list)
    candidate_questions: List[str] = field(default_factory=list)
    selected_questions: List[str] = field(default_factory=list)
    qa_pairs: List[Dict[str, Any]] = field(default_factory=list)
    contrastive_summary: str = ""
    kg_text: str = ""
    raw_outputs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
