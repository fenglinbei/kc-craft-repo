from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .api import OpenAICompatibleChatClient
from .prompts import build_operational_kg_extraction_prompt
from .schemas import Entity, KnowledgeGraph, Triple
from .utils import normalize_text, safe_json_loads


@dataclass
class KGExtractionOutput:
    kg: KnowledgeGraph
    raw_text: str
    raw_response: dict


class KGExtractor:
    def __init__(self, client: OpenAICompatibleChatClient):
        self.client = client

    def extract(self, text: str) -> KGExtractionOutput:
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: List[str]) -> List[KGExtractionOutput]:
        if not texts:
            return []
        prompts = [build_operational_kg_extraction_prompt(text) for text in texts]
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        responses = self.client.chat_batch(messages_batch=messages_batch)
        if len(responses) != len(texts):
            raise ValueError(
                f"Batch KG extraction response size mismatch: expected={len(texts)} got={len(responses)}"
            )
        outputs: List[KGExtractionOutput] = []
        for text, response in zip(texts, responses):
            parsed = safe_json_loads(response.content)
            kg = parse_kg_json(parsed)
            outputs.append(KGExtractionOutput(kg=kg, raw_text=response.content, raw_response=response.raw))
        return outputs

def parse_kg_json(payload: dict) -> KnowledgeGraph:
    entities_raw = payload.get("entities", []) or []
    triples_raw = payload.get("triples", []) or []

    entities: List[Entity] = []
    entity_seen = set()
    for item in entities_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "Other")).strip() or "Other"
        if not name:
            continue
        key = (normalize_text(name), normalize_text(etype))
        if key in entity_seen:
            continue
        entity_seen.add(key)
        entities.append(Entity(name=name, type=etype))

    triple_seen = set()
    triples: List[Triple] = []
    for item in triples_raw:
        if not isinstance(item, dict):
            continue
        head = str(item.get("head", "")).strip()
        relation = str(item.get("relation", "")).strip()
        tail = str(item.get("tail", "")).strip()
        if not head or not relation or not tail:
            continue
        triple = Triple(head=head, relation=relation, tail=tail)
        if triple in triple_seen:
            continue
        triple_seen.add(triple)
        triples.append(triple)

    # Make sure every triple endpoint exists in entities.
    known_entity_names = {normalize_text(e.name): e.type for e in entities}
    for triple in triples:
        if normalize_text(triple.head) not in known_entity_names:
            entities.append(Entity(name=triple.head, type="Other"))
            known_entity_names[normalize_text(triple.head)] = "Other"
        if normalize_text(triple.tail) not in known_entity_names:
            entities.append(Entity(name=triple.tail, type="Other"))
            known_entity_names[normalize_text(triple.tail)] = "Other"

    return KnowledgeGraph(entities=entities, triples=triples)



def merge_kgs(kgs: List[KnowledgeGraph]) -> KnowledgeGraph:
    type_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}
    merged_entities: List[Entity] = []

    for kg in kgs:
        for entity in kg.entities:
            norm = normalize_text(entity.name)
            if norm not in name_map:
                name_map[norm] = entity.name
                type_map[norm] = entity.type
                merged_entities.append(Entity(name=entity.name, type=entity.type))
            elif type_map.get(norm, "Other") == "Other" and entity.type != "Other":
                type_map[norm] = entity.type

    canonical_entities = {
        norm: Entity(name=name_map[norm], type=type_map.get(norm, "Other"))
        for norm in name_map
    }

    triple_seen = set()
    merged_triples: List[Triple] = []
    for kg in kgs:
        for triple in kg.triples:
            head_norm = normalize_text(triple.head)
            tail_norm = normalize_text(triple.tail)
            canonical_head = canonical_entities.get(head_norm, Entity(triple.head, "Other")).name
            canonical_tail = canonical_entities.get(tail_norm, Entity(triple.tail, "Other")).name
            canonical_triple = Triple(
                head=canonical_head,
                relation=triple.relation,
                tail=canonical_tail,
            )
            if canonical_triple in triple_seen:
                continue
            triple_seen.add(canonical_triple)
            merged_triples.append(canonical_triple)

    return KnowledgeGraph(entities=list(canonical_entities.values()), triples=merged_triples)



def entity_type_map(kg: KnowledgeGraph) -> Dict[str, str]:
    return {normalize_text(entity.name): entity.type for entity in kg.entities}



def triples_from_claim_in_merged_graph(claim_kg: KnowledgeGraph, merged_kg: KnowledgeGraph) -> List[Triple]:
    merged_set = set(merged_kg.triples)
    matched: List[Triple] = []
    for triple in claim_kg.triples:
        if triple in merged_set:
            matched.append(triple)
    # Fallback: if exact triple is missing due to small normalization differences, just return claim triples.
    return matched if matched else list(claim_kg.triples)
