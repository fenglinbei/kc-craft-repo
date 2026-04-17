from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .api import OpenAICompatibleChatClient
from .prompts import (
    build_kg_phase_a_prompt,
    build_kg_phase_b_prompt,
    build_kg_phase_c_prompt,
)
from .schemas import Entity, KnowledgeGraph, Triple
from .utils import (
    normalize_text, 
    safe_json_loads, 
    normalize_relation,
)


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
        outputs: List[KGExtractionOutput] = []
        phase_a_payloads = self._run_phase_a(texts)
        phase_b_payloads = self._run_phase_b(texts, phase_a_payloads)
        phase_c_payloads, raw_phase_c = self._run_phase_c(texts, phase_b_payloads)

        for idx, text in enumerate(texts):
            kg = parse_phased_kg_json(
                phase_b_payload=phase_b_payloads[idx],
                phase_c_payload=phase_c_payloads[idx],
            )
            outputs.append(
                KGExtractionOutput(
                    kg=kg,
                    raw_text=raw_phase_c[idx],
                    raw_response={
                        "phase_a": phase_a_payloads[idx],
                        "phase_b": phase_b_payloads[idx],
                        "phase_c": phase_c_payloads[idx],
                    },
                )
            )
        return outputs

    def _run_phase_a(self, texts: List[str]) -> List[dict]:
        prompts = [build_kg_phase_a_prompt(text) for text in texts]
        responses = self._chat_for_prompts(prompts, "A")
        return [safe_json_loads(response.content) for response in responses]

    def _run_phase_b(self, texts: List[str], phase_a_payloads: List[dict]) -> List[dict]:
        prompts = []
        for text, phase_a in zip(texts, phase_a_payloads):
            mentions = collect_mentions(phase_a)
            prompts.append(build_kg_phase_b_prompt(text=text, mentions=mentions))
        responses = self._chat_for_prompts(prompts, "B")
        return [safe_json_loads(response.content) for response in responses]

    def _run_phase_c(self, texts: List[str], phase_b_payloads: List[dict]) -> Tuple[List[dict], List[str]]:
        prompts = []
        for text, phase_b in zip(texts, phase_b_payloads):
            canonical_entities = collect_canonical_entities(phase_b)
            prompts.append(build_kg_phase_c_prompt(text=text, canonical_entities=canonical_entities))
        responses = self._chat_for_prompts(prompts, "C")
        return [safe_json_loads(response.content) for response in responses], [response.content for response in responses]

    def _chat_for_prompts(self, prompts: List[str], phase: str):
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        responses = self.client.chat_batch(messages_batch=messages_batch)
        if len(responses) != len(prompts):
            raise ValueError(
                f"Batch KG extraction phase {phase} response size mismatch: expected={len(prompts)} got={len(responses)}"
            )
        return responses


def collect_mentions(phase_a_payload: dict) -> List[str]:
    mentions: List[str] = []
    seen = set()
    for item in (phase_a_payload.get("entities", []) or []):
        if not isinstance(item, dict):
            continue
        mention = str(item.get("mention", "")).strip()
        if not mention:
            continue
        key = normalize_text(mention)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(mention)
    return mentions


def collect_canonical_entities(phase_b_payload: dict) -> List[dict]:
    entities: List[dict] = []
    seen = set()
    for item in (phase_b_payload.get("entities", []) or []):
        if not isinstance(item, dict):
            continue
        canonical_name = str(item.get("canonical_name", "")).strip()
        etype = str(item.get("type", "Other")).strip() or "Other"
        if not canonical_name:
            continue
        key = (normalize_text(canonical_name), normalize_text(etype))
        if key in seen:
            continue
        seen.add(key)
        entities.append({"canonical_name": canonical_name, "type": etype})
    return entities


def parse_phased_kg_json(phase_b_payload: dict, phase_c_payload: dict) -> KnowledgeGraph:
    entities_raw = phase_b_payload.get("entities", []) or []
    triples_raw = phase_c_payload.get("triples", []) or []

    entities: List[Entity] = []
    entity_seen = set()
    for item in entities_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("canonical_name") or item.get("name", "")).strip()
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
        relation = normalize_relation(str(item.get("relation", "")).strip())
        tail = str(item.get("tail", "")).strip()
        if not head or not relation or not tail:
            continue

        key = (normalize_text(head), relation, normalize_text(tail))
        if key in triple_seen:
            continue
        triple_seen.add(key)
        triples.append(Triple(head=head, relation=relation, tail=tail))

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


def parse_kg_json(payload: dict) -> KnowledgeGraph:
    """Backward-compatible parser for single-pass extraction payloads."""
    return parse_phased_kg_json(phase_b_payload=payload, phase_c_payload=payload)



def merge_kgs(kgs: List[KnowledgeGraph]) -> KnowledgeGraph:
    type_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}

    for kg in kgs:
        for entity in kg.entities:
            norm = normalize_text(entity.name)
            if norm not in name_map:
                name_map[norm] = entity.name
                type_map[norm] = entity.type
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
