from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from .api import OpenAICompatibleChatClient
from .embeddings import LocalSentenceEmbedder
from .kg_extraction import entity_type_map
from .prompts import (
    build_answer_summary_prompt,
    build_contrastive_answer_prompt,
    build_llm_question_generation_prompt,
    build_question_from_triple,
)
from .schemas import KnowledgeGraph, QAPair, Triple
from .utils import (
    deduplicate_preserve_order, 
    join_reports, 
    normalize_text, 
    truncate_text,
    near_duplicate_entity, 
    is_clause_like_entity
)

ALLOWED_TYPES = {
    "Person", "Organization", "Location", "Event", "Policy",
    "Law", "Group", "Concept", "Program", "Date"
}

def is_substitutable(name: str, etype: str) -> bool:
    if etype not in ALLOWED_TYPES:
        return False
    if is_clause_like_entity(name):
        return False
    return True

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    return normalized @ normalized.T



def generate_candidate_questions_from_kg(
    merged_kg: KnowledgeGraph,
    claim_triples: Sequence[Triple],
) -> List[str]:
    type_map = entity_type_map(merged_kg)
    entities_by_type: Dict[str, List[str]] = {}
    for entity in merged_kg.entities:
        entities_by_type.setdefault(entity.type, []).append(entity.name)

    questions: List[str] = []
    for triple in claim_triples:
        head_type = type_map.get(normalize_text(triple.head), "Other")
        tail_type = type_map.get(normalize_text(triple.tail), "Other")

        if not is_substitutable(triple.head, head_type):
            continue
        if not is_substitutable(triple.tail, tail_type):
            continue

        head_alts = [
            x for x in entities_by_type.get(head_type, [])
            if not near_duplicate_entity(x, triple.head) and is_substitutable(x, head_type)
        ]
        tail_alts = [
            x for x in entities_by_type.get(tail_type, [])
            if not near_duplicate_entity(x, triple.tail) and is_substitutable(x, tail_type)
        ]

        for alt_head in head_alts:
            questions.append(build_question_from_triple(triple, alt_head, replace="head"))
        for alt_tail in tail_alts:
            questions.append(build_question_from_triple(triple, alt_tail, replace="tail"))

    return deduplicate_preserve_order(questions)



def mmr_rerank_questions(
    questions: Sequence[str],
    embedder: LocalSentenceEmbedder,
    top_k: int,
    mmr_lambda: float = 1.0,
) -> List[str]:
    questions = list(questions)
    if not questions:
        return []
    if len(questions) <= top_k:
        return questions

    embeddings = embedder.encode(questions)
    sim = cosine_similarity_matrix(embeddings)

    avg_sim = sim.mean(axis=1)
    query_idx = int(np.argmax(avg_sim))
    selected = [query_idx]
    candidates = set(range(len(questions))) - set(selected)

    while candidates and len(selected) < top_k:
        best_idx = None
        best_score = -1e18
        for idx in candidates:
            relevance = sim[idx, query_idx]
            redundancy = max(sim[idx, s] for s in selected) if selected else 0.0
            score = relevance - mmr_lambda * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        candidates.remove(best_idx)

    return [questions[i] for i in selected]



def answer_questions(
    client: OpenAICompatibleChatClient,
    claim: str,
    reports: List[str],
    questions: Sequence[str],
    max_context_chars: int,
    progress_callback: Callable[[], None] | None = None,
) -> List[QAPair]:
    context = truncate_text(join_reports(reports), max_context_chars)
    qa_pairs: List[QAPair] = []
    for question in questions:
        prompt = build_contrastive_answer_prompt(context=context, claim=claim, question=question)
        response = client.chat(messages=[{"role": "user", "content": prompt}])
        qa_pairs.append(QAPair(question=question, answer=response.content.strip()))
        if progress_callback is not None:
            progress_callback()
    return qa_pairs



def summarize_qa_pairs(
    client: OpenAICompatibleChatClient,
    claim: str,
    qa_pairs: Sequence[QAPair],
) -> str:
    prompt = build_answer_summary_prompt(claim=claim, qa_pairs=list(qa_pairs))
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    return response.content.strip()



def generate_questions_with_llm(
    client: OpenAICompatibleChatClient,
    claim: str,
    reports: List[str],
    num_questions: int,
    examples: List[dict] | None = None,
) -> List[str]:
    return generate_questions_with_llm_batch(
        client=client,
        claims=[claim],
        reports_list=[reports],
        num_questions=num_questions,
        examples=examples,
    )[0]


def generate_questions_with_llm_batch(
    client: OpenAICompatibleChatClient,
    claims: Sequence[str],
    reports_list: Sequence[List[str]],
    num_questions: int,
    examples: List[dict] | None = None,
) -> List[List[str]]:
    if len(claims) != len(reports_list):
        raise ValueError(f"claims/reports_list size mismatch: {len(claims)} vs {len(reports_list)}")
    if not claims:
        return []
    prompts = [
        build_llm_question_generation_prompt(
            claim=claim,
            reports=join_reports(reports),
            num_questions=num_questions,
            examples=examples,
        )
        for claim, reports in zip(claims, reports_list)
    ]
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    responses = client.chat_batch(messages_batch=messages_batch)
    if len(responses) != len(claims):
        raise ValueError(
            f"Batch question generation response size mismatch: expected={len(claims)} got={len(responses)}"
        )
    all_questions: List[List[str]] = []
    for response in responses:
        questions: List[str] = []
        for line in response.content.splitlines():
            line = line.strip()
            if not line:
                continue
            line = line.lstrip("-•0123456789. ")
            if line:
                questions.append(line)
        all_questions.append(deduplicate_preserve_order(questions)[:num_questions])
    return all_questions
