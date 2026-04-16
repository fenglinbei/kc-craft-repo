from __future__ import annotations

from typing import Dict, List, Sequence

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
from .utils import deduplicate_preserve_order, join_reports, normalize_text, truncate_text



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

        head_alts = [x for x in entities_by_type.get(head_type, []) if normalize_text(x) != normalize_text(triple.head)]
        tail_alts = [x for x in entities_by_type.get(tail_type, []) if normalize_text(x) != normalize_text(triple.tail)]

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
) -> List[QAPair]:
    context = truncate_text(join_reports(reports), max_context_chars)
    qa_pairs: List[QAPair] = []
    for question in questions:
        prompt = build_contrastive_answer_prompt(context=context, claim=claim, question=question)
        response = client.chat(messages=[{"role": "user", "content": prompt}])
        qa_pairs.append(QAPair(question=question, answer=response.content.strip()))
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
    prompt = build_llm_question_generation_prompt(
        claim=claim,
        reports=join_reports(reports),
        num_questions=num_questions,
        examples=examples,
    )
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    questions = []
    for line in response.content.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-•0123456789. ")
        if line:
            questions.append(line)
    return deduplicate_preserve_order(questions)[:num_questions]
