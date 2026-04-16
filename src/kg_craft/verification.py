from __future__ import annotations

from typing import Dict, List, Optional

from .api import OpenAICompatibleChatClient
from .prompts import (
    build_kg_only_verification_prompt,
    build_naive_verification_prompt,
    build_verification_prompt,
)
from .utils import truncate_text



def normalize_prediction(prediction: str, labels: List[str]) -> str:
    cleaned = prediction.strip()
    for label in labels:
        if cleaned == label:
            return label
    lowered = cleaned.lower()
    for label in labels:
        if lowered == label.lower():
            return label
    # fallback: try exact containment of any label as a line or token
    for label in labels:
        if label.lower() in lowered:
            return label
    return cleaned



def verify_claim(
    client: OpenAICompatibleChatClient,
    claim: str,
    context: str,
    labels: List[str],
    label_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    prompt = build_verification_prompt(
        context=context,
        claim=claim,
        labels=labels,
        label_descriptions=label_descriptions,
    )
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    return normalize_prediction(response.content, labels)



def verify_naive(
    client: OpenAICompatibleChatClient,
    claim: str,
    context: str,
    labels: List[str],
    label_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    prompt = build_naive_verification_prompt(
        context=context,
        claim=claim,
        labels=labels,
        label_descriptions=label_descriptions,
    )
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    return normalize_prediction(response.content, labels)



def verify_with_kg_only(
    client: OpenAICompatibleChatClient,
    claim: str,
    kg_text: str,
    labels: List[str],
    label_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    prompt = build_kg_only_verification_prompt(
        claim=claim,
        kg_text=kg_text,
        labels=labels,
        label_descriptions=label_descriptions,
    )
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    return normalize_prediction(response.content, labels)
