from __future__ import annotations

from typing import Dict, List, Optional

from .schemas import QAPair, Triple


PAPER_KG_EXTRACTION_TEMPLATE = '''You are a top-tier algorithm designed
for extracting information in
structured formats to build a
knowledge graph.
Knowledge graphs consist of a set of
triples. Each triple contains two
entities (subject and object) and one
relation that connects these subject
and object.
Try to capture as much information
from the text as possible without
sacrificing accuracy. Do not add any
information that is not explicitly
mentioned in the text.
This is the process to extract
information and build a knowledge
graph:
1. Extract nodes [...]
2. Label nodes [...]
3. Extract relationships [...]
Compliance criteria: [...]
Text: {claim_or_report}'''


PAPER_CONTRASTIVE_QA_TEMPLATE = '''You are an expert answering questions
based only on the provided context.
## Task:
Using the context provided and being
aware of the claim, answer the
question regarding the claim aiming
to fact-check it. Limit your answer
to 200 words at most.
## Desired Outcome:
- Base the concise answer strictly on
the context.
- Present the information neutrally,
without judging or labeling the
claim.
- Do not re-state the claim in the
answer.
- Write in continuous prose (no lists,
bullet points, or meta-commentary).
- Limit your answer to 200 words at
most.
## Input:
* Context: {context}
* Claim: {claim}
* Question: {contrastive_question}
## Output:'''


PAPER_SUMMARY_TEMPLATE = '''You are an expert writing summarizing
information from pairs of question
and answer.
## Task:
Your task is to generate an one
paragraph summary of the information
based on given pairs of question and
answer.
## Desired Outcome:
- A one paragraph summary of the
information contained in the question
and answer.
- Present the information neutrally,
without judging or labeling the
claim.
- Ensure that the summary is clear
and accurately based on the provided
context.
- Write in continuous prose (no lists,
bullet points, or meta-commentary).
- Do not add any utterances (for
example "Here are" statements) to the
final answer.
- Limit your answer to 200 words at
most.''' 


PAPER_VERIFICATION_TEMPLATE = '''You are an expert fact-checking
claims based solely on the provided
context of the claim.
## Task:
Your task is to categorize the claim
based only on the context as:
- {veracity_labels}
## Desired Outcome:
- The veracity of the claim based on
the context provided.
- Your response must be only one of
the above options. Do not include
any other text.
## Input:
* Context: {context}
* Claim: {claim}
## Output:'''


PAPER_CONTRASTIVE_Q_FORM_TEMPLATE = '''You are an expert writing analyzing a
given claim and generating
contrastive questions based on given
context. Your task is to generate
contrastive questions of given claim
based on given context.
## Desired Outcome:
- Create contrastive questions of an
input claim based on given context.
- Present a list of five contrastive
questions.
- Do not add any utterances (for
example "Here are" statements) to the
final answer.
## Example Prompt:
* Claim: {claim example}
* Context: {reports examples}
## Example Output:
"""
{contrastive questions examples}
"""
## Additional Notes:
- Ensure that the contrastive
questions are clear and accurately
contrasts the claim based on the
provided context.
- Maintain a consistent and readable
format for the output.
- Ensure that the output is only the
contrastive questions, no other
additional text or utterances.
## Input:
* Claim: {claim}
* Context: {reports}
## Output:'''



def build_kg_phase_a_prompt(text: str) -> str:
    """Phase A prompt: mention discovery only (few-shot)."""
    return f"""You are a top-tier algorithm for extracting entities from text to build a knowledge graph.

Phase A goal:
- Extract mention-level entities only.
- Do not output triples.
- Keep surface mentions as they appear in text; do not canonicalize yet.

Few-shot example:
Input text:
JoeBiden discussed an economic stimulus package for the American people. Joe Biden said it would reduce costs.

Output JSON:
{{
  "entities": [
    {{"mention": "JoeBiden"}},
    {{"mention": "economic stimulus package"}},
    {{"mention": "American people"}},
    {{"mention": "Joe Biden"}}
  ]
}}

Rules:
- Include salient people, organizations, places, events, objects, policies, or concepts explicitly mentioned.
- Remove duplicates only if the exact mention string repeats.
- Return valid JSON only.
- Schema must be:
{{
  "entities": [
    {{"mention": "string"}}
  ]
}}

Input text:
{text}"""


def build_kg_phase_b_prompt(text: str, mentions: List[str]) -> str:
    """Phase B prompt: mention typing + canonicalization (few-shot)."""
    mention_lines = "\n".join(f"- {m}" for m in mentions) if mentions else "- (none)"
    return f"""You are a top-tier algorithm for entity disambiguation and typing in knowledge graph construction.

Phase B goal:
- Given the text and extracted mentions, map each mention to:
  1) canonical_name
  2) semantic type

Few-shot example:
Input text:
JoeBiden discussed an economic stimulus package for the American people. Joe Biden said it would reduce costs.
Mentions:
- JoeBiden
- Joe Biden
- American people
- economic stimulus package

Output JSON:
{{
  "entities": [
    {{"mention": "JoeBiden", "canonical_name": "Joe Biden", "type": "Person"}},
    {{"mention": "Joe Biden", "canonical_name": "Joe Biden", "type": "Person"}},
    {{"mention": "American people", "canonical_name": "American people", "type": "Group"}},
    {{"mention": "economic stimulus package", "canonical_name": "economic stimulus package", "type": "Policy"}}
  ]
}}

Rules:
- Every input mention must appear exactly once in output.
- canonical_name must be concise and consistent across aliases.
- type should be one of: Person, Organization, Location, Event, Date, Policy, Product, Group, Concept, Claim, Other.
- Do not add mentions that are not in the list.
- Return valid JSON only.

Input text:
{text}

Mentions:
{mention_lines}"""


def build_kg_phase_c_prompt(text: str, canonical_entities: List[dict]) -> str:
    """Phase C prompt: relation extraction constrained to canonical entities."""
    if canonical_entities:
        entity_lines = "\n".join(
            f"- {item.get('canonical_name', '')} [{item.get('type', 'Other')}]"
            for item in canonical_entities
        )
    else:
        entity_lines = "- (none)"

    return f"""You are a top-tier algorithm for extracting relationships for knowledge graphs.

Phase C goal:
- Extract relation triples supported by the text.
- You may only use canonical entity names from the provided entity list as triple head/tail.

Few-shot example:
Input text:
JoeBiden discussed an economic stimulus package for the American people. Joe Biden said it would reduce costs.
Canonical entities:
- Joe Biden [Person]
- economic stimulus package [Policy]
- American people [Group]

Output JSON:
{{
  "triples": [
    {{"head": "Joe Biden", "relation": "DISCUSSED", "tail": "economic stimulus package"}},
    {{"head": "American people", "relation": "AFFECTED_BY", "tail": "economic stimulus package"}}
  ]
}}

Rules:
- Use uppercase relation labels (prefer UPPER_SNAKE_CASE).
- Do not invent facts not explicitly supported.
- Do not output triples with head/tail outside the provided canonical entities.
- Do not output duplicate triples.
- Return valid JSON only with schema:
{{
  "triples": [
    {{"head": "string", "relation": "RELATION", "tail": "string"}}
  ]
}}

Input text:
{text}

Canonical entities:
{entity_lines}"""



def build_contrastive_answer_prompt(context: str, claim: str, question: str) -> str:
    return f"""You are an expert answering questions based only on the provided context.

## Task:
Using the context provided and being aware of the claim, answer the question regarding the claim aiming to fact-check it. Limit your answer to 200 words at most.

## Desired Outcome:
- Base the concise answer strictly on the context.
- Present the information neutrally, without judging or labeling the claim.
- Do not re-state the claim in the answer.
- Write in continuous prose (no lists, bullet points, or meta-commentary).
- Limit your answer to 200 words at most.

## Input:
* Context: {context}
* Claim: {claim}
* Question: {question}

## Output:"""



def build_answer_summary_prompt(claim: str, qa_pairs: List[QAPair]) -> str:
    qa_blocks = [f"* Claim: {claim}"]
    for idx, qa in enumerate(qa_pairs, start=1):
        qa_blocks.append(f"* Question {idx}: {qa.question}")
        qa_blocks.append(f"* Answer {idx}: {qa.answer}")
    joined = "\n".join(qa_blocks)
    return f"""You are an expert writing summarizing information from pairs of question and answer.

## Task:
Your task is to generate an one paragraph summary of the information based on given pairs of question and answer.

## Desired Outcome:
- A one paragraph summary of the information contained in the question and answer.
- Present the information neutrally, without judging or labeling the claim.
- Ensure that the summary is clear and accurately based on the provided context.
- Write in continuous prose (no lists, bullet points, or meta-commentary).
- Do not add any utterances (for example \"Here are\" statements) to the final answer.
- Limit your answer to 200 words at most.

## Input:
{joined}

## Output:"""



def build_verification_prompt(
    context: str,
    claim: str,
    labels: List[str],
    label_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    label_lines = []
    for label in labels:
        desc = (label_descriptions or {}).get(label, "")
        if desc:
            label_lines.append(f"- {label}: {desc}")
        else:
            label_lines.append(f"- {label}")
    labels_text = "\n".join(label_lines)
    return f"""You are an expert fact-checking claims based solely on the provided context of the claim.

## Task:
Your task is to categorize the claim based only on the context as:
{labels_text}

## Desired Outcome:
- The veracity of the claim based on the context provided.
- Your response must be only one of the above options. Do not include any other text.

## Input:
* Context: {context}
* Claim: {claim}

## Output:"""



def build_llm_question_generation_prompt(
    claim: str,
    reports: str,
    num_questions: int,
    examples: Optional[List[dict]] = None,
) -> str:
    example_block = ""
    if examples:
        chunks = []
        for ex in examples:
            q_text = "\n".join(f"- {q}" for q in ex.get("questions", []))
            chunks.append(
                "## Example Prompt:\n"
                f"* Claim: {ex.get('claim', '')}\n"
                f"* Context: {ex.get('reports', '')}\n"
                "## Example Output:\n"
                f"{q_text}"
            )
        example_block = "\n\n" + "\n\n".join(chunks)

    return f"""You are an expert writing analyzing a given claim and generating contrastive questions based on given context. Your task is to generate contrastive questions of given claim based on given context.

## Desired Outcome:
- Create contrastive questions of an input claim based on given context.
- Present a list of {num_questions} contrastive questions.
- Do not add any utterances (for example \"Here are\" statements) to the final answer.{example_block}

## Additional Notes:
- Ensure that the contrastive questions are clear and accurately contrasts the claim based on the provided context.
- Maintain a consistent and readable format for the output.
- Ensure that the output is only the contrastive questions, no other additional text or utterances.

## Input:
* Claim: {claim}
* Context: {reports}

## Output:"""



def format_kg_as_text(entities: List[dict], triples: List[dict]) -> str:
    lines = ["Entities:"]
    for ent in entities:
        lines.append(f"- {ent['name']} [{ent['type']}]")
    lines.append("\nTriples:")
    for triple in triples:
        lines.append(f"- ({triple['head']}, {triple['relation']}, {triple['tail']})")
    return "\n".join(lines)



def build_naive_verification_prompt(context: str, claim: str, labels: List[str], label_descriptions: Optional[Dict[str, str]] = None) -> str:
    return build_verification_prompt(context=context, claim=claim, labels=labels, label_descriptions=label_descriptions)



def build_kg_only_verification_prompt(claim: str, kg_text: str, labels: List[str], label_descriptions: Optional[Dict[str, str]] = None) -> str:
    label_lines = []
    for label in labels:
        desc = (label_descriptions or {}).get(label, "")
        if desc:
            label_lines.append(f"- {label}: {desc}")
        else:
            label_lines.append(f"- {label}")
    labels_text = "\n".join(label_lines)
    return f"""You are an expert fact-checking claims based solely on the provided claim and the extracted knowledge graph.

## Task:
Your task is to categorize the claim based only on the provided knowledge graph as:
{labels_text}

## Desired Outcome:
- Use the knowledge graph as structured evidence.
- Your response must be only one of the above options. Do not include any other text.

## Input:
* Claim: {claim}
* Knowledge Graph:
{kg_text}

## Output:"""



def build_question_from_triple(triple: Triple, alternative: str, replace: str) -> str:
    if replace == "head":
        return f"Why {triple.head} {triple.relation} {triple.tail}, rather than {alternative} {triple.relation} {triple.tail}?"
    if replace == "tail":
        return f"Why {triple.head} {triple.relation} {alternative}, rather than {triple.head} {triple.relation} {triple.tail}?"
    raise ValueError(f"replace must be 'head' or 'tail', got {replace!r}")
