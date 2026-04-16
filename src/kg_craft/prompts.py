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



def build_operational_kg_extraction_prompt(text: str) -> str:
    """A runnable operationalization of Appendix D.1.

    The paper PDF exposes a shortened template with elided details. This prompt keeps
    the same three phases but makes the output machine-readable JSON.
    """
    return f"""You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.

Knowledge graphs consist of a set of triples. Each triple contains two entities (subject and object) and one relation that connects these subject and object.

Try to capture as much information from the text as possible without sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text.

This is the process to extract information and build a knowledge graph:
1. Extract nodes:
- Identify salient entities, concepts, organizations, people, locations, dates, events, or named objects explicitly mentioned in the text.
- Keep node names short, faithful, and canonical when possible.
2. Label nodes:
- Assign one concise semantic type to each node, such as Person, Organization, Location, Event, Date, Policy, Product, Concept, Claim, or Other.
3. Extract relationships:
- Create triples (head, relation, tail) only when the relation is clearly supported by the text.
- Use short relation phrases in uppercase snake case or uppercase words when possible, such as WORKS_FOR, LOCATED_IN, ANNOUNCED, PAID, PART_OF.

Compliance criteria:
- Do not invent facts.
- Do not include duplicate entities or duplicate triples.
- Every triple head and tail must appear in the entities list.
- If the text is too vague, return fewer triples rather than hallucinating.
- Return valid JSON only. Do not include markdown fences or any explanatory text.

Return exactly this JSON schema:
{{
  "entities": [
    {{"name": "entity name", "type": "entity type"}}
  ],
  "triples": [
    {{"head": "entity name", "relation": "RELATION", "tail": "entity name"}}
  ]
}}

Text: {text}"""



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
