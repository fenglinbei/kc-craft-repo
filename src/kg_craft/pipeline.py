from __future__ import annotations

from dataclasses import asdict
import logging
from typing import List

from tqdm import tqdm

from .api import OpenAICompatibleChatClient
from .config import AppConfig
from .contrastive import (
    answer_questions,
    generate_candidate_questions_from_kg,
    generate_questions_with_llm,
    mmr_rerank_questions,
    summarize_qa_pairs,
)
from .embeddings import LocalSentenceEmbedder
from .kg_extraction import KGExtractor, merge_kgs, triples_from_claim_in_merged_graph
from .prompts import format_kg_as_text
from .schemas import PipelineResult, Sample
from .utils import join_reports, truncate_text
from .verification import verify_claim, verify_naive, verify_with_kg_only

LOGGER = logging.getLogger(__name__)


class KGCRAFTPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        if "kg_llm" not in config.models:
            raise KeyError("config.models must include 'kg_llm'")
        if "reasoning_llm" not in config.models:
            raise KeyError("config.models must include 'reasoning_llm'")

        self.kg_client = OpenAICompatibleChatClient(
            config.models["kg_llm"],
            cache_cfg=config.cache,
            namespace="kg_llm",
        )
        self.reasoning_client = OpenAICompatibleChatClient(
            config.models["reasoning_llm"],
            cache_cfg=config.cache,
            namespace="reasoning_llm",
        )
        self.kg_extractor = KGExtractor(self.kg_client)
        self.embedder = LocalSentenceEmbedder(
            model_path=config.embedding.model_path,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            normalize=config.embedding.normalize,
        )

    def run(self, samples: List[Sample], mode: str | None = None) -> List[PipelineResult]:
        mode = mode or self.config.run.mode
        results: List[PipelineResult] = []
        scoped_samples = samples[: self.config.run.limit]
        LOGGER.info("Pipeline started. mode=%s, samples=%d", mode, len(scoped_samples))
        for sample in tqdm(scoped_samples, desc=f"KG-CRAFT ({mode})", unit="sample"):
            results.append(self.run_one(sample, mode=mode))
        LOGGER.info("Pipeline finished. mode=%s, results=%d", mode, len(results))
        return results

    def run_one(self, sample: Sample, mode: str | None = None) -> PipelineResult:
        mode = mode or self.config.run.mode
        LOGGER.debug("Processing sample_id=%s mode=%s", sample.sample_id, mode)
        labels = self.config.verification.labels
        label_descriptions = self.config.verification.label_descriptions

        result = PipelineResult(
            sample_id=sample.sample_id,
            claim=sample.claim,
            reports=sample.reports,
            label=sample.label,
            prediction="",
            mode=mode,
            meta=sample.meta,
        )

        if mode == "naive_llm":
            context = truncate_text(
                join_reports(sample.reports),
                self.config.pipeline.max_context_chars_for_verification,
            )
            result.prediction = verify_naive(
                client=self.reasoning_client,
                claim=sample.claim,
                context=context,
                labels=labels,
                label_descriptions=label_descriptions,
            )
            return result

        # Shared KG extraction for full / kg_only / llm_questions
        claim_kg_out = self.kg_extractor.extract(sample.claim)
        report_kgs_out = [self.kg_extractor.extract(report) for report in sample.reports]
        merged_kg = merge_kgs([claim_kg_out.kg] + [x.kg for x in report_kgs_out])
        claim_triples = triples_from_claim_in_merged_graph(claim_kg_out.kg, merged_kg)
        kg_text = format_kg_as_text(
            entities=merged_kg.to_dict()["entities"],
            triples=merged_kg.to_dict()["triples"],
        )

        result.claim_kg = claim_kg_out.kg.to_dict()
        result.report_kgs = [x.kg.to_dict() for x in report_kgs_out]
        result.merged_kg = merged_kg.to_dict()
        result.claim_triples = [t.to_dict() for t in claim_triples]
        result.kg_text = kg_text
        if self.config.pipeline.save_raw_api_responses:
            result.raw_outputs["claim_kg_response"] = claim_kg_out.raw_response
            result.raw_outputs["report_kg_responses"] = [x.raw_response for x in report_kgs_out]

        if mode == "kg_only":
            result.prediction = verify_with_kg_only(
                client=self.reasoning_client,
                claim=sample.claim,
                kg_text=kg_text,
                labels=labels,
                label_descriptions=label_descriptions,
            )
            return result

        if mode == "full":
            candidate_questions = generate_candidate_questions_from_kg(
                merged_kg=merged_kg,
                claim_triples=claim_triples,
            )
            selected_questions = mmr_rerank_questions(
                questions=candidate_questions,
                embedder=self.embedder,
                top_k=self.config.pipeline.max_contrastive_questions,
                mmr_lambda=self.config.pipeline.mmr_lambda,
            )
        elif mode == "llm_questions":
            candidate_questions = generate_questions_with_llm(
                client=self.reasoning_client,
                claim=sample.claim,
                reports=sample.reports,
                num_questions=self.config.pipeline.max_contrastive_questions,
                examples=self.config.prompts.llm_question_examples,
            )
            selected_questions = candidate_questions[: self.config.pipeline.max_contrastive_questions]
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        qa_pairs = answer_questions(
            client=self.reasoning_client,
            claim=sample.claim,
            reports=sample.reports,
            questions=selected_questions,
            max_context_chars=self.config.pipeline.max_context_chars_for_answers,
        )
        summary = summarize_qa_pairs(
            client=self.reasoning_client,
            claim=sample.claim,
            qa_pairs=qa_pairs,
        )
        prediction = verify_claim(
            client=self.reasoning_client,
            claim=sample.claim,
            context=summary,
            labels=labels,
            label_descriptions=label_descriptions,
        )

        result.candidate_questions = candidate_questions
        result.selected_questions = selected_questions
        result.qa_pairs = [qa.to_dict() for qa in qa_pairs]
        result.contrastive_summary = summary
        result.prediction = prediction
        return result
