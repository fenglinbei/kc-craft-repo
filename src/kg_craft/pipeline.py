from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import logging
from pathlib import Path
import time
from typing import Any, List

from tqdm import tqdm

from .api import OpenAICompatibleChatClient
from .config import AppConfig
from .contrastive import (
    answer_questions,
    generate_candidate_questions_from_kg,
    generate_questions_with_llm_batch,
    generate_questions_with_llm,
    mmr_rerank_questions,
    summarize_qa_pairs,
)
from .embeddings import LocalSentenceEmbedder
from .evaluation import compute_metrics, save_metrics_figure
from .kg_extraction import KGExtractor, merge_kgs, triples_from_claim_in_merged_graph
from .prompts import format_kg_as_text
from .schemas import PipelineResult, Sample
from .utils import join_reports, truncate_text
from .verification import verify_claim, verify_naive, verify_with_kg_only

LOGGER = logging.getLogger(__name__)


class WandbTracker:
    def __init__(self, wandb_config: dict[str, Any], run_config: dict[str, Any], data_config: dict[str, Any]):
        self._enabled = bool(wandb_config.get("enabled", False))
        self._run = None
        self._step = 0
        self._wandb = None
        if not self._enabled:
            return
        try:
            self._wandb = importlib.import_module("wandb")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("W&B enabled in config but import failed, skip logging: %s", exc)
            self._enabled = False
            return

        init_kwargs: dict[str, Any] = {
            "project": wandb_config.get("project", "kg-craft"),
            "entity": wandb_config.get("entity"),
            "name": wandb_config.get("name"),
            "group": wandb_config.get("group"),
            "job_type": wandb_config.get("job_type", "pipeline"),
            "tags": wandb_config.get("tags"),
            "mode": wandb_config.get("mode"),
            "config": {
                "run": run_config,
                "data": data_config,
            },
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        self._run = self._wandb.init(**init_kwargs)
        LOGGER.info("Initialized W&B run. project=%s", init_kwargs.get("project"))

    @property
    def enabled(self) -> bool:
        return self._enabled and self._run is not None

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        if step is None:
            self._step += 1
            step = self._step
        self._wandb.log(payload, step=step)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._run.finish()


class KGCRAFTPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        enable_messages_batch_api = max(1, config.run.batch_size) > 1
        if "kg_llm" not in config.models:
            raise KeyError("config.models must include 'kg_llm'")
        if "reasoning_llm" not in config.models:
            raise KeyError("config.models must include 'reasoning_llm'")

        self.kg_client = OpenAICompatibleChatClient(
            config.models["kg_llm"],
            cache_cfg=config.cache,
            namespace="kg_llm",
            enable_messages_batch_api=enable_messages_batch_api,
            debug=config.run.debug,
            debug_preview_chars=config.run.debug_preview_chars,
            debug_head_chars=config.run.debug_head_chars,
            debug_tail_chars=config.run.debug_tail_chars,
        )
        self.reasoning_client = OpenAICompatibleChatClient(
            config.models["reasoning_llm"],
            cache_cfg=config.cache,
            namespace="reasoning_llm",
            enable_messages_batch_api=enable_messages_batch_api,
            debug=config.run.debug,
            debug_preview_chars=config.run.debug_preview_chars,
            debug_head_chars=config.run.debug_head_chars,
            debug_tail_chars=config.run.debug_tail_chars,
        )
        self.kg_extractor = KGExtractor(self.kg_client)
        self.embedder = LocalSentenceEmbedder(
            model_path=config.embedding.model_path,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            normalize=config.embedding.normalize,
        )
        wandb_cfg = config.extras.get("wandb", {}) if isinstance(config.extras, dict) else {}
        self.wandb = WandbTracker(
            wandb_config=wandb_cfg if isinstance(wandb_cfg, dict) else {},
            run_config=asdict(config.run),
            data_config=asdict(config.data),
        )

    def run(self, samples: List[Sample], mode: str | None = None) -> List[PipelineResult]:
        mode = mode or self.config.run.mode
        scoped_samples = samples[: self.config.run.limit]
        total_samples = len(scoped_samples)
        num_workers = max(1, self.config.run.num_workers)
        batch_size = max(1, self.config.run.batch_size)
        show_overall_progress = self.config.run.verbose

        LOGGER.info(
            "Pipeline started. mode=%s, samples=%d, num_workers=%d, batch_size=%d",
            mode,
            total_samples,
            num_workers,
            batch_size,
        )
        progress_bar: tqdm | None = None
        if show_overall_progress:
            progress_bar = tqdm(total=total_samples, desc=f"KG-CRAFT ({mode})", unit="sample")

        results: List[PipelineResult] = []
        try:
            if num_workers == 1:
                for batch_start in range(0, total_samples, batch_size):
                    batch = scoped_samples[batch_start : batch_start + batch_size]
                    batch_started = time.perf_counter()
                    batch_results = self.run_batch(batch, mode=mode)
                    batch_elapsed = time.perf_counter() - batch_started
                    avg_batch_sample_latency = batch_elapsed / max(1, len(batch_results))
                    for sample, result in zip(batch, batch_results):
                        result.meta.setdefault("processing_latency_seconds", avg_batch_sample_latency)
                        results.append(result)
                        self._log_running_metrics(
                            results,
                            sample_id=sample.sample_id,
                            progress_bar=progress_bar,
                            total_samples=total_samples,
                        )
            else:
                indexed_results: list[tuple[int, PipelineResult]] = []
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for batch_start in range(0, total_samples, batch_size):
                        batch_items = list(enumerate(scoped_samples[batch_start : batch_start + batch_size], start=batch_start))
                        futures = {
                            executor.submit(self.run_one, sample, mode): idx
                            for idx, sample in batch_items
                        }
                        for future in as_completed(futures):
                            idx = futures[future]
                            result = future.result()
                            indexed_results.append((idx, result))
                            self._log_running_metrics(
                                [x[1] for x in indexed_results],
                                sample_id=result.sample_id,
                                progress_bar=progress_bar,
                                total_samples=total_samples,
                            )
                results = [result for _, result in sorted(indexed_results, key=lambda item: item[0])]
            self._log_final_metrics(results)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            self.wandb.finish()
        LOGGER.info("Pipeline finished. mode=%s, results=%d", mode, len(results))
        return results

    def run_batch(self, samples: List[Sample], mode: str | None = None) -> List[PipelineResult]:
        mode = mode or self.config.run.mode
        if not samples:
            return []
        if mode != "llm_questions":
            return [self.run_one(sample, mode=mode) for sample in samples]
        return self._run_batch_llm_questions(samples=samples, mode=mode)

    def _run_batch_llm_questions(self, samples: List[Sample], mode: str) -> List[PipelineResult]:
        base_results = [self._run_one_before_question_generation(sample, mode) for sample in samples]
        question_batches = generate_questions_with_llm_batch(
            client=self.reasoning_client,
            claims=[sample.claim for sample in samples],
            reports_list=[sample.reports for sample in samples],
            num_questions=self.config.pipeline.max_contrastive_questions,
            examples=self.config.prompts.llm_question_examples,
        )
        for result, questions in zip(base_results, question_batches):
            selected_questions = questions[: self.config.pipeline.max_contrastive_questions]
            result.candidate_questions = selected_questions
            result.selected_questions = selected_questions
            qa_pairs = answer_questions(
                client=self.reasoning_client,
                claim=result.claim,
                reports=result.reports,
                questions=selected_questions,
                max_context_chars=self.config.pipeline.max_context_chars_for_answers,
            )
            result.qa_pairs = [asdict(qa) for qa in qa_pairs]
            summary = summarize_qa_pairs(client=self.reasoning_client, claim=result.claim, qa_pairs=qa_pairs)
            result.qa_summary = summary
            context_for_verification = truncate_text(
                summary,
                self.config.pipeline.max_context_chars_for_verification,
            )
            result.prediction = verify_claim(
                client=self.reasoning_client,
                claim=result.claim,
                context=context_for_verification,
                labels=self.config.verification.labels,
                label_descriptions=self.config.verification.label_descriptions,
            )
        return base_results

    def _run_one_before_question_generation(self, sample: Sample, mode: str) -> PipelineResult:
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
        kg_outputs = self.kg_extractor.extract_batch([sample.claim] + sample.reports)
        claim_kg_out = kg_outputs[0]
        report_kgs_out = kg_outputs[1:]
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

    @staticmethod
    def _labeled_pairs(results: List[PipelineResult]) -> tuple[list[str], list[str]]:
        y_true: list[str] = []
        y_pred: list[str] = []
        for result in results:
            if result.label is None:
                continue
            gold = str(result.label).strip()
            pred = str(result.prediction).strip()
            if not gold or not pred:
                continue
            y_true.append(gold)
            y_pred.append(pred)
        return y_true, y_pred

    @staticmethod
    def _format_running_metrics_postfix(metrics: dict[str, Any], n: int) -> str:
        return (
            f"P/R/F1={metrics['macro_precision']:.4f}/"
            f"{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}, n={n}"
        )

    def _log_running_metrics(
        self,
        results: List[PipelineResult],
        sample_id: str,
        progress_bar: tqdm | None = None,
        total_samples: int | None = None,
    ) -> None:
        y_true, y_pred = self._labeled_pairs(results)
        completed_samples = len(results)
        latest_result = next((x for x in reversed(results) if x.sample_id == sample_id), None)
        latest_latency = None if latest_result is None else latest_result.meta.get("processing_latency_seconds")
        if progress_bar is not None:
            progress_bar.update(1)
        if not y_true:
            LOGGER.info(
                "sample_id=%s classification finished. P/R/F1 unavailable (missing labels or predictions).",
                sample_id,
            )
            if progress_bar is not None:
                progress_bar.set_postfix_str("running macro P/R/F1 unavailable")
            self.wandb.log(
                {
                    "progress/completed_samples": completed_samples,
                    "progress/total_samples": total_samples,
                    "progress/ratio": (completed_samples / total_samples) if total_samples else 0.0,
                    "latency/sample_seconds": latest_latency,
                },
                step=completed_samples,
            )
            return
        metrics = compute_metrics(y_true, y_pred)
        running_postfix = self._format_running_metrics_postfix(metrics, n=len(y_true))
        if progress_bar is not None:
            progress_bar.set_postfix_str(running_postfix)
        self.wandb.log(
            {
                "progress/completed_samples": completed_samples,
                "progress/total_samples": total_samples,
                "progress/ratio": (completed_samples / total_samples) if total_samples else 0.0,
                "latency/sample_seconds": latest_latency,
                "metrics/running/macro_precision": metrics.get("macro_precision"),
                "metrics/running/macro_recall": metrics.get("macro_recall"),
                "metrics/running/macro_f1": metrics.get("macro_f1"),
                "metrics/running/accuracy": metrics.get("accuracy"),
                "metrics/running/labeled_count": len(y_true),
            },
            step=completed_samples,
        )
        # LOGGER.info(
        #     "sample_id=%s classification finished. running macro P/R/F1 = %.4f / %.4f / %.4f (n=%d)",
        #     sample_id,
        #     metrics["macro_precision"],
        #     metrics["macro_recall"],
        #     metrics["macro_f1"],
        #     len(y_true),
        # )

    def _log_final_metrics(self, results: List[PipelineResult]) -> None:
        y_true, y_pred = self._labeled_pairs(results)
        if not y_true:
            LOGGER.info("Final metrics unavailable (missing labels or predictions).")
            return

        metrics = compute_metrics(y_true, y_pred)
        report = metrics["classification_report"]
        LOGGER.info(
            "Final macro P/R/F1 = %.4f / %.4f / %.4f (accuracy=%.4f, n=%d)",
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
            metrics["accuracy"],
            len(y_true),
        )
        for label, values in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            LOGGER.info(
                "Final per-class %s: P/R/F1 = %.4f / %.4f / %.4f (support=%s)",
                label,
                values.get("precision", 0.0),
                values.get("recall", 0.0),
                values.get("f1-score", 0.0),
                values.get("support", 0),
            )
        macro_values = report.get("macro avg", {})
        LOGGER.info(
            "Final report macro avg: P/R/F1 = %.4f / %.4f / %.4f",
            macro_values.get("precision", 0.0),
            macro_values.get("recall", 0.0),
            macro_values.get("f1-score", 0.0),
        )
        labels = metrics.get("labels", [])
        matrix = metrics.get("confusion_matrix", [])
        if labels and matrix:
            LOGGER.info("Final confusion matrix labels: %s", labels)
            for idx, row in enumerate(matrix):
                LOGGER.info("Final confusion matrix row true=%s: %s", labels[idx], row)
        self.wandb.log(
            {
                "metrics/final/macro_precision": metrics.get("macro_precision"),
                "metrics/final/macro_recall": metrics.get("macro_recall"),
                "metrics/final/macro_f1": metrics.get("macro_f1"),
                "metrics/final/weighted_f1": metrics.get("weighted_f1"),
                "metrics/final/accuracy": metrics.get("accuracy"),
                "metrics/final/labeled_count": len(y_true),
            }
        )
        self._save_final_metrics_figure(metrics, mode=results[0].mode if results else self.config.run.mode)

    def _save_final_metrics_figure(self, metrics: dict[str, Any], mode: str) -> None:
        output_path = self.config.data.output_path
        if output_path:
            output_dir = Path(output_path).resolve().parent
        else:
            output_dir = Path.cwd() / "outputs"
        figure_path = output_dir / f"metrics_{mode}.png"
        try:
            saved_path = save_metrics_figure(metrics, figure_path, title=f"KG-CRAFT Final Metrics ({mode})")
            LOGGER.info("Saved final metrics figure to %s", saved_path)
        except ImportError as exc:
            LOGGER.warning("Skip metrics figure export because plotting dependency is missing: %s", exc)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to save metrics figure: %s", exc)

    def run_one(self, sample: Sample, mode: str | None = None) -> PipelineResult:
        mode = mode or self.config.run.mode
        LOGGER.debug("Processing sample_id=%s mode=%s", sample.sample_id, mode)
        sample_started = time.perf_counter()
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
        show_stage_progress = (
            self.config.run.verbose
            and self.config.run.show_sample_stage_progress
            and self.config.run.num_workers == 1
        )

        def make_stage_bar(stage_name: str, total: int = 1) -> tqdm | None:
            if not show_stage_progress:
                return None
            return tqdm(
                total=max(1, total),
                desc=f"sample={sample.sample_id} | {stage_name}",
                unit="step",
                leave=False,
            )

        def advance_stage_bar(stage_bar: tqdm | None, step: int = 1) -> None:
            if stage_bar is None:
                return
            stage_bar.update(step)

        def close_stage_bar(stage_bar: tqdm | None, ensure_complete: bool = True) -> None:
            if stage_bar is None:
                return
            if ensure_complete and stage_bar.total is not None and stage_bar.n < stage_bar.total:
                stage_bar.update(stage_bar.total - stage_bar.n)
            stage_bar.close()

        if mode == "naive_llm":
            kg_bar = make_stage_bar("KG extract（skip）")
            close_stage_bar(kg_bar)
            qa_bar = make_stage_bar("Contrastive QA（skip）")
            close_stage_bar(qa_bar)
            classification_bar = make_stage_bar("Classification")
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
            close_stage_bar(classification_bar)
            if self.config.run.debug:
                LOGGER.debug(
                    "[debug][pipeline] sample_id=%s step=naive_verification elapsed=%.3fs prediction=%s",
                    sample.sample_id,
                    time.perf_counter() - sample_started,
                    result.prediction,
                )
            result.meta["processing_latency_seconds"] = time.perf_counter() - sample_started
            return result

        # Shared KG extraction for full / kg_only / llm_questions
        kg_total_steps = 1 + len(sample.reports)
        kg_bar = make_stage_bar("KG extract", total=kg_total_steps)
        kg_outputs = self.kg_extractor.extract_batch([sample.claim] + sample.reports)
        claim_kg_out = kg_outputs[0]
        advance_stage_bar(kg_bar)
        report_kgs_out = kg_outputs[1:]
        for _ in sample.reports:
            advance_stage_bar(kg_bar)
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
        close_stage_bar(kg_bar)

        if mode == "kg_only":
            qa_bar = make_stage_bar("Contrastive QA（skip）")
            close_stage_bar(qa_bar)
            classification_bar = make_stage_bar("Classification")
            result.prediction = verify_with_kg_only(
                client=self.reasoning_client,
                claim=sample.claim,
                kg_text=kg_text,
                labels=labels,
                label_descriptions=label_descriptions,
            )
            close_stage_bar(classification_bar)
            if self.config.run.debug:
                LOGGER.debug(
                    "[debug][pipeline] sample_id=%s step=kg_only_verification elapsed=%.3fs prediction=%s",
                    sample.sample_id,
                    time.perf_counter() - sample_started,
                    result.prediction,
                )
            result.meta["processing_latency_seconds"] = time.perf_counter() - sample_started
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

        qa_total_steps = max(1, len(selected_questions))
        qa_bar = make_stage_bar("Contrastive QA", total=qa_total_steps)
        qa_pairs = answer_questions(
            client=self.reasoning_client,
            claim=sample.claim,
            reports=sample.reports,
            questions=selected_questions,
            max_context_chars=self.config.pipeline.max_context_chars_for_answers,
            progress_callback=lambda: advance_stage_bar(qa_bar),
        )
        summary = summarize_qa_pairs(
            client=self.reasoning_client,
            claim=sample.claim,
            qa_pairs=qa_pairs,
        )
        close_stage_bar(qa_bar)

        classification_bar = make_stage_bar("Classification")
        prediction = verify_claim(
            client=self.reasoning_client,
            claim=sample.claim,
            context=summary,
            labels=labels,
            label_descriptions=label_descriptions,
        )
        close_stage_bar(classification_bar)

        result.candidate_questions = candidate_questions
        result.selected_questions = selected_questions
        result.qa_pairs = [qa.to_dict() for qa in qa_pairs]
        result.contrastive_summary = summary
        result.prediction = prediction
        if self.config.run.debug:
            LOGGER.debug(
                "[debug][pipeline] sample_id=%s step=full_pipeline_done elapsed=%.3fs selected_questions=%d prediction=%s",
                sample.sample_id,
                time.perf_counter() - sample_started,
                len(selected_questions),
                prediction,
            )
        result.meta["processing_latency_seconds"] = time.perf_counter() - sample_started
        return result
