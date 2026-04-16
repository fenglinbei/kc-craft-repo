#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kg_craft.config import load_config
from kg_craft.data import load_jsonl, rows_to_samples, save_results
from kg_craft.pipeline import KGCRAFTPipeline
from kg_craft.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG-CRAFT reproduction pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--input", default=None, help="Override input JSONL path.")
    parser.add_argument("--output", default=None, help="Override output JSONL path.")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["full", "naive_llm", "kg_only", "llm_questions"],
        help="Pipeline mode override.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logs for each pipeline/API step.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads used to process samples in parallel.",
    )
    parser.add_argument(
        "--no-sample-progress",
        action="store_true",
        help="Disable per-sample stage progress bars (KG/QA/分类).",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    log_level = "DEBUG" if args.debug else args.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    if args.num_workers is not None:
        if args.num_workers < 1:
            raise ValueError("--num-workers must be >= 1")
        config.run.num_workers = args.num_workers
    if args.no_sample_progress:
        config.run.show_sample_stage_progress = False
    if args.debug:
        config.run.debug = True
        if args.log_level != "DEBUG":
            logger.warning("--debug enabled; log level forced to DEBUG for complete traces.")

    input_path = args.input or config.data.input_path
    output_path = args.output or config.data.output_path
    mode = args.mode or config.run.mode
    config.data.output_path = output_path

    if not input_path:
        raise ValueError("Input path is required via --input or data.input_path in config.")
    if not output_path:
        raise ValueError("Output path is required via --output or data.output_path in config.")

    rows = load_jsonl(input_path)
    samples = rows_to_samples(
        rows=rows,
        id_field=config.data.id_field,
        claim_field=config.data.claim_field,
        reports_field=config.data.reports_field,
        label_field=config.data.label_field,
    )

    pipeline = KGCRAFTPipeline(config)
    results = pipeline.run(samples, mode=mode)
    save_results(output_path, results)
    logger.info("Saved %d results to %s", len(results), output_path)
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
