#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kg_craft.data import load_jsonl
from kg_craft.evaluation import compute_metrics
from kg_craft.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KG-CRAFT predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--pred-field", default="prediction")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    rows = load_jsonl(args.predictions)
    y_true = []
    y_pred = []
    for row in rows:
        gold = row.get(args.label_field)
        pred = row.get(args.pred_field)
        if gold is None or pred is None:
            continue
        y_true.append(str(gold))
        y_pred.append(str(pred))

    if not y_true:
        raise ValueError("No rows with both gold label and prediction found.")

    metrics = compute_metrics(y_true, y_pred)
    logger.info("Evaluated %d rows.", len(y_true))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
