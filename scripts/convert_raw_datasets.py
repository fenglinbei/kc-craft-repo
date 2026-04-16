#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kg_craft.data import write_jsonl
from kg_craft.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LIAR-RAW / RAWFC raw datasets to KG-CRAFT input JSONL format: "
            "{'id','claim','reports','label', ...}."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["liar_raw", "rawfc", "both"],
        help="Which dataset to convert.",
    )
    parser.add_argument(
        "--input-root",
        default="data/raw",
        help="Root folder containing LIAR-RAW and/or RAWFC.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/converted",
        help="Output directory for converted jsonl files.",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        choices=["label", "original_label"],
        help="Label field to use for RAWFC (LIAR-RAW always uses 'label').",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_reports(reports: Iterable[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    for report in reports:
        content = str(report.get("content", "")).strip()
        if content:
            values.append(content)
    return values


def _convert_record(record: Dict[str, Any], sample_id: str, split: str, dataset: str, label_field: str) -> Dict[str, Any]:
    claim = str(record.get("claim", "")).strip()
    reports = _extract_reports(record.get("reports", []))
    row: Dict[str, Any] = {
        "id": sample_id,
        "claim": claim,
        "reports": reports,
        "label": str(record.get(label_field, "")).strip() if record.get(label_field) is not None else None,
        "dataset": dataset,
        "split": split,
    }
    if record.get("event_id") is not None:
        row["event_id"] = str(record["event_id"])
    if record.get("original_label") is not None:
        row["original_label"] = str(record["original_label"])
    if record.get("explain") is not None:
        row["explain"] = str(record["explain"])
    return row


def convert_liar_raw(input_root: Path, output_dir: Path) -> List[Path]:
    dataset_dir = input_root / "LIAR-RAW"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"LIAR-RAW folder not found: {dataset_dir}")

    outputs: List[Path] = []
    for split in ["train", "val", "test"]:
        src_path = dataset_dir / f"{split}.json"
        if not src_path.exists():
            LOGGER.warning("Skip missing LIAR-RAW split file: %s", src_path)
            continue

        raw_records = _read_json(src_path)
        if not isinstance(raw_records, list):
            raise TypeError(f"Expected list in {src_path}, got {type(raw_records)!r}")

        converted_rows: List[Dict[str, Any]] = []
        for idx, record in enumerate(
            tqdm(raw_records, desc=f"LIAR-RAW {split}", unit="sample")
        ):
            event_id = record.get("event_id")
            sample_id = str(event_id if event_id is not None else f"liar_raw-{split}-{idx}")
            converted_rows.append(
                _convert_record(
                    record=record,
                    sample_id=sample_id,
                    split=split,
                    dataset="LIAR-RAW",
                    label_field="label",
                )
            )

        out_path = output_dir / f"liar_raw_{split}.jsonl"
        write_jsonl(out_path, converted_rows)
        outputs.append(out_path)
        LOGGER.info("Converted LIAR-RAW %s: %d samples -> %s", split, len(converted_rows), out_path)
    return outputs


def convert_rawfc(input_root: Path, output_dir: Path, label_field: str) -> List[Path]:
    dataset_dir = input_root / "RAWFC"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"RAWFC folder not found: {dataset_dir}")

    outputs: List[Path] = []
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            LOGGER.warning("Skip missing RAWFC split folder: %s", split_dir)
            continue

        json_files = sorted(split_dir.glob("*.json"))
        converted_rows: List[Dict[str, Any]] = []
        for path in tqdm(json_files, desc=f"RAWFC {split}", unit="file"):
            record = _read_json(path)
            sample_id = str(record.get("event_id") or path.stem)
            converted_rows.append(
                _convert_record(
                    record=record,
                    sample_id=sample_id,
                    split=split,
                    dataset="RAWFC",
                    label_field=label_field,
                )
            )

        out_path = output_dir / f"rawfc_{split}.jsonl"
        write_jsonl(out_path, converted_rows)
        outputs.append(out_path)
        LOGGER.info("Converted RAWFC %s: %d samples -> %s", split, len(converted_rows), out_path)
    return outputs


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced_files: List[Path] = []
    if args.dataset in {"liar_raw", "both"}:
        produced_files.extend(convert_liar_raw(input_root, output_dir))
    if args.dataset in {"rawfc", "both"}:
        produced_files.extend(convert_rawfc(input_root, output_dir, label_field=args.label_field))

    if not produced_files:
        LOGGER.warning("No output files were generated.")
        return

    LOGGER.info("Done. Generated %d files.", len(produced_files))
    for path in produced_files:
        LOGGER.info(" - %s", path)


if __name__ == "__main__":
    main()
