from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schemas import PipelineResult, Sample


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def rows_to_samples(
    rows: List[Dict[str, Any]],
    id_field: str,
    claim_field: str,
    reports_field: str,
    label_field: str,
) -> List[Sample]:
    samples: List[Sample] = []
    for idx, row in enumerate(rows):
        reports = row.get(reports_field, [])
        if not isinstance(reports, list):
            raise TypeError(f"reports field must be a list[str], got: {type(reports)!r}")

        sample_id = str(row.get(id_field, idx))
        claim = str(row[claim_field])
        label = row.get(label_field)
        samples.append(
            Sample(
                sample_id=sample_id,
                claim=claim,
                reports=[str(x) for x in reports],
                label=str(label) if label is not None else None,
                meta={
                    k: v for k, v in row.items()
                    if k not in {id_field, claim_field, reports_field, label_field}
                },
            )
        )
    return samples



def save_results(path: str | Path, results: List[PipelineResult]) -> None:
    write_jsonl(path, [r.to_dict() for r in results])
