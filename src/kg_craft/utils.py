from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, List


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def stable_hash(payload: Any) -> str:
    dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def join_reports(reports: list[str], sep: str = "\n\n") -> str:
    blocks = []
    for idx, report in enumerate(reports, start=1):
        blocks.append(f"[Report {idx}]\n{report.strip()}")
    return sep.join(blocks)


def batched(items: List[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def extract_first_json_object(text: str) -> str:
    text = strip_code_fence(text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text.")

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Unbalanced JSON object in text.")


def safe_json_loads(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate = extract_first_json_object(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        if "Invalid \\escape" not in str(error):
            raise
    return json.loads(_escape_invalid_backslashes_in_json_strings(candidate))


def _escape_invalid_backslashes_in_json_strings(text: str) -> str:
    valid_escape_chars = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    repaired: list[str] = []
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
        if not in_string:
            if ch == '"':
                in_string = True
            repaired.append(ch)
            continue

        if ch == "\\":
            next_ch = text[idx + 1] if idx + 1 < len(text) else ""
            if next_ch in valid_escape_chars:
                repaired.append(ch)
                escaped = True
            else:
                repaired.append("\\\\")
                escaped = False
            continue

        if ch == '"' and not escaped:
            in_string = False
        escaped = False
        repaired.append(ch)

    return "".join(repaired)


def deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = normalize_text(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
