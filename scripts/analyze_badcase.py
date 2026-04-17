#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kg_craft.data import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bad cases from a KG-CRAFT predictions JSONL file.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations and summary.")
    parser.add_argument("--label-field", default="label", help="Ground-truth label field name.")
    parser.add_argument("--pred-field", default="prediction", help="Prediction label field name.")
    parser.add_argument("--id-field", default="sample_id", help="Sample id field name.")
    parser.add_argument("--limit", type=int, default=20, help="Max number of bad cases to analyze.")
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated sample ids to analyze; if provided, only these ids are considered.",
    )
    return parser.parse_args()


def _wrap_text(text: str, width: int = 52) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def _safe_name(sample_id: str) -> str:
    keep = []
    for ch in str(sample_id):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:80]


def _plot_kg_graph(sample: dict[str, Any], output_path: Path) -> None:
    merged_kg = sample.get("merged_kg") or {}
    triples = merged_kg.get("triples") or []
    entities = merged_kg.get("entities") or []

    node_names: list[str] = []
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        if name:
            node_names.append(name)
    for triple in triples:
        for key in ("head", "tail"):
            name = str(triple.get(key, "")).strip()
            if name and name not in node_names:
                node_names.append(name)

    if not node_names:
        node_names = ["(empty KG)"]

    n = len(node_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = max(1.0, 0.35 * n)
    pos = {node: (radius * np.cos(a), radius * np.sin(a)) for node, a in zip(node_names, angles)}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Merged KG Graph")
    ax.axis("off")

    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=700, color="#D7EAFB", edgecolor="#3A7CA5", linewidth=1.2, zorder=3)
        ax.text(x, y, _wrap_text(node, width=16), ha="center", va="center", fontsize=8, zorder=4)

    for triple in triples:
        head = str(triple.get("head", "")).strip()
        tail = str(triple.get("tail", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        if head not in pos or tail not in pos:
            continue
        x1, y1 = pos[head]
        x2, y2 = pos[tail]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.2, alpha=0.8),
            zorder=2,
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, _wrap_text(relation, width=18), fontsize=7, color="#444", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top5_qa(sample: dict[str, Any], output_path: Path) -> None:
    qa_pairs = sample.get("qa_pairs") or []
    top5 = qa_pairs[:5]

    fig_h = max(4.5, 1.8 * max(1, len(top5)))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title("Top-5 QA", fontsize=13)

    if not top5:
        ax.text(0.02, 0.9, "No QA pairs found.", fontsize=12, transform=ax.transAxes)
    else:
        y = 0.95
        for idx, qa in enumerate(top5, start=1):
            q_text = _wrap_text(f"Q{idx}: {qa.get('question', '')}", width=95)
            a_text = _wrap_text(f"A{idx}: {qa.get('answer', '')}", width=95)
            ax.text(
                0.02,
                y,
                q_text + "\n" + a_text,
                fontsize=9,
                va="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7", edgecolor="#DDDDDD"),
            )
            y -= 0.18

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_mispred_labels(badcases: list[dict[str, Any]], output_path: Path, label_field: str, pred_field: str) -> None:
    pair_counter = Counter(f"{row.get(label_field)} -> {row.get(pred_field)}" for row in badcases)
    labels = list(pair_counter.keys())
    values = [pair_counter[label] for label in labels]

    fig_w = max(8, 0.9 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    if not labels:
        ax.text(0.5, 0.5, "No bad cases found.", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        bars = ax.bar(range(len(labels)), values, color="#E45756")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Misclassified Label Distribution")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.predictions)
    selected_ids = {x.strip() for x in args.sample_ids.split(",") if x.strip()}

    badcases = []
    for row in rows:
        gold = row.get(args.label_field)
        pred = row.get(args.pred_field)
        if gold is None or pred is None:
            continue
        if str(gold).strip() == str(pred).strip():
            continue
        sid = str(row.get(args.id_field, row.get("id", "unknown")))
        if selected_ids and sid not in selected_ids:
            continue
        row = dict(row)
        row[args.id_field] = sid
        badcases.append(row)
        if len(badcases) >= max(1, args.limit):
            break

    sample_summaries: list[dict[str, Any]] = []
    for sample in badcases:
        sample_id = str(sample.get(args.id_field))
        sample_slug = _safe_name(sample_id)
        sample_dir = output_dir / sample_slug
        sample_dir.mkdir(parents=True, exist_ok=True)

        kg_path = sample_dir / "kg_graph.png"
        qa_path = sample_dir / "top5_qa.png"
        _plot_kg_graph(sample, kg_path)
        _plot_top5_qa(sample, qa_path)

        sample_summaries.append(
            {
                "sample_id": sample_id,
                "label": sample.get(args.label_field),
                "prediction": sample.get(args.pred_field),
                "kg_graph": str(kg_path.relative_to(output_dir)),
                "top5_qa": str(qa_path.relative_to(output_dir)),
            }
        )

    mispred_path = output_dir / "mispred_labels.png"
    _plot_mispred_labels(badcases, mispred_path, label_field=args.label_field, pred_field=args.pred_field)

    summary = {
        "predictions": str(Path(args.predictions).resolve()),
        "total_rows": len(rows),
        "num_badcases_analyzed": len(badcases),
        "sample_summaries": sample_summaries,
        "mispred_labels_figure": str(mispred_path.name),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
