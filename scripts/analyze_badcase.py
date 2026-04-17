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
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kg_craft.data import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cases from a KG-CRAFT predictions JSONL file.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations and summary.")
    parser.add_argument("--label-field", default="label", help="Ground-truth label field name.")
    parser.add_argument("--pred-field", default="prediction", help="Prediction label field name.")
    parser.add_argument("--id-field", default="sample_id", help="Sample id field name.")
    parser.add_argument("--wrong-limit", type=int, default=20, help="Max number of wrong cases to analyze.")
    parser.add_argument("--correct-limit", type=int, default=0, help="Max number of correct cases to analyze.")
    parser.add_argument(
        "--max-kg-triples",
        type=int,
        default=10,
        help="Max number of KG triples to show in the simplified KG view.",
    )
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


def _simplified_nodes_and_triples(sample: dict[str, Any], max_kg_triples: int) -> tuple[list[str], list[dict[str, Any]]]:
    merged_kg = sample.get("merged_kg") or {}
    triples = (merged_kg.get("triples") or [])[: max(1, max_kg_triples)]
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
    return node_names, triples


def _force_layout(node_names: list[str], triples: list[dict[str, Any]], iterations: int = 220) -> dict[str, tuple[float, float]]:
    """Simple deterministic force-directed layout to avoid the single-ring effect."""
    if len(node_names) == 1:
        return {node_names[0]: (0.0, 0.0)}

    rng = np.random.default_rng(42)
    n = len(node_names)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))
    node_index = {name: idx for idx, name in enumerate(node_names)}

    edges: list[tuple[int, int]] = []
    for triple in triples:
        head = str(triple.get("head", "")).strip()
        tail = str(triple.get("tail", "")).strip()
        if head in node_index and tail in node_index and head != tail:
            edges.append((node_index[head], node_index[tail]))

    k = np.sqrt(4.0 / max(1, n))
    temperature = 0.25
    for _ in range(iterations):
        disp = np.zeros_like(pos)

        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, 1.0)
        repulsive_force = (k * k) / np.maximum(dist, 1e-4)
        disp += (delta / dist[:, :, None] * repulsive_force[:, :, None]).sum(axis=1)

        for u, v in edges:
            d = pos[u] - pos[v]
            l = np.linalg.norm(d) + 1e-6
            attractive = (l * l) / k
            f = (d / l) * attractive
            disp[u] -= f
            disp[v] += f

        norms = np.linalg.norm(disp, axis=1)
        norms = np.maximum(norms, 1e-9)
        step = (disp / norms[:, None]) * np.minimum(norms[:, None], temperature)
        pos += step
        temperature *= 0.985

    pos = pos - pos.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(pos))
    if max_abs > 0:
        pos /= max_abs
    return {name: (float(pos[idx, 0]), float(pos[idx, 1])) for idx, name in enumerate(node_names)}


def _plot_case_overview(
    sample: dict[str, Any],
    output_path: Path,
    label_field: str,
    pred_field: str,
    max_kg_triples: int,
) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.3], height_ratios=[0.28, 0.72])
    ax_meta = fig.add_subplot(gs[0, :])
    ax_kg = fig.add_subplot(gs[1, 0])
    ax_qa = fig.add_subplot(gs[1, 1])

    sample_id = str(sample.get("sample_id", sample.get("id", "unknown")))
    gold = str(sample.get(label_field, ""))
    pred = str(sample.get(pred_field, ""))
    is_correct = gold.strip() == pred.strip() and gold.strip() != ""
    icon = "✓" if is_correct else "✗"
    icon_color = "#2E7D32" if is_correct else "#C62828"

    ax_meta.axis("off")
    ax_meta.set_title(f"Case Analysis: {sample_id}", fontsize=14, loc="left")
    meta_text = (
        f"Label: {gold}\n"
        f"Prediction: {pred}\n"
        f"Result: {'Correct' if is_correct else 'Wrong'}"
    )
    ax_meta.text(
        0.01,
        0.65,
        _wrap_text(meta_text, width=55),
        transform=ax_meta.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F8F8", edgecolor="#DDDDDD"),
    )
    ax_meta.text(0.29, 0.62, icon, transform=ax_meta.transAxes, fontsize=44, color=icon_color, fontweight="bold")

    claim = str(sample.get("claim", ""))
    if claim:
        ax_meta.text(
            0.35,
            0.64,
            "Claim:\n" + _wrap_text(claim, width=110),
            transform=ax_meta.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFDF7", edgecolor="#E7D8A4"),
        )

    node_names, triples = _simplified_nodes_and_triples(sample, max_kg_triples=max_kg_triples)
    pos = _force_layout(node_names, triples)

    ax_kg.set_title(f"Simplified KG (top {max(1, max_kg_triples)} triples)")
    ax_kg.axis("off")
    ax_kg.set_aspect("equal")
    edge_pair_counter: Counter[tuple[str, str]] = Counter()
    edge_pair_seen: Counter[tuple[str, str]] = Counter()
    for triple in triples:
        head = str(triple.get("head", "")).strip()
        tail = str(triple.get("tail", "")).strip()
        if head in pos and tail in pos:
            edge_pair_counter[(head, tail)] += 1

    for node, (x, y) in pos.items():
        ax_kg.scatter(x, y, s=740, color="#D7EAFB", edgecolor="#3A7CA5", linewidth=1.2, zorder=3)
        ax_kg.text(x, y, _wrap_text(node, width=16), ha="center", va="center", fontsize=8, zorder=4)

    for triple in triples:
        head = str(triple.get("head", "")).strip()
        tail = str(triple.get("tail", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        if head not in pos or tail not in pos:
            continue
        x1, y1 = pos[head]
        x2, y2 = pos[tail]
        pair = (head, tail)
        offset = edge_pair_seen[pair]
        total = edge_pair_counter[pair]
        edge_pair_seen[pair] += 1
        if total <= 1:
            rad = 0.0
        else:
            center = (total - 1) / 2.0
            rad = (offset - center) * 0.22

        arrow = FancyArrowPatch(
            posA=(x1, y1),
            posB=(x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.1,
            color="#666666",
            alpha=0.85,
            zorder=2,
            shrinkA=18,
            shrinkB=18,
        )
        ax_kg.add_patch(arrow)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if rad != 0:
            nx, ny = y2 - y1, -(x2 - x1)
            norm = np.hypot(nx, ny) + 1e-9
            mx += (nx / norm) * rad * 0.35
            my += (ny / norm) * rad * 0.35
        ax_kg.text(
            mx,
            my,
            _wrap_text(relation, width=18),
            fontsize=7,
            color="#444",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=5,
        )

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    if xs and ys:
        pad = 0.35
        ax_kg.set_xlim(min(xs) - pad, max(xs) + pad)
        ax_kg.set_ylim(min(ys) - pad, max(ys) + pad)

    qa_pairs = sample.get("qa_pairs") or []
    top5 = qa_pairs[:5]
    ax_qa.axis("off")
    ax_qa.set_title("Top-5 QA")
    if not top5:
        ax_qa.text(0.02, 0.92, "No QA pairs found.", fontsize=11, transform=ax_qa.transAxes)
    else:
        y = 0.96
        for idx, qa in enumerate(top5, start=1):
            q_text = _wrap_text(f"Q{idx}: {qa.get('question', '')}", width=68)
            a_text = _wrap_text(f"A{idx}: {qa.get('answer', '')}", width=68)
            ax_qa.text(
                0.02,
                y,
                q_text + "\n" + a_text,
                fontsize=8.8,
                va="top",
                transform=ax_qa.transAxes,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#DDDDDD"),
            )
            y -= 0.19

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_mispred_labels(cases: list[dict[str, Any]], output_path: Path, label_field: str, pred_field: str) -> None:
    wrong_cases = [row for row in cases if str(row.get(label_field)).strip() != str(row.get(pred_field)).strip()]
    pair_counter = Counter(f"{row.get(label_field)} -> {row.get(pred_field)}" for row in wrong_cases)
    labels = list(pair_counter.keys())
    values = [pair_counter[label] for label in labels]

    fig_w = max(8, 0.9 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    if not labels:
        ax.text(0.5, 0.5, "No wrong cases found.", ha="center", va="center", transform=ax.transAxes)
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

    wrong_cases = []
    correct_cases = []
    for row in rows:
        gold = row.get(args.label_field)
        pred = row.get(args.pred_field)
        if gold is None or pred is None:
            continue
        sid = str(row.get(args.id_field, row.get("id", "unknown")))
        if selected_ids and sid not in selected_ids:
            continue
        row = dict(row)
        row[args.id_field] = sid
        if str(gold).strip() == str(pred).strip():
            if len(correct_cases) < max(0, args.correct_limit):
                correct_cases.append(row)
        else:
            if len(wrong_cases) < max(0, args.wrong_limit):
                wrong_cases.append(row)
        if len(correct_cases) >= max(0, args.correct_limit) and len(wrong_cases) >= max(0, args.wrong_limit):
            break

    cases = wrong_cases + correct_cases

    sample_summaries: list[dict[str, Any]] = []
    for sample in cases:
        sample_id = str(sample.get(args.id_field))
        sample_slug = _safe_name(sample_id)
        sample_dir = output_dir / sample_slug
        sample_dir.mkdir(parents=True, exist_ok=True)

        case_overview_path = sample_dir / "case_overview.png"
        _plot_case_overview(
            sample,
            case_overview_path,
            label_field=args.label_field,
            pred_field=args.pred_field,
            max_kg_triples=args.max_kg_triples,
        )
        sample_correct = str(sample.get(args.label_field)).strip() == str(sample.get(args.pred_field)).strip()

        sample_summaries.append(
            {
                "sample_id": sample_id,
                "label": sample.get(args.label_field),
                "prediction": sample.get(args.pred_field),
                "is_correct": sample_correct,
                "case_overview": str(case_overview_path.relative_to(output_dir)),
            }
        )

    mispred_path = output_dir / "mispred_labels.png"
    _plot_mispred_labels(cases, mispred_path, label_field=args.label_field, pred_field=args.pred_field)

    summary = {
        "predictions": str(Path(args.predictions).resolve()),
        "total_rows": len(rows),
        "num_wrong_cases_analyzed": len(wrong_cases),
        "num_correct_cases_analyzed": len(correct_cases),
        "num_cases_analyzed": len(cases),
        "sample_summaries": sample_summaries,
        "mispred_labels_figure": str(mispred_path.name),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
