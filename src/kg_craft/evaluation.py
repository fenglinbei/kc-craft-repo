from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support



def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = sorted(set(y_true) | set(y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_metrics_figure(metrics: Dict[str, Any], output_path: str | Path, title: str = "Final Metrics") -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scalar_metrics = [
        ("accuracy", float(metrics.get("accuracy", 0.0))),
        ("macro_precision", float(metrics.get("macro_precision", 0.0))),
        ("macro_recall", float(metrics.get("macro_recall", 0.0))),
        ("macro_f1", float(metrics.get("macro_f1", 0.0))),
        ("weighted_precision", float(metrics.get("weighted_precision", 0.0))),
        ("weighted_recall", float(metrics.get("weighted_recall", 0.0))),
        ("weighted_f1", float(metrics.get("weighted_f1", 0.0))),
    ]

    labels = list(metrics.get("labels", []))
    matrix = np.array(metrics.get("confusion_matrix", []), dtype=float)
    if matrix.size == 0:
        matrix = np.zeros((1, 1), dtype=float)
        labels = ["N/A"]

    fig, (ax_metrics, ax_cm) = plt.subplots(
        1,
        2,
        figsize=(max(14, 1.6 * max(1, len(labels)) + 8), 6),
        constrained_layout=True,
    )

    x_labels = [name.replace("_", "\n") for name, _ in scalar_metrics]
    values = [value for _, value in scalar_metrics]
    bars = ax_metrics.bar(range(len(values)), values, color="#4C78A8")
    ax_metrics.set_xticks(range(len(values)))
    ax_metrics.set_xticklabels(x_labels, fontsize=9)
    ax_metrics.set_ylim(0.0, 1.05)
    ax_metrics.set_title("Scalar Metrics")
    ax_metrics.set_ylabel("Score")
    ax_metrics.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax_metrics.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    im = ax_cm.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_xticks(range(len(labels)))
    ax_cm.set_xticklabels(labels, rotation=45, ha="right")
    ax_cm.set_yticks(range(len(labels)))
    ax_cm.set_yticklabels(labels)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j]) if float(matrix[i, j]).is_integer() else matrix[i, j]
            ax_cm.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path



def class_distance_weight(pred_label: str, gold_label: str, ordered_labels: Sequence[str]) -> float:
    if pred_label not in ordered_labels:
        raise ValueError(f"pred_label {pred_label!r} not in ordered_labels")
    if gold_label not in ordered_labels:
        raise ValueError(f"gold_label {gold_label!r} not in ordered_labels")
    ymin = 1
    ymax = len(ordered_labels)
    pred_value = ordered_labels.index(pred_label) + 1
    gold_value = ordered_labels.index(gold_label) + 1
    if ymax == ymin:
        return 1.0
    return 1.0 - ((pred_value - gold_value) ** 2) / ((ymax - ymin) ** 2)



def apply_class_distance_weight(base_score: float, pred_label: str, gold_label: str, ordered_labels: Sequence[str]) -> float:
    return class_distance_weight(pred_label, gold_label, ordered_labels) * base_score
