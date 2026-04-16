from __future__ import annotations

from typing import Any, Dict, List, Sequence

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support



def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

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
    }



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
