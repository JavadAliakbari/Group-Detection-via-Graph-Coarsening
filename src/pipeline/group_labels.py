r"""Group-level label prediction by majority vote over member nodes.

The label head predicts per *node*; the unit of interest is the *group*.  For
every ground-truth group that the coarsening actually detected, the group's
predicted class is the majority vote of its members' predicted classes.

Ties are broken deterministically: among the tied classes take the one with the
highest mean softmax probability over the group's nodes, and if that still ties,
the lowest class index.

Every metric here is **conditional on detection** and is therefore reported with
its coverage,

    coverage = detected groups / total groups,

because a detector that only finds the easy groups would otherwise look strong.
The names carry the condition explicitly (``detected_group_label_*``).

Overlapping groups are evaluated independently, each over its own member nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["GroupLabelReport", "evaluate_group_labels"]


@dataclass
class GroupLabelReport:
    """Per-group rows plus the aggregate metrics, all conditional on detection."""

    rows: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.metrics.get("detected_group_label_coverage", 0.0))


def _majority_vote(node_classes: np.ndarray, probabilities: np.ndarray, n_classes: int):
    """``(label, votes, mean_probabilities)`` with deterministic tie-breaking."""

    votes = np.bincount(node_classes, minlength=n_classes)
    mean_probabilities = probabilities.mean(axis=0)
    tied = np.flatnonzero(votes == votes.max())
    if tied.size == 1:
        return int(tied[0]), votes, mean_probabilities
    best = tied[
        np.flatnonzero(mean_probabilities[tied] == mean_probabilities[tied].max())
    ]
    return int(best.min()), votes, mean_probabilities


def evaluate_group_labels(
    graph,
    probabilities: torch.Tensor,
    cut,
    *,
    evaluation_groups: "list | None" = None,
) -> GroupLabelReport:
    """Majority-vote group labels at one coarsening cut.

    ``cut`` is a :class:`~src.pipeline.coarsening.CutEvaluation`; only the groups
    it marks detected enter the metrics, and its ``mode`` is recorded on every
    row so oracle and stopping-rule evaluations never get mixed up.
    """

    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
    )

    groups = (
        list(evaluation_groups) if evaluation_groups is not None else list(graph.groups)
    )
    detected = {row["group_id"]: row for row in cut.per_group}
    probabilities = probabilities.detach().cpu().numpy()
    n_classes = int(probabilities.shape[1])
    node_classes = probabilities.argmax(axis=1)

    rows, truth, predicted = [], [], []
    for group in groups:
        info = detected.get(group.group_id)
        if info is None:
            continue
        nodes = group.nodes.cpu().numpy()
        label, votes, mean_probabilities = _majority_vote(
            node_classes[nodes], probabilities[nodes], n_classes
        )
        row = {
            "graph_id": graph.graph_id,
            "group_id": group.group_id,
            "size": int(group.num_nodes),
            "split": group.split,
            "true_label": int(group.label),
            "predicted_label": label,
            "votes": votes.tolist(),
            "mean_probabilities": [float(v) for v in mean_probabilities],
            "detected": bool(info["detected"]),
            "evaluation_mode": cut.mode,
            "correct": bool(info["detected"]) and label == int(group.label),
        }
        rows.append(row)
        if info["detected"]:
            truth.append(int(group.label))
            predicted.append(label)

    total = len(rows)
    coverage = len(truth) / total if total else 0.0
    metrics = {
        "detected_group_label_coverage": coverage,
        "n_detected_groups": len(truth),
        "n_groups": total,
        "evaluation_mode": cut.mode,
    }
    if not truth:
        return GroupLabelReport(rows=rows, metrics=metrics)

    truth_a, predicted_a = np.asarray(truth), np.asarray(predicted)
    classes = sorted(set(truth_a.tolist()) | set(predicted_a.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth_a, predicted_a, labels=classes, zero_division=0
    )
    macro = precision_recall_fscore_support(
        truth_a, predicted_a, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        truth_a, predicted_a, average="weighted", zero_division=0
    )
    metrics.update(
        detected_group_label_accuracy=float((truth_a == predicted_a).mean()),
        detected_group_label_precision_macro=float(macro[0]),
        detected_group_label_recall_macro=float(macro[1]),
        detected_group_label_f1_macro=float(macro[2]),
        detected_group_label_precision_weighted=float(weighted[0]),
        detected_group_label_recall_weighted=float(weighted[1]),
        detected_group_label_f1_weighted=float(weighted[2]),
        per_class={
            int(c): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        confusion_matrix=confusion_matrix(
            truth_a, predicted_a, labels=classes
        ).tolist(),
        confusion_matrix_labels=[int(c) for c in classes],
    )
    if n_classes == 2:  # binary: the positive class is reported alongside the macro
        positive = precision_recall_fscore_support(
            truth_a, predicted_a, labels=[1], average="binary", zero_division=0
        )
        metrics.update(
            detected_group_label_precision_positive=float(positive[0]),
            detected_group_label_recall_positive=float(positive[1]),
            detected_group_label_f1_positive=float(positive[2]),
        )
    return GroupLabelReport(rows=rows, metrics=metrics)
