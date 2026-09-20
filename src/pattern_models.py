"""Class-based pattern domain model for GangPrediction evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
import torch

# from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

from src.utils.utils import *


@dataclass
class PatternLevelSnapshot:
    """Per-level lineage details for pattern nodes."""

    super_nodes: Optional[torch.Tensor] = field(default_factory=torch.tensor)
    pseudo_labels: Optional[torch.Tensor] = field(default_factory=torch.tensor)


@dataclass
class PatternMetrics:
    """Mutable metrics store for each pattern instance."""

    values: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> None:
        self.values.update(kwargs)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.values)


@dataclass
class Pattern:
    """Base pattern model storing topology, lineage, and evaluation metrics."""

    pattern_id: Any
    nodes: torch.Tensor
    pattern_type: str
    label: str = "unknown"
    level_data: List[PatternLevelSnapshot] = field(default_factory=list)
    metrics: PatternMetrics = field(default_factory=PatternMetrics)

    @property
    def id(self) -> Any:
        return self.pattern_id

    @property
    def node_indices(self) -> torch.Tensor:
        return self.nodes

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def compute_detection_metrics(
        self,
        node_to_supernode: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Compute detection-related metrics for this pattern and store them."""

        target_label = 1 if self.label == "alert" else 0
        probs = self.level_data[-1].pseudo_labels if self.level_data else None
        predicted_labels = probs.max(1)[1]
        target_mask = predicted_labels == target_label
        n_target = int(target_mask.sum().item())
        target_ratio = n_target / self.num_nodes

        super_nodes = self.level_data[-1].super_nodes
        super_ids, counts = torch.unique(super_nodes, return_counts=True)
        self.super_node = super_ids[counts.argmax().item()]
        max_count = counts.max().item()

        recall = max_count / self.num_nodes
        super_node_size = (node_to_supernode == self.super_node).sum().item()
        precision = max_count / super_node_size if super_node_size > 0 else 0.0

        filtered_supernode_mask = (super_nodes == self.super_node) & target_mask
        filtered_supernode_count = filtered_supernode_mask.sum().item()

        recall_filtered = (
            filtered_supernode_count / self.num_nodes if self.num_nodes > 0 else 0.0
        )
        precision_filtered = (
            filtered_supernode_count / super_node_size if super_node_size > 0 else 0.0
        )

        f1_scores = (2.0 * recall * precision) / np.maximum(recall + precision, 1e-12)

        # condition1_met = target_ratio > majority_threshold
        # condition2_met = recall > coarsening_threshold
        # condition3_met = precision > coarsening_threshold

        result = {
            # "n_nodes": self.num_nodes,
            # "n_target": n_target,
            "target_ratio": target_ratio,
            "coarsening_ratio": recall,
            # "condition1_met": condition1_met,
            # "condition2_met": condition2_met,
            # "condition3_met": condition3_met,
            # "detected1": bool(condition1_met and condition2_met),
            # "detected2": bool(condition1_met and condition3_met),
            # "detected": bool(condition1_met and condition2_met and condition3_met),
            "recall": recall,
            "precision": precision,
            "f1": f1_scores,
            "recall_filtered": recall_filtered,
            "precision_filtered": precision_filtered,
        }
        self.metrics.update(**result)
        return result

    def capture_level(
        self,
        node_to_supernode: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Store node-level supernode and pseudo-label lineage for one level."""
        if node_to_supernode is not None:
            super_nodes = node_to_supernode[self.nodes]

        if pseudo_labels is not None:
            pseudo = pseudo_labels[self.nodes]

        self.level_data.append(
            PatternLevelSnapshot(
                super_nodes=super_nodes,
                pseudo_labels=pseudo,
            )
        )

        return self.compute_detection_metrics(node_to_supernode=node_to_supernode)

    @staticmethod
    def average_metrics(
        patterns: Sequence["Pattern"],
        metric_keys: Optional[Sequence[str]] = None,
        majority_threshold: float = 0.5,
        coarsening_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute mean value for all numeric metrics across patterns."""
        if not patterns:
            return {}

        metric_names: set[str] = set()
        if metric_keys is not None:
            metric_names.update(metric_keys)
        else:
            for pattern in patterns:
                metric_names.update(pattern.metrics.values.keys())

        total = len(patterns)
        averages: Dict[str, float] = {}
        for metric_name in metric_names:
            values: List[float] = []
            num_nodes = []
            for pattern in patterns:
                value = pattern.metrics.values.get(metric_name)
                if isinstance(value, (int, float, np.floating, np.integer)):
                    values.append(float(value))

                num_nodes.append(pattern.num_nodes)
            if values:
                averages[metric_name] = float(
                    np.sum(np.array(values) * np.array(num_nodes)) / np.sum(num_nodes)
                )

        recal_detected = np.array(
            [
                pattern.metrics.values.get("recall", 0.0) > majority_threshold
                for pattern in patterns
            ],
            dtype=bool,
        )
        precission_detected = np.array(
            [
                pattern.metrics.values.get("precision", 0.0) > majority_threshold
                for pattern in patterns
            ],
            dtype=bool,
        )
        detected = (recal_detected & precission_detected).sum()
        averages["detection_rate"] = detected / total if total > 0 else 0.0
        averages["detected"] = detected
        averages["total"] = total

        recal_detected_filtered = np.array(
            [
                pattern.metrics.values.get("recall_filtered", 0.0)
                > coarsening_threshold
                for pattern in patterns
            ],
            dtype=bool,
        )
        precission_detected_filtered = np.array(
            [
                pattern.metrics.values.get("precision_filtered", 0.0)
                > coarsening_threshold
                for pattern in patterns
            ],
            dtype=bool,
        )
        detected_filtered = (
            recal_detected_filtered & precission_detected_filtered
        ).sum()
        averages["detection_rate_filtered"] = (
            detected_filtered / total if total > 0 else 0.0
        )

        # th_list = np.arange(0.0, 1.01, 0.05)
        # roc_data = []
        # for th in th_list:
        #     # for pattern in patterns:
        #     recal_ratio = (
        #         np.sum(
        #             [
        #                 pattern.metrics.values.get("recall", 0.0) > th
        #                 for pattern in patterns
        #             ],
        #             # dtype=bool,
        #         )
        #         / total
        #     )  # if total > 0 else 0.0
        #     precission_ratio = (
        #         np.sum(
        #             [
        #                 pattern.metrics.values.get("precision", 0.0) > th
        #                 for pattern in patterns
        #             ],
        #             # dtype=bool,
        #         )
        #         / total
        #     )  # if total > 0 else 0.0
        #     # detected = (recal_detected & precission_detected).sum()
        #     roc_data.append((recal_ratio, precission_ratio))

        # auc_data = auc(
        #     [point[0] for point in roc_data], [point[1] for point in roc_data]
        # )
        # averages["roc_auc"] = auc_data

        return averages


class UnknownPattern(Pattern):
    pass


class FanOutPattern(Pattern):
    pass


class FanInPattern(Pattern):
    pass


class CyclePattern(Pattern):
    pass


class BipartitePattern(Pattern):
    pass


class StackPattern(Pattern):
    pass


class RandomPattern(Pattern):
    pass


class ScatterGatherPattern(Pattern):
    pass


class GatherScatterPattern(Pattern):
    pass


class SinglePattern(Pattern):
    pass


class ForwardPattern(Pattern):
    pass


class MutualPattern(Pattern):
    pass


class PeriodicalPattern(Pattern):
    pass


PATTERN_CLASS_REGISTRY: Dict[str, Type[Pattern]] = {
    "fan_out": FanOutPattern,
    "fan_in": FanInPattern,
    "cycle": CyclePattern,
    "bipartite": BipartitePattern,
    "stack": StackPattern,
    "random": RandomPattern,
    "scatter_gather": ScatterGatherPattern,
    "gather_scatter": GatherScatterPattern,
    "single": SinglePattern,
    "forward": ForwardPattern,
    "mutual": MutualPattern,
    "periodical": PeriodicalPattern,
}


def create_pattern(
    pattern_id: Any, nodes: Iterable[int], pattern_type: str, label: str = "unknown"
) -> Pattern:
    """Factory method for creating concrete pattern subclasses by type."""
    klass = PATTERN_CLASS_REGISTRY.get(pattern_type, UnknownPattern)
    return klass(
        pattern_id=pattern_id, nodes=list(nodes), pattern_type=pattern_type, label=label
    )