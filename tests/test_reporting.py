"""Group-label majority vote and the reporting artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.pipeline.coarsening import CutEvaluation
from src.pipeline.data import Graph, Group
from src.pipeline.group_labels import evaluate_group_labels


def _graph(groups) -> Graph:
    n = 40
    src = list(range(n - 1))
    edge_index = torch.tensor(
        [src + list(range(1, n)), list(range(1, n)) + src], dtype=torch.long
    )
    return Graph(
        graph_id="g",
        num_nodes=n,
        edge_index=edge_index,
        features=torch.zeros(n, 3),
        node_labels=torch.zeros(n, dtype=torch.long),
        groups=groups,
    )


def _cut(graph, detected_ids, mode="stopping_rule") -> CutEvaluation:
    return CutEvaluation(
        mode=mode,
        rule="f1",
        merges=0,
        n_coarse=graph.num_nodes,
        reduction=0.0,
        retained=1.0,
        epsilon=0.0,
        node_to_supernode=torch.arange(graph.num_nodes),
        metrics={},
        row={},
        per_group=[
            {
                "group_id": g.group_id,
                "split": g.split,
                "group_type": g.group_type,
                "size": g.num_nodes,
                "recall": 1.0,
                "precision": 1.0,
                "f1": 1.0,
                "detected": g.group_id in detected_ids,
                "supernode_size": g.num_nodes,
            }
            for g in graph.groups
        ],
    )


def _probabilities(n, assignments, n_classes=3):
    probabilities = torch.full((n, n_classes), 0.01, dtype=torch.float64)
    for node, cls in assignments.items():
        probabilities[node] = 0.01
        probabilities[node, cls] = 0.9
    return probabilities / probabilities.sum(1, keepdim=True)


def test_majority_vote_picks_the_modal_class():
    groups = [Group("a", torch.tensor([0, 1, 2, 3]), 2, "train")]
    graph = _graph(groups)
    # three nodes vote class 2, one votes class 1
    probabilities = _probabilities(40, {0: 2, 1: 2, 2: 2, 3: 1})
    report = evaluate_group_labels(graph, probabilities, _cut(graph, {"a"}))
    row = report.rows[0]
    assert row["predicted_label"] == 2
    assert row["votes"][2] == 3 and row["votes"][1] == 1
    assert row["correct"] is True
    assert report.metrics["detected_group_label_accuracy"] == 1.0


def test_ties_break_on_mean_probability_then_lowest_index():
    groups = [Group("a", torch.tensor([0, 1]), 1, "train")]
    graph = _graph(groups)
    probabilities = torch.zeros(40, 3, dtype=torch.float64)
    probabilities[:, 0] = 1.0
    probabilities[0] = torch.tensor([0.1, 0.6, 0.3])  # votes class 1
    probabilities[1] = torch.tensor([0.1, 0.2, 0.7])  # votes class 2
    report = evaluate_group_labels(graph, probabilities, _cut(graph, {"a"}))
    # 1-1 tie; class 2 has the higher mean probability (0.5 vs 0.4)
    assert report.rows[0]["predicted_label"] == 2

    exact = torch.zeros(40, 3, dtype=torch.float64)
    exact[:, 0] = 1.0
    exact[0] = torch.tensor([0.0, 0.5, 0.5])
    exact[1] = torch.tensor([0.0, 0.5, 0.5])
    # both nodes tie internally; argmax takes class 1 for both -> no tie in votes
    assert (
        evaluate_group_labels(graph, exact, _cut(graph, {"a"})).rows[0][
            "predicted_label"
        ]
        == 1
    )


def test_metrics_are_conditional_on_detection_and_carry_coverage():
    groups = [
        Group("a", torch.tensor([0, 1]), 1, "train"),
        Group("b", torch.tensor([10, 11]), 2, "train"),
        Group("c", torch.tensor([20, 21]), 1, "test"),
        Group("d", torch.tensor([30, 31]), 2, "test"),
    ]
    graph = _graph(groups)
    probabilities = _probabilities(40, {0: 1, 1: 1, 10: 2, 11: 2, 20: 2, 21: 2})
    report = evaluate_group_labels(graph, probabilities, _cut(graph, {"a", "b", "c"}))
    metrics = report.metrics
    assert metrics["n_groups"] == 4
    assert metrics["n_detected_groups"] == 3
    assert metrics["detected_group_label_coverage"] == pytest.approx(0.75)
    # a and b correct, c predicted 2 but truly 1
    assert metrics["detected_group_label_accuracy"] == pytest.approx(2 / 3)
    assert "detected_group_label_f1_macro" in metrics
    assert len(metrics["confusion_matrix"]) == len(metrics["confusion_matrix_labels"])
    # the undetected group still gets a row, flagged as missed
    missed = [r for r in report.rows if not r["detected"]]
    assert len(missed) == 1 and missed[0]["correct"] is False


def test_overlapping_groups_vote_independently():
    groups = [
        Group("a", torch.tensor([0, 1, 2]), 1, "train"),
        Group("b", torch.tensor([2, 3, 4]), 2, "train"),
    ]
    graph = _graph(groups)
    probabilities = _probabilities(40, {0: 1, 1: 1, 2: 1, 3: 2, 4: 2})
    report = evaluate_group_labels(graph, probabilities, _cut(graph, {"a", "b"}))
    by_id = {row["group_id"]: row for row in report.rows}
    assert by_id["a"]["predicted_label"] == 1
    assert by_id["b"]["predicted_label"] == 2  # node 2 votes for both, majority wins
    assert by_id["a"]["correct"] and by_id["b"]["correct"]


def test_binary_task_also_reports_the_positive_class():
    groups = [
        Group("a", torch.tensor([0, 1]), 1, "train"),
        Group("b", torch.tensor([10, 11]), 0, "train"),
    ]
    graph = _graph(groups)
    probabilities = _probabilities(40, {0: 1, 1: 1, 10: 0, 11: 0}, n_classes=2)
    metrics = evaluate_group_labels(
        graph, probabilities, _cut(graph, {"a", "b"})
    ).metrics
    assert "detected_group_label_f1_positive" in metrics
    assert metrics["detected_group_label_f1_positive"] == pytest.approx(1.0)


def test_evaluation_mode_is_recorded_on_every_row():
    groups = [Group("a", torch.tensor([0, 1]), 1, "train")]
    graph = _graph(groups)
    probabilities = _probabilities(40, {0: 1, 1: 1})
    for mode in ("stopping_rule", "oracle"):
        report = evaluate_group_labels(graph, probabilities, _cut(graph, {"a"}, mode))
        assert all(row["evaluation_mode"] == mode for row in report.rows)
        assert report.metrics["evaluation_mode"] == mode


# --------------------------------------------------------------------------- #
# reporting artifacts
# --------------------------------------------------------------------------- #
def test_reporter_writes_the_documented_layout(bundle, tmp_path):
    from src.pipeline.coarsening import Coarsening, CoarseningConfig
    from src.pipeline.learning import LearningConfig
    from src.pipeline.learning.polynomial import PolynomialFilterLearning
    from src.pipeline.logging_visualization import (
        GraphEvaluation,
        LoggingVisualization,
        LoggingVisualizationConfig,
    )
    from src.pipeline.objective import ObjectiveConfig

    learner = PolynomialFilterLearning(
        LearningConfig(
            architecture="polynomial",
            training_mode="closed_form",
            objective=ObjectiveConfig(gamma=2.0),
            degree=4,
            tau=0.5,
            seed=0,
        )
    )
    result = learner.run(bundle.train_graphs)
    graph = bundle.train_graphs[0]
    coarsening = Coarsening(
        CoarseningConfig(
            method="ward_tree", tau=0.5, cut_rule="f1", exact_epsilon_budget=8
        )
    ).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )

    reporter = LoggingVisualization(
        LoggingVisualizationConfig(
            output_dir=tmp_path,
            make_plots=True,
            per_group_top_k=5,
            halo_groups_per_graph=1,
            halo_max_nodes=40,
        )
    )
    evaluation = GraphEvaluation(
        graph=graph,
        coarsening=coarsening,
        level=learner.level_of(graph).values.detach(),
        scope="train",
    )
    reporter.log_coarsening(evaluation)
    summaries = reporter.log_summaries([evaluation], labels_enabled=False)

    for mode in ("stopping_rule", "oracle"):
        assert (tmp_path / mode / "tables" / "summary.csv").exists()
        assert (tmp_path / mode / "per_group" / f"{graph.graph_id}.png").exists()
        assert (tmp_path / mode / "edge_diagnostics" / f"{graph.graph_id}.png").exists()
        assert (tmp_path / mode / "edge_diagnostics" / f"{graph.graph_id}.csv").exists()
        assert list((tmp_path / mode / "halo").glob("*.png"))
    assert (tmp_path / "trajectories" / f"{graph.graph_id}.png").exists()
    assert (tmp_path / "trajectories" / f"{graph.graph_id}.csv").exists()
    assert (tmp_path / "comparison.csv").exists()
    payload = json.loads((tmp_path / "oracle" / "tables" / "summary.json").read_text())
    assert payload["rows"][0]["graph"] == graph.graph_id
    for column in (
        "pr_auc",
        "epsilon",
        "reduction",
        "n_coarse",
        "recall",
        "precision",
        "f1",
        "detection_rate",
        "levels",
        "groups",
    ):
        assert column in payload["rows"][0]
    assert summaries["oracle"]["overall"]["train"]["graphs"] == 1


def test_training_figure_needs_a_gradient_history(bundle, tmp_path):
    from src.pipeline.learning import LearningConfig
    from src.pipeline.learning.polynomial import PolynomialFilterLearning
    from src.pipeline.logging_visualization import (
        LoggingVisualization,
        LoggingVisualizationConfig,
    )
    from src.pipeline.objective import ObjectiveConfig

    learner = PolynomialFilterLearning(
        LearningConfig(
            architecture="polynomial",
            training_mode="gradient",
            objective=ObjectiveConfig(
                gamma=2.0,
                host_mode="neighbours",
                host_weight=0.5,
                label_enabled=True,
                label_weight=1.0,
            ),
            degree=4,
            epochs=40,
            tau=0.5,
            diagnostic_interval=1,
            seed=0,
        )
    )
    result = learner.run(bundle.train_graphs)
    reporter = LoggingVisualization(
        LoggingVisualizationConfig(output_dir=tmp_path, make_plots=True)
    )
    reporter.log_learning(result)
    assert (tmp_path / "training" / "training_curves.png").exists()
    history = (tmp_path / "training" / "learning_history.csv").read_text()
    for column in (
        "train_capture",
        "confusability",
        "internal",
        "boundary",
        "label_loss",
        "host_loss",
        "graph_id",
    ):
        assert column in history
    # every epoch carries the diagnostics, not just a subsample
    assert all(np.isfinite(row["train_capture"]) for row in result.history)
