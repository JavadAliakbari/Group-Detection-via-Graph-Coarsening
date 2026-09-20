"""Coarsening: complete hierarchies, trajectories, cut rules, leakage, epsilon."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from src.pipeline.coarsening import Coarsening, CoarseningConfig, CutRule
from src.pipeline.learning import LearningConfig, PolynomialFilterLearning
from src.pipeline.objective import ObjectiveConfig

METHODS = ["ward_tree", "raw_ward", "deflated_ward"]


@pytest.fixture(scope="module")
def fitted(bundle):
    config = LearningConfig(
        architecture="polynomial",
        training_mode="closed_form",
        objective=ObjectiveConfig(gamma=2.0),
        degree=4,
        num_heads=1,
        tau=0.5,
        seed=0,
    )
    learner = PolynomialFilterLearning(config)
    result = learner.run(bundle.train_graphs)
    return learner, result


def _config(**kw) -> CoarseningConfig:
    base = dict(method="ward_tree", tau=0.5, cut_rule="f1", exact_epsilon_budget=25)
    base.update(kw)
    return CoarseningConfig(**base)


@pytest.mark.parametrize("method", METHODS)
def test_hierarchy_reaches_two_supernodes(method, bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(method=method)).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    assert out.hierarchy.children.shape[0] == graph.num_nodes - 1
    assert out.trajectory[0]["n_coarse"] == graph.num_nodes
    assert out.trajectory[-1]["n_coarse"] == 2
    counts = [row["n_coarse"] for row in out.trajectory]
    assert counts == list(range(graph.num_nodes, 1, -1))
    assert out.hierarchy.labels_at(2).max() == 1


@pytest.mark.parametrize("method", METHODS)
def test_trajectory_fields(method, bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(method=method)).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    row = out.trajectory[len(out.trajectory) // 2]
    required = {
        "n_coarse",
        "reduction",
        "retained",
        "epsilon",
        "epsilon_is_exact",
        "selected",
        "train_mean_recall",
        "train_mean_precision",
        "train_mean_f1",
        "train_detection_rate",
        "train_detected",
        "train_total",
        "train_micro_f1",
    }
    assert required <= set(row)
    assert sum(r["selected"] for r in out.trajectory) == 1


def test_reduction_convention_is_the_fraction_removed(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config()).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    for row in out.trajectory[::37]:
        assert row["reduction"] == pytest.approx(
            1.0 - row["n_coarse"] / graph.num_nodes
        )
        assert row["retained"] == pytest.approx(row["n_coarse"] / graph.num_nodes)
        assert row["reduction"] + row["retained"] == pytest.approx(1.0)


def test_epsilon_budget_and_exactness(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(
        _config(cut_rule="epsilon", epsilon_budget=0.35, exact_epsilon_budget=20)
    ).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    assert out.selected["epsilon"] <= 0.35 + 1e-9
    assert out.selected["epsilon_is_exact"]
    exact = [row for row in out.trajectory if row["epsilon_is_exact"]]
    assert 2 <= len(exact) <= 22  # budget + the selected level
    assert len(exact) < len(out.trajectory)


def test_exact_epsilon_budget_default_is_200():
    assert CoarseningConfig().exact_epsilon_budget == 200


def test_epsilon_q_cut_rule_uses_the_harmonic_constant(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(
        _config(cut_rule="epsilon_q", epsilon_budget=0.35, exact_epsilon_budget=20)
    ).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    assert out.selected["rule"] == "epsilon_q"
    assert out.selected["epsilon_q"] <= 0.35 + 1e-9
    assert out.selected["epsilon_q_is_exact"]
    # the common epsilon is still measured, it is simply not what selected the cut
    assert "epsilon" in out.selected
    values = [row["epsilon_q"] for row in out.trajectory]
    assert values[0] <= values[-1]
    assert min(values) >= 0.0


def test_epsilon_q_axis_is_monotone_along_nested_partitions(bundle, fitted):
    """The bisection in _budgeted_cut is only valid if the axis is monotone."""

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    coarsener = Coarsening(_config(cut_rule="epsilon_q"))
    hierarchy, _ctx = coarsener.build_hierarchy(
        graph, result.representations[graph.graph_id]
    )
    evaluate = coarsener._epsilon_q_evaluator(
        graph, result.representations[graph.graph_id], hierarchy
    )
    levels = np.linspace(0, hierarchy.n_leaves - 2, 12).astype(int)
    values = [evaluate(int(t)) for t in levels]
    assert all(b - a >= -1e-9 for a, b in zip(values, values[1:]))


def test_score_sum_cut_rule_is_a_normalized_cumulative_cost(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(cut_rule="score_sum", epsilon_budget=0.4)).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    assert out.selected["rule"] == "score_sum"
    assert out.selected["score_sum"] <= 0.4 + 1e-9
    assert out.selected["score_sum_raw"] > 0.0

    values = [row["score_sum"] for row in out.trajectory]
    assert values[0] == pytest.approx(0.0)
    assert all(b - a >= -1e-12 for a, b in zip(values, values[1:]))
    assert max(values) <= 1.0 + 1e-12
    # the raw sum is the plain cumulative merge score
    raw = [row["score_sum_raw"] for row in out.trajectory]
    assert raw[1] == pytest.approx(out.hierarchy.scores[0])


def test_the_new_axes_are_only_measured_when_a_rule_reads_them(bundle, fitted):
    """eps_Q is an eigensolve per level, so it must not be paid for by default."""

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(cut_rule="reduction", transfer_cut_rule="reduction")).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    assert "epsilon_q" not in out.trajectory[0]
    assert "score_sum" not in out.trajectory[0]
    assert "epsilon" in out.trajectory[0]


@pytest.mark.parametrize("transfer_rule", ["epsilon", "epsilon_q", "score_sum"])
def test_every_budgeted_rule_transfers(transfer_rule, bundle, fitted):
    learner, result = fitted
    train = bundle.train_graphs[0]
    transfer = bundle.test_graphs[0]
    coarsener = Coarsening(
        _config(cut_rule="f1", transfer_cut_rule=transfer_rule, exact_epsilon_budget=20)
    )
    learned = coarsener.run(
        train,
        result.representations[train.graph_id],
        train_groups=train.train_groups,
    ).cut_rule
    assert math.isfinite(learned.budget_for(transfer_rule))

    moved = coarsener.run(transfer, learner.represent(transfer), cut_rule=learned)
    assert moved.selected["rule"] == transfer_rule
    assert moved.selected[transfer_rule] <= learned.budget_for(transfer_rule) + 1e-9


def test_a_cut_rule_without_the_requested_coordinate_is_refused():
    rule = CutRule(epsilon=0.4, reduction=0.5, n_coarse=10, source_graph="G0")
    assert rule.budget_for("epsilon") == pytest.approx(0.4)
    with pytest.raises(ValueError, match="carries no 'epsilon_q' coordinate"):
        rule.budget_for("epsilon_q")


def test_reduction_cut_rule(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(cut_rule="reduction", reduction=0.6)).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    assert out.selected["reduction"] == pytest.approx(0.6, abs=0.01)


def test_f1_cut_rule_maximizes_training_f1(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(cut_rule="f1")).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    best = max(row["train_mean_f1"] for row in out.trajectory)
    selected = [row for row in out.trajectory if row["selected"]][0]
    assert selected["train_mean_f1"] == pytest.approx(best)
    assert out.cut_rule is not None
    assert out.cut_rule.epsilon == pytest.approx(out.selected["epsilon"])


def test_oracle_is_the_hierarchys_best_cut_over_all_groups(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(cut_rule="f1")).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    assert out.oracle.mode == "oracle"
    assert out.stopping_rule.mode == "stopping_rule"
    best = max(row["all_mean_f1"] for row in out.trajectory)
    assert out.oracle.metrics["all"]["mean_f1"] == pytest.approx(best)
    # the oracle is an upper bound on every split it reports
    for split, metrics in out.stopping_rule.metrics.items():
        assert out.oracle.metrics["all"]["mean_f1"] >= 0.0
        assert split in out.oracle.metrics
    assert out.oracle.node_to_supernode.shape == (graph.num_nodes,)
    assert len(out.oracle.per_group) == len(graph.groups)
    assert 0.0 <= out.pr_auc <= 1.0
    assert out.levels >= 2


def test_a_misplaced_cut_is_flagged(bundle, fitted, caplog):
    """Regression: a too-tight epsilon budget must not look like a broken model."""

    import logging

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    with caplog.at_level(logging.WARNING):
        out = Coarsening(_config(cut_rule="epsilon", epsilon_budget=0.05)).run(
            graph,
            result.representations[graph.graph_id],
            train_groups=graph.train_groups,
        )
    assert (
        out.stopping_rule.metrics["all"]["mean_f1"]
        < out.oracle.metrics["all"]["mean_f1"]
    )
    assert any("operating point" in record.message for record in caplog.records)


def test_f1_rule_is_refused_without_training_groups(bundle, fitted):
    learner, _result = fitted
    transfer = bundle.test_graphs[0]
    with pytest.raises(ValueError, match="requires train_groups"):
        Coarsening(_config(cut_rule="f1")).run(transfer, learner.represent(transfer))


def test_f1_rule_refuses_groups_outside_the_declared_training_set(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    with pytest.raises(ValueError, match="outside train_groups"):
        Coarsening(_config(cut_rule="f1")).run(
            graph,
            result.representations[graph.graph_id],
            train_groups=graph.train_groups[:1],
        )


def test_transfer_uses_the_learned_rule_not_test_labels(bundle, fitted):
    learner, result = fitted
    graph = bundle.train_graphs[0]
    coarsener = Coarsening(_config(cut_rule="f1", transfer_cut_rule="epsilon"))
    trained = coarsener.run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    transfer = bundle.test_graphs[0]
    applied = coarsener.run(
        transfer, learner.represent(transfer), cut_rule=trained.cut_rule
    )
    assert applied.selected["rule"] == "epsilon"
    assert applied.selected["epsilon"] <= trained.cut_rule.epsilon + 1e-9
    # the F1-optimal cut for the transfer graph is generally NOT the applied one
    best = max(row["test_mean_f1"] for row in applied.trajectory)
    assert best >= applied.metrics["test"]["mean_f1"] - 1e-12
    assert applied.cut_rule is None


def test_transfer_reduction_rule(bundle, fitted):
    learner, _result = fitted
    transfer = bundle.test_graphs[0]
    rule = CutRule(epsilon=0.9, reduction=0.5, n_coarse=0, source_graph="G0")
    out = Coarsening(_config(transfer_cut_rule="reduction")).run(
        transfer, learner.represent(transfer), cut_rule=rule
    )
    assert out.selected["reduction"] == pytest.approx(0.5, abs=0.01)


def test_overlapping_groups_are_scored_independently(fitted):
    from src.pipeline.data import Graph, Group

    n = 30
    src = list(range(n - 1))
    edge_index = torch.tensor(
        [src + list(range(1, n)), list(range(1, n)) + src], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(0)
    graph = Graph(
        graph_id="overlap",
        num_nodes=n,
        edge_index=edge_index,
        features=torch.randn(n, 4, dtype=torch.float64, generator=generator),
        node_labels=torch.zeros(n, dtype=torch.long),
        groups=[
            Group("a", torch.tensor([3, 4, 5]), 1, "train"),
            Group("b", torch.tensor([5, 6, 7]), 1, "train"),
        ],
    )
    Z = torch.randn(n, 3, dtype=torch.float64, generator=generator)
    out = Coarsening(
        _config(cut_rule="reduction", reduction=0.5, exact_epsilon_budget=10)
    ).run(graph, Z, train_groups=graph.train_groups)
    assert {row["group_id"] for row in out.per_group} == {"a", "b"}
    assert out.metrics["train"]["total"] == 2


def test_deflated_certificate_is_reported(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config(method="deflated_ward")).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    cert = out.deflated_certificate
    assert {
        "epsilon_q",
        "epsilon_pi",
        "mu",
        "sandwich_upper",
        "sandwich_ok",
        "epsilon_common",
    } == set(cert)
    assert cert["epsilon_q"] <= cert["epsilon_pi"] + 1e-8


def test_other_methods_report_no_certificate(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    for method in ("ward_tree", "raw_ward"):
        out = Coarsening(_config(method=method)).run(
            graph,
            result.representations[graph.graph_id],
            train_groups=graph.train_groups,
        )
        assert out.deflated_certificate is None


def test_config_validation():
    with pytest.raises(ValueError, match="method must be"):
        CoarseningConfig(method="kmeans")
    with pytest.raises(ValueError, match="cut_rule must be"):
        CoarseningConfig(cut_rule="oracle")
    with pytest.raises(ValueError, match="transfer_cut_rule"):
        CoarseningConfig(transfer_cut_rule="f1")
    with pytest.raises(ValueError, match="reduction must lie"):
        CoarseningConfig(reduction=1.0)


def test_result_serializes(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(_config()).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    payload = json.loads(json.dumps(out.to_serializable(), default=str))
    assert payload["selected"]["n_coarse"] > 0
    assert payload["per_group"]
    assert payload["metrics"]["train"]["mean_f1"] >= 0.0


def test_deterministic_with_a_fixed_seed(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    Z = result.representations[graph.graph_id]
    first = Coarsening(_config(method="deflated_ward")).run(
        graph, Z, train_groups=graph.train_groups
    )
    second = Coarsening(_config(method="deflated_ward")).run(
        graph, Z, train_groups=graph.train_groups
    )
    assert torch.equal(first.node_to_supernode, second.node_to_supernode)
    assert first.selected == second.selected
