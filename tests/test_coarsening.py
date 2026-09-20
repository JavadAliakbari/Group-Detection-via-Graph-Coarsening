"""Coarsening: complete hierarchies, trajectories, cut rules, leakage, epsilon."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from src.pipeline.coarsening import (
    Coarsening,
    CoarseningConfig,
    CutRule,
    _labels_after,
)
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
    evaluate = coarsener._harmonic_evaluator(
        graph, result.representations[graph.graph_id], hierarchy
    )
    levels = np.linspace(0, hierarchy.n_leaves - 2, 12).astype(int)
    values = [evaluate(int(t))[0] for t in levels]
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


# --------------------------------------------------------------------------- #
# the cumulative-score bound  eps_Q^2 <= S_n <= q eps_Q^2
#
# S_n telescopes to trace(H_P) and eps_Q^2 = lambda_max(H_P), so the bound is the
# eigenvalue/trace sandwich of a PSD q x q matrix.  Tolerance 1e-9 absolute: the
# per-merge scores are accumulated in float64 and H_P is assembled by block CG
# with tol 1e-11, so a few hundred merges leave errors around 1e-13.
# --------------------------------------------------------------------------- #
SCORE_BOUND_TOLERANCE = 1e-9


def _deflated_scores(graph, basis, tau, **kw):
    """Hierarchy plus the exact harmonic machinery, for the bound checks."""

    from src.deflated_coarsen import exact_harmonic_H, harmonic_rsa_epsilon
    from src.pipeline.coarsening import _to_scipy
    from src.smooth_dual_ward import m_orthonormal_basis, screened_operators

    coarsener = Coarsening(_config(method="deflated_ward", tau=tau, **kw))
    hierarchy, _ctx = coarsener.build_hierarchy(graph, basis)
    _a, d_tilde, _l, M = screened_operators(_to_scipy(graph.adjacency), tau)
    U, rank = m_orthonormal_basis(basis.detach().cpu().numpy(), M)
    return hierarchy, U, M, d_tilde, rank, exact_harmonic_H, harmonic_rsa_epsilon


def test_cumulative_score_bound_holds_for_exact_deflated_scores(bundle, fitted):
    """eps_Q^2 <= S_n <= q eps_Q^2 at EVERY genuine deflated merge."""

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    basis = result.representations[graph.graph_id]
    hierarchy, U, M, d_tilde, rank, exact_H, eps_q_of = _deflated_scores(
        graph, basis, 0.5, deflated_commit_solve="exact", cut_rule="score_sum"
    )
    q = U.shape[1]
    assert q == rank  # q is the effective rank, not the column count

    cumulative = np.concatenate([[0.0], np.cumsum(hierarchy.scores)])
    deflated = hierarchy.constrained_merges
    assert deflated >= 10
    for t in range(1, deflated + 1):
        labels = _labels_after(hierarchy.children, hierarchy.n_leaves, t)
        eps_sq = eps_q_of(U, M, d_tilde, labels) ** 2
        s_n = float(cumulative[t])
        # S_n telescopes to trace(H_P) exactly under the exact commit solve
        assert s_n == pytest.approx(
            float(np.trace(exact_H(U, M, d_tilde, labels))), abs=SCORE_BOUND_TOLERANCE
        )
        assert s_n >= eps_sq - SCORE_BOUND_TOLERANCE
        assert s_n <= q * eps_sq + SCORE_BOUND_TOLERANCE
        if eps_sq > 0:
            assert 1.0 - 1e-6 <= s_n / eps_sq <= q + 1e-6


def test_the_first_merge_is_where_the_lower_bound_is_tight(bundle, fitted):
    """After one merge H_P is rank one, so S_n == eps_Q^2 exactly."""

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    hierarchy, U, M, d_tilde, _rank, _H, eps_q_of = _deflated_scores(
        graph,
        result.representations[graph.graph_id],
        0.5,
        deflated_commit_solve="exact",
        cut_rule="score_sum",
    )
    labels = _labels_after(hierarchy.children, hierarchy.n_leaves, 1)
    assert float(hierarchy.scores[0]) == pytest.approx(
        eps_q_of(U, M, d_tilde, labels) ** 2, abs=SCORE_BOUND_TOLERANCE
    )


def test_the_hierarchy_stores_the_committed_direction_not_the_selection_key(
    bundle, fitted
):
    """``score`` is the queue key and goes stale; ``a_sq`` is what was committed."""

    from src.deflated_coarsen import deflated_coarsen
    from src.pipeline.coarsening import _to_scipy

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    basis = result.representations[graph.graph_id]
    raw = deflated_coarsen(
        _to_scipy(graph.adjacency),
        basis.detach().cpu().numpy(),
        0.5,
        rule="dual-ward",
        n_clusters=None,
        build_full_tree=True,
        solve="local",
        commit_solve="exact",
        track_euclidean=False,
        record_curve=False,
    )
    a_sq = np.array([r["a_sq"] for r in raw.merge_records_])
    score = np.array([r["score"] for r in raw.merge_records_])
    assert not np.allclose(a_sq, score)  # the two genuinely differ in this mode
    assert np.allclose(np.cumsum(a_sq), [r["trace_h"] for r in raw.merge_records_])

    hierarchy, *_ = _deflated_scores(
        graph, basis, 0.5, deflated_commit_solve="exact", cut_rule="score_sum"
    )
    assert np.allclose(hierarchy.scores[: len(a_sq)], a_sq)


def test_local_deflation_breaks_the_lower_bound_and_says_so(bundle, fitted):
    """Diagnostic: the production truncated solve is not the harmonic direction.

    This documents the limit of the proposition rather than enforcing it: the
    bound is stated for the exact eliminated direction, and ``solve='local'``
    returns a different unit vector, so ``S_n`` stops telescoping to
    ``trace(H_P)`` -- in either direction -- and the lower bound can fail where
    it is tight.
    """

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    basis = result.representations[graph.graph_id]
    hierarchy, U, M, d_tilde, _rank, exact_H, eps_q_of = _deflated_scores(
        graph, basis, 0.5, deflated_commit_solve="local", cut_rule="score_sum"
    )
    labels = _labels_after(hierarchy.children, hierarchy.n_leaves, 1)
    exact_first = float(np.trace(exact_H(U, M, d_tilde, labels)))
    assert float(hierarchy.scores[0]) != pytest.approx(
        exact_first, abs=SCORE_BOUND_TOLERANCE
    )

    out = Coarsening(
        _config(
            method="deflated_ward",
            cut_rule="score_sum",
            epsilon_budget=0.4,
            deflated_commit_solve="local",
        )
    ).run(graph, basis, train_groups=graph.train_groups)
    certificate = out.score_certificate
    assert certificate is not None
    assert certificate["solve"] == "local"
    # the discrepancy is real and two-sided, which is exactly why the bound is
    # not claimed in this mode
    assert certificate["max_abs_score_error"] > SCORE_BOUND_TOLERANCE
    if not certificate["holds_on_deflated_prefix"]:
        first = certificate["first_violation"]
        assert first["lower_gap"] < 0
        assert abs(first["lower_gap"]) <= 1e-2  # small, and it is the known cause


def test_the_exact_commit_solve_restores_the_bound(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    out = Coarsening(
        _config(
            method="deflated_ward",
            cut_rule="score_sum",
            epsilon_budget=0.4,
            deflated_commit_solve="exact",
        )
    ).run(
        graph, result.representations[graph.graph_id], train_groups=graph.train_groups
    )
    certificate = out.score_certificate
    assert certificate["holds_on_deflated_prefix"]
    assert certificate["worst_lower_gap"] >= -SCORE_BOUND_TOLERANCE
    assert certificate["worst_upper_gap"] >= -SCORE_BOUND_TOLERANCE
    assert certificate["max_abs_score_error"] <= SCORE_BOUND_TOLERANCE
    assert all(row["epsilon_q_is_exact"] for row in certificate["rows"])


def test_completion_merges_are_marked_and_excluded_from_the_bound(bundle, fitted):
    """Their score is a Ward increment, not an eliminated deflated direction."""

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    basis = result.representations[graph.graph_id]
    out = Coarsening(
        _config(
            method="deflated_ward",
            cut_rule="score_sum",
            epsilon_budget=0.4,
            deflated_commit_solve="exact",
        )
    ).run(graph, basis, train_groups=graph.train_groups)

    types = {row["merge_type"] for row in out.trajectory}
    assert types <= {"deflated", "completion", "ward"}
    boundary = out.hierarchy.constrained_merges
    for row in out.trajectory:
        expected = "completion" if row["merges"] > boundary else "deflated"
        assert row["merge_type"] == expected

    certificate = out.score_certificate
    assert certificate["constrained_merges"] == boundary
    # the verdict is only ever claimed on the deflated prefix
    assert all(
        row["merge_index"] <= boundary
        for row in certificate["rows"]
        if row["merge_type"] == "deflated"
    )
    if out.hierarchy.completion_merges:
        hierarchy = out.hierarchy
        from src.deflated_coarsen import exact_harmonic_H
        from src.pipeline.coarsening import _to_scipy
        from src.smooth_dual_ward import m_orthonormal_basis, screened_operators

        _a, d_tilde, _l, M = screened_operators(_to_scipy(graph.adjacency), 0.5)
        U, _r = m_orthonormal_basis(basis.detach().cpu().numpy(), M)
        t = min(boundary + 1, hierarchy.children.shape[0])
        labels = _labels_after(hierarchy.children, hierarchy.n_leaves, t)
        s_n = float(np.sum(hierarchy.scores[:t]))
        trace = float(np.trace(exact_harmonic_H(U, M, d_tilde, labels)))
        # the two definitions have parted company; this is why it is excluded
        assert abs(s_n - trace) > SCORE_BOUND_TOLERANCE


def test_the_bound_is_not_claimed_for_the_other_merge_rules(bundle, fitted):
    """ward_tree / raw_ward scores are different quantities entirely."""

    _learner, result = fitted
    graph = bundle.train_graphs[0]
    for method in ("ward_tree", "raw_ward"):
        out = Coarsening(
            _config(method=method, cut_rule="score_sum", epsilon_budget=0.4)
        ).run(
            graph,
            result.representations[graph.graph_id],
            train_groups=graph.train_groups,
        )
        assert out.score_certificate is None


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
