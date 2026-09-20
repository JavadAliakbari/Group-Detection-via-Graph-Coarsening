"""Learning: architectures, bases, heads, training modes, label head, determinism."""

from __future__ import annotations

import math

import pytest
import torch

from src.pipeline.data import Data, DataConfig
from src.pipeline.learning import (
    GCNLearning,
    GraphSAGELearning,
    LearningConfig,
    PolynomialFilterLearning,
    build_learner,
)
from src.pipeline.objective import ObjectiveConfig


def _learning_config(**kw) -> LearningConfig:
    base = dict(
        architecture="polynomial",
        training_mode="gradient",
        objective=ObjectiveConfig(gamma=2.0),
        degree=4,
        num_heads=1,
        epochs=6,
        tau=0.5,
        hidden_dim=6,
        num_layers=2,
        diagnostic_interval=5,
        label_head_epochs=5,
        seed=0,
    )
    base.update(kw)
    return LearningConfig(**base)


@pytest.mark.parametrize("basis", ["chebyshev", "monomial"])
def test_polynomial_bases(basis, bundle):
    result = PolynomialFilterLearning(_learning_config(basis=basis)).run(
        bundle.train_graphs
    )
    Z = result.representations[bundle.train_graphs[0].graph_id]
    assert Z.shape == (bundle.train_graphs[0].num_nodes, bundle.feature_dim)
    assert torch.isfinite(Z).all()


def test_lanczos_basis_is_not_supported():
    with pytest.raises(ValueError, match="basis must be"):
        _learning_config(basis="lanczos")


@pytest.mark.parametrize("heads", [1, 3])
def test_multi_head_polynomial_width(heads, bundle):
    result = PolynomialFilterLearning(_learning_config(num_heads=heads)).run(
        bundle.train_graphs
    )
    Z = result.representations[bundle.train_graphs[0].graph_id]
    assert Z.shape[1] == heads * bundle.feature_dim


def test_shared_and_per_channel_filters(bundle):
    shared = PolynomialFilterLearning(
        _learning_config(num_heads=2, shared_filters=True)
    ).run(bundle.train_graphs)
    per_channel = PolynomialFilterLearning(
        _learning_config(num_heads=2, shared_filters=False)
    ).run(bundle.train_graphs)
    theta_shared = shared.model.theta
    theta_channel = per_channel.model.theta
    assert theta_shared.shape == theta_channel.shape
    # shared: every channel carries the same coefficient vector
    assert torch.allclose(theta_shared[:, :, 0:1].expand_as(theta_shared), theta_shared)
    assert not torch.allclose(
        theta_channel[:, :, 0:1].expand_as(theta_channel), theta_channel
    )


@pytest.mark.parametrize(
    "architecture,cls", [("gcn", GCNLearning), ("graphsage", GraphSAGELearning)]
)
@pytest.mark.parametrize("heads", [1, 2])
def test_nonlinear_output_shapes(architecture, cls, heads, bundle):
    config = _learning_config(architecture=architecture, num_heads=heads, hidden_dim=5)
    result = cls(config).run(bundle.train_graphs)
    graph = bundle.train_graphs[0]
    assert result.representations[graph.graph_id].shape == (graph.num_nodes, heads * 5)


@pytest.mark.parametrize("aggregation", ["mean", "max", "sum"])
def test_graphsage_aggregations(aggregation, bundle):
    config = _learning_config(
        architecture="graphsage", aggregation=aggregation, hidden_dim=4
    )
    result = GraphSAGELearning(config).run(bundle.train_graphs)
    Z = result.representations[bundle.train_graphs[0].graph_id]
    assert Z.shape[1] == 4 and torch.isfinite(Z).all()


@pytest.mark.parametrize("architecture", ["gcn", "graphsage"])
def test_closed_form_is_rejected_for_nonlinear_models(architecture):
    with pytest.raises(ValueError, match="no closed-form solution"):
        _learning_config(architecture=architecture, training_mode="closed_form")


def test_closed_form_returns_before_and_after_diagnostics(bundle):
    result = PolynomialFilterLearning(
        _learning_config(training_mode="closed_form", num_heads=2)
    ).run(bundle.train_graphs)
    report = result.closed_form
    assert report is not None
    for graph in bundle.train_graphs:
        for stage in ("initial", "final"):
            row = report[stage][graph.graph_id]
            assert {
                "boundary",
                "internal",
                "ratio",
                "edge_objective",
                "host_loss",
                "train_capture",
                "confusability",
            } <= set(row)
    assert report["gram_condition_median"] > 0
    assert len(report["positive_directions_per_channel"]) == bundle.feature_dim


def test_closed_form_solution_lives_in_the_gram_support(bundle):
    """Regression: the solve must not return a filter from the Gram's null space.

    A degree-K Chebyshev block Gram is numerically rank deficient, and solving in
    a ridge metric that keeps those directions returns a roundoff-sized
    "positive" eigenvalue whose Z carries no screened energy -- the level then
    collapses and capture goes to zero.
    """

    config = _learning_config(
        training_mode="closed_form",
        degree=24,
        num_heads=2,
        objective=ObjectiveConfig(gamma=0.5),
    )
    learner = PolynomialFilterLearning(config)
    result = learner.run(bundle.train_graphs)
    report = result.closed_form
    assert report["gram_rank_median"] <= report["gram_dimension"]
    for graph in bundle.train_graphs:
        before = report["initial"][graph.graph_id]
        after = report["final"][graph.graph_id]
        # the level keeps a comparable amount of contrast, and the objective rose
        assert after["boundary"] > 1e-3 * before["boundary"]
        assert after["edge_objective"] >= before["edge_objective"] - 1e-12
        level = learner.level_of(graph)
        assert float(level.gram.diagonal().mean()) > 1e-6


def test_closed_form_reports_the_cliff_and_flags_gamma_above_it(bundle, caplog):
    """Regression: gamma >= lambda_max(N, P) has no positive direction."""

    import logging

    config = _learning_config(
        training_mode="closed_form", objective=ObjectiveConfig(gamma=1e6)
    )
    with caplog.at_level(logging.WARNING):
        result = PolynomialFilterLearning(config).run(bundle.train_graphs)
    report = result.closed_form
    assert report["penalty_star_median"] < 1e6
    assert max(report["positive_directions_per_channel"]) == 0
    assert any("above the pencil cliff" in record.message for record in caplog.records)


def test_screened_level_is_scale_invariant_by_default(graph, geometry):
    """Regression: an absolute ridge would drive a small-energy Z to a zero level."""

    from src.pipeline.geometry import screened_level

    generator = torch.Generator().manual_seed(0)
    Z = torch.randn(graph.num_nodes, 4, dtype=torch.float64, generator=generator)
    base = screened_level(Z, geometry, 0.5)
    tiny = screened_level(1e-8 * Z, geometry, 0.5)
    assert torch.allclose(base.values.abs(), tiny.values.abs(), atol=1e-8)


def test_closed_form_ratio_reuses_the_dinkelbach_solver(bundle):
    result = PolynomialFilterLearning(
        _learning_config(
            training_mode="closed_form", objective=ObjectiveConfig(gamma=None)
        )
    ).run(bundle.train_graphs)
    ratio = result.closed_form["ratio_solver"]
    assert ratio["converged"]
    assert ratio["iterations_max"] >= 1
    assert result.closed_form["form"] == "ratio"


def test_gradient_ratio_runs_an_outer_dinkelbach_loop(bundle):
    config = _learning_config(
        objective=ObjectiveConfig(gamma=None), epochs=12, dinkelbach_iters=4
    )
    result = PolynomialFilterLearning(config).run(bundle.train_graphs)
    report = result.dinkelbach
    assert report["outer_iterations"] == 4
    assert report["inner_epochs_per_iteration"] == 3
    assert len(report["rho_trace"]) == 4
    assert len(result.history) == 12
    assert {row["outer_iteration"] for row in result.history} == {0, 1, 2, 3}


def test_dinkelbach_iterate_selection_ignores_the_moving_surrogate(bundle):
    """The linearized objective shrinks as rho grows; ranking by it would always
    return an iterate from the first outer stage."""

    config = _learning_config(
        objective=ObjectiveConfig(gamma=None),
        epochs=200,
        dinkelbach_iters=5,
        diagnostic_interval=10,
    )
    result = PolynomialFilterLearning(config).run(bundle.train_graphs)
    history = result.history
    best_epoch = result.dinkelbach["best_epoch"]

    # rho climbs, so the surrogate's minimum sits in the very first outer stage
    naive = min(history, key=lambda r: r["total_loss"])
    assert naive["outer_iteration"] == 0
    assert history[best_epoch]["outer_iteration"] > 0

    measured = [r for r in history if math.isfinite(r["selection_score"])]
    assert result.dinkelbach["best_ratio"] >= measured[0]["all_graphs_ratio"]
    assert result.dinkelbach["best_ratio"] == pytest.approx(
        max(r["all_graphs_ratio"] for r in measured)
    )


def test_iterate_selection_scores_every_training_graph(bundle):
    """A lucky single-graph draw must not be mistaken for an improvement."""

    config = _learning_config(
        objective=ObjectiveConfig(gamma=None),
        epochs=40,
        dinkelbach_iters=2,
        diagnostic_interval=5,
        multi_graph_mode="sample",
    )
    result = PolynomialFilterLearning(config).run(bundle.train_graphs)
    measured = [r for r in result.history if math.isfinite(r["selection_score"])]
    assert measured
    for row in measured:
        assert row["selection_score"] == pytest.approx(-row["all_graphs_ratio"])
        # the per-epoch row still reports the graph the step was taken on
        assert row["graph_id"] in {g.graph_id for g in bundle.train_graphs}
    assert all(
        math.isnan(r["selection_score"])
        for r in result.history
        if "all_graphs_ratio" not in r
    )
    assert len(measured) < len(result.history)  # the cadence is the diagnostic one


def test_gradient_trajectory_is_complete(bundle):
    result = PolynomialFilterLearning(
        _learning_config(epochs=10, diagnostic_interval=5)
    ).run(bundle.train_graphs)
    assert len(result.history) == 10
    required = {
        "epoch",
        "total_loss",
        "edge_objective",
        "boundary",
        "internal",
        "ratio",
        "host_loss",
        "label_loss",
        "train_capture",
        "heldout_capture",
        "confusability",
        "learning_rate",
        "graph_id",
    }
    assert required <= set(result.history[0])


def test_collapse_onto_the_origin_is_flagged(bundle, caplog):
    """internal -> 0 bought by killing capture is a degenerate optimum, not a fit."""

    learner = PolynomialFilterLearning(
        _learning_config(objective=ObjectiveConfig(gamma=None))
    )
    history = [
        dict(train_capture=0.18, heldout_capture=0.17, ratio=1.3),
        dict(train_capture=0.02, heldout_capture=0.11, ratio=640.0),
    ]
    with caplog.at_level("WARNING"):
        learner._warn_about_the_trajectory(history)
    text = caplog.text
    assert "collapsing onto the origin" in text
    assert "fitting the *identities* of the training groups" in text

    caplog.clear()
    with caplog.at_level("WARNING"):
        learner._warn_about_the_trajectory(
            [
                dict(train_capture=0.18, heldout_capture=0.17, ratio=1.3),
                dict(train_capture=0.23, heldout_capture=0.26, ratio=1.7),
            ]
        )
    assert caplog.text == ""


def test_a_healthy_polynomial_run_is_not_flagged(bundle, caplog):
    with caplog.at_level("WARNING"):
        result = PolynomialFilterLearning(
            _learning_config(
                objective=ObjectiveConfig(gamma=None),
                epochs=60,
                dinkelbach_iters=3,
                diagnostic_interval=10,
            )
        ).run(bundle.train_graphs)
    assert "collapsing onto the origin" not in caplog.text
    measured = [r for r in result.history if math.isfinite(r["train_capture"])]
    assert measured[-1]["train_capture"] >= 0.5 * measured[0]["train_capture"]


def test_gradient_label_head_reaches_the_group_label_report(bundle):
    """Under gradient the head is trained jointly and still yields probabilities."""

    config = _learning_config(
        objective=ObjectiveConfig(gamma=1.0, label_enabled=True, label_weight=1.0),
        epochs=20,
    )
    learner = PolynomialFilterLearning(config)
    result = learner.run(bundle.train_graphs)
    assert result.label_head is not None
    assert result.label_history and "label_loss" in result.label_history[0]
    assert all(row["label_loss"] > 0 for row in result.history)

    graph = bundle.train_graphs[0]
    probabilities = learner.predict_proba(graph)
    assert probabilities.shape == (graph.num_nodes, result.diagnostics["num_classes"])
    assert torch.allclose(probabilities.sum(1), torch.ones(graph.num_nodes))


def test_multi_graph_modes_step_on_the_right_graph(bundle):
    ids = {g.graph_id for g in bundle.train_graphs}
    sampled = PolynomialFilterLearning(
        _learning_config(multi_graph_mode="sample", epochs=20)
    ).run(bundle.train_graphs)
    assert {row["graph_id"] for row in sampled.history} == ids
    for mode in ("mean", "min"):
        result = PolynomialFilterLearning(
            _learning_config(multi_graph_mode=mode, epochs=6)
        ).run(bundle.train_graphs)
        assert len(result.history) == 6


def test_label_head_consumes_the_screened_level_not_z(bundle):
    """The head's input width is the level width, and its input IS the level."""

    config = _learning_config(
        num_heads=2,
        objective=ObjectiveConfig(gamma=2.0, label_enabled=True, label_weight=1.0),
    )
    learner = PolynomialFilterLearning(config)
    result = learner.run(bundle.train_graphs)
    graph = bundle.train_graphs[0]
    head = result.label_head
    assert head is not None
    level = learner.level_of(graph)
    Z = result.representations[graph.graph_id]
    assert head.linear.in_features == level.width
    probabilities = learner.predict_proba(graph)
    assert probabilities.shape == (graph.num_nodes, 2)
    expected = torch.softmax(head(level.values), dim=1)
    assert torch.allclose(probabilities, expected)
    # feeding Z instead of the level gives a different answer, so they are not
    # accidentally the same object
    assert not torch.allclose(level.values, Z)


def test_multiclass_label_head_dimensions():
    from src.pipeline.data import Graph, Group

    n = 40
    src = list(range(n - 1))
    edge_index = torch.tensor(
        [src + list(range(1, n)), list(range(1, n)) + src], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(0)
    graph = Graph(
        graph_id="g",
        num_nodes=n,
        edge_index=edge_index,
        features=torch.randn(n, 4, dtype=torch.float64, generator=generator),
        node_labels=torch.zeros(n, dtype=torch.long),
        groups=[
            Group("a", torch.tensor([1, 2, 3]), 1, "train"),
            Group("b", torch.tensor([10, 11, 12]), 2, "train"),
            Group("c", torch.tensor([20, 21, 22]), 3, "train"),
        ],
    )
    config = _learning_config(
        epochs=4,
        objective=ObjectiveConfig(gamma=1.0, label_enabled=True, label_weight=1.0),
    )
    learner = PolynomialFilterLearning(config)
    learner.run([graph])
    assert learner.num_classes_ == 4
    assert learner.label_head_.linear.out_features == 4
    assert learner.predict_proba(graph).shape == (n, 4)


def test_closed_form_label_head_is_fitted_on_a_frozen_level(bundle):
    config = _learning_config(
        training_mode="closed_form",
        objective=ObjectiveConfig(gamma=2.0, label_enabled=True, label_weight=1.0),
        label_head_epochs=15,
    )
    learner = PolynomialFilterLearning(config)
    result = learner.run(bundle.train_graphs)
    assert len(result.label_history) == 15
    assert {"epoch", "label_loss", "train_accuracy", "graph_id"} == set(
        result.label_history[0]
    )
    before = {k: v.clone() for k, v in learner.model_.state_dict().items()}
    learner._fit_frozen_label_head(
        [learner.build_context(g, i) for i, g in enumerate(bundle.train_graphs)]
    )
    for key, value in learner.model_.state_dict().items():
        assert torch.equal(before[key], value), "the closed-form subspace moved"


def test_training_never_reads_held_out_group_labels(bundle):
    """Contexts carry held-out node sets for diagnostics but no held-out labels."""

    learner = PolynomialFilterLearning(_learning_config(epochs=4))
    learner.feature_dim_ = bundle.feature_dim
    learner.num_classes_ = 2
    context = learner.build_context(bundle.train_graphs[0], 0)
    train_nodes = {
        v for g in bundle.train_graphs[0].train_groups for v in g.nodes.tolist()
    }
    assert set(context.label_nodes.tolist()) <= train_nodes
    held_nodes = {
        v for g in bundle.train_graphs[0].test_groups for v in g.nodes.tolist()
    }
    assert not (set(context.label_nodes.tolist()) & (held_nodes - train_nodes))


def test_determinism_with_a_fixed_seed(bundle):
    first = PolynomialFilterLearning(_learning_config(epochs=8, seed=5)).run(
        bundle.train_graphs
    )
    second = PolynomialFilterLearning(_learning_config(epochs=8, seed=5)).run(
        bundle.train_graphs
    )
    assert [row["total_loss"] for row in first.history] == [
        row["total_loss"] for row in second.history
    ]
    key = bundle.train_graphs[0].graph_id
    assert torch.allclose(first.representations[key], second.representations[key])


def test_transfer_requires_matching_feature_dimension(bundle):
    learner = PolynomialFilterLearning(_learning_config(epochs=3))
    learner.run(bundle.train_graphs)
    other = (
        Data(
            DataConfig(
                source="synthetic",
                num_nodes=120,
                num_groups=3,
                group_size=5,
                feature_dim=bundle.feature_dim + 4,
                seed=9,
            )
        )
        .run()
        .train_graphs[0]
    )
    with pytest.raises(ValueError, match="feature dimension"):
        learner.represent(other)


def test_build_learner_dispatch():
    assert isinstance(build_learner(_learning_config()), PolynomialFilterLearning)
    assert isinstance(build_learner(_learning_config(architecture="gcn")), GCNLearning)
    assert isinstance(
        build_learner(_learning_config(architecture="graphsage")), GraphSAGELearning
    )


def test_learning_result_serializes(bundle):
    result = PolynomialFilterLearning(_learning_config(epochs=4)).run(
        bundle.train_graphs
    )
    import json

    payload = json.loads(json.dumps(result.to_serializable(), default=str))
    assert payload["config"]["architecture"] == "polynomial"
    assert payload["history"]
