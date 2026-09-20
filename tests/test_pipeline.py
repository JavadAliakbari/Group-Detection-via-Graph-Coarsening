"""End-to-end smoke tests for the four supported configurations."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.pipeline.coarsening import CoarseningConfig
from src.pipeline.data import DataConfig
from src.pipeline.learning import LearningConfig
from src.pipeline.logging_visualization import LoggingVisualizationConfig
from src.pipeline.objective import ObjectiveConfig
from src.pipeline.pipeline import PipelineConfig, run_pipeline

DATA = dict(
    source="synthetic",
    num_train_graphs=2,
    num_test_graphs=1,
    num_nodes=180,
    group_size=[5, 8],
    num_groups=4,
    feature_dim=6,
    seed=17,
)


def _pipeline_config(tmp_path, learning_kw, objective_kw=None, method="deflated_ward"):
    objective = ObjectiveConfig(**{"gamma": 2.0, **(objective_kw or {})})
    return PipelineConfig(
        data=DataConfig(**DATA),
        learning=LearningConfig(
            objective=objective,
            tau=0.5,
            degree=4,
            epochs=8,
            hidden_dim=5,
            num_layers=2,
            diagnostic_interval=4,
            label_head_epochs=5,
            seed=0,
            **learning_kw,
        ),
        coarsening=CoarseningConfig(
            method=method, tau=0.5, cut_rule="f1", exact_epsilon_budget=15
        ),
        logging=LoggingVisualizationConfig(output_dir=tmp_path, make_plots=False),
    )


@pytest.mark.parametrize(
    "name, learning_kw, objective_kw",
    [
        (
            "polynomial-gradient",
            dict(architecture="polynomial", training_mode="gradient"),
            None,
        ),
        (
            "polynomial-closed-form",
            dict(architecture="polynomial", training_mode="closed_form"),
            dict(label_enabled=True, label_weight=1.0),
        ),
        (
            "gcn",
            dict(architecture="gcn", training_mode="gradient"),
            dict(host_mode="neighbours", host_weight=0.25),
        ),
        ("graphsage", dict(architecture="graphsage", training_mode="gradient"), None),
    ],
)
def test_end_to_end(name, learning_kw, objective_kw, tmp_path):
    config = _pipeline_config(tmp_path / name, learning_kw, objective_kw)
    result = run_pipeline(config)

    assert set(result.coarsenings) == {g.graph_id for g in result.bundle.graphs}
    for graph in result.bundle.train_graphs:
        assert result.coarsenings[graph.graph_id].metrics["train"]["total"] > 0
    transfer = result.bundle.test_graphs[0]
    assert result.coarsenings[transfer.graph_id].selected["rule"] == "epsilon"
    assert result.cut_rule is not None

    output = config.logging.output_dir
    payload = json.loads((output / "results.json").read_text())
    assert payload["dataset"]["feature_dim"] == DATA["feature_dim"]
    assert payload["summaries"]["stopping_rule"]["overall"]["train"]["graphs"] == 2
    assert payload["summaries"]["oracle"]["overall"]["test"]["graphs"] == 1
    for graph_id in result.coarsenings:
        assert (output / "trajectories" / f"{graph_id}.csv").exists()
        for mode in ("stopping_rule", "oracle"):
            assert (output / mode / "per_group" / f"{graph_id}.csv").exists()
    for mode in ("stopping_rule", "oracle"):
        assert (output / mode / "tables" / "summary.csv").exists()
        assert (output / mode / "tables" / "summary.json").exists()
    assert (output / "comparison.csv").exists()
    assert (output / "config.json").exists()


def test_oracle_bounds_the_stopping_rule(tmp_path):
    """The oracle is an upper bound and is never allowed to move the cut."""

    config = _pipeline_config(
        tmp_path / "bound", dict(architecture="polynomial", training_mode="closed_form")
    )
    result = run_pipeline(config)
    for evaluation in result.evaluations:
        coarsening = evaluation.coarsening
        assert (
            coarsening.oracle.metrics["all"]["mean_f1"]
            >= coarsening.stopping_rule.metrics["all"]["mean_f1"] - 1e-12
        )
    # the deployable rule was learned on a training graph, not on a transfer one
    assert result.cut_rule.source_graph in {
        g.graph_id for g in result.bundle.train_graphs
    }


def test_group_label_majority_vote_is_reported_with_coverage(tmp_path):
    config = _pipeline_config(
        tmp_path / "labels",
        dict(architecture="polynomial", training_mode="closed_form"),
        dict(label_enabled=True, label_weight=1.0),
    )
    result = run_pipeline(config)
    payload = json.loads((config.logging.output_dir / "results.json").read_text())
    for mode in ("stopping_rule", "oracle"):
        metrics = payload["group_labels"][mode]
        assert metrics
        for graph_metrics in metrics.values():
            assert "detected_group_label_coverage" in graph_metrics
            assert 0.0 <= graph_metrics["detected_group_label_coverage"] <= 1.0
    assert (config.logging.output_dir / "group_label_votes.csv").exists()
    rows = pd.read_csv(config.logging.output_dir / "group_label_votes.csv")
    assert {
        "graph_id",
        "group_id",
        "size",
        "true_label",
        "predicted_label",
        "votes",
        "mean_probabilities",
        "detected",
        "evaluation_mode",
        "correct",
    } <= set(rows.columns)


def test_pipeline_rejects_mismatched_tau(tmp_path):
    config = _pipeline_config(
        tmp_path, dict(architecture="polynomial", training_mode="gradient")
    )
    with pytest.raises(ValueError, match="same screened metric"):
        PipelineConfig(
            data=config.data,
            learning=config.learning,
            coarsening=CoarseningConfig(tau=0.9),
            logging=config.logging,
        )


def test_ratio_pipeline(tmp_path):
    config = _pipeline_config(
        tmp_path / "ratio",
        dict(architecture="polynomial", training_mode="gradient", dinkelbach_iters=2),
        dict(gamma=None),
        method="raw_ward",
    )
    result = run_pipeline(config)
    assert result.learning.dinkelbach["outer_iterations"] == 2
    assert len(result.learning.dinkelbach["rho_trace"]) == 2


def test_pipeline_is_deterministic(tmp_path):
    first = run_pipeline(
        _pipeline_config(
            tmp_path / "a", dict(architecture="polynomial", training_mode="gradient")
        )
    )
    second = run_pipeline(
        _pipeline_config(
            tmp_path / "b", dict(architecture="polynomial", training_mode="gradient")
        )
    )
    for graph_id, result in first.coarsenings.items():
        other = second.coarsenings[graph_id]
        assert result.selected["n_coarse"] == other.selected["n_coarse"]
        assert result.metrics == other.metrics
