"""Real-data adapters, exercised only when the local datasets are present."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.pipeline.data import Data, DataConfig

ELLIPTIC = Path("data/elliptic_actors")
SNAP = Path("data/community")

elliptic_available = pytest.mark.skipif(
    not (ELLIPTIC / "wallets_features.csv").exists(),
    reason="Elliptic++ Actors data not available locally",
)
snap_available = pytest.mark.skipif(
    not (SNAP / "com-amazon.ungraph.txt").exists(),
    reason="SNAP ground-truth-community data not available locally",
)


def _check_bundle(bundle):
    for graph in bundle.graphs:
        assert graph.features.shape[0] == graph.num_nodes
        assert torch.isfinite(graph.features).all()
        assert graph.node_labels.shape == (graph.num_nodes,)
        assert graph.groups
        for group in graph.groups:
            assert int(group.nodes.max()) < graph.num_nodes
    for graph in bundle.test_graphs:
        assert graph.train_groups == []


@elliptic_available
@pytest.mark.parametrize("feature_mode", ["random", "wallet+random"])
def test_elliptic_adapter(feature_mode):
    bundle = Data(
        DataConfig(
            source="elliptic",
            data_dir=ELLIPTIC,
            elliptic_train_days=(25,),
            elliptic_test_days=(26,),
            elliptic_feature_mode=feature_mode,
            structural_feature_dim=8,
            min_group_size=2,
            train_ratio=0.5,
            seed=0,
        )
    ).run()
    assert len(bundle.train_graphs) == 1 and len(bundle.test_graphs) == 1
    assert bundle.train_graphs[0].node_id_map is not None
    _check_bundle(bundle)


@elliptic_available
def test_elliptic_pins_feature_columns_across_days():
    """A filter fit on one day must apply to the next, so the width is pinned."""

    bundle = Data(
        DataConfig(
            source="elliptic",
            data_dir=ELLIPTIC,
            elliptic_train_days=(25,),
            elliptic_test_days=(26, 27),
            elliptic_feature_mode="wallet",
            min_group_size=2,
            seed=0,
        )
    ).run()
    assert len({g.feature_dim for g in bundle.graphs}) == 1


@elliptic_available
def test_elliptic_end_to_end_smoke(tmp_path):
    from src.pipeline.coarsening import CoarseningConfig
    from src.pipeline.learning import LearningConfig
    from src.pipeline.logging_visualization import LoggingVisualizationConfig
    from src.pipeline.objective import ObjectiveConfig
    from src.pipeline.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        data=DataConfig(
            source="elliptic",
            data_dir=ELLIPTIC,
            elliptic_train_days=(25,),
            elliptic_feature_mode="random",
            structural_feature_dim=8,
            min_group_size=2,
            train_ratio=0.5,
            seed=0,
        ),
        learning=LearningConfig(
            architecture="polynomial",
            training_mode="closed_form",
            objective=ObjectiveConfig(gamma=10.0),
            degree=8,
            num_heads=2,
            shared_filters=True,
            tau=0.5,
            seed=0,
        ),
        coarsening=CoarseningConfig(
            method="raw_ward", tau=0.5, cut_rule="f1", exact_epsilon_budget=12
        ),
        logging=LoggingVisualizationConfig(output_dir=tmp_path, make_plots=False),
    )
    result = run_pipeline(config)
    graph_id = config.data.elliptic_train_days[0]
    out = result.coarsenings[f"d{graph_id}"]
    assert out.trajectory[-1]["n_coarse"] == 2
    assert out.metrics["train"]["total"] > 0


@snap_available
def test_snap_adapter_keeps_overlapping_communities():
    bundle = Data(
        DataConfig(
            source="snap",
            data_dir=SNAP,
            snap_name="amazon",
            snap_max_groups=40,
            structural_feature_dim=8,
            min_group_size=3,
            train_ratio=0.5,
            seed=0,
        )
    ).run()
    graph = bundle.train_graphs[0]
    _check_bundle(bundle)
    assert graph.train_groups and graph.test_groups
    members: dict = {}
    for group in graph.groups:
        for node in group.nodes.tolist():
            members.setdefault(node, []).append(group.group_id)
    assert any(len(v) > 1 for v in members.values()), "expected overlapping communities"
    # documented conventions: structural features, membership labels
    assert graph.feature_dim == 8
    assert set(graph.node_labels.unique().tolist()) <= {0, 1}
