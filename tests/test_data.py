"""Data component: sizing, splitting, validation, determinism."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pipeline.data import Data, DataConfig, DatasetBundle, Graph, Group


def _config(**kw) -> DataConfig:
    base = dict(
        source="synthetic",
        num_train_graphs=1,
        num_test_graphs=0,
        num_nodes=200,
        group_size=6,
        num_groups=4,
        feature_dim=8,
        seed=1,
    )
    base.update(kw)
    return DataConfig(**base)


def test_fixed_graph_size_is_exact():
    bundle = Data(_config(num_nodes=250, num_train_graphs=3)).run()
    assert [g.num_nodes for g in bundle.train_graphs] == [250, 250, 250]


def test_ranged_graph_size_is_sampled_per_graph():
    bundle = Data(_config(num_nodes=[150, 400], num_train_graphs=6)).run()
    sizes = [g.num_nodes for g in bundle.train_graphs]
    assert all(150 <= n <= 400 for n in sizes)
    assert len(set(sizes)) > 1


def test_fixed_and_ranged_group_size():
    fixed = Data(_config(group_size=7)).run().train_graphs[0]
    assert {g.num_nodes for g in fixed.groups} == {7}
    ranged = Data(_config(group_size=[3, 12], num_groups=10, num_nodes=400)).run()
    sizes = [g.num_nodes for g in ranged.train_graphs[0].groups]
    assert all(3 <= s <= 12 for s in sizes) and len(set(sizes)) > 1


def test_every_graph_has_valid_features_and_labels():
    bundle = Data(_config(num_train_graphs=2, num_test_graphs=2)).run()
    for graph in bundle.graphs:
        assert graph.features.shape == (graph.num_nodes, bundle.feature_dim)
        assert torch.isfinite(graph.features).all()
        assert graph.node_labels.shape == (graph.num_nodes,)
        assert int(graph.node_labels.min()) >= 0


def test_split_has_no_leakage_and_test_graphs_hold_out_everything():
    bundle = Data(
        _config(
            num_groups=8,
            num_train_graphs=2,
            num_test_graphs=2,
            num_nodes=400,
            train_ratio=0.5,
        )
    ).run()
    for graph in bundle.train_graphs:
        assert graph.train_groups and graph.test_groups
        assert not (
            {g.group_id for g in graph.train_groups}
            & {g.group_id for g in graph.test_groups}
        )
    for graph in bundle.test_graphs:
        assert graph.train_groups == []
        assert len(graph.test_groups) == len(graph.groups)


def test_bundle_rejects_training_groups_on_a_test_graph(graph):
    with pytest.raises(ValueError, match="carries training groups"):
        DatasetBundle([graph], [graph])


def test_bundle_rejects_mismatched_feature_dimensions():
    a = Data(_config(feature_dim=8)).run().train_graphs[0]
    b = Data(_config(feature_dim=16, seed=2)).run().train_graphs[0]
    with pytest.raises(ValueError, match="feature dimension"):
        DatasetBundle([a, b])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda kw: kw.update(num_nodes=0), "num_nodes must be positive"),
        (
            lambda kw: kw.update(edge_index=torch.tensor([[0, 99]], dtype=torch.long)),
            "edge_index must be",
        ),
        (lambda kw: kw.update(features=torch.zeros(3, 4)), "features must be"),
        (
            lambda kw: kw.update(node_labels=torch.zeros(3, dtype=torch.long)),
            "one entry per node",
        ),
    ],
)
def test_graph_validation_rejects_bad_shapes(mutate, message):
    kw = dict(
        graph_id="g",
        num_nodes=5,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        features=torch.zeros(5, 4),
        node_labels=torch.zeros(5, dtype=torch.long),
        groups=[],
    )
    mutate(kw)
    with pytest.raises(ValueError, match=message):
        Graph(**kw)


def test_graph_validation_rejects_out_of_range_group():
    with pytest.raises(ValueError, match="out-of-range node"):
        Graph(
            graph_id="g",
            num_nodes=4,
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            features=torch.zeros(4, 2),
            node_labels=torch.zeros(4, dtype=torch.long),
            groups=[Group("a", torch.tensor([0, 9]), 1, "train")],
        )


def test_group_validation():
    with pytest.raises(ValueError, match="empty"):
        Group("a", torch.tensor([], dtype=torch.long), 1, "train")
    with pytest.raises(ValueError, match="repeats a node"):
        Group("a", torch.tensor([1, 1]), 1, "train")
    with pytest.raises(ValueError, match="split must be"):
        Group("a", torch.tensor([1]), 1, "validation")


def test_overlapping_groups_are_allowed_when_labels_agree():
    graph = Graph(
        graph_id="g",
        num_nodes=6,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        features=torch.zeros(6, 2),
        node_labels=torch.zeros(6, dtype=torch.long),
        groups=[
            Group("a", torch.tensor([0, 1, 2]), 1, "train"),
            Group("b", torch.tensor([2, 3]), 1, "train"),
        ],
    )
    labels, mask = graph.node_group_labels("train")
    assert int(mask.sum()) == 4
    assert int(labels[2]) == 1


def test_conflicting_labels_on_a_shared_node_raise():
    graph = Graph(
        graph_id="g",
        num_nodes=6,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        features=torch.zeros(6, 2),
        node_labels=torch.zeros(6, dtype=torch.long),
        groups=[
            Group("a", torch.tensor([0, 1]), 1, "train"),
            Group("b", torch.tensor([1, 2]), 2, "train"),
        ],
    )
    with pytest.raises(ValueError, match="resolve the conflict"):
        graph.node_group_labels("train")


def test_config_validation():
    with pytest.raises(ValueError, match="source must be"):
        DataConfig(source="nope")
    with pytest.raises(ValueError, match="group_types"):
        _config(group_types=("triangle",))
    with pytest.raises(ValueError, match="train_ratio"):
        _config(train_ratio=0.0)
    with pytest.raises(ValueError, match="requires data_dir"):
        DataConfig(source="snap")


def test_deterministic_with_a_fixed_seed():
    first = Data(_config(num_train_graphs=2, num_test_graphs=1, seed=42)).run()
    second = Data(_config(num_train_graphs=2, num_test_graphs=1, seed=42)).run()
    for a, b in zip(first.graphs, second.graphs):
        assert a.num_nodes == b.num_nodes
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.allclose(a.features, b.features)
        assert [g.group_id for g in a.groups] == [g.group_id for g in b.groups]
        assert [g.split for g in a.groups] == [g.split for g in b.groups]


def test_multi_graph_sampling_modes():
    from src.pipeline.data import sample_training_graph

    bundle = Data(_config(num_train_graphs=3)).run()
    rng = np.random.default_rng(0)
    drawn = {sample_training_graph(bundle, "sample", rng).graph_id for _ in range(50)}
    assert drawn == {g.graph_id for g in bundle.train_graphs}
    assert sample_training_graph(bundle, "mean", rng) is None
    assert sample_training_graph(bundle, "min", rng) is None
    with pytest.raises(ValueError):
        sample_training_graph(bundle, "median", rng)
