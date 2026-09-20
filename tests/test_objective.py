"""Objective: edge terms, overlap handling, host modes, gamma/ratio, label head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pipeline.data import Graph, Group
from src.pipeline.geometry import screened_geometry, screened_level
from src.pipeline.objective import (
    LabelHead,
    ObjectiveConfig,
    build_edge_sets,
    edge_terms,
    evaluate_objective,
    select_host_nodes,
)


def _line_graph(n: int, groups) -> Graph:
    src = list(range(n - 1))
    dst = list(range(1, n))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return Graph(
        graph_id="line",
        num_nodes=n,
        edge_index=edge_index,
        features=torch.eye(n, dtype=torch.float64),
        node_labels=torch.zeros(n, dtype=torch.long),
        groups=groups,
    )


def test_edge_sets_partition_a_disjoint_group_correctly():
    graph = _line_graph(6, [Group("a", torch.tensor([1, 2, 3]), 1, "train")])
    edges = build_edge_sets(graph.adjacency, [torch.tensor([1, 2, 3])])
    assert edges.counts == {"internal": 2, "boundary": 2}


def test_overlapping_groups_share_edges_instead_of_last_group_winning():
    """An edge internal to one group can be boundary for another; both count."""

    sets = [torch.tensor([0, 1, 2]), torch.tensor([2, 3])]
    graph = _line_graph(
        5, [Group("a", sets[0], 1, "train"), Group("b", sets[1], 1, "train")]
    )
    edges = build_edge_sets(graph.adjacency, sets)
    # group a: internal (0,1),(1,2); boundary (2,3).  group b: internal (2,3); boundary (2,1),(3,4)
    assert edges.counts == {"internal": 3, "boundary": 3}
    assert sorted(edges.int_g.tolist()) == [0, 0, 1]
    assert sorted(edges.bnd_g.tolist()) == [0, 1, 1]


def test_edge_terms_match_a_direct_computation():
    sets = [torch.tensor([1, 2, 3])]
    graph = _line_graph(7, [Group("a", sets[0], 1, "train")])
    geometry = screened_geometry(graph.a_hat, graph.adjacency)
    generator = torch.Generator().manual_seed(0)
    Z = torch.randn(7, 3, dtype=torch.float64, generator=generator)
    level = screened_level(Z, geometry, 0.5)
    edges = build_edge_sets(graph.adjacency, sets)
    boundary, internal = edge_terms(level, edges)

    def distance(u, v):
        return float((level.values[u] - level.values[v]).square().sum())

    assert internal == pytest.approx(np.mean([distance(1, 2), distance(2, 3)]))
    assert boundary == pytest.approx(np.mean([distance(1, 0), distance(3, 4)]))


def test_difference_and_ratio_forms():
    sets = [torch.tensor([1, 2, 3])]
    graph = _line_graph(7, [Group("a", sets[0], 1, "train")])
    geometry = screened_geometry(graph.a_hat, graph.adjacency)
    generator = torch.Generator().manual_seed(1)
    level = screened_level(
        torch.randn(7, 3, dtype=torch.float64, generator=generator), geometry, 0.5
    )
    edges = build_edge_sets(graph.adjacency, sets)

    difference = evaluate_objective(level, edges, ObjectiveConfig(gamma=3.0))
    assert difference.edge_objective == pytest.approx(
        float(difference.boundary) - 3.0 * float(difference.internal)
    )
    assert difference.rho is None

    ratio_config = ObjectiveConfig(gamma=None)
    assert ratio_config.is_ratio
    with pytest.raises(ValueError, match="Dinkelbach rho"):
        evaluate_objective(level, edges, ratio_config)
    linearized = evaluate_objective(level, edges, ratio_config, rho=2.0)
    assert linearized.edge_objective == pytest.approx(
        float(linearized.boundary) - 2.0 * float(linearized.internal)
    )
    assert linearized.ratio == pytest.approx(
        float(linearized.boundary) / float(linearized.internal)
    )


def test_loss_sign_convention_and_components():
    sets = [torch.tensor([1, 2, 3])]
    graph = _line_graph(7, [Group("a", sets[0], 1, "train")])
    geometry = screened_geometry(graph.a_hat, graph.adjacency)
    generator = torch.Generator().manual_seed(2)
    level = screened_level(
        torch.randn(7, 3, dtype=torch.float64, generator=generator), geometry, 0.5
    )
    edges = build_edge_sets(graph.adjacency, sets)
    config = ObjectiveConfig(
        gamma=1.0,
        host_mode="neighbours",
        host_weight=0.5,
        label_enabled=True,
        label_weight=2.0,
    )
    hosts = select_host_nodes(graph.adjacency, sets, config)
    head = LabelHead(level.width, 2)
    targets, mask = graph.node_group_labels("train")
    nodes = torch.nonzero(mask, as_tuple=False).flatten()
    loss = evaluate_objective(
        level,
        edges,
        config,
        host_index=hosts,
        label_head=head,
        label_targets=targets,
        label_nodes=nodes,
    )
    assert float(loss.total) == pytest.approx(
        -float(loss.edge_objective) + 0.5 * float(loss.host) + 2.0 * float(loss.label)
    )
    assert set(loss.to_dict()) == {
        "total_loss",
        "edge_objective",
        "boundary",
        "internal",
        "ratio",
        "host_loss",
        "label_loss",
        "rho",
    }


@pytest.mark.parametrize("mode", ["none", "neighbours", "random"])
def test_all_host_modes(mode):
    sets = [torch.tensor([2, 3, 4])]
    graph = _line_graph(12, [Group("a", sets[0], 1, "train")])
    config = ObjectiveConfig(gamma=1.0, host_mode=mode, host_weight=1.0, host_count=4)
    hosts = select_host_nodes(graph.adjacency, sets, config)
    if mode == "none":
        assert hosts.numel() == 0
        return
    assert hosts.numel() > 0
    assert not set(hosts.tolist()) & set(sets[0].tolist())
    if mode == "neighbours":
        assert sorted(hosts.tolist()) == [1, 5]
    else:
        assert hosts.numel() == 4


def test_host_random_sampling_is_seed_deterministic():
    sets = [torch.tensor([2, 3, 4])]
    graph = _line_graph(30, [Group("a", sets[0], 1, "train")])
    config = ObjectiveConfig(
        gamma=1.0, host_mode="random", host_weight=1.0, host_count=6, seed=7
    )
    first = select_host_nodes(graph.adjacency, sets, config)
    second = select_host_nodes(graph.adjacency, sets, config)
    assert torch.equal(first, second)


def test_label_head_dimensions_and_probabilities():
    head = LabelHead(level_width=5, num_classes=4)
    rows = torch.randn(9, 5, dtype=torch.float64)
    logits = head(rows)
    assert logits.shape == (9, 4)
    probabilities = head.probabilities(rows)
    assert torch.allclose(probabilities.sum(1), torch.ones(9, dtype=torch.float64))
    with pytest.raises(ValueError, match="at least two classes"):
        LabelHead(5, 1)


def test_objective_config_validation():
    with pytest.raises(ValueError, match="host_mode"):
        ObjectiveConfig(host_mode="everything")
    with pytest.raises(ValueError, match="gamma"):
        ObjectiveConfig(gamma=-1.0)
    with pytest.raises(ValueError, match="host_weight"):
        ObjectiveConfig(host_weight=-0.5)


def test_no_negative_group_sampling_in_the_public_api():
    import src.pipeline.objective as module

    names = " ".join(dir(module)).lower()
    assert "negative" not in names
    assert not any("neg_" in field for field in ObjectiveConfig.__dataclass_fields__)
