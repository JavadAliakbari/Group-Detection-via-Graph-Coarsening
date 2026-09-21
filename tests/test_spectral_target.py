"""The static spectral target and the density sweep's paired construction.

The convention under test is the paper's, Sec. "Preliminaries":
``L = I - D~^{-1/2}(W+I)D~^{-1/2}`` with ``0 = lambda_0 <= ... < 2`` and
``U_K = [u_0, ..., u_{K-1}]``, so "top q" means the q *lowest* eigenvectors and
the constant one is kept.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.loukas_sgc_detection import _normalized_laplacian
from src.pipeline.learning import LearningConfig, StaticSpectralLearning, build_learner
from src.pipeline.learning.spectral import _DENSE_MAX_NODES, spectral_basis

TAU = 0.5


def _config(**kw) -> LearningConfig:
    base = dict(
        architecture="static_spectral",
        training_mode="closed_form",
        tau=TAU,
        num_heads=2,
        seed=0,
    )
    base.update(kw)
    return LearningConfig(**base)


# --------------------------------------------------------------------------- #
# the convention
# --------------------------------------------------------------------------- #
def test_the_basis_is_the_lowest_eigenvectors_of_the_normalized_laplacian(graph):
    record = spectral_basis(graph, 12)
    U = record["basis"].numpy()
    L = _normalized_laplacian(graph.adjacency).to_dense().numpy()
    reference = np.linalg.eigvalsh(L)[:12]
    assert record["returned_q"] == 12
    assert np.allclose(record["eigenvalues"], reference, atol=1e-8)
    # each column really is an eigenvector of that L, at that eigenvalue
    for i in range(12):
        residual = L @ U[:, i] - record["eigenvalues"][i] * U[:, i]
        assert np.abs(residual).max() < 1e-8


def test_eigenvalues_come_back_ascending_and_start_at_zero(graph):
    record = spectral_basis(graph, 10)
    values = np.asarray(record["eigenvalues"])
    assert np.all(np.diff(values) >= -1e-10)
    assert abs(values[0]) < 1e-8  # lambda_0 = 0 on a connected component
    assert record["ordering"].startswith("ascending")


def test_the_constant_direction_is_kept(graph):
    """``D~^{1/2} 1`` lies in ``span(U_q)``.

    Stated as a span membership rather than "``u_0`` *is* that vector": the
    fixture graph has two components, so ``lambda_0 = 0`` is degenerate and the
    solver may return any basis of its kernel.  What the convention promises is
    that the constant direction is not discarded, and that holds either way.
    """

    record = spectral_basis(graph, 6)
    assert record["includes_constant_eigenvector"] is True
    U = record["basis"].numpy()
    degrees = torch.sparse.sum(graph.adjacency, dim=1).to_dense().numpy() + 1.0
    constant = np.sqrt(degrees)
    constant = constant / np.linalg.norm(constant)
    residual = constant - U @ (U.T @ constant)
    assert np.linalg.norm(residual) < 1e-8


def test_the_basis_is_euclidean_orthonormal(graph):
    U = spectral_basis(graph, 15)["basis"].numpy()
    assert np.abs(U.T @ U - np.eye(15)).max() < 1e-9


def test_q_is_clamped_and_reported(graph):
    record = spectral_basis(graph, 10_000)
    assert record["requested_q"] == 10_000
    assert record["returned_q"] == int(graph.num_nodes) - 1


def test_the_sparse_and_dense_paths_agree(graph):
    """The routing threshold must not change the subspace, only the cost."""

    import src.pipeline.learning.spectral as spectral

    dense = spectral_basis(graph, 8)
    original = spectral._DENSE_MAX_NODES
    try:
        spectral._DENSE_MAX_NODES = 0  # force Lanczos
        sparse = spectral_basis(graph, 8)
    finally:
        spectral._DENSE_MAX_NODES = original
    assert dense["solver"] == "dense_eigh"
    assert sparse["solver"] == "sparse_shift_invert"
    assert np.allclose(dense["eigenvalues"], sparse["eigenvalues"], atol=1e-8)
    # compare SUBSPACES, not columns: lambda_0 = 0 is degenerate on this graph,
    # so the two solvers may return different bases of the same eigenspace
    a, b = dense["basis"].numpy(), sparse["basis"].numpy()
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    assert np.abs(singular - 1.0).max() < 1e-6


def test_the_default_basis_is_reproducible_column_by_column(graph):
    """Same span is not enough: the coarsener's heap ties see the basis itself.

    The default (dense) path must be bitwise reproducible, including after other
    eigensolves in the same process -- ARPACK's hidden state broke exactly that.
    """

    a = spectral_basis(graph, 10)["basis"].numpy()
    for other in range(3):
        spectral_basis(graph, 5 + other)
    b = spectral_basis(graph, 10)["basis"].numpy()
    assert spectral_basis(graph, 10)["solver"] == "dense_eigh"
    assert np.array_equal(a, b)


def test_the_sparse_path_is_span_reproducible(graph):
    """The sparse path promises the span, not the basis (see the module notes)."""

    import src.pipeline.learning.spectral as spectral

    original = spectral._DENSE_MAX_NODES
    try:
        spectral._DENSE_MAX_NODES = 0
        a = spectral_basis(graph, 10)["basis"].numpy()
        spectral_basis(graph, 7)
        b = spectral_basis(graph, 10)["basis"].numpy()
    finally:
        spectral._DENSE_MAX_NODES = original
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    assert np.abs(singular - 1.0).max() < 1e-8


def test_the_sparse_path_finds_the_true_lowest_set_in_a_clustered_spectrum():
    """The regression that motivated shift-invert.

    ``eigsh(A_hat, which="LA")`` returned a wrong eigenvalue set (error 0.10)
    on a 3,000-node planted graph with q = 256.  A graph with many small planted
    groups has exactly that clustered low end; the sparse path must reproduce
    the dense spectrum on it.
    """

    import src.pipeline.learning.spectral as spectral
    from src.pipeline.data import DataConfig, build_synthetic_graph

    config = DataConfig(
        source="synthetic", num_nodes=900, num_groups=30, group_size=[7, 12],
        group_density=0.5, avg_degree=4.0, feature_dim=4, seed=3,
    )
    graph = build_synthetic_graph(config, "C", 3)
    L = _normalized_laplacian(graph.adjacency).to_dense().numpy()
    reference = np.linalg.eigvalsh(L)[:120]
    original = spectral._DENSE_MAX_NODES
    try:
        spectral._DENSE_MAX_NODES = 0
        record = spectral_basis(graph, 120)
    finally:
        spectral._DENSE_MAX_NODES = original
    assert record["solver"] == "sparse_shift_invert"
    assert np.abs(np.asarray(record["eigenvalues"]) - reference).max() < 1e-9


assert _DENSE_MAX_NODES > 0  # the module ships with the dense path reachable


# --------------------------------------------------------------------------- #
# the learner
# --------------------------------------------------------------------------- #
def test_the_target_is_a_function_of_the_graph_alone(small_config):
    """Relabelling every group cannot change the subspace.

    Built from its own bundle rather than the shared fixture, so flipping the
    splits cannot leak into another test.
    """

    from dataclasses import replace

    from src.pipeline.data import Data

    graph = Data(small_config).run().train_graphs[0]
    learner = StaticSpectralLearning(_config())
    learner.build_model(graph.feature_dim, torch.float64)
    before = learner.encode(None, graph, None).clone()
    learner.spectra_.clear()
    graph.groups = [replace(g, split="test") for g in graph.groups]
    after = learner.encode(None, graph, None)
    assert torch.allclose(before, after)
    assert {g.split for g in graph.groups} == {"test"}


def test_widths_are_matched_per_graph(bundle):
    widths = {g.graph_id: 7 for g in bundle.train_graphs}
    learner = StaticSpectralLearning(_config(), target_widths=widths)
    learner.build_model(bundle.train_graphs[0].feature_dim, torch.float64)
    for graph in bundle.train_graphs:
        assert learner.encode(None, graph, None).shape[1] == 7
    # a graph outside the map falls back to the bank's width, not to a crash
    other = bundle.test_graphs[0]
    assert learner.encode(None, other, None).shape[1] == 2 * other.feature_dim


def test_each_graph_gets_its_own_subspace(bundle):
    learner = StaticSpectralLearning(_config())
    learner.build_model(bundle.train_graphs[0].feature_dim, torch.float64)
    a = learner.encode(None, bundle.train_graphs[0], None)
    b = learner.encode(None, bundle.train_graphs[1], None)
    assert not torch.allclose(a, b)
    assert set(learner.spectra_) == {
        bundle.train_graphs[0].graph_id,
        bundle.train_graphs[1].graph_id,
    }


def test_the_registry_builds_it_and_the_config_validates(graph):
    learner = build_learner(_config())
    assert isinstance(learner, StaticSpectralLearning)
    with pytest.raises(ValueError, match="nothing to train by gradient"):
        _config(training_mode="gradient")


def test_a_closed_form_report_is_produced_without_fitting_anything(bundle):
    learner = StaticSpectralLearning(_config())
    result = learner.run(bundle.train_graphs)
    report = result.closed_form
    assert report["solver"] == "static_spectral"
    assert report["learned_parameters"] == 0
    assert set(report["per_graph"]) == {g.graph_id for g in bundle.train_graphs}
    assert all(
        record["operator"].startswith("L_sym") for record in report["per_graph"].values()
    )


# --------------------------------------------------------------------------- #
# the sweep's own invariants
# --------------------------------------------------------------------------- #
def test_the_density_grid_is_the_specified_one():
    from src.run_synthetic_density_sweep import DENSITIES, METHOD, SUBSPACES

    assert len(DENSITIES) == 21
    assert DENSITIES[0] == 0.0 and DENSITIES[-1] == 1.0
    assert all(
        abs((DENSITIES[i + 1] - DENSITIES[i]) - 0.05) < 1e-9
        for i in range(len(DENSITIES) - 1)
    )
    assert METHOD == "deflated_ward_tight"
    assert SUBSPACES == ("learned", "static_spectral")


def test_only_the_density_the_seeds_and_the_architecture_change():
    from src.run_synthetic_density_sweep import _fixed_fingerprint, build_config
    from pathlib import Path

    a = build_config(0.0, 1, Path("x"), subspace="learned")
    b = build_config(1.0, 9, Path("y"), subspace="static_spectral")
    assert a.coarsening.method == b.coarsening.method == "deflated_ward_tight"
    assert a.data.group_density == 0.0 and b.data.group_density == 1.0
    assert _fixed_fingerprint(a) == _fixed_fingerprint(b)


def test_density_zero_still_plants_a_spanning_path():
    """The sweep's floor caveat, pinned: density 0 is a path, not an empty group."""

    from dataclasses import replace

    from src.pipeline.data import DataConfig, build_synthetic_graph

    config = DataConfig(
        source="synthetic", num_nodes=300, num_groups=4, group_size=10,
        group_density=0.0, avg_degree=4.0, feature_dim=4, seed=5,
    )
    graph = build_synthetic_graph(config, "G", 5)
    edges = graph.edge_index.numpy()
    upper = edges[0] < edges[1]
    for group in graph.groups:
        members = set(int(v) for v in group.nodes)
        inside = sum(
            1
            for u, v in zip(edges[0][upper], edges[1][upper])
            if int(u) in members and int(v) in members
        )
        assert inside >= len(members) - 1  # at least the spanning path
