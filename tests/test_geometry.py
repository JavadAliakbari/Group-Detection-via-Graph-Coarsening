"""Screened level: definition, invariance, and agreement with the legacy Gram path."""

from __future__ import annotations

import numpy as np
import torch

from src.pipeline.geometry import (
    group_capture,
    group_confusability_bound,
    screened_level,
)


def test_level_is_the_gram_whitened_screened_response(graph, geometry, level):
    """``ell = D^{-1/2} M_tau Z (G + rho I)^{-1/2}`` up to a right rotation."""

    generator = torch.Generator().manual_seed(0)
    Z = torch.randn(graph.num_nodes, 6, dtype=torch.float64, generator=generator)
    m_z = geometry.m_apply(Z, 0.5)
    nodes = torch.arange(graph.num_nodes)
    raw = m_z / geometry.node_weights(nodes).unsqueeze(1)
    gram = level.gram + level.ridge * torch.eye(6, dtype=torch.float64)
    # a right-rotation-invariant fingerprint of the whitening
    expected = raw @ torch.linalg.inv(gram) @ raw.T
    assert torch.allclose(level.values @ level.values.T, expected, atol=1e-9)


def test_pair_distance_is_the_gram_inverse_distance(graph, geometry):
    generator = torch.Generator().manual_seed(1)
    Z = torch.randn(graph.num_nodes, 6, dtype=torch.float64, generator=generator)
    level = screened_level(Z, geometry, 0.5)
    m_z = geometry.m_apply(Z, 0.5)
    nodes = torch.arange(graph.num_nodes)
    raw = m_z / geometry.node_weights(nodes).unsqueeze(1)
    gram = level.gram + level.ridge * torch.eye(6, dtype=torch.float64)
    a = torch.tensor([0, 5, 9])
    b = torch.tensor([3, 7, 11])
    difference = raw[a] - raw[b]
    expected = (difference @ torch.linalg.inv(gram) * difference).sum(1)
    assert torch.allclose(level.pair_distances(a, b), expected, atol=1e-9)


def test_level_is_invariant_to_an_invertible_right_factor(graph, geometry):
    """``R_Theta`` depends on the subspace only, so ``Z -> Z R`` cannot move it."""

    generator = torch.Generator().manual_seed(2)
    Z = torch.randn(graph.num_nodes, 5, dtype=torch.float64, generator=generator)
    R = torch.randn(5, 5, dtype=torch.float64, generator=generator) + 3 * torch.eye(5)
    sets = [g.nodes for g in graph.train_groups]
    a = screened_level(Z, geometry, 0.5, ridge=0.0, ridge_relative=0.0)
    b = screened_level(Z @ R, geometry, 0.5, ridge=0.0, ridge_relative=0.0)
    assert torch.allclose(
        group_capture(a, geometry, sets), group_capture(b, geometry, sets), atol=1e-8
    )
    assert torch.allclose(
        group_confusability_bound(a, geometry, sets),
        group_confusability_bound(b, geometry, sets),
        atol=1e-8,
    )


def test_capture_matches_the_legacy_collective_gram(graph, geometry):
    """``C_S`` off the level equals ``Gamma_jj`` of the legacy Gram path."""

    from src.run_collective_bank_detection import _collective_gamma, _make_indicators

    generator = torch.Generator().manual_seed(3)
    Z = torch.randn(graph.num_nodes, 5, dtype=torch.float64, generator=generator)
    tau = 0.5
    level = screened_level(Z, geometry, tau, ridge=0.0, ridge_relative=0.0)
    patterns = [g.to_pattern() for g in graph.train_groups]
    _v, m_vhat = _make_indicators(
        graph.a_hat, graph.adjacency, patterns, tau, "geometry", geometry=geometry
    )
    gamma = _collective_gamma(graph.a_hat, Z, m_vhat, 0.0, tau, geometry=geometry)
    legacy = torch.diagonal(gamma)
    ours = group_capture(level, geometry, [g.nodes for g in graph.train_groups])
    assert torch.allclose(ours, legacy, atol=1e-8)


def test_confusability_bound_is_zero_for_the_screened_dual(graph, geometry):
    """``Z* = M_tau^{-1} V`` makes the level constant on every group, so chibar = 0."""

    import scipy.sparse as sp

    from src.raw_ward import screened_operators_kappa

    tau = 0.5
    coalesced = graph.adjacency.coalesce()
    idx = coalesced.indices().numpy()
    val = coalesced.values().numpy()
    n = graph.num_nodes
    W = sp.coo_matrix((val, (idx[0], idx[1])), shape=(n, n)).tocsr()
    _a, _kappa, _l, M = screened_operators_kappa(W, tau, "symmetric")
    sets = [g.nodes for g in graph.train_groups]
    V = geometry.indicator_columns([s.tolist() for s in sets]).numpy()
    Z = torch.from_numpy(np.linalg.solve(M.toarray(), V))
    level = screened_level(Z, geometry, tau, ridge=0.0, ridge_relative=1e-12)
    chi = group_confusability_bound(level, geometry, sets)
    assert float(chi.max()) < 1e-6


def test_rank_reports_the_effective_target_width(graph, geometry):
    generator = torch.Generator().manual_seed(4)
    base = torch.randn(graph.num_nodes, 3, dtype=torch.float64, generator=generator)
    Z = torch.cat([base, base], dim=1)  # rank 3 in a width-6 target
    assert screened_level(Z, geometry, 0.5).rank == 3
