"""Controlled collective learnable-filter (``M_tau``) gang-detection experiment.

This is an end-to-end, *controlled* implementation of the collective detect-all
objective of Section "Collective Detection":

    Theta* = argmax_{||theta^(a)||=1 for all a}  lambda_min(Gamma(Theta)),
    Gamma(Theta) = Vhat^T M Z (Z^T M Z)^+ Z^T M Vhat  in R^{m x m},   M = M_tau = L + tau*I,
    Z[:, a] = g_{theta^(a)}(A_hat) x_a = sum_k Theta_{ka} A_hat^k x_a  (one filter
                                                                       per channel),

with everything measured in the *screened* metric ``M_tau = L + tau*I`` -- the
one-parameter family that interpolates between the ``L_sym`` seminorm at
``tau = 0`` (Loukas' analysis; ``L = I - A_hat``) and the plain ``l2`` metric as
``tau -> inf``, and is positive *definite* for every ``tau > 0``.  ``Vhat`` are the
``M_tau``-normalized *degree-weighted* gang indicators
``v_S = D_tilde^{1/2} 1_S / sqrt(vol(S))``, so ``||v_S||_L^2 = Phi = cut/vol`` and
``||v_S||_{M_tau}^2 = Phi + tau``; the screened channel Gram factorizes as
``Z^T M_tau Z = Z^T L Z + tau*Z^T Z`` (screening = a Tikhonov ridge on the Gram).
Pass ``--tau`` (a single value or a comma-separated list, e.g. ``0,0.1,0.3,1.0``)
to switch metric and compare detection across ``tau``.

Pipeline (the seven requested steps):

1. build a random (Erdos-Renyi) background graph with ``--num-nodes`` nodes;
2. plant ``--num-motifs`` dense motifs of a chosen ``--motif-type``
   (``clique`` / ``cycle`` / ``star`` / ``random``, the last with a tunable
   ``--motif-density``);
3. hold out a fraction, training on ``--train-ratio`` of the motifs;
4. learn the per-channel filter bank ``Theta`` by ascending
   ``lambda_min(Gamma(Theta))`` on the *training* motifs only;
5. form the embedding ``Z = g_Theta(A_hat) X`` and the target subspace
   ``R = span(Z)`` from the learned filters (``span(Z)`` is itself
   ``tau``-independent, but the learned ``Theta*`` -- and hence ``Z`` -- is not);
6. hand ``R`` to the Loukas RSA coarsening (reusing the existing
   :func:`loukas_coarsen_pytorch`);
7. report post-coarsening recall / precision / detection rate (reusing
   :func:`evaluate_loukas_patterns`), broken out over train / test / all motifs.

Note on the coarsening metric.  The Loukas coarsening historically measured RSA
distortion in the *combinatorial* Laplacian ``L = D - W``; the algorithm above is
derived in the *symmetric normalized* ``L = I - A_hat``.  Both are exposed through
``--coarsening-laplacian`` (default ``symmetric``).  The screening ``tau`` enters
the *learning* objective and the capture diagnostic (where the ``L_sym`` norm used
to live); it flows into detection through the learned filter ``Theta*(tau)``.

Run, e.g.::

    python -m src.run_collective_bank_detection \
        --motif-type clique --num-motifs 10 --num-nodes 2000 \
        --train-ratio 0.4 --degree 10 --feature-dim 64 --reduction 0.85 \
        --tau 0,0.1,0.3,1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from src.utils.utils import *
from src.sgc_detection import (
    chebyshev_stack,
    fit_collective_sgc,
    propagation_stack,
)
from src.loukas_sgc_detection import (
    LoukasCoarseningResult,
    _degrees,
    _exact_rsa_epsilon,
    _l_orthonormalize,
    _laplacian,
    _normalized_laplacian,
    _screened_metric,
    build_laplacian_subspace,
    build_sgc_subspace,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.pattern_models import create_pattern
from src.bank_visualize import save_rich_plots
from src.propagation_encoders import GCN2Encoder, fit_encoder

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def random_structural_features(num_nodes: int, width: int, seed: int) -> torch.Tensor:
    """Isotropic random range-finder ``Omega`` (the structural feature channel)."""

    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(num_nodes, width, dtype=torch.float64, generator=gen)
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)


# --------------------------------------------------------------------------- #
# 1-2.  synthetic random graph with planted dense motifs
# --------------------------------------------------------------------------- #
def _motif_edges(
    nodes: list[int],
    motif_type: str,
    *,
    density: float = 1.0,
    rng: "np.random.Generator | None" = None,
) -> list[tuple[int, int]]:
    """Return the undirected edge list of one planted motif on ``nodes``.

    ``motif_type="random"`` plants an Erdos-Renyi ``G(s, density)`` subgraph: a
    random spanning path is always included first (so the motif is connected /
    a valid single gang), then extra random pairs are added until the edge count
    reaches ``round(density * s(s-1)/2)``.  ``density=1.0`` reproduces the clique.
    """

    s = len(nodes)
    if motif_type == "clique":
        return [(int(u), int(v)) for u, v in combinations(nodes, 2)]
    if motif_type == "cycle":
        return [(int(nodes[i]), int(nodes[(i + 1) % s])) for i in range(s)]
    if motif_type == "star":
        hub = int(nodes[0])
        return [(hub, int(nodes[i])) for i in range(1, s)]
    if motif_type == "random":
        if rng is None:
            raise ValueError("random motif requires a numpy Generator via `rng`")
        if not 0.0 <= density <= 1.0:
            raise ValueError("--motif-density must be in [0, 1]")
        perm = [int(nodes[i]) for i in rng.permutation(s)]
        motif: set[tuple[int, int]] = set()
        for i in range(s - 1):  # spanning path -> guaranteed connected
            a, b = perm[i], perm[i + 1]
            motif.add((min(a, b), max(a, b)))
        max_edges = s * (s - 1) // 2
        target = max(len(motif), int(round(float(density) * max_edges)))
        pairs = [
            (int(u), int(v)) for u, v in combinations(sorted(int(n) for n in nodes), 2)
        ]
        for idx in rng.permutation(len(pairs)):
            if len(motif) >= target:
                break
            u, v = pairs[idx]
            motif.add((min(u, v), max(u, v)))
        return sorted(motif)
    raise ValueError("motif_type must be 'clique', 'cycle', 'star', or 'random'")


def build_synthetic_graph(
    *,
    num_nodes: int,
    num_motifs: int,
    motif_type: str,
    motif_size: int,
    avg_degree: float,
    feature_dim: int,
    rng_seed: int,
    motif_density: float = 1.0,
    motif_conductance: float = -1.0,
) -> tuple[Data, list]:
    """Erdos-Renyi background + ``num_motifs`` disjoint planted motifs.

    Returns the graph (``x``, ``edge_index``, ``y``) and the list of
    :class:`Pattern` objects (label ``"alert"``) for evaluation.  Node labels
    ``y`` mark every motif node as class 1 so the coarsening evaluation can pool
    pseudo-labels.

    ``motif_conductance`` (when ``>= 0``) tunes each motif's conductance
    ``Phi = cut/vol`` by adding random motif->host edges; a negative value leaves
    the motifs as planted (disjoint, minimal conductance).
    """

    if num_motifs * motif_size > num_nodes:
        raise ValueError("num_motifs * motif_size exceeds num_nodes")

    rng = np.random.default_rng(rng_seed)

    # --- disjoint node blocks for the motifs ---------------------------------
    perm = rng.permutation(num_nodes)
    motif_nodes = perm[: num_motifs * motif_size].reshape(num_motifs, motif_size)

    edges: set[tuple[int, int]] = set()

    # --- Erdos-Renyi background ---------------------------------------------
    n_background = int(num_nodes * avg_degree / 2)
    src = rng.integers(0, num_nodes, size=n_background)
    dst = rng.integers(0, num_nodes, size=n_background)
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            edges.add((min(u, v), max(u, v)))

    # --- planted motifs ------------------------------------------------------
    patterns = []
    y = np.zeros(num_nodes, dtype=np.int64)
    for m in range(num_motifs):
        nodes = motif_nodes[m].tolist()
        for u, v in _motif_edges(nodes, motif_type, density=motif_density, rng=rng):
            edges.add((min(u, v), max(u, v)))
        y[nodes] = 1
        patterns.append(
            create_pattern(f"{motif_type}_{m}", nodes, motif_type, label="alert")
        )

    # --- optional: tune each motif's conductance by wiring it to host nodes ---
    # Phi(S) = cut(S)/vol(S).  Each added random motif->host edge raises both cut
    # and vol by 1, so to hit a target Phi we solve k = (Phi*vol - cut)/(1 - Phi)
    # from the motif's *current* (background + internal) cut/vol.  A negative
    # target leaves the motifs as planted (disjoint, minimal conductance).
    if motif_conductance is not None and motif_conductance >= 0.0:
        if motif_conductance >= 1.0:
            raise ValueError("motif_conductance must be < 1 (a conductance in [0, 1))")
        motif_of = -np.ones(num_nodes, dtype=np.int64)
        for m in range(num_motifs):
            motif_of[motif_nodes[m]] = m
        host_nodes = np.nonzero(motif_of < 0)[0]
        if host_nodes.size == 0:
            raise ValueError("no host nodes available to tune motif conductance")
        cut = np.zeros(num_motifs, dtype=np.int64)
        vol = np.zeros(num_motifs, dtype=np.int64)
        for u, v in edges:
            mu, mv = motif_of[u], motif_of[v]
            if mu >= 0:
                vol[mu] += 1
            if mv >= 0:
                vol[mv] += 1
            if mu != mv:
                if mu >= 0:
                    cut[mu] += 1
                if mv >= 0:
                    cut[mv] += 1
        for m in range(num_motifs):
            k = int(
                round((motif_conductance * vol[m] - cut[m]) / (1.0 - motif_conductance))
            )
            if k <= 0:  # already at/above target -- we only add, never cut
                continue
            nodes = motif_nodes[m]
            added, attempts, cap = 0, 0, 20 * k + 100
            while added < k and attempts < cap:
                attempts += 1
                u = int(rng.choice(nodes))
                w = int(rng.choice(host_nodes))
                e = (min(u, w), max(u, w))
                if e in edges:
                    continue
                edges.add(e)
                added += 1

    # --- undirected edge_index ----------------------------------------------
    edge_array = np.array(sorted(edges), dtype=np.int64).T  # (2, E)
    edge_index = torch.from_numpy(
        np.concatenate([edge_array, edge_array[::-1]], axis=1)
    ).long()

    # --- isotropic node features (the reachability channel of the theory) ----
    gen = torch.Generator().manual_seed(rng_seed)
    X = torch.randn(num_nodes, feature_dim, dtype=torch.float64, generator=gen)
    X = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)

    graph = Data(
        x=X,
        edge_index=edge_index,
        y=torch.from_numpy(y),
        num_nodes=num_nodes,
    )
    return graph, patterns


def inject_gang_features(
    X: torch.Tensor,
    patterns: list,
    *,
    shared: float = 0.0,
    signature: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """Add *class-consistent, block-aligned* feature structure on the gang nodes.

    Implements the "when do features help" analysis (Theorem 5.6 / Prop 6.8): the
    capture ceiling is *reachability*, so features raise ``C_S^tau`` exactly when
    they carry a component *along* ``v_S`` (roughly constant inside the gang and
    offset from the host).  Two additive block-constant components are injected on
    each gang's member nodes:

    * ``shared`` -- a single unit direction ``u_shared`` common to *every* gang (a
      class-consistent "gangness" signature).  This is what generalizes train->test
      and drives *union* detection / the node head (Q2): the same statistic recurs
      on held-out gangs.
    * ``signature`` -- a *fresh* random unit direction per gang.  Distinct
      per-gang signatures give the *collective* objective linearly independent
      feature signatures within the shared conductance band, lifting
      ``rank(Psi|_c)`` so ``lambda_min(Gamma)`` does not collapse (Prop 6.8).

    Host nodes are untouched (mean 0), so the injected mass is offset from the
    host.  ``shared = signature = 0`` returns ``X`` unchanged (isotropic baseline).
    """

    if shared <= 0.0 and signature <= 0.0:
        return X
    N, d = X.shape
    gen = torch.Generator().manual_seed(int(seed) + 777)
    Xg = X.clone()
    u_shared = torch.randn(d, generator=gen, dtype=X.dtype)
    u_shared = u_shared / u_shared.norm().clamp_min(1e-12)
    for p in patterns:
        idx = torch.as_tensor(p.node_indices, dtype=torch.long)
        offset = torch.zeros(d, dtype=X.dtype)
        if shared > 0.0:
            offset = offset + shared * u_shared
        if signature > 0.0:
            sig = torch.randn(d, generator=gen, dtype=X.dtype)
            sig = sig / sig.norm().clamp_min(1e-12)
            offset = offset + signature * sig
        Xg[idx] = Xg[idx] + offset.unsqueeze(0)
    return Xg


def make_negative_sampler(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    num_sets: int,
    size_min: int,
    size_max: int,
    avoid: "list[int] | np.ndarray",
    rng: "np.random.Generator",
) -> "callable":
    """Build a resampler that draws a *fresh* batch of negative sets on each call.

    Each negative "repeller" set grows from a random seed by random neighbor
    accretion (a shuffled-frontier BFS) up to a random size in
    ``[size_min, size_max]``.  Seeds and grown nodes avoid ``avoid`` (the planted
    motif nodes) so the negatives are genuine background neighborhoods -- random
    sets of neighboring nodes of random size -- rather than gangs.

    Neighbor lists and the seed-candidate pool are precomputed once, so the
    returned ``sample()`` closure is cheap to call every epoch: resampling the
    negatives each step stops the filter bank from overfitting one fixed batch.
    """

    if not 2 <= size_min <= size_max:
        raise ValueError("require 2 <= --neg-size-min <= --neg-size-max")

    neighbors: "dict[int, list[int]]" = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        neighbors[u].append(v)
    avoid_set = {
        int(a) for a in (avoid.tolist() if hasattr(avoid, "tolist") else avoid)
    }
    candidates = [n for n in range(num_nodes) if n not in avoid_set and neighbors[n]]

    def sample() -> list[list[int]]:
        if num_sets <= 0 or not candidates:
            return []
        sets: list[list[int]] = []
        for _ in range(num_sets):
            size = int(rng.integers(size_min, size_max + 1))
            seed = int(rng.choice(candidates))
            blob, seen = [seed], {seed}
            frontier = [w for w in neighbors[seed] if w not in avoid_set]
            rng.shuffle(frontier)
            while len(blob) < size and frontier:
                nxt = int(frontier.pop())
                if nxt in seen or nxt in avoid_set:
                    continue
                blob.append(nxt)
                seen.add(nxt)
                extra = [
                    w for w in neighbors[nxt] if w not in seen and w not in avoid_set
                ]
                rng.shuffle(extra)
                frontier.extend(extra)
            if len(blob) >= 2:
                sets.append(blob)
        return sets

    return sample


# --------------------------------------------------------------------------- #
# 3-4.  collective L_sym filter-bank learning
# --------------------------------------------------------------------------- #
def _geometry_or_symmetric(geometry, a_hat: torch.Tensor, adjacency: torch.Tensor):
    """The caller's :class:`~src.screened_geometry.ScreenedGeometry`, or the default.

    Every geometry-aware helper below takes ``geometry=None`` and funnels through
    here, so omitting it reproduces the historical symmetric behaviour
    (``L = I - A_hat``, degree-weighted ``v_S``, propagation on ``A_hat``)
    bit-for-bit -- the combinatorial convention is opt-in and never leaks into a
    call site that did not ask for it.
    """

    if geometry is not None:
        return geometry
    from src.screened_geometry import symmetric_geometry

    return symmetric_geometry(a_hat, adjacency)


def _degree_weighted_columns(
    adjacency: torch.Tensor, node_sets: "list"
) -> torch.Tensor:
    """Degree-weighted indicators ``v_S = D_tilde^{1/2} 1_S / sqrt(vol(S))``.

    ``D_tilde = D + I`` matches the self-loop renormalization of ``A_hat``, so
    ``||v_S||_L^2 = Phi(S)`` under ``L = I - A_hat``.  ``node_sets`` is any list of
    node-index sequences (planted gangs or sampled negatives).

    This is the *symmetric* convention, hard-coded.  Geometry-aware code should
    call :meth:`src.screened_geometry.ScreenedGeometry.indicator_columns`
    instead, which returns this under ``kind="symmetric"`` and the uniform
    ``v_S = 1_S/sqrt(|S|)`` under ``kind="combinatorial"``.
    """

    n = adjacency.shape[0]
    dtype, device = adjacency.dtype, adjacency.device
    d_tilde = _degrees(adjacency) + 1.0  # self-loop augmented degree
    columns = []
    for nodes in node_sets:
        nodes = torch.as_tensor(nodes, dtype=torch.long, device=device)
        column = torch.zeros(n, dtype=dtype, device=device)
        column[nodes] = d_tilde[nodes].sqrt()
        column = column / d_tilde[nodes].sum().clamp_min(torch.finfo(dtype).eps).sqrt()
        columns.append(column)
    return torch.stack(columns, dim=1)  # (N, m)


def degree_weighted_indicators(adjacency: torch.Tensor, patterns: list) -> torch.Tensor:
    """Degree-weighted gang indicators for the evaluation :class:`Pattern` list."""

    return _degree_weighted_columns(adjacency, [p.node_indices for p in patterns])


def _l_apply(a_hat: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
    """Apply ``L = I - A_hat`` to dense ``signals`` (columns are graph signals).

    The *symmetric* convention, hard-coded; geometry-aware code calls
    :meth:`src.screened_geometry.ScreenedGeometry.l_apply` instead.
    """

    return signals - torch.sparse.mm(a_hat, signals)


def _m_apply(a_hat: torch.Tensor, signals: torch.Tensor, tau: float) -> torch.Tensor:
    """Apply the screened metric ``M_tau = L + tau*I = (I - A_hat) + tau*I``.

    ``tau = 0`` recovers the ``L_sym`` seminorm; ``tau > 0`` makes the metric
    positive *definite* (a true norm) and adds ``tau*||x||_2^2`` of within-supernode
    (l2) energy to every inner product, per the screened-metric family
    ``||x||_{M_tau}^2 = ||x||_L^2 + tau*||x||_2^2``.

    The *symmetric* convention, hard-coded; geometry-aware code calls
    :meth:`src.screened_geometry.ScreenedGeometry.m_apply` instead.
    """

    out = _l_apply(a_hat, signals)
    return out + tau * signals if tau else out


def as_multihead(theta: torch.Tensor) -> torch.Tensor:
    """Canonical filter shape ``(H, K+1, d)``; a legacy ``(K+1, d)`` becomes ``H=1``."""

    return theta.unsqueeze(0) if theta.dim() == 2 else theta


def theta_degree(theta: torch.Tensor) -> int:
    """Polynomial degree ``K`` of a filter in either layout."""

    return as_multihead(theta).shape[1] - 1


def _filtered_bank(propagated: list[torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
    """Filter bank of ``H`` heads: ``Z = [Z^(1) | ... | Z^(H)]``, shape ``(N, H*d)``.

    Head ``h`` contributes ``Z^(h)[:, a] = sum_k theta[h, k, a] (phi_k(A_hat) X)[:, a]``
    -- one hop-profile per (head, feature channel).  ``H = 1`` (or a legacy
    ``(K+1, d)`` filter) reproduces the single shared bank exactly, so every
    downstream consumer is unchanged in that case.

    Multiple heads matter because the coarsening target is the *span* of these
    columns: with one head all gangs must share one hop-profile per channel and
    compete; with ``H`` heads each gang can align with whichever head suits its
    motif, and the projection picks the combination per gang for free.  Duplicate
    heads add no new directions to the span, so the objective has no incentive to
    keep them equal -- see ``head_diversity`` in :func:`fit_collective_bank`.
    """

    th = as_multihead(theta)
    heads = [
        sum(propagated[k] * th[h, k].unsqueeze(0) for k in range(th.shape[1]))
        for h in range(th.shape[0])
    ]
    return heads[0] if len(heads) == 1 else torch.cat(heads, dim=1)


def lanczos_stack(
    a_hat: torch.Tensor, X: torch.Tensor, degree: int, tau: float, geometry=None
) -> list[torch.Tensor]:
    """Per-channel ``M_tau``-orthonormal Krylov basis ``q_k = p_k(A_hat) x_j``.

    The *instance-optimal* dictionary: ``{p_k}`` are the orthogonal polynomials of
    the actual ``M_tau``-weighted spectral density of ``(A_hat, x_j)``, so the
    per-channel Gram ``<q_k, q_l>_{M_tau} = delta_kl`` is the identity **by
    construction** (contrast Chebyshev, which is orthogonal against the arcsine
    weight, i.e. optimal only for the graph whose density happens to be arcsine).

    This is a genuine polynomial basis -- not merely an orthogonalized stack --
    because ``A_hat`` is self-adjoint in the ``M_tau`` inner product: ``M_tau =
    (1 + tau) I - A_hat`` is itself a polynomial in ``A_hat``, so the two commute
    and the three-term Lanczos recurrence applies.  Each ``q_k`` is therefore
    exactly ``p_k(A_hat) x_j`` with ``deg p_k = k``, and ``span{q_0..q_K}`` is the
    Krylov space ``span{x_j, A_hat x_j, ..., A_hat^K x_j}`` -- the *same* span as
    the monomial and Chebyshev stacks (Lemma 6.1), so this is a drop-in third
    basis that changes only conditioning, not the reachable filter set.

    Runs the ``d`` channels simultaneously (every op is a shared sparse mat-mul
    plus per-column elementwise work) with **full reorthogonalization** against the
    cached ``M_tau q_i`` -- plain three-term Lanczos loses orthogonality within a
    few steps at these degrees, which would silently destroy the identity Gram
    that is the whole point.  A channel whose Krylov space is exhausted
    (``beta ~ 0``, e.g. a signal lying in a small invariant subspace) yields zero
    vectors from that order on, correctly telling the filter those orders carry no
    energy rather than amplifying numerical noise.
    """

    eps = torch.finfo(X.dtype).eps
    tol = eps**0.5

    def _unit(v, mv):
        """Normalize each column to unit ``M_tau`` norm; zero out dead channels."""
        nrm = (v * mv).sum(0).clamp_min(0.0).sqrt()
        inv = torch.where(nrm > tol, 1.0 / nrm.clamp_min(eps), torch.zeros_like(nrm))
        return v * inv.unsqueeze(0), mv * inv.unsqueeze(0), nrm

    geo = _geometry_or_symmetric(geometry, a_hat, None)
    q, mq, _ = _unit(X, geo.m_apply(X, tau))
    Q, MQ = [q], [mq]
    beta = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
    for k in range(degree):
        w = torch.sparse.mm(a_hat, Q[-1])  # A_hat q_k
        w = w - (w * MQ[-1]).sum(0).unsqueeze(0) * Q[-1]  # - alpha_k q_k
        if k:
            w = w - beta.unsqueeze(0) * Q[-2]  # - beta_k q_{k-1}
        for Qi, MQi in zip(Q, MQ):  # full reorthogonalization (cached M_tau q_i)
            w = w - (w * MQi).sum(0).unsqueeze(0) * Qi
        q, mq, beta = _unit(w, geo.m_apply(w, tau))
        Q.append(q)
        MQ.append(mq)
    return Q


def _basis_stack(
    a_hat: torch.Tensor,
    X: torch.Tensor,
    degree: int,
    basis: str,
    tau: float = 0.0,
    geometry=None,
) -> list[torch.Tensor]:
    """Dictionary the filter bank is built on: monomial ``A_hat^k X`` or Chebyshev.

    ``basis="chebyshev"`` returns ``[T_k(A_hat) X]_{k=0..K}`` (paper eq. 30, the
    well-conditioned bank of Section 6.2); ``basis="monomial"`` returns the legacy
    ``[A_hat^k X]_{k=0..K}``.  Both span the same subspace (Lemma 6.1), so the
    learned filter ``theta``, the collective Gram ``Gamma``, and the whole
    detection path are identical in exact arithmetic -- only the conditioning of
    the coefficient solve differs (Prop 6.3).

    ``basis="lanczos"`` (:func:`lanczos_stack`) is the third, *instance-optimal*
    option.  Chebyshev is *minimax*-optimal, not optimal for this graph: it
    orthogonalizes against the arcsine weight on ``[-1, 1]``, whereas the quantity
    that actually conditions the solve is the ``M_tau``-weighted spectral density
    of ``(A_hat, X)``.  The genuinely optimal dictionary is the sequence of
    orthogonal polynomials w.r.t. *that* measure -- exactly what Lanczos on
    ``(A_hat, X)`` computes -- which makes the channel Gram ``Z^T M_tau Z`` the
    identity by construction and so dominates Chebyshev on conditioning whenever
    the empirical density is far from arcsine (on heavy-tailed hosts the mass
    concentrates near ``lambda ~ 1``, which is precisely that regime).  The price
    is that the basis becomes graph-dependent:
    the explicit witness polynomials and the sharp *universal* leakage constants
    of Section 6.2 are stated for a fixed, graph-independent dictionary and do not
    survive re-derivation against a per-graph measure, and the learned ``theta``
    is no longer comparable across graphs (or across days, for the frozen-filter
    transfer this code relies on).  So the two are a deliberate trade: Chebyshev
    is the minimax-robust choice with transferable coefficients and universal
    guarantees; Lanczos is the instance-optimal one with a perfectly conditioned
    Gram but graph-specific coefficients and no universal constants.  We take the
    robust side because the transfer experiments freeze ``theta`` across days.
    """

    if basis == "chebyshev":
        return chebyshev_stack(a_hat, X, degree)
    if basis == "monomial":
        return propagation_stack(a_hat, X, degree)
    if basis == "lanczos":
        return lanczos_stack(a_hat, X, degree, tau, geometry=geometry)
    raise ValueError("basis must be 'chebyshev', 'monomial' or 'lanczos'")


def _collective_gamma(
    a_hat: torch.Tensor,
    Z: torch.Tensor,
    m_vhat: torch.Tensor,
    ridge: float,
    tau: float,
    geometry=None,
) -> torch.Tensor:
    """Collective ``M_tau``-Gram ``Gamma = Vhat^T M Z (Z^T M Z)^+ Z^T M Vhat``.

    ``m_vhat = M_tau Vhat`` is precomputed (independent of ``theta``).  The channel
    Gram ``Z^T M_tau Z = Z^T L Z + tau*Z^T Z`` is the ``tau=0`` Gram plus ``tau``
    times the plain dictionary Gram -- a structured Tikhonov ridge that stabilizes
    the (otherwise Hankel-ill-conditioned) solve, exactly the paper's
    ``screening is a ridge on the Gram``.  A small ``ridge`` adds a further guard.
    """

    geo = _geometry_or_symmetric(geometry, a_hat, None)
    m_z = geo.m_apply(Z, tau)  # M_tau Z          (N, d)
    return _collective_gamma_mz(Z, m_z, m_vhat, ridge * geo.ridge_scale)


def _collective_gamma_mz(
    Z: torch.Tensor, m_z: torch.Tensor, m_vhat: torch.Tensor, ridge: float
) -> torch.Tensor:
    """``Gamma`` with ``m_z = M_tau Z`` supplied (no sparse mat-vec).

    ``M_tau Z`` is linear in ``theta`` (``M_tau`` and the filter both commute with
    ``A_hat``), so the training loop precomputes the screened dictionary stack
    ``[M_tau phi_k(A_hat) X]`` once and rebuilds ``m_z`` per epoch by an elementwise
    filter combine -- turning the previous per-epoch sparse mat-vec into a cheap
    dense contraction (:func:`_collective_gamma` still offers the standalone form).
    """

    g_z = Z.T @ m_z  # Z^T M_tau Z                    (d, d)
    g_z = 0.5 * (g_z + g_z.T)
    m = Z.T @ m_vhat  # Z^T M_tau Vhat                (d, m)
    eye = torch.eye(g_z.shape[0], dtype=g_z.dtype, device=g_z.device)
    g_inv_m = torch.linalg.solve(g_z + ridge * eye, m)  # (d, m)
    gamma = m.T @ g_inv_m  # (m, m)
    return 0.5 * (gamma + gamma.T)


def _soft_lambda_max(
    gamma: torch.Tensor, temperature: float, *, sharpen: bool = False
) -> torch.Tensor:
    """Smooth surrogate for ``lambda_max(gamma)``: ``sum_i softmax(lam/tau)_i lam_i``.

    Differentiable and ``-> lambda_max`` as ``temperature -> 0``.  Penalizing it
    for the *negative* sets drives every negative neighborhood's retained
    ``L``-energy toward zero (none is preserved by ``R``), so the RSA coarsening
    pulls those sets apart.

    ``temperature`` is **relative to the top eigenvalue** (``tau_eff = temperature
    * lambda_max``): the softmax weights decay by ``e`` per ``temperature``
    fraction of the peak, so the surrogate stays a genuine soft-*max* regardless
    of the absolute eigenvalue scale.  With a fixed *absolute* temperature the
    weights collapse to uniform (a plain *mean*, with a weak diffuse gradient)
    whenever the retained energies are tiny -- exactly the regime the negatives
    live in -- which is why ``sharpen=False`` (mean-like) is the calmer default
    and ``sharpen=True`` targets the single worst negative blob.
    """

    eigs = torch.linalg.eigvalsh(gamma)
    if sharpen:
        top = eigs[-1].detach().clamp_min(1e-12)  # eigvalsh is ascending
        tau = max(float(temperature), 1e-8) * top
    else:
        tau = max(float(temperature), 1e-8)
    weights = torch.softmax((eigs - eigs[-1]) / tau, dim=0)
    return (weights * eigs).sum()


def _soft_min_values(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Boltzmann soft-min over a plain vector (used on ``Gamma``'s *diagonal*).

    :func:`_soft_lambda_min` soft-mins the *eigenvalues* of ``Gamma``, which couples
    capture to cross-gang separation.  This one soft-mins whatever vector it is
    handed -- e.g. the per-gang captures ``Gamma_jj`` -- so the objective can ask for
    "every gang well captured" without also demanding the gangs be mutually
    independent (see ``capture_objective="softmin_diag"``).
    """

    if temperature <= 0.0:
        return values.min()
    t = max(float(temperature), 1e-8)
    weights = torch.softmax((values.min().detach() - values) / t, dim=0)
    return (weights * values).sum()


def _capture_objective(
    gamma: torch.Tensor, kind: str, temperature: float
) -> torch.Tensor:
    """The quantity the bank ascends.

    * ``lambda_min``   -- ``lambda_min(Gamma)``: capture of the hardest gang *after
      discounting alignment with the others*.  Enforces capture **and** cross-gang
      separation, and carries the capacity wall ``lambda_min = 0`` when ``m > d``
      (Theorem 7.3).
    * ``trace``        -- ``mean_j Gamma_jj``: mean per-gang capture.  Pure capture,
      no separation term, **no capacity wall**.
    * ``softmin_diag`` -- soft-min of ``Gamma_jj``: the *worst gang's* capture, still
      with no cross-gang separation term.

    ``trace`` / ``softmin_diag`` are motivated by the connectivity constraint: a
    local-variation / Ward coarsener only ever merges *adjacent* nodes, so spatially
    disjoint gangs cannot be merged into one supernode however aligned their
    embeddings are -- Prop 8.5's "copies do not merge ... repetition is thus harmless
    for detection".  The separation that *is* needed (from the gang's own
    neighbourhood) is the confusability ``chi``, which is a separate term.
    """

    diag = torch.diagonal(gamma)
    if kind == "trace":
        return diag.mean()
    if kind == "softmin_diag":
        return _soft_min_values(diag, temperature)
    if kind == "lambda_min":
        return (
            _soft_lambda_min(gamma, temperature)
            if temperature > 0.0
            else torch.linalg.eigvalsh(gamma)[0]
        )
    raise ValueError(
        "capture_objective must be 'lambda_min', 'trace', or 'softmin_diag'"
    )


def _soft_lambda_min(
    gamma: torch.Tensor, temperature: float, *, relative: bool = False
) -> torch.Tensor:
    """Smooth surrogate for ``lambda_min(gamma)``: ``sum_i softmax(-lam/T)_i lam_i``.

    The Boltzmann *soft-min* of the eigenvalues: weights ``w_i = softmax((lam_0 -
    lam_i)/T)`` put the most mass on the smallest eigenvalues and decay upward, so
    the surrogate ``-> lambda_min`` as ``T -> 0`` and ``-> mean(lam)`` as
    ``T -> inf``.  Unlike the hard ``lambda_min`` (whose gradient sees only the
    single worst eigenvector and is non-smooth at spectral crossings), the
    soft-min spreads the ascent pressure over *all* poorly-reconstructed gang
    directions at once -- a smoother, better-conditioned objective that lifts the
    whole low end of the spectrum rather than chasing one eigenvalue.

    ``relative=True`` scales the temperature by the spectral spread
    (``T_eff = T * (lam_max - lam_min)``) so the softness is invariant to the
    absolute energy scale; the default uses an absolute ``T`` (mirroring
    :func:`_soft_lambda_max`).
    """

    eigs = torch.linalg.eigvalsh(gamma)  # ascending
    if relative:
        spread = (eigs[-1] - eigs[0]).detach().clamp_min(1e-12)
        tau = max(float(temperature), 1e-8) * spread
    else:
        tau = max(float(temperature), 1e-8)
    weights = torch.softmax((eigs[0] - eigs) / tau, dim=0)
    return (weights * eigs).sum()


# --------------------------------------------------------------------------- #
# confusability regularizer (Prop 6.8 / eq. 36; margin objective eq. 40)
# --------------------------------------------------------------------------- #
def build_confusability_tables(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    patterns: list,
    X: torch.Tensor,
    *,
    tau: float,
    degree: int,
    basis: str = "chebyshev",
    delta: float = 0.0,
    halo_hops: int = 1,
    geometry=None,
) -> list:
    """Precompute the per-gang, ``Theta``-independent pieces of the confusability.

    With ``delta == 0`` (the default, hard confusability eq. 36) each gang ``S``
    gets the S-localized Chebyshev table ``Y_S in R^{(K+1) x s x d}``,
    ``(Y_S)_{k,i,a} = (M_tau T_k(P) x_a)_i * weight_i``, the exact local
    ``M_tau``-form ``Q_S^tau = L^int_S + diag(d_partial) + tau * diag(weight^2)``,
    and ``weight^2|_S`` for the mean-zero constraint ``sum_i weight_i^2 z_i = 0``.

    ``weight`` is the coordinate change ``w = diag(weight) z`` that makes the
    fluctuation ``w`` supported on ``S`` and comes from ``geometry``:
    ``sqrt(d_tilde_i)`` under the symmetric convention (so ``Q_S`` picks up
    ``tau * D_tilde_S`` and the constraint is ``sum_i d_tilde_i z_i = 0``) and
    ``1`` under the combinatorial one (``Q_S = M_tau[S, S]`` exactly, constraint
    ``sum_i z_i = 0``).  The Laplacian part ``L^int_S + diag(d_partial)`` is the
    same quadratic form either way, which is why only these two pieces move.

    With ``delta > 0`` (the delta-leaky cone, Definitions 4.4/4.6) the confuser may
    place up to a ``delta`` fraction of its ell2 mass *outside* ``S`` -- in the
    ``halo_hops``-hop halo ``H = S union boundary(S)`` (the r-hop confuser of Remark
    4.12).  Each gang then gets the *halo*-localized table ``Y_H`` (no degree
    weighting -- ``w``-coordinates), the principal submatrix of ``M_tau`` on the
    halo ``B_H`` (so ``w^T B_H w = ||w||^2_{M_tau}`` for signals supported on
    ``H``; ``(1+tau) I - A_hat[H,H]`` under the symmetric convention and
    ``(D - W + tau I)[H,H]`` under the combinatorial one), the ell2 orthogonality
    vector ``c`` (``geometry.node_weights`` on ``S``, ``0`` on the halo, so that
    ``c^T w = <w, v_S>_{l2}``), and a boolean ``leak_mask`` marking the halo
    (outside-``S``) coordinates.  None depend on the learned filter, so they are
    built once and reused every epoch.
    """

    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    dtype, device = X.dtype, X.device
    # M_tau phi_k(P) X for k=0..K, shared across gangs (Theta-independent).
    propagated = _basis_stack(geo.prop, X, degree, basis, tau, geometry=geo)
    m_prop = [geo.m_apply(propagated[k], tau) for k in range(degree + 1)]

    d_total = _degrees(adjacency)  # (N,) weighted degree (no self-loops in W)
    n = adjacency.shape[0]
    coalesced = adjacency.coalesce()
    ii, jj = coalesced.indices()
    vv = coalesced.values()
    pos = torch.full((n,), -1, dtype=torch.long, device=device)

    if delta > 0.0:
        return _build_leaky_tables(
            a_hat, m_prop, patterns, tau, delta, halo_hops, degree, geometry=geo
        )

    tables = []
    for p in patterns:
        nodes = torch.as_tensor(p.node_indices, dtype=torch.long, device=device)
        s = int(nodes.numel())
        pos.fill_(-1)
        pos[nodes] = torch.arange(s, device=device)
        # internal weight submatrix W_SS (edges with both endpoints in S)
        mask = (pos[ii] >= 0) & (pos[jj] >= 0)
        w_sub = torch.zeros(s, s, dtype=dtype, device=device)
        w_sub[pos[ii[mask]], pos[jj[mask]]] = vv[mask].to(dtype)
        w_sub = 0.5 * (w_sub + w_sub.T)  # undirected
        d_int = w_sub.sum(1)  # internal degree
        d_tot_s = d_total[nodes].to(dtype)
        d_bnd = (d_tot_s - d_int).clamp_min(0.0)  # boundary degree d_partial
        # The Laplacian part of the local M_tau form is convention-free: for w
        # supported on S, w^T L w = w^T (L^int_S + diag(d_partial)) w for BOTH
        # L = D - W and L = I - A_hat (the latter after the w = D_tilde^{1/2} z
        # change of variables).  Only the screening term and the coordinate
        # weight carry the geometry, so both come from ``geo``.
        weight = geo.node_weights(nodes).to(dtype)  # sqrt(d_tilde) | 1
        q = (
            (torch.diag(d_int) - w_sub)  # L^int_S
            + torch.diag(d_bnd)  # diag(d_partial)
            + tau * torch.diag(geo.screening_diagonal(nodes).to(dtype))  # tau*D_tilde|tau*I
        )
        q = 0.5 * (q + q.T)
        # Y[k] = (M_tau phi_k X)[S] * weight   -> (K+1, s, d)
        Y = torch.stack(
            [m_prop[k][nodes] * weight.unsqueeze(1) for k in range(degree + 1)], dim=0
        )
        # constraint vector of F_S: <w, v_S>_2 = 0 <=> sum_i weight_i^2 z_i = 0
        tables.append({"Y": Y, "Q": q, "dtilde": weight**2})
    return tables


def _build_leaky_tables(
    a_hat, m_prop, patterns, tau, delta, halo_hops, degree, *, geometry=None
):
    """Per-gang halo tables for the delta-leaky cone (Definitions 4.4/4.6)."""

    geo = _geometry_or_symmetric(geometry, a_hat, None)
    combinatorial = geo.kind == "combinatorial"
    dtype, device = m_prop[0].dtype, m_prop[0].device
    n = a_hat.shape[0]
    a_coo = (geo.adjacency if combinatorial else a_hat).coalesce()
    ai, aj = a_coo.indices()
    av = a_coo.values()
    pos = torch.full((n,), -1, dtype=torch.long, device=device)

    tables = []
    for p in patterns:
        core = torch.as_tensor(p.node_indices, dtype=torch.long, device=device)
        # grow the r-hop halo H = S union (r-hop boundary) by neighbour accretion
        halo = core
        for _ in range(max(1, int(halo_hops))):
            halo = torch.unique(torch.cat([halo, aj[torch.isin(ai, halo)]]))
        h = int(halo.numel())
        pos.fill_(-1)
        pos[halo] = torch.arange(h, device=device)
        core_local = pos[core]
        leak_mask = torch.ones(h, dtype=torch.bool, device=device)
        leak_mask[core_local] = False  # True on the halo (outside-S) coordinates
        # local M_tau form on the halo: ||w||^2_{M_tau} = w^T B_H w for any w
        # supported on H -- the principal submatrix of M_tau on H either way.
        m = (pos[ai] >= 0) & (pos[aj] >= 0)
        sub_hh = torch.zeros(h, h, dtype=dtype, device=device)
        sub_hh[pos[ai[m]], pos[aj[m]]] = av[m].to(dtype)
        sub_hh = 0.5 * (sub_hh + sub_hh.T)
        if combinatorial:
            # (D - W + tau I)[H, H]: the diagonal is the FULL degree (edges to
            # outside H included), which is what makes this the exact restriction.
            deg_h = _degrees(geo.adjacency)[halo].to(dtype)
            b_h = torch.diag(deg_h + tau) - sub_hh
        else:
            b_h = (1.0 + tau) * torch.eye(h, dtype=dtype, device=device) - sub_hh
        b_h = 0.5 * (b_h + b_h.T)
        # ell2 orthogonality to v_S: c^T w = <w, v_S>_{l2}
        c = torch.zeros(h, dtype=dtype, device=device)
        c[core_local] = geo.node_weights(core).to(dtype)
        # halo-localized bank table in w-coordinates (no degree weighting)
        Y = torch.stack([m_prop[k][halo] for k in range(degree + 1)], dim=0)
        tables.append(
            {"Y": Y, "B": b_h, "c": c, "leak_mask": leak_mask, "delta": float(delta)}
        )
    return tables


def _top_confuser(
    a_num: torch.Tensor, q: torch.Tensor, dtilde: torch.Tensor, eps: float
) -> torch.Tensor:
    """Top generalized eigenvector of ``(A_num, Q)`` over ``{z : d_tilde^T z = 0}``.

    Returns the maximizing ``z*`` of the eq.-36 Rayleigh quotient on the
    degree-mean-zero subspace (the constant gang mode ``z ~ 1`` is the captured
    direction and is excluded).  Solved by whitening on an orthonormal basis of the
    constraint subspace; the caller detaches this (Danskin: ``z*`` is held fixed
    when differentiating the ratio through ``Theta``).
    """

    s = a_num.shape[0]
    dvec = dtilde / dtilde.norm().clamp_min(eps)
    # orthonormal basis P (s x s-1) of the complement of d_tilde
    P = torch.linalg.svd(dvec.reshape(s, 1), full_matrices=True).U[:, 1:]
    a_r = P.T @ a_num @ P
    b_r = P.T @ q @ P
    a_r = 0.5 * (a_r + a_r.T)
    b_r = 0.5 * (b_r + b_r.T)
    jitter = eps * (b_r.diagonal().mean().abs() + 1.0)
    b_r = b_r + jitter * torch.eye(s - 1, dtype=q.dtype, device=q.device)
    # generalized eig via Cholesky whitening: C = Lc^{-1} A_r Lc^{-T}
    lc = torch.linalg.cholesky(b_r)
    tmp = torch.linalg.solve_triangular(lc, a_r, upper=False)
    c = torch.linalg.solve_triangular(lc, tmp.T, upper=False).T
    c = 0.5 * (c + c.T)
    y = _sym_top_eigvec(c)  # largest generalized eigenvalue
    x = torch.linalg.solve_triangular(lc.T, y.unsqueeze(1), upper=True).squeeze(1)
    z = P @ x
    return z / z.norm().clamp_min(eps)


# LAPACK convergence failures surface as torch._C._LinAlgError, which subclasses
# RuntimeError; naming both keeps the catch working across torch versions.
_LINALG_ERROR = (getattr(torch._C, "_LinAlgError", RuntimeError), RuntimeError)


def _sym_top_eigvec(m: torch.Tensor) -> torch.Tensor:
    """Top (largest-eigenvalue) eigenvector of a symmetric matrix, robustly.

    ``torch.linalg.eigh`` uses LAPACK's divide-and-conquer ``syevd``, which can
    fail to converge precisely when the leading eigenvalues are nearly
    degenerate.  That is not a rare corner here: the S-procedure ``mu``-search in
    :func:`_top_leaky_confuser` bisects *towards* the KKT boundary, and the
    boundary is by definition where the maximizer switches, i.e. where the top
    two eigenvalues cross.  So the closer the bisection gets to the answer, the
    likelier ``syevd`` is to abort with "too many repeated eigenvalues".

    On that failure we shift the matrix to positive semi-definite (Gershgorin
    bound on ``|lambda|``) and take the top *singular* vector instead.  For a
    symmetric PSD matrix the singular and eigen decompositions coincide, so this
    returns the same vector via a different LAPACK path (``gesdd``), which does
    not share ``syevd``'s degeneracy failure mode.
    """

    try:
        return torch.linalg.eigh(m).eigenvectors[:, -1]
    except _LINALG_ERROR:
        shift = float(m.abs().sum(1).max()) + 1.0  # >= |lambda|_max by Gershgorin
        eye = torch.eye(m.shape[0], dtype=m.dtype, device=m.device)
        return torch.linalg.svd(m + shift * eye).U[:, 0]


def _top_leaky_confuser(
    a_h: torch.Tensor,
    b_h: torch.Tensor,
    c: torch.Tensor,
    leak_mask: torch.Tensor,
    delta: float,
    eps: float,
) -> torch.Tensor:
    """Maximizer ``w*`` of the delta-leaky Rayleigh quotient (Definition 4.6).

    ``max  w^T A_H w / w^T B_H w`` over the cone ``{c^T w = 0,  w^T E w <= 0}`` with
    ``E = diag(leak_mask) - delta I`` (so ``w^T E w <= 0`` is exactly
    ``||w_leak||^2 <= delta ||w||^2``).  Reduced to the constraint subspace
    ``c^T w = 0`` and solved by the S-procedure: the KKT points satisfy
    ``(A_H - mu E) w = lam B_H w`` with ``mu >= 0`` and complementary slackness, so
    we bisect ``mu`` until the top eigenvector of ``(A_r - mu E_r, B_r)`` sits on the
    leak boundary ``w^T E w = 0`` (``mu = 0`` if the leak cap is already slack).

    ``B_r`` is fixed across the mu-search, so it is whitened *once*: with
    ``B_r = Lc Lc^T`` the pencil becomes the standard symmetric eigenproblem
    ``(Ca - mu Ce) y = lam y`` where ``Ca = Lc^{-1} A_r Lc^{-T}`` and
    ``Ce = Lc^{-1} E_r Lc^{-T}`` are precomputed -- every mu step is then one small
    ``eigh`` on the same-size matrix, no re-factorization.  Detached; the caller
    differentiates the ratio at fixed ``w*`` (Danskin).
    """

    h = a_h.shape[0]
    cn = c / c.norm().clamp_min(eps)
    P = torch.linalg.svd(cn.reshape(h, 1), full_matrices=True).U[:, 1:]  # (h, h-1)
    a_r = 0.5 * (P.T @ a_h @ P + (P.T @ a_h @ P).T)
    b_r = P.T @ b_h @ P
    b_r = 0.5 * (b_r + b_r.T)
    b_r = b_r + eps * (b_r.diagonal().mean().abs() + 1.0) * torch.eye(
        h - 1, dtype=b_h.dtype, device=b_h.device
    )
    e_mat = torch.diag(leak_mask.to(a_h.dtype)) - delta * torch.eye(
        h, dtype=a_h.dtype, device=a_h.device
    )
    e_r = 0.5 * (P.T @ e_mat @ P + (P.T @ e_mat @ P).T)

    # whiten B_r once; work in y = Lc^T x coordinates
    lc = torch.linalg.cholesky(b_r)

    def _whiten(mat):
        t = torch.linalg.solve_triangular(lc, mat, upper=False)  # Lc^{-1} mat
        m = torch.linalg.solve_triangular(lc, t.T, upper=False).T  # Lc^{-1} mat Lc^{-T}
        return 0.5 * (m + m.T)

    ca, ce = _whiten(a_r), _whiten(e_r)
    # eigenvector maps back as x = Lc^{-T} y, w = P x, so w = (P Lc^{-T}) y with
    # P Lc^{-T} = (Lc^{-1} P^T)^T.  In the search we only need sign(w^T E w), which
    # equals sign(y^T Ce y) (== w^T E w up to the positive w-normalization), so we
    # avoid the back-transform until the final vector is returned.
    p_lct_inv = torch.linalg.solve_triangular(lc, P.T, upper=False).T  # (h, h-1)

    def _top_y(mu):
        # Scale by 1/(1+mu): eigenvectors are invariant under positive scaling, so
        # this is the same KKT point, but the matrix norm stays O(|ca| + |ce|)
        # instead of growing with mu.  Without it a large bracket hands ``eigh`` a
        # matrix whose entries dwarf ``ca``, and the search degrades to solving
        # ``-ce`` in floating point.
        return _sym_top_eigvec((ca - mu * ce) / (1.0 + mu))

    def _leak(y):  # sign-consistent proxy for w^T E w
        return float(y @ (ce @ y))

    def _back(y):
        w = p_lct_inv @ y
        return w / w.norm().clamp_min(eps)

    y0 = _top_y(0.0)
    if _leak(y0) <= 1e-9:  # leak cap already slack -> unconstrained (on c) optimum
        return _back(y0)
    # bracket a mu with leak(mu) < 0, then bisect to the boundary leak(mu*) = 0
    mu_hi, bracketed = 1.0, False
    for _ in range(40):
        if _leak(_top_y(mu_hi)) < 0.0:
            bracketed = True
            break
        mu_hi *= 2.0
    if not bracketed:
        # No mu enforces the leak cap (the cone is effectively empty for this
        # gang at this Theta).  Returning the mu_hi iterate would be an arbitrary
        # point on a failed search, so return the leak-minimizing direction: the
        # most feasible vector available, and a continuous limit of the search.
        return _back(_sym_top_eigvec(-ce))
    mu_lo = 0.0
    for _ in range(24):
        mu = 0.5 * (mu_lo + mu_hi)
        if _leak(_top_y(mu)) > 0.0:
            mu_lo = mu
        else:
            mu_hi = mu
    return _back(_top_y(mu_hi))  # feasible (leak-satisfying) side


def _gang_confusability_leaky(
    theta: torch.Tensor,
    chol: torch.Tensor,
    table: dict,
    eps: float,
) -> torch.Tensor:
    """Differentiable delta-leaky confusability ``chi^{tau,delta}_{R_Theta}(S)``.

    Halo ``w``-coordinates: ``u = Y_H^T w = (M_tau Z)[H]^T w`` (per channel
    ``u_a = <Z_{:,a}, w>_{M_tau}``), numerator ``u^T G_Z^+ u = ||Pi^{M_tau}_{span Z}
    w||^2_{M_tau}``, denominator ``w^T B_H w = ||w||^2_{M_tau}``.  ``chi`` is the
    largest ratio over the delta-leaky cone; ``w*`` is found detached and the ratio
    differentiated at fixed ``w*`` (Danskin), exactly as in the hard case.  ``chol``
    is the shared Cholesky factor of ``G_Z + ridge I`` (built once per step).
    """

    Y, b_h, c = table["Y"], table["B"], table["c"]
    leak_mask, delta = table["leak_mask"], table["delta"]
    h = b_h.shape[0]
    if h < 2 or not bool(leak_mask.any()):
        # no halo to leak into -> falls back to the hard internal-support problem
        # (handled by the delta=0 path); return 0 rather than a degenerate solve.
        return chol.new_zeros(())
    # (h, H*d): M_tau Z restricted to the halo, all heads concatenated
    Y_H = torch.einsum("kia,hka->iha", Y, as_multihead(theta)).reshape(Y.shape[1], -1)
    g_inv = torch.cholesky_solve(Y_H.T, chol)  # G_Z^+ Y_H^T   (d, h)
    a_h = Y_H @ g_inv  # w^T (Y_H G_Z^+ Y_H^T) w              (h, h)
    a_h = 0.5 * (a_h + a_h.T)
    with torch.no_grad():
        w_star = _top_leaky_confuser(a_h.detach(), b_h, c, leak_mask, delta, eps)
    num = w_star @ (a_h @ w_star)
    den = (w_star @ (b_h @ w_star)).clamp_min(eps)
    return num / den


def _gang_confusability(
    theta: torch.Tensor,
    chol: torch.Tensor,
    table: dict,
    eps: float,
) -> torch.Tensor:
    """Differentiable hard confusability ``chi^tau_{R_Theta}(S)`` of eq. 36.

    ``u = M(Theta) z`` with ``M[a,:] = sum_k theta[k,a] Y[k,:,a]`` (so
    ``u_a = <Z_{:,a}, D_tilde^{1/2} z>_{M_tau}``), the numerator
    ``u^T G_Z^+ u = ||Pi^{M_tau}_{span Z} w||^2_{M_tau}`` is the retained energy of
    the internal fluctuation ``w``, and ``chi`` is its largest fraction of
    ``||w||^2_{M_tau} = z^T Q z``.  The top eigenvector ``z*`` is found once
    (detached); the returned scalar carries gradient through ``M(Theta)`` and
    ``G_Z(Theta)`` only, which by Danskin's rule is exactly ``d chi / d Theta``.
    ``chol`` is the shared Cholesky factor of ``G_Z + ridge I`` (built once per
    step in :func:`collective_confusability` and reused across gangs).

    Dispatches to the delta-leaky cone (:func:`_gang_confusability_leaky`) when the
    table carries a halo (``delta > 0``); otherwise the hard eq.-36 problem below.
    """

    if "leak_mask" in table:
        return _gang_confusability_leaky(theta, chol, table, eps)

    Y, q, dtilde = table["Y"], table["Q"], table["dtilde"]
    s = q.shape[0]
    if s < 2:
        return chol.new_zeros(())
    th = as_multihead(theta)
    # (H, K+1, d) x (K+1, s, d) -> (H*d, s): head h's channel a row is
    # sum_k theta[h,k,a] Y[k,:,a], i.e. the bank column (h,a) tested against z
    M = torch.einsum("hka,kia->hai", th, Y).reshape(-1, Y.shape[1])
    g_inv_m = torch.cholesky_solve(M, chol)  # G_Z^+ M   (d, s)
    a_num = M.T @ g_inv_m  # z^T (M^T G_Z^+ M) z            (s, s)
    a_num = 0.5 * (a_num + a_num.T)
    with torch.no_grad():
        z_star = _top_confuser(a_num.detach(), q, dtilde, eps)
    num = z_star @ (a_num @ z_star)
    den = (z_star @ (q @ z_star)).clamp_min(eps)
    return num / den


def collective_confusability(
    theta: torch.Tensor,
    Z: torch.Tensor,
    a_hat: torch.Tensor,
    tau: float,
    ridge: float,
    tables: list,
    eps: float,
    *,
    reduce: str = "max",
    m_z: "torch.Tensor | None" = None,
    chol: "torch.Tensor | None" = None,
    geometry=None,
) -> torch.Tensor:
    """Worst-gang confusability ``max_{j<=m} chi^tau_{R_Theta}(S_j)`` (eq. 40 term).

    Danskin over the ``max`` is automatic: the gradient flows through the single
    worst gang each step.  ``reduce="mean"`` uses the average instead (a smoother,
    all-gang pressure if the hard max is too spiky).

    The per-gang solves need only the shared Cholesky factor of ``G_Z + ridge I``
    and the (small, precomputed) gang tables -- **not** the full embedding.  Pass
    ``chol`` directly (the training loop builds it N-free from the precomputed Gram
    kernel) to skip this call's ``O(N)`` work entirely; otherwise supply ``m_z =
    M_tau Z`` to reuse the caller's mat-vec, or fall back to computing it here.
    """

    if not tables:
        return theta.new_zeros(())
    # cholesky/cholesky_solve are differentiable, so gradient flows through
    # G_Z(Theta) whether chol is precomputed or built here.
    if chol is None:
        if m_z is None:
            m_z = _geometry_or_symmetric(geometry, a_hat, None).m_apply(Z, tau)
        g_z = Z.T @ m_z
        g_z = 0.5 * (g_z + g_z.T)
        d = g_z.shape[0]
        chol = torch.linalg.cholesky(
            g_z + ridge * torch.eye(d, dtype=g_z.dtype, device=g_z.device)
        )
    chis = torch.stack([_gang_confusability(theta, chol, t, eps) for t in tables])
    return chis.mean() if reduce == "mean" else chis.max()


class _RiemannianAdam:
    """Adam on the product of per-channel unit spheres ``{||theta[:,a]||=1}``.

    The constraint ``||theta^(a)|| = 1`` (one sphere per feature channel ``a``) is
    kept *exactly* every step rather than restored by a post-hoc renormalization.
    Each step (Absil-Mahony-Sepulchre / Becigneul-Ganea RAdam):

    1. project the Euclidean gradient to each sphere's tangent
       ``g_tan = g - <g, theta> theta`` (per column);
    2. run Adam on the tangent (bias-corrected moments), which respects the
       manifold instead of the redundant radial direction the normalize-in-forward
       reparametrization leaves in the optimizer state;
    3. retract by normalization ``theta <- (theta - lr*d)/||.||`` and vector-
       transport the first moment to the new tangent by re-projection.

    ``step(theta, egrad)`` takes the *loss* Euclidean gradient (minimization) and
    returns the updated on-manifold ``theta``.
    """

    def __init__(self, shape, lr, *, betas=(0.9, 0.999), eps=1e-8, dtype, device,
                 sphere_dim: int = 0):
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        # axis along which each unit sphere lives: the hop axis, i.e. 0 for a
        # legacy (K+1, d) filter and 1 for the multi-head (H, K+1, d) layout
        self.sd = int(sphere_dim)
        self.m = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    def _tangent(self, theta, g):  # remove the radial (per-sphere) component
        return g - (g * theta).sum(self.sd, keepdim=True) * theta

    def step(self, theta, egrad):
        self.t += 1
        rgrad = self._tangent(theta, egrad)  # Riemannian gradient
        self.m = self.b1 * self.m + (1 - self.b1) * rgrad
        self.v = self.b2 * self.v + (1 - self.b2) * rgrad * rgrad
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        direction = self._tangent(theta, m_hat / (v_hat.sqrt() + self.eps))
        new = theta - self.lr * direction  # descend the loss
        new = new / new.norm(dim=self.sd, keepdim=True).clamp_min(1e-12)  # retraction
        self.m = self._tangent(new, self.m)  # transport moment to the new point
        return new.detach()


def _graph_bundle(
    label,
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    train_patterns: list,
    X: torch.Tensor,
    test_patterns: "list | None",
    *,
    degree: int,
    tau: float,
    basis: str,
    conf_active: bool,
    conf_delta: float,
    conf_halo_hops: int,
    keep_dense: bool,
    geometry=None,
) -> dict:
    """All ``O(N)`` work for ONE graph, precomputed once.

    Everything the epoch loop needs from a graph lives in the returned bundle, so
    the loop itself is ``N``-independent no matter which graph it steps on:

    * ``gram_kernel[k,l,a,b] = <phi_k x_a, M_tau phi_l x_b>``  (``P``)
    * ``rhs_kernel[k,a,j]   = <phi_k x_a, M_tau vhat_j>``      (``Q``)
    * ``rhs_test_kernel``   -- the same for the HELD-OUT gangs, so their ``Gamma``
      is a genuine validation curve (the filter never sees their gradient)
    * ``conf_tables``       -- the Theta-independent confusability tables (eq. 36)

    ``keep_dense`` additionally retains the propagated dictionary stacks.  They
    are ``(N, (K+1)d)`` each, i.e. the one genuinely large object here, and are
    needed only by the features that resample signals every epoch (negatives) or
    read rows of ``Z`` directly (the joint label head) plus the initial
    ``O(N)`` objective -- so only the graphs that use them pay for them.
    """

    eps = torch.finfo(X.dtype).eps
    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    # v_j is unit-l2 in BOTH geometries (degree-weighted / uniform, per ``geo``),
    # so Phi_j = ||v_j||_L^2 is the conductance cut/vol in the symmetric case and
    # the cardinality-normalized cut/|S| of the paper's eq. 1-2 in the
    # combinatorial one -- and the vhat normalization below is the same formula.
    V = geo.indicators(train_patterns)  # (N, m)
    l_v = geo.l_apply(V)
    phi = (V * l_v).sum(0).clamp_min(eps)  # Phi_j = ||v_j||_L^2
    m_norm = (phi + tau).sqrt().unsqueeze(0)  # ||v_j||_{M_tau} = sqrt(Phi_j + tau)
    m_vhat = (l_v + tau * V) / m_norm  # M_tau Vhat = (L + tau I) v_j / sqrt(Phi_j+tau)

    propagated = _basis_stack(geo.prop, X, degree, basis, tau, geometry=geo)
    # Screened dictionary stack [M_tau phi_k(P) X] (K+1 sparse mat-vecs).  M_tau Z
    # is linear in theta, so M_tau Z = _filtered_bank(m_prop, theta) each epoch -- no
    # per-epoch sparse mat-vec in the Gram/confusability.
    m_propagated = [geo.m_apply(propagated[k], tau) for k in range(degree + 1)]

    # N-independent Gram kernel.  The screened channel Gram and the RHS are
    # quadratic / linear in the per-channel filter ``theta`` with theta-independent
    # coefficients:
    #   g_z[a,b] = Z[:,a]^T M_tau Z[:,b] = sum_{k,l} theta[k,a] theta[l,b] P[k,l,a,b],
    #   m[a,j]   = Z[:,a]^T M_tau vhat_j = sum_k theta[k,a] Q[k,a,j],
    # with P[k,l,a,b] = <phi_k X[:,a], M_tau phi_l X[:,b]> and
    #      Q[k,a,j]   = <phi_k X[:,a], M_tau vhat_j>.
    # Precomputing ``P`` (via one (K+1)d x (K+1)d Gram of the stacked dictionaries)
    # and ``Q`` ONCE turns every epoch's Gram from an O(K N d) pass over the
    # propagated stack into an O(K^2 d^2) einsum -- fully independent of N.
    d_feat = X.shape[1]
    _pr = torch.cat(propagated, dim=1)  # (N, (K+1) d), column k*d+a = phi_k X[:,a]
    _mp = torch.cat(m_propagated, dim=1)  # (N, (K+1) d)
    gram_kernel = (
        (_pr.T @ _mp)  # ((K+1)d, (K+1)d)
        .reshape(degree + 1, d_feat, degree + 1, d_feat)
        .permute(0, 2, 1, 3)
        .contiguous()
    )  # P: (K+1, K+1, d, d)
    rhs_kernel = (_pr.T @ m_vhat).reshape(degree + 1, d_feat, -1)  # Q: (K+1, d, m)

    rhs_test_kernel = None
    if test_patterns:
        v_t = geo.indicators(test_patterns)
        l_vt = geo.l_apply(v_t)
        phi_t = (v_t * l_vt).sum(0).clamp_min(eps)
        m_vt = (l_vt + tau * v_t) / (phi_t + tau).sqrt().unsqueeze(0)
        rhs_test_kernel = (_pr.T @ m_vt).reshape(degree + 1, d_feat, -1)
    del _pr, _mp

    conf_tables = (
        build_confusability_tables(
            a_hat,
            adjacency,
            train_patterns,
            X,
            tau=tau,
            degree=degree,
            basis=basis,
            delta=conf_delta,
            halo_hops=conf_halo_hops,
            geometry=geo,
        )
        if conf_active and train_patterns
        else []
    )

    return {
        "label": label,
        "a_hat": a_hat,
        "geometry": geo,
        "adjacency": adjacency,
        "X": X,
        "patterns": train_patterns,
        "m_vhat": m_vhat,
        "gram_kernel": gram_kernel,
        "rhs_kernel": rhs_kernel,
        "rhs_test_kernel": rhs_test_kernel,
        "conf_tables": conf_tables,
        "propagated": propagated if keep_dense else None,
        "m_propagated": m_propagated if keep_dense else None,
        "m": len(train_patterns),
        "m_test": len(test_patterns or []),
        "n": int(X.shape[0]),
    }


def fit_collective_bank(
    days: list,
    *,
    degree: int,
    epochs: int,
    learning_rate: float,
    ridge: float,
    fit_seed: int,
    tau: float = 0.0,
    neg_sampler: "callable | None" = None,
    neg_weight: float = 0.0,
    neg_temperature: float = 0.1,
    neg_project: bool = False,
    neg_sharpen: bool = False,
    snapshot_interval: int = 20,
    softmin_temperature: float = 0.0,
    basis: str = "chebyshev",
    conf_weight: float = 0.0,
    conf_reduce: str = "max",
    conf_delta: float = 0.0,
    conf_halo_hops: int = 1,
    optimizer_kind: str = "projected",
    capture_objective: str = "lambda_min",
    label_weight: float = 0.0,
    label_y: "torch.Tensor | None" = None,
    label_idx: "torch.Tensor | None" = None,
    warm_start: str = "ones",
    softmin_anneal: float = 1.0,
    heads: int = 1,
    head_diversity: float = 0.0,
    day_aggregate: str = "sample",
    geometries: "list | None" = None,
) -> dict:
    """Ascend ``lambda_min(Gamma(Theta))`` over the per-channel unit spheres.

    ``days`` is a list of ``(label, a_hat, adjacency, train_patterns, X)`` tuples,
    optionally with a sixth ``test_patterns`` element.  **A single graph is a
    one-element list** -- there is no separate single-graph path: the precompute
    is a per-graph bundle loop and the epoch loop selects a bundle, which for
    ``D = 1`` is the identity.  Every graph must share the feature dimension
    ``d`` (the filter is shared); graph size and gang count may differ freely.

    Training across graphs is the natural regularizer for the regime where the
    in-sample objective and out-of-sample detection come apart: the filter must
    work for gangs living on *different* graphs, so graph-specific structure
    cannot be memorized.  ``day_aggregate`` picks what each epoch ascends:

    * ``"sample"`` -- one graph drawn uniformly per epoch.  Stochastic gradient
      ascent whose "minibatch" is a whole graph: unbiased for the average-over-
      graphs objective ``E[capture_objective(Gamma_day(Theta))]`` and far cheaper
      than summing every graph per step.
    * ``"mean"``   -- average over all graphs (deterministic).
    * ``"min"``    -- the worst graph only (maximin).

    The price of several graphs is memory: the Gram kernel is ``(K+1)^2 d^2`` per
    graph (~123 MB at K=32, d=119), and all graphs are held simultaneously, so the
    count should stay modest.

    ``optimizer_kind`` chooses how the constraint ``||theta^(a)|| = 1`` is enforced:
    ``"projected"`` (default) optimizes an unconstrained ``raw`` with Adam and
    normalizes in the forward pass (weight-norm style); ``"riemannian"`` optimizes
    ``theta`` directly on the product of per-channel spheres with
    :class:`_RiemannianAdam`, keeping the norm exact and the step geometry-aware.
    ``"lbfgs"`` runs full-batch L-BFGS on the normalized parameterization: with the
    N-independent kernel the objective is deterministic and low-dimensional
    (``(K+1)d`` unknowns), i.e. exactly the regime where a quasi-Newton method
    beats a diagonally-preconditioned first-order one -- Adam's per-coordinate
    scaling carries no curvature information about the Gram's ill-conditioning,
    which is what makes ``lambda_min`` ascent crawl.  (L-BFGS needs a
    deterministic objective, so it falls back to Adam when negatives resample.)

    ``warm_start="closed_form"`` initializes ``theta`` at the best single filter
    consistent with the per-gang Theorem 6.2 optima: for each channel the top
    left-singular vector of that channel's closed-form coefficient vectors across
    the training gangs.  This starts the ascent inside the reachable set instead
    of at the flat all-ones low-pass, and typically buys more than any optimizer
    change.  ``softmin_anneal > 1`` starts at ``softmin_temperature *
    softmin_anneal`` and geometrically anneals to the requested value: a warmer
    soft-min spreads gradient over the whole low end of the spectrum, so early
    steps move all the weakly-captured gangs instead of only the argmin one.

    With ``conf_weight`` (``beta``) ``> 0`` the objective becomes the collective
    *margin* of eq. 40, ``lambda_min(Gamma) - beta * max_j chi^tau_{R_Theta}(S_j)``:
    the confusability penalty (eq. 36, :func:`collective_confusability`) drives down
    the worst gang's retained internal-fluctuation energy so no gang's *parts*
    survive as a separate supernode (the necessity branch of Theorem 4.10).

    ``heads = H > 1`` learns ``H`` independent filter banks at once; the target is
    their concatenated span ``Z = [Z^(1) | ... | Z^(H)]`` (``H*d`` columns), so a
    gang no longer has to share one hop-profile per channel with every other gang
    -- the ``M_tau``-projection inside ``Gamma`` picks each gang's best combination
    of heads for free.  **Everything else is unchanged**: same objective
    (``capture_objective``, including ``lambda_min``), same confusability penalty,
    same optimizer, same epochs.  ``heads = 1`` reproduces the single shared bank
    bit-for-bit, so it is a strict generalization rather than a separate path.

    Nothing in the objective forces two heads apart, but nothing rewards keeping
    them together either: capture depends only on ``span(Z)``, and duplicate heads
    contribute duplicate columns, i.e. no new directions -- so the all-heads-equal
    configuration is a *critical point* that wastes ``(H-1) d`` of the capacity.
    Escaping it needs symmetry breaking (the random init below) and, optionally,
    an explicit push: ``head_diversity > 0`` adds
    ``-head_diversity * mean_{h<h'} <theta_h, theta_h'>^2`` (mean squared cosine
    between heads' per-channel filters) to the ascended objective.

    ``geometries`` is the per-graph :class:`~src.screened_geometry.ScreenedGeometry`
    (same length and order as ``days``), which fixes the screened metric
    ``M_tau``, the group indicator ``v_S``, and the operator the bank propagates
    on.  ``None`` means the symmetric default for every graph, i.e. exactly the
    behaviour that predates the combinatorial option.

    Returns the learned ``theta`` (``(H, K+1, d)``, unit per-(head, channel)
    columns) plus the initial and final objective and the optimization history,
    with a per-graph breakdown of the final objective in ``per_day``.
    """

    if not days:
        raise ValueError("days must be a non-empty list of graph specs")
    if day_aggregate not in ("sample", "mean", "min"):
        raise ValueError("day_aggregate must be 'sample', 'mean' or 'min'")
    if geometries is not None and len(geometries) != len(days):
        raise ValueError(
            f"geometries has {len(geometries)} entries but {len(days)} graphs "
            "were given; pass one geometry per graph (or None for all-symmetric)."
        )
    heads = max(1, int(heads))

    # negative "repeller" sets: their softmax-lambda_max is *minimized*, so ``R``
    # preserves none of them and the coarsening splits them apart.  The sets are
    # resampled every epoch (see ``neg_sampler``) so the bank cannot overfit a
    # single fixed batch of negatives.
    neg_active = neg_sampler is not None and neg_weight > 0.0
    # optional joint supervised head on the same embedding, trained jointly with theta
    label_active = label_weight > 0.0 and label_y is not None and label_idx is not None
    # Both are genuinely single-graph features: the negative sampler is built from
    # one graph's edge index and the label split indexes one graph's nodes.  Rather
    # than silently applying them to whichever graph an epoch happened to draw,
    # refuse the combination outright.
    if len(days) > 1 and (neg_active or label_active):
        which = " and ".join(
            [n for n, on in (("neg_sampler", neg_active), ("label head", label_active)) if on]
        )
        raise ValueError(
            f"{which} is a single-graph feature (it is built from one graph's "
            f"nodes/edges) but {len(days)} graphs were given; drop it or fit on "
            "one graph."
        )

    dtype, device = days[0][1].dtype, days[0][1].device
    eps = torch.finfo(dtype).eps
    d_feat = int(days[0][4].shape[1])
    n_col = d_feat * heads  # target width: one block of d columns per head
    conf_active = conf_weight > 0.0 and any(len(spec[3]) > 0 for spec in days)

    if len(days) > 1:
        LOGGER.info(
            f"  multi-graph fit: building moment kernels for {len(days)} graphs "
            f"(K={degree}, d={d_feat}, aggregate={day_aggregate}) ..."
        )
    bundles = []
    for i, spec in enumerate(days):
        label, a_hat_i, adjacency_i, patterns_i, X_i = spec[:5]
        test_i = spec[5] if len(spec) > 5 else None
        if int(X_i.shape[1]) != d_feat:
            raise ValueError(
                f"graph {label} has feature-dim {X_i.shape[1]} != {d_feat}; the "
                "filter is shared so the feature dimension must match."
            )
        if len(patterns_i) > n_col:
            LOGGER.warning(
                f"  capacity: m_train={len(patterns_i)} > H*d={n_col}"
                f"{f' on graph {label}' if len(days) > 1 else ''}; "
                "lambda_min(Gamma) is 0 by Theorem (Capacity threshold) -- raise "
                "--feature-dim / --heads or lower --num-motifs / --train-ratio."
            )
        bundles.append(
            _graph_bundle(
                label, a_hat_i, adjacency_i, patterns_i, X_i, test_i,
                degree=degree, tau=tau, basis=basis, conf_active=conf_active,
                conf_delta=conf_delta, conf_halo_hops=conf_halo_hops,
                geometry=(geometries[i] if geometries is not None else None),
                # the dense stacks are only ever read on the first graph (initial
                # objective) and by the single-graph negatives / label head
                keep_dense=(i == 0),
            )
        )
        if len(days) > 1:
            LOGGER.info(
                f"    graph {label}: N={bundles[-1]['n']:,}  "
                f"train gangs={bundles[-1]['m']}"
            )

    # the graph the O(N) initial objective and the single-graph features read
    first = bundles[0]
    # The absolute ridge on Z^T M_tau Z is carried into each graph's own metric
    # scale -- per graph, since lambda_max(D - W) differs day to day.  A no-op in
    # the symmetric geometry, where ridge_scale is exactly 1 (see
    # ScreenedGeometry.ridge_scale).
    eye_n_col = torch.eye(n_col, dtype=dtype, device=device)
    for bundle in bundles:
        bundle["ridge_eye"] = (ridge * bundle["geometry"].ridge_scale) * eye_n_col

    def _neg_l_vhat(sets: "list | None"):
        if not sets:
            return None
        v_neg = first["geometry"].indicator_columns(sets)
        l_v_neg = first["geometry"].l_apply(v_neg)
        phi_neg = (v_neg * l_v_neg).sum(0).clamp_min(eps)
        return (l_v_neg + tau * v_neg) / (phi_neg + tau).sqrt().unsqueeze(0)

    def _gamma_and_chol(theta: torch.Tensor, bundle: dict, rhs_key="rhs_kernel"):
        """``(Gamma, chol)`` on one graph, from its precomputed kernel -- no O(N) work.

        With ``H`` heads the channel Gram gains cross-head blocks:
        ``g_z[(h,a), (g,b)] = sum_{k,l} theta[h,k,a] theta[g,l,b] P[k,l,a,b]``.
        For ``H = 1`` this is exactly the single-head contraction.  ``rhs_key``
        selects the training gangs or the held-out ones (the validation curve).
        """
        th = as_multihead(theta)
        g_z = torch.einsum(
            "hka,glb,klab->hagb", th, th, bundle["gram_kernel"]
        ).reshape(n_col, n_col)
        g_z = 0.5 * (g_z + g_z.T)
        m = torch.einsum("hka,kaj->haj", th, bundle[rhs_key]).reshape(n_col, -1)
        chol = torch.linalg.cholesky(g_z + bundle["ridge_eye"])
        gamma = m.T @ torch.cholesky_solve(m, chol)  # Vhat^T M Z (Z^T M Z)^+ Z^T M Vhat
        return 0.5 * (gamma + gamma.T), chol

    def _diversity(theta: torch.Tensor) -> torch.Tensor:
        """Mean squared cosine between distinct heads' per-channel filters.

        Zero when every pair of heads is per-channel orthogonal, one when the
        heads are identical -- subtracting it from the ascended objective pushes
        the bank off the degenerate all-heads-equal critical point.
        """
        th = as_multihead(theta)
        if th.shape[0] < 2:
            return theta.new_zeros(())
        # theta is already unit-norm per (head, channel), so the inner product
        # over the hop axis is the cosine
        cos = torch.einsum("hka,gka->hga", th, th)  # (H, H, d)
        off = ~torch.eye(th.shape[0], dtype=torch.bool, device=th.device)
        return (cos[off] ** 2).mean()

    # optional joint label head: restrict the propagated stack to the labelled rows
    # once, so the per-epoch CE costs O(K n_lab d) instead of O(K N d).
    head = None
    if label_active:
        prop_lab = [first["propagated"][k][label_idx] for k in range(degree + 1)]
        # the head sees the full bank, i.e. all H heads concatenated (n_col = H*d)
        head = torch.nn.Linear(n_col, 2).to(dtype=dtype, device=device)
        head_opt = torch.optim.Adam(
            head.parameters(), lr=learning_rate, weight_decay=5e-4
        )
        y_lab = label_y.to(torch.long)[label_idx]
        counts = torch.bincount(y_lab, minlength=2).to(dtype=dtype)
        cls_w = (counts.sum() / counts.clamp_min(1.0)) / 2.0

    torch.manual_seed(fit_seed)
    if warm_start == "closed_form" and any(b["m"] > 0 for b in bundles):
        # per-gang Theorem 6.2 coefficients W_j = (T^T M T)^+ T^T M vhat_j, then
        # the best SINGLE filter per channel: the top left-singular vector of that
        # channel's {W_j[:, a]}_j (maximizes the summed squared alignment).  Each
        # graph contributes its own gangs' coefficients to that channel stack, so
        # with one graph this is exactly the classic init.
        with torch.no_grad():
            per_graph_w = []
            for b in bundles:
                gram_full = torch.einsum("klab->kalb", b["gram_kernel"]).reshape(
                    (degree + 1) * d_feat, (degree + 1) * d_feat
                )
                rhs_full = b["rhs_kernel"].reshape((degree + 1) * d_feat, -1)
                evals_g, evecs_g = torch.linalg.eigh(0.5 * (gram_full + gram_full.T))
                keep_g = evals_g > evals_g.max() * 1e-10
                coeff = evecs_g[:, keep_g] @ (
                    (evecs_g[:, keep_g].T @ rhs_full) / evals_g[keep_g].unsqueeze(1)
                )  # ((K+1)d, m)
                per_graph_w.append(coeff.reshape(degree + 1, d_feat, -1))
                del gram_full, rhs_full, evals_g, evecs_g, coeff
            W = (
                per_graph_w[0]
                if len(per_graph_w) == 1
                else torch.cat(per_graph_w, dim=2)
            )  # (K+1, d, sum_i m_i)
            # head h takes the h-th singular direction of each channel's closed-form
            # coefficients: the heads start on an orthogonal frame by construction
            init = torch.stack([
                torch.stack([
                    torch.linalg.svd(W[:, a, :], full_matrices=True)[0][:, min(h, degree)]
                    for a in range(d_feat)
                ], dim=1)
                for h in range(heads)
            ], dim=0)  # (H, K+1, d)
            del per_graph_w, W
        raw = torch.nn.Parameter(init.contiguous())
    else:
        raw = torch.ones(heads, degree + 1, d_feat, dtype=dtype, device=device)
        if heads > 1:
            # all-heads-equal is a critical point of a span objective (duplicate
            # heads add no directions), so break the symmetry at init
            raw = raw + 0.5 * torch.randn(
                heads, degree + 1, d_feat, dtype=dtype, device=device
            )
        raw = torch.nn.Parameter(raw)

    def _unit(theta_raw: torch.Tensor) -> torch.Tensor:
        """Unit norm per (head, feature channel) -- the hop axis is axis 1."""
        th = as_multihead(theta_raw)
        return th / th.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)

    def _neg_softmax(embedding: torch.Tensor, l_vhat_neg, m_z=None) -> torch.Tensor:
        if l_vhat_neg is None:
            return torch.zeros((), dtype=dtype, device=device)
        gamma_neg = (
            _collective_gamma_mz(embedding, m_z, l_vhat_neg, ridge)
            if m_z is not None
            else _collective_gamma(
                first["a_hat"], embedding, l_vhat_neg, ridge, tau,
                geometry=first["geometry"],
            )
        )
        return _soft_lambda_max(gamma_neg, neg_temperature, sharpen=neg_sharpen)

    # baseline on the first graph, via the O(N) embedding rather than the kernel
    with torch.no_grad():
        Z0 = _filtered_bank(first["propagated"], _unit(raw))
        init_obj = float(
            torch.linalg.eigvalsh(
                _collective_gamma(
                    first["a_hat"], Z0, first["m_vhat"], ridge, tau,
                    geometry=first["geometry"],
                )
            )[0]
        )
        init_neg = float(
            _neg_softmax(Z0, _neg_l_vhat(neg_sampler() if neg_active else None))
        )
        init_conf = (
            float(
                collective_confusability(
                    _unit(raw),
                    Z0,
                    first["a_hat"],
                    tau,
                    ridge,
                    first["conf_tables"],
                    eps,
                    reduce=conf_reduce,
                )
            )
            if conf_active and first["conf_tables"]
            else 0.0
        )
        del Z0
    # the dense stacks exist only for the baseline above and the single-graph
    # features; with several graphs nothing reads them again, so drop the
    # (N, (K+1)d) pair rather than carrying it through the whole fit.
    if len(bundles) > 1:
        first["propagated"] = first["m_propagated"] = None

    use_lbfgs = optimizer_kind == "lbfgs"
    if use_lbfgs and neg_active:
        LOGGER.warning(
            "  optimizer=lbfgs needs a deterministic objective but negatives "
            "resample every step -- falling back to projected Adam."
        )
        use_lbfgs, optimizer_kind = False, "projected"
    if use_lbfgs and len(bundles) > 1 and day_aggregate == "sample":
        LOGGER.warning(
            "  optimizer=lbfgs needs a deterministic objective but "
            "day_aggregate='sample' redraws a graph every step -- falling back "
            "to projected Adam (use day_aggregate='mean' to keep L-BFGS)."
        )
        use_lbfgs, optimizer_kind = False, "projected"
    riemannian = optimizer_kind == "riemannian"
    if riemannian:
        theta_param = torch.nn.Parameter(_unit(raw).detach().clone())  # unit columns
        r_opt = _RiemannianAdam(
            theta_param.shape, learning_rate, dtype=dtype, device=device,
            sphere_dim=1,  # (H, K+1, d): each sphere lives on the hop axis
        )
        optimizer = None
    elif use_lbfgs:
        theta_param = None
        optimizer = torch.optim.LBFGS(
            (raw,),
            lr=learning_rate if learning_rate > 0.1 else 1.0,
            max_iter=20,
            history_size=25,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )
    else:
        theta_param = None
        optimizer = torch.optim.Adam((raw,), lr=learning_rate)
    # the live parameter whose gradient the ascent is taken w.r.t. each step
    param = theta_param if riemannian else raw
    best_theta = _unit(raw).detach().clone()
    # track the best iterate by the *margin* lambda_min - beta*conf (eq. 40); when
    # beta=0 this reduces to the plain lambda_min criterion.
    best_lam, best_neg, best_conf = init_obj, init_neg, init_conf
    best_crit = -float("inf")  # first epoch always sets the baseline (any objective)
    history: list[float] = []
    neg_history: list[float] = []
    conf_history: list[float] = []
    energy_history: list[float] = []
    # the objective actually ascended each epoch (capture term - beta*chi -
    # label_weight*CE): its negation is the training loss, so plotting it shows
    # what the optimizer optimized rather than a proxy.
    margin_history: list[float] = []
    ce_history: list[float] = []
    capture_min_history: list[float] = []
    # RMS cosine between distinct heads: 1 = collapsed to one filter, 0 = orthogonal
    head_sim_history: list[float] = []
    # held-out Gamma on the stepped graph: the filter never sees these gangs'
    # gradient, so this is a genuine validation curve (nan when none were given)
    history_test: list[float] = []
    energy_history_test: list[float] = []
    # which graph each epoch stepped on
    day_history: list = []
    snapshots: list = []
    snap_interval = max(1, epochs // 20) if snapshot_interval > 0 else 0
    day_rng = np.random.default_rng(int(fit_seed))
    epoch_bar = tqdm(range(epochs), desc="fitting collective bank", leave=False)

    def _capture_term(theta: torch.Tensor, temperature: float, bundle: dict):
        """``(pos_obj, gamma, chol)`` -- capture objective before the penalties."""

        # N-independent (precomputed per-graph kernel)
        gamma_, chol_ = _gamma_and_chol(theta, bundle)
        # the hard lambda_min (T=0) or a differentiable soft-min over the whole
        # low end of the spectrum (T>0)
        obj = _capture_objective(gamma_, capture_objective, temperature)
        if head_diversity > 0.0:
            obj = obj - head_diversity * _diversity(theta)
        return obj, gamma_, chol_

    def _graph_objective(theta: torch.Tensor, temperature: float, bundle: dict):
        """``(obj, gamma, chol, conf_val)`` on ONE graph: capture minus eq. 40's chi.

        The eq. 40 margin subtracts the worst-gang confusability penalty from the
        capture floor, so the same step lifts lambda_min AND shrinks chi.  The
        confusability reuses the Gram's Cholesky factor -- also N-independent.
        """

        obj, gamma_, chol_ = _capture_term(theta, temperature, bundle)
        if not (conf_active and bundle["conf_tables"]):
            return obj, gamma_, chol_, 0.0
        conf = collective_confusability(
            theta,
            None,
            bundle["a_hat"],
            tau,
            ridge,
            bundle["conf_tables"],
            eps,
            reduce=conf_reduce,
            chol=chol_,
        )
        return obj - conf_weight * conf, gamma_, chol_, float(conf.detach())

    def _positive(theta: torch.Tensor, temperature: float, drawn: "dict | None"):
        """The epoch's ascended objective over the graph list.

        ``drawn`` is the graph sampled for this epoch under ``day_aggregate=
        "sample"``; ``None`` means aggregate over every graph.  With a single
        graph both routes reduce to that graph's own objective, so ``D = 1`` is
        the same computation the classic single-graph fit performed, not an
        approximation of it.  Returns ``(obj, gamma, chol, bundle, conf_val)``
        where the reported ``gamma``/``conf_val`` belong to the binding graph.
        """

        if drawn is not None:
            obj, gamma_, chol_, conf_v = _graph_objective(theta, temperature, drawn)
            return obj, gamma_, chol_, drawn, conf_v
        per = [_graph_objective(theta, temperature, b) for b in bundles]
        objs = [p[0] for p in per]
        if day_aggregate == "min":  # maximin: step on the worst graph only
            k = int(torch.stack([o.detach() for o in objs]).argmin())
            obj = objs[k]
        else:  # "mean": step on every graph, report the weakest
            k = int(np.argmin([float(torch.linalg.eigvalsh(p[1])[0]) for p in per]))
            obj = torch.stack(objs).mean()
        return obj, per[k][1], per[k][2], bundles[k], per[k][3]

    for _ep in epoch_bar:
        # geometric anneal of the soft-min temperature: warm early (gradient
        # spread over the whole low end of the spectrum) -> sharp late.
        temp_ep = softmin_temperature
        if softmin_anneal > 1.0 and epochs > 1:
            temp_ep = softmin_temperature * softmin_anneal ** (1.0 - _ep / (epochs - 1))

        # one graph per epoch under "sample" -- drawn once so L-BFGS's inner
        # iterations all step on the same graph.  With one graph this draw is the
        # identity.
        drawn = (
            bundles[int(day_rng.integers(len(bundles)))]
            if day_aggregate == "sample"
            else None
        )

        if use_lbfgs:
            # L-BFGS runs its own inner iterations via the closure (each "epoch"
            # is up to `max_iter` quasi-Newton steps with a strong-Wolfe line
            # search); the bookkeeping below then records the accepted iterate.
            def _closure(_drawn=drawn):
                optimizer.zero_grad(set_to_none=True)
                obj_c, _, _, _, _ = _positive(_unit(raw), temp_ep, _drawn)
                if label_active:
                    obj_c = obj_c - label_weight * F.cross_entropy(
                        head(_filtered_bank(prop_lab, _unit(raw))), y_lab, weight=cls_w
                    )
                (-obj_c).backward()
                return -obj_c

            optimizer.step(_closure)

        theta = theta_param if riemannian else _unit(raw)
        pos_obj, gamma, _, bundle, conf_val = _positive(theta, temp_ep, drawn)

        # optional supervised head on the SAME embedding, trained jointly.  Only the
        # labelled rows of Z are needed, so this stays cheap (the propagated stack is
        # pre-restricted to label_idx) and the Gram path stays N-independent.
        if label_active:
            Z_lab = _filtered_bank(prop_lab, theta)
            ce = F.cross_entropy(head(Z_lab), y_lab, weight=cls_w)
            pos_obj = pos_obj - label_weight * ce
            ce_val = float(ce.detach())
        else:
            ce_val = 0.0

        if use_lbfgs:
            # the closure already took the step (and the head's, if any)
            ascent_grad, soft_neg_val = None, 0.0
        elif not neg_active:
            if label_active:
                # one backward for theta AND the head: both ascend pos_obj, so the
                # head's Adam (which minimizes) is fed the negated ascent.
                grads = torch.autograd.grad(
                    pos_obj, [param, *head.parameters()], allow_unused=True
                )
                ascent_grad = grads[0]
                head_opt.zero_grad(set_to_none=True)
                for p, g in zip(head.parameters(), grads[1:]):
                    p.grad = None if g is None else -g
                head_opt.step()
            else:
                (ascent_grad,) = torch.autograd.grad(pos_obj, param)
            soft_neg_val = 0.0
        else:
            # fresh negatives every epoch -> stochastic repeller (no overfitting).
            # Negatives resample new indicators each step, so they still need the
            # full embedding Z / M_tau Z (built here only when negatives are on).
            Z = _filtered_bank(first["propagated"], theta)
            m_z = _filtered_bank(first["m_propagated"], theta)
            l_vhat_neg = _neg_l_vhat(neg_sampler())
            soft_neg = _neg_softmax(Z, l_vhat_neg, m_z=m_z)
            # Take the two gradients separately so the negative step can be made
            # non-conflicting with the positive margin (gradient surgery).
            (grad_pos,) = torch.autograd.grad(pos_obj, param, retain_graph=True)
            grad_pos = grad_pos.detach().clone()  # ascent dir for the positive obj
            (grad_neg,) = torch.autograd.grad(soft_neg, param)
            desc_neg = -grad_neg.detach().clone()  # descent dir for the negatives
            if neg_project:
                conflict = (desc_neg * grad_pos).sum()
                if conflict < 0:  # this step would lower lambda_min -> remove it
                    denom = grad_pos.pow(2).sum().clamp_min(eps)
                    desc_neg = desc_neg - (conflict / denom) * grad_pos
            ascent_grad = grad_pos + neg_weight * desc_neg
            soft_neg_val = float(soft_neg.detach())

        # apply the ascent: Adam minimizes, so both optimizers get the negated
        # ascent as the loss gradient.  (L-BFGS already stepped in its closure.)
        if use_lbfgs:
            pass
        elif riemannian:
            with torch.no_grad():
                theta_param.copy_(r_opt.step(theta_param.detach(), -ascent_grad))
        else:
            optimizer.zero_grad(set_to_none=True)
            raw.grad = -ascent_grad
            optimizer.step()

        value = float(torch.linalg.eigvalsh(gamma)[0].detach())
        history.append(value)
        neg_history.append(soft_neg_val)
        conf_history.append(conf_val)
        gamma_diag = torch.diagonal(gamma.detach()).clamp(0, 1)
        energy_history.append(float(gamma_diag.mean()))
        capture_min_history.append(float(gamma_diag.min()))
        margin_history.append(float(pos_obj.detach()))
        ce_history.append(ce_val)
        day_history.append(bundle["label"])
        # held-out gangs on the stepped graph: no gradient, so this is a pure
        # validation reading of the same iterate
        lam_te = cap_te = float("nan")
        if bundle["rhs_test_kernel"] is not None:
            with torch.no_grad():
                g_te, _ = _gamma_and_chol(theta, bundle, rhs_key="rhs_test_kernel")
                lam_te = float(torch.linalg.eigvalsh(g_te)[0])
                cap_te = float(torch.diagonal(g_te).clamp(0.0, 1.0).mean())
        history_test.append(lam_te)
        energy_history_test.append(cap_te)
        if heads > 1:
            with torch.no_grad():
                head_sim_history.append(float(_diversity(theta).sqrt()))
        epoch_bar.set_postfix(
            lambda_min=f"{value:.4g}", margin=f"{float(pos_obj.detach()):.4g}",
            **({"graph": str(bundle["label"])} if len(bundles) > 1 else {}),
            **({"lam_te": f"{lam_te:.4g}"} if lam_te == lam_te else {}),
        )
        if snap_interval > 0 and (_ep % snap_interval == 0 or _ep == epochs - 1):
            snapshots.append(
                {
                    "epoch": _ep,
                    "theta": theta.detach().clone(),
                    "lam_min": value,
                    "gamma_diag": torch.diagonal(gamma.detach()).clamp(0, 1).tolist(),
                    # the per-gang panel is only comparable within one graph
                    "day": bundle["label"],
                }
            )
        # snapshot by the FULL objective actually being ascended (capture term -
        # beta*conf - label_weight*CE), so the criterion matches whichever
        # capture_objective is in use rather than always assuming lambda_min.  The
        # negatives are a stochastic regularizer resampled each step, so their
        # per-epoch value is noisy and stays out of the selection criterion.
        crit = float(pos_obj.detach())
        if crit > best_crit:
            best_crit = crit
            best_lam = value
            best_neg = soft_neg_val
            best_conf = conf_val
            best_theta = (
                theta_param.detach().clone()
                if riemannian
                else _unit(raw).detach().clone()
            )

    # the objective on EVERY graph at the retained iterate: with one graph this is
    # just a breakdown of ``objective``, with several it shows the spread the
    # shared filter actually achieved.
    per_day = {}
    with torch.no_grad():
        for b in bundles:
            g, _ = _gamma_and_chol(best_theta, b)
            dg = torch.diagonal(g).clamp(0.0, 1.0)
            per_day[str(b["label"])] = {
                "lambda_min": float(torch.linalg.eigvalsh(g)[0]),
                "mean_capture": float(dg.mean()),
                "min_capture": float(dg.min()),
                "n_train_gangs": b["m"],
            }
    if len(bundles) > 1:
        LOGGER.info("  per-graph objective at the retained filter:")
        for lbl, v in per_day.items():
            LOGGER.info(
                f"    graph {lbl}: lambda_min={v['lambda_min']:.4g}  "
                f"mean C={v['mean_capture']:.4g}  ({v['n_train_gangs']} gangs)"
            )

    return {
        "theta": best_theta,
        "init_objective": init_obj,
        "objective": best_lam,
        "margin": best_crit,
        "confusability_init": init_conf,
        "confusability": best_conf,
        "confusability_mean": float(np.mean(conf_history)) if conf_history else 0.0,
        "neg_objective_init": init_neg,
        "neg_objective": best_neg,
        "neg_objective_mean": float(np.mean(neg_history)) if neg_history else 0.0,
        "history": history,
        "history_test": history_test,
        "energy_history_test": energy_history_test,
        "neg_history": neg_history,
        "conf_history": conf_history,
        "energy_history": energy_history,
        "margin_history": margin_history,
        "ce_history": ce_history,
        "capture_min_history": capture_min_history,
        "head_sim_history": head_sim_history,
        "heads": heads,
        "head_similarity": head_sim_history[-1] if head_sim_history else 0.0,
        "snapshots": snapshots,
        "n_train_patterns": sum(b["m"] for b in bundles),
        "capture_objective": capture_objective,
        "conf_weight": conf_weight,
        "label_weight": label_weight,
        "train_days": [str(b["label"]) for b in bundles],
        "epoch_days": day_history,
        "day_aggregate": day_aggregate,
        "per_day": per_day,
    }


def channel_gram_cond(
    a_hat: torch.Tensor,
    X: torch.Tensor,
    theta: torch.Tensor,
    tau: float,
    basis: str,
    geometry=None,
) -> float:
    """Condition number ``kappa`` of the screened channel Gram ``Z^T M_tau Z``.

    This is the quantity Prop 6.3 bounds: with the Chebyshev basis the Gram is
    uniformly well-conditioned (``kappa = O(C/c)``), while the monomial Hankel Gram
    blows up like ``e^{cK}``.  Reported per basis so the numerical payoff of the
    switch is visible directly (a smaller ``kappa`` is the reason the Chebyshev
    solve needs no ridge and trains stably at large ``K``).
    """

    geo = _geometry_or_symmetric(geometry, a_hat, None)
    propagated = _basis_stack(geo.prop, X, theta_degree(theta), basis, tau, geometry=geo)
    Z = _filtered_bank(propagated, theta)
    g_z = Z.T @ geo.m_apply(Z, tau)
    g_z = 0.5 * (g_z + g_z.T)
    eigs = torch.linalg.eigvalsh(g_z)
    lo = (
        eigs[eigs > 0].min() if (eigs > 0).any() else eigs.abs().min().clamp_min(1e-300)
    )
    return float(eigs[-1] / lo)


def build_bank_subspace(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    theta: torch.Tensor,
    ridge: float,
    train_patterns: list | None = None,
    tau: float = 0.0,
    structural_width: int = 0,
    seed: int | None = None,
    coarsen_target: str = "bank",
    basis: str = "chebyshev",
    geometry=None,
) -> torch.Tensor:
    """Target ``R`` handed to the coarsener.

    Two choices of ``R``, selected by ``coarsen_target``:

    * ``"bank"`` (default) -- the *whole learned filter-bank subspace* ``span(Z)``
      (``d`` columns, ``Z = g_Theta(A_hat) X``).  This is the subspace the filter
      actually generates; it is *not* built from any particular gang, so it treats
      train and test gangs identically -- a held-out gang the filter can reach
      (e.g. a clique) is captured by ``span(Z)`` just as well as a training one.
      Its per-gang retained ``M_tau``-energy is exactly the capture ``C_S = Gamma_jj``
      the learner optimizes, so the reported energy now matches the objective.

    * ``"indicators"`` -- the ``M_tau``-projected gang indicators of Remark C.18,
      ``z_hat_j = Pi^{M_tau}_{span Z} v_hat_{S_j}`` (``m`` columns).  This is the
      *cheaper* RSA target (``span(Z)`` drags in within-gang-varying directions
      that never carry a full indicator and only spend RSA budget), but it is
      assembled from the *training* gangs, so a held-out indicator is ~0 inside it
      by construction -- its per-split energy is not a generalization diagnostic.

    ``Pi^{M_tau}_{span Z} = Z (Z^T M_tau Z)^+ Z^T M_tau`` is the ``M_tau``-orthogonal
    projector onto ``span(Z)``; applied to ``v_hat_j`` it returns the same
    reconstruction whose ``M_tau``-Gram is the ``Gamma`` the bank optimizes.  The
    coarsener then ``M_tau``-orthonormalizes these columns internally.

    ``structural_width > 0`` concatenates a *class-agnostic* low-frequency range
    finder ``g_theta_bar(A_hat) Omega`` (random Gaussian ``Omega``, shared filter
    ``theta_bar = mean_channel(theta)``) alongside the projected indicators -- the
    ``build_joint_subspace`` "indicator channel + structural channel" trick.  The
    projected indicators only cover the *train* gangs the filter learned; a wide
    low-frequency structural target additionally preserves the bottom-Laplacian
    subspace where *sparse* low-conductance motifs (cycles / stars / fans) live,
    so held-out gangs survive RSA coarsening even when the learned filter did not
    reconstruct them.  ``structural_width = 0`` leaves the target unchanged.
    """

    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    propagated = _basis_stack(geo.prop, X, theta_degree(theta), basis, tau, geometry=geo)
    Z = _filtered_bank(propagated, theta)  # (N, d)

    if coarsen_target == "bank":
        target = Z  # R = span(Z), the full learned filter-bank subspace     (N, d)
    elif coarsen_target == "indicators":
        eps = torch.finfo(X.dtype).eps
        V = geo.indicators(train_patterns)  # (N, m)
        l_v = geo.l_apply(V)
        phi = (V * l_v).sum(0).clamp_min(eps)  # Phi_j = ||v_j||_L^2
        m_norm = (phi + tau).sqrt().unsqueeze(0)  # ||v_j||_{M_tau}
        m_vhat = (l_v + tau * V) / m_norm  # M_tau v_hat_j                     (N, m)
        m_z = geo.m_apply(Z, tau)  # M_tau Z                                 (N, d)
        g_z = Z.T @ m_z  # Z^T M_tau Z                                       (d, d)
        g_z = 0.5 * (g_z + g_z.T)
        rhs = Z.T @ m_vhat  # Z^T M_tau v_hat                                (d, m)
        eye = torch.eye(g_z.shape[0], dtype=g_z.dtype, device=g_z.device)
        coeffs = torch.linalg.solve(
            g_z + (ridge * geo.ridge_scale) * eye, rhs
        )  # (Z^T M Z)^+ Z^T M v_hat
        target = Z @ coeffs  # Pi^{M_tau}_{span Z} v_hat                     (N, m)
    elif coarsen_target in ("dictionary", "bank+dictionary"):
        # Theorem 6.2 closed form: project each v_hat_j onto the FULL dictionary
        # col T = span{T_k(A_hat) x_a} ((K+1)d columns), not onto span(Z).  The
        # "indicators" branch above caps every gang at the *bank's* capture
        # (Gamma_jj of span Z); this branch attains the per-gang reachability
        # ceiling of Theorem 6.5 -- the best capture ANY degree-K filter of these
        # features can give that gang.  theta plays no role here beyond fixing
        # K and the polynomial basis.
        eps = torch.finfo(X.dtype).eps
        V = geo.indicators(train_patterns)  # (N, m)
        l_v = geo.l_apply(V)
        phi = (V * l_v).sum(0).clamp_min(eps)
        m_vhat = (l_v + tau * V) / (phi + tau).sqrt().unsqueeze(0)  # (N, m)
        B = torch.cat(propagated, dim=1)  # (N, (K+1)d)
        m_b = geo.m_apply(B, tau)
        gram = B.T @ m_b
        gram = 0.5 * (gram + gram.T)
        rhs = B.T @ m_vhat  # ((K+1)d, m)
        # rank-revealing pseudo-inverse: Krylov channels saturate, so the Gram is
        # far from full rank and a plain ridge solve would blur the projection.
        evals, evecs = torch.linalg.eigh(gram)
        keep = evals > evals.max() * 1e-10
        coeffs = evecs[:, keep] @ ((evecs[:, keep].T @ rhs) / evals[keep].unsqueeze(1))
        target = B @ coeffs  # Pi^{M_tau}_{col T} v_hat  (N, m), per-gang optimal
        if coarsen_target == "bank+dictionary":
            # ceiling capture on the training gangs (dictionary columns) plus the
            # shared bank span so *held-out* gangs keep a generalizing target.
            target = torch.cat([Z, target], dim=1)  # (N, d + m)
    else:
        raise ValueError(
            "coarsen_target must be 'bank', 'indicators', 'dictionary' "
            "or 'bank+dictionary'"
        )

    if structural_width and structural_width > 0:
        # Class-agnostic structural channel g_theta_bar(A_hat) Omega. The shared
        # scalar filter theta_bar = mean over the learned feature channels keeps
        # the same low-pass shape the bank learned, applied to a random range
        # finder that does *not* depend on which nodes are gangs.
        gen = torch.Generator(device=a_hat.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        omega = torch.randn(
            a_hat.shape[0],
            int(structural_width),
            dtype=a_hat.dtype,
            device=a_hat.device,
            generator=gen,
        )
        theta_bar = as_multihead(theta).mean(dim=(0, 2))  # (K+1,) mean over heads+channels
        prop_struct = _basis_stack(
            geo.prop, omega, theta_bar.shape[0] - 1, basis, tau, geometry=geo
        )
        z_struct = _filtered_bank(prop_struct, theta_bar)  # (N, structural_width)
        target = torch.cat([target, z_struct], dim=1)  # (N, m + structural_width)
    return target


def retained_energy(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    patterns: list,
    X: torch.Tensor,
    theta: torch.Tensor,
    ridge: float,
    tau: float = 0.0,
    indicator: str = "degree_weighted",
    basis: str = "chebyshev",
    geometry=None,
) -> dict:
    """Per-gang retained ``M_tau``-energy ``C_S = Gamma_jj`` and the collective margin.

    ``indicator`` selects the gang signal (see :func:`_make_indicators`).
    """

    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    _, m_vhat = _make_indicators(
        a_hat, adjacency, patterns, tau, indicator, geometry=geo
    )
    propagated = _basis_stack(geo.prop, X, theta_degree(theta), basis, tau, geometry=geo)
    Z = _filtered_bank(propagated, theta)
    gamma = _collective_gamma(a_hat, Z, m_vhat, ridge, tau, geometry=geo)
    diag = torch.diagonal(gamma).clamp(0.0, 1.0)
    return {
        "per_gang_capture": [float(v) for v in diag],
        "min_capture": float(diag.min()),
        "mean_capture": float(diag.mean()),
        "lambda_min_gamma": float(torch.linalg.eigvalsh(gamma)[0]),
    }


# --------------------------------------------------------------------------- #
# 6-7.  coarsen + evaluate
# --------------------------------------------------------------------------- #
def _alert_metrics(patterns: list, node_to_supernode, node_labels, threshold: float):
    """Run the Loukas pattern evaluation and pull out the alert-class summary."""

    _, by_label = evaluate_loukas_patterns(
        patterns, node_to_supernode, node_labels, threshold=threshold
    )
    return by_label.get("alert", {})


def _make_indicators(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    patterns: list,
    tau: float,
    indicator: str = "degree_weighted",
    geometry=None,
) -> tuple:
    """Build ``V`` and the precomputed ``M_tau Vhat`` for the chosen indicator.

    ``indicator='geometry'`` (what every geometry-aware caller passes) takes the
    indicator from the screened geometry itself, so the *reported* capture is
    measured on the same signal the training objective ascended:
    ``v_S = D_tilde^{1/2} 1_S / sqrt(vol(S))`` under the symmetric convention and
    ``v_S = 1_S / sqrt(|S|)`` under the combinatorial one.  This is the setting
    that cannot silently disagree with the fit.

    ``indicator='degree_weighted'`` and ``indicator='plain'`` pin the signal
    explicitly, whatever the geometry: the conductance-normalized
    ``D_tilde^{1/2} 1_S / sqrt(vol(S))`` and the raw 0/1 membership vector
    normalized to unit ``M_tau``-energy (``||1_S||_{M_tau}^2 = 1_S^T L 1_S +
    tau*|S|``).  All three yield ``||vhat||_{M_tau} = 1``; the difference is what
    notion of "which nodes belong to the gang" is privileged.  Pinning is useful
    for cross-geometry reporting -- the same signal scored in two metrics -- but
    ``'degree_weighted'`` under a combinatorial fit reports a capture the filter
    never optimized.

    Returns ``(V, m_vhat)`` where ``V`` is ``(N, m)`` unnormalized and
    ``m_vhat = M_tau Vhat`` is ``(N, m)``.
    """
    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    n = a_hat.shape[0]
    eps = torch.finfo(a_hat.dtype).eps
    if indicator == "plain":
        V = torch.zeros(n, len(patterns), dtype=a_hat.dtype, device=a_hat.device)
        for j, p in enumerate(patterns):
            V[list(p.node_indices), j] = 1.0
        l_v = geo.l_apply(V)
        phi = (V * l_v).sum(0)  # 1_S^T L 1_S
        sq = (V * V).sum(0)  # |S|
        denom = (phi + tau * sq).clamp_min(eps)
        m_vhat = (l_v + tau * V) / denom.sqrt().unsqueeze(0)
    else:  # "geometry" (the geometry's own v_S) or "degree_weighted" (pinned)
        V = (
            geo.indicators(patterns)
            if indicator == "geometry"
            else degree_weighted_indicators(adjacency, patterns)
        )
        l_v = geo.l_apply(V)
        phi = (V * l_v).sum(0).clamp_min(eps)
        m_vhat = (l_v + tau * V) / (phi + tau).sqrt().unsqueeze(0)
    return V, m_vhat


def _basis_retained_energy(
    a_hat,
    adjacency,
    patterns,
    basis,
    ridge,
    tau,
    indicator: str = "degree_weighted",
    geometry=None,
):
    """Mean/min retained ``M_tau``-energy of the gang indicators under ``span(basis)``.

    ``C_S = Gamma_jj`` with ``Gamma = Vhat^T M R (R^T M R)^+ R^T M Vhat`` for the
    coarsening *target subspace* ``R = span(basis)`` -- i.e. the fraction of each
    gang indicator's ``M_tau``-energy that the target subspace preserves (exactly
    the quantity the RSA coarsening is asked to keep together).  Because it is
    computed straight from the handed-in ``basis``, every encoder (collective
    bank, ``structural``, ``laplacian``) is measured on the same footing.

    ``indicator`` selects the gang signal whose energy is measured:
    ``'degree_weighted'`` uses the conductance-normalized ``v_S = D_tilde^{1/2}
    1_S / sqrt(vol(S))``;  ``'plain'`` uses the raw 0/1 membership vector
    ``1_S`` normalized to unit ``M_tau``-energy.
    """

    if not patterns:
        return {"mean": None, "min": None}
    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    _, m_vhat = _make_indicators(
        a_hat, adjacency, patterns, tau, indicator, geometry=geo
    )
    gamma = _collective_gamma(a_hat, basis, m_vhat, ridge, tau, geometry=geo)
    diag = torch.diagonal(gamma).clamp(0.0, 1.0)
    return {"mean": float(diag.mean()), "min": float(diag.min())}


def _labels_from_tree(children: "np.ndarray", n_samples: int, k: int) -> "np.ndarray":
    """Cut an agglomerative merge tree to ``k`` clusters -> contiguous labels.

    ``children`` is the sklearn ``children_`` array (row ``i`` records the two
    nodes merged to form node ``n_samples + i``; leaves are ``0..n_samples-1``).
    Applying the first ``n_samples - k`` merges and path-compressed union-find
    labels every leaf by its surviving root (ported from ``Coarsening_test``).
    """

    n_merges = max(0, n_samples - k)
    parent = np.arange(n_samples + len(children))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for i in range(n_merges):
        a, b = int(children[i, 0]), int(children[i, 1])
        parent[find(a)] = n_samples + i
        parent[find(b)] = n_samples + i

    roots: dict = {}
    labels = np.empty(n_samples, dtype=np.int64)
    for leaf in range(n_samples):
        r = find(leaf)
        labels[leaf] = roots.setdefault(r, len(roots))
    return labels


def ward_tree_coarsen(
    adjacency: torch.Tensor,
    basis: torch.Tensor,
    train_patterns: list,
    node_labels: torch.Tensor,
    *,
    tau: float,
    laplacian: str = "symmetric",
    threshold: float = 0.51,
    stop: str = "epsilon",
    epsilon_budget: float = float("inf"),
    num_cuts: int = 200,
) -> tuple:
    """Contiguity-constrained Ward tree over ``R = span(basis)``, cut fine->coarse.

    Builds the *whole* Ward merge tree once (on the ``M_tau``-orthonormal rows of
    ``R``, connectivity = graph adjacency so every supernode stays connected), then
    walks cluster counts from the *first combination* (``k = n-1``) toward the root
    (fewer, coarser supernodes).  At each cut it records the exact RSA distortion
    ``epsilon`` (Loukas Def. 2) and the mean **training** F1.  Two stop rules:

    * ``stop="epsilon"`` -- keep coarsening while the exact ``epsilon`` stays within
      ``epsilon_budget``; return the *coarsest* cut that still satisfies it (epsilon
      is monotone non-decreasing along the nested tree cuts, so this is the crossing).
    * ``stop="f1"``      -- return the cut with the best mean training F1 (the peak
      of the recall-vs-precision trade-off, chosen on *training* gangs only).

    Returns ``(LoukasCoarseningResult, trajectory)`` where ``trajectory`` is the
    per-cut list of ``{n_coarse, epsilon, train_f1, recall, precision}`` dicts.
    """

    try:
        from sklearn.cluster import AgglomerativeClustering
        from scipy.sparse import csr_matrix
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ward-tree coarsening needs scikit-learn and scipy") from exc

    base_fn = (
        _laplacian if laplacian in ("combinatorial", "comb") else _normalized_laplacian
    )
    metric = _screened_metric(base_fn(adjacency), tau)  # M_tau (sparse)
    a0 = _l_orthonormalize(basis, metric)  # M_tau-orthonormal basis of R (original)
    A = a0.detach().cpu().numpy()
    n = int(adjacency.shape[0])
    if A.shape[1] == 0:
        raise ValueError("target subspace has no positive-energy direction for Ward")

    # connectivity = original adjacency sparsity -> Ward merges stay connected
    idx = adjacency.coalesce().indices().cpu().numpy()
    conn = csr_matrix(
        (np.ones(idx.shape[1], dtype=np.float64), (idx[0], idx[1])), shape=(n, n)
    )
    model = AgglomerativeClustering(
        n_clusters=2, linkage="ward", connectivity=conn, compute_full_tree=True
    ).fit(A)
    children = np.asarray(model.children_)

    # cluster-count schedule, descending: first combination (k=n-1) -> coarse
    ks = np.unique(np.round(np.geomspace(2, n - 1, max(num_cuts, 2))).astype(int))
    ks = ks[(ks >= 2) & (ks <= n - 1)][::-1]

    trajectory: list = []
    best = None
    for k in ks.tolist():
        labels = _labels_from_tree(children, n, int(k))
        n2s = torch.from_numpy(labels).to(node_labels.device)
        eps = _exact_rsa_epsilon(a0, metric, n2s)
        results, by_label = evaluate_loukas_patterns(
            train_patterns, n2s, node_labels, threshold=threshold
        )
        f1 = float(np.mean([r.f1 for r in results])) if results else 0.0
        alert = by_label.get("alert", {})
        rec = float(alert.get("mean_recall", 0.0) or 0.0)
        prec = float(alert.get("mean_precision", 0.0) or 0.0)
        n_coarse = int(labels.max()) + 1
        entry = {
            "n_coarse": n_coarse,
            "epsilon": eps,
            "train_f1": f1,
            "recall": rec,
            "precision": prec,
            "labels": labels,
        }
        trajectory.append(entry)
        if stop == "epsilon":
            if eps <= epsilon_budget:
                best = entry  # last (coarsest) feasible cut so far
            else:
                break  # monotone: once over budget, coarser cuts only get worse
        else:  # f1
            if best is None or f1 > best["train_f1"]:
                best = entry

    if best is None:  # even the first combination overshot the budget
        # Returning the finest cut here means returning an essentially UNcoarsened
        # graph (one merge) and calling it a result: every gang stays split across
        # its own nodes, so detection is 0 while precision reads 1.  That is a
        # silent failure worth naming, and it is the normal outcome when the
        # budget was calibrated in one geometry and spent in another -- Pi_P is
        # only Euclidean-orthogonal, so the block-averaging distortion mu_P^tau
        # <= sqrt((lambda_max + tau)/tau) is ~2.2 under L_sym and orders of
        # magnitude larger under L = D - W.
        if trajectory:
            LOGGER.warning(
                f"  ward-tree: the FINEST cut (n_coarse={trajectory[0]['n_coarse']:,}) "
                f"already has epsilon={trajectory[0]['epsilon']:.4g} > budget "
                f"{epsilon_budget:g}, so no coarsening is feasible and the graph is "
                f"returned near-uncoarsened.  Raise --epsilon, or switch to "
                f"--ward-stop f1; note the budget is metric-dependent "
                f"(laplacian={laplacian!r})."
            )
        best = trajectory[0] if trajectory else None
    if best is None:
        raise ValueError("ward-tree produced no valid cut")

    labels = best["labels"]
    result = LoukasCoarseningResult(
        node_to_supernode=torch.from_numpy(labels).to(node_labels.device),
        n_original=n,
        n_coarse=best["n_coarse"],
        epsilon=best["epsilon"],
        epsilon_bound=float(
            "nan"
        ),  # per-level product bound not defined for a tree cut
        sigmas=[],
        sizes=[n, best["n_coarse"]],
    )
    return result, trajectory


def _coarsen_and_detect(normalized, adjacency, basis, splits, node_labels, args, tau):
    """Coarsen with ``basis`` and score alert recall/precision/detection per split.

    Shared by the collective-bank target and the ``laplacian`` / ``structural``
    baselines so every encoder is coarsened by the *identical* Loukas RSA
    procedure (same method, laplacian, budget, and screening ``tau``) and only
    the target subspace ``R = span(basis)`` differs.  Each split also gets the
    retained ``M_tau``-energy of its gang indicators under ``span(basis)``.
    """

    ward_trajectory = None
    if args.coarsening_method == "ward-tree":
        # Build the Ward tree once and cut it fine->coarse, stopping either at the
        # RSA epsilon budget or at the best *training* F1 (--ward-stop).
        coarsening, ward_trajectory = ward_tree_coarsen(
            adjacency,
            basis,
            splits["train"],
            node_labels,
            tau=tau,
            laplacian=args.coarsening_laplacian,
            threshold=args.threshold,
            stop=args.ward_stop,
            epsilon_budget=(args.epsilon if args.epsilon is not None else float("inf")),
            num_cuts=args.ward_num_cuts,
        )
    else:
        if args.epsilon is not None:
            budget = dict(
                reduction=args.reduction,
                epsilon=args.epsilon,
                epsilon_ramp_levels=args.epsilon_ramp_levels,
            )
        else:
            budget = dict(reduction=args.reduction)
        coarsening = loukas_coarsen_pytorch(
            adjacency,
            basis,
            method=args.coarsening_method,
            laplacian=args.coarsening_laplacian,
            max_levels=args.max_levels,
            tau=tau,
            **budget,
        )
    report = {}
    for name, split in splits.items():
        metrics = _alert_metrics(
            split, coarsening.node_to_supernode, node_labels, args.threshold
        )
        energy = _basis_retained_energy(
            normalized,
            adjacency,
            split,
            basis,
            args.ridge,
            tau,
            indicator=args.indicator,
        )
        report[name] = {
            "detection_rate": metrics.get("detection_rate"),
            "mean_recall": metrics.get("mean_recall"),
            "mean_precision": metrics.get("mean_precision"),
            "retained_energy": energy["mean"],
            "retained_energy_min": energy["min"],
            "detected": metrics.get("detected"),
            "total": metrics.get("total"),
        }
    if ward_trajectory is not None:
        # stash the fine->coarse sweep for logging / JSON (no schema change needed)
        coarsening.ward_trajectory = ward_trajectory
    return coarsening, report


def _baseline_encoder_reports(
    normalized, adjacency, train_patterns, splits, node_labels, args, tau
):
    """Coarsening reports for the ``structural`` and ``laplacian`` baselines.

    Ported from :mod:`run_joint_encoder_comparison`: both are *learning-free of
    the bank* target subspaces handed to the same coarsener.

    * ``structural`` -- ``theta`` fit on the structural Gram (``Sigma_X = I``),
      then ``R = span(g_theta(A_hat) Omega)`` (random range finder).
    * ``laplacian``  -- ``R = span(U_K)``, the bottom-``K`` combinatorial
      Laplacian eigenvectors (classical Loukas target, no learning).
    """

    reports = {}
    if not args.baselines or args.baseline_width <= 0:
        return reports

    structural_fit = fit_collective_sgc(
        normalized,
        train_patterns,
        features=None,
        mode="lambda_min",
        degree=args.degree,
        epochs=args.baseline_epochs,
        learning_rate=args.learning_rate,
    )
    structural_basis = build_sgc_subspace(
        normalized,
        structural_fit.theta,
        None,
        width=args.baseline_width,
        seed=args.seed,
    )
    _, reports["structural"] = _coarsen_and_detect(
        normalized, adjacency, structural_basis, splits, node_labels, args, tau
    )

    n_nodes = normalized.shape[0]
    if n_nodes <= args.baseline_laplacian_max_nodes:
        laplacian_basis = build_laplacian_subspace(adjacency, width=args.baseline_width)
        _, reports["laplacian"] = _coarsen_and_detect(
            normalized, adjacency, laplacian_basis, splits, node_labels, args, tau
        )
    else:
        LOGGER.info(
            f"  [tau={tau:g}] laplacian baseline skipped (N={n_nodes} > "
            f"{args.baseline_laplacian_max_nodes}; dense eigh too costly -- raise "
            "--baseline-laplacian-max-nodes to force it)"
        )
    return reports


def _parse_taus(spec: str) -> list:
    """Parse ``--tau`` into a list of floats (single value or comma-separated)."""

    vals = [float(t) for t in str(spec).replace(" ", "").split(",") if t != ""]
    if not vals:
        raise ValueError("--tau must contain at least one value")
    return vals


# --------------------------------------------------------------------------- #
# 8.  supervised node-level classification: linear head on Z  vs.  a GNN
# --------------------------------------------------------------------------- #
def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    """Accuracy / precision / recall / AUC for the gang (positive) class."""

    pred = (prob >= thr).astype(np.int64)
    try:
        auc = (
            float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else float("nan")
        )
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "auc": auc,
    }


def build_node_split(
    train_patterns: list,
    test_patterns: list,
    num_nodes: int,
    gang_mask: torch.Tensor,
    *,
    neg_per_pos: float,
    seed: int,
) -> tuple:
    """Node-level gang(1)/non-gang(0) labels with disjoint train/test node indices.

    Positives are the *train* / *test* gang nodes (the same gang split used for
    detection, so the head is scored on *held-out* gangs).  Negatives are random
    non-gang (host) nodes -- the conductance-matched benign background of
    Definition 8.4 -- split disjointly into train and test so no host node is
    shared.  Returns ``(y, train_idx, test_idx)``.
    """

    rng = np.random.default_rng(int(seed) + 99)
    y = gang_mask.to(torch.long).clone()
    pos_tr = torch.cat(
        [torch.as_tensor(p.node_indices, dtype=torch.long) for p in train_patterns]
    )
    pos_te = torch.cat(
        [torch.as_tensor(p.node_indices, dtype=torch.long) for p in test_patterns]
    )

    host = np.nonzero(gang_mask.cpu().numpy() == 0)[0]
    rng.shuffle(host)
    n_neg_tr = int(round(neg_per_pos * len(pos_tr)))
    n_neg_te = int(round(neg_per_pos * len(pos_te)))
    n_neg_tr = min(n_neg_tr, len(host))
    neg_tr = host[:n_neg_tr]
    neg_te = host[n_neg_tr : n_neg_tr + min(n_neg_te, len(host) - n_neg_tr)]

    train_idx = torch.cat([pos_tr, torch.as_tensor(neg_tr, dtype=torch.long)])
    test_idx = torch.cat([pos_te, torch.as_tensor(neg_te, dtype=torch.long)])
    return y, train_idx, test_idx


def train_linear_head(
    Z: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict:
    """Class-weighted logistic head on the frozen collective embedding ``Z``.

    This is the operational form of the union-detection / Definition 8.4 head: a
    *linear* probe on ``Z`` suffices exactly when the union indicator is captured
    (``v_union approx Z w``).  Metrics are reported on the held-out gang nodes.
    """

    torch.manual_seed(int(seed))
    d = Z.shape[1]
    head = nn.Linear(d, 2).to(dtype=Z.dtype)
    opt = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=5e-4)
    yl = y.to(torch.long)
    counts = torch.bincount(yl[train_idx], minlength=2).to(dtype=Z.dtype)
    w = (counts.sum() / counts.clamp_min(1.0)) / 2.0
    for _ in range(epochs):
        head.train()
        opt.zero_grad(set_to_none=True)
        logits = head(Z[train_idx])
        loss = F.cross_entropy(logits, yl[train_idx], weight=w.to(Z.dtype))
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        prob = torch.softmax(head(Z[test_idx]), dim=1)[:, 1].cpu().numpy()
    return _binary_metrics(yl[test_idx].cpu().numpy(), prob)


def _train_gang_m_vhat(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    patterns: list,
    tau: float,
    geometry=None,
) -> torch.Tensor:
    """``M_tau v_hat_S`` for the training gangs -- the RHS of the collective Gram.

    The indicator comes from the geometry, so this matches the RHS kernel
    :func:`_graph_bundle` precomputes for the fit under either convention.
    """

    eps = torch.finfo(a_hat.dtype).eps
    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    V = geo.indicators(patterns)  # (N, m)
    l_v = geo.l_apply(V)
    phi = (V * l_v).sum(0).clamp_min(eps)  # Phi_j = ||v_j||_L^2
    return (l_v + tau * V) / (phi + tau).sqrt().unsqueeze(0)  # M_tau v_hat_j


def _class_weights(y: torch.Tensor, train_idx: torch.Tensor, dtype) -> torch.Tensor:
    counts = torch.bincount(y[train_idx].to(torch.long), minlength=2).to(dtype=dtype)
    return (counts.sum() / counts.clamp_min(1.0)) / 2.0


def fit_joint_bank_head(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_patterns: list,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    degree: int,
    epochs: int,
    learning_rate: float,
    ridge: float,
    tau: float,
    label_weight: float,
    seed: int,
    basis: str = "chebyshev",
    capture_weight: float = 1.0,
) -> tuple:
    """Filter bank trained on the **two-term** objective (coarsening + labels).

    ``loss = -capture_weight * lambda_min(Gamma(Z)) + label_weight * CE(head(Z), y)``.
    ``capture_weight = 0`` drops the collective term entirely, giving a *purely
    supervised* embedding (the ``beta -> inf`` limit) without needing a numerically
    awkward huge ``label_weight``; ``label_weight = 0`` recovers the pure
    detect-all bank.  The first term is the collective capture objective that makes
    ``Z`` a good RSA coarsening target (so the embedding still detects gangs); the
    second is a label
    regularizer that flows into **both** the head and the per-channel filter
    coefficients.  The single learned ``Z`` therefore serves coarsening *and*
    prediction.  Returns ``(theta, Z, metrics, lam_min_final)``.
    """

    torch.manual_seed(int(seed))
    dtype = X.dtype
    eps = torch.finfo(dtype).eps
    m_vhat = _train_gang_m_vhat(a_hat, adjacency, train_patterns, tau)
    propagated = _basis_stack(a_hat, X, degree, basis, tau)  # [phi_k(A_hat) X], k=0..K
    d = X.shape[1]
    raw = nn.Parameter(torch.ones(degree + 1, d, dtype=dtype))
    head = nn.Linear(d, 2).to(dtype=dtype)
    opt = torch.optim.Adam(
        [raw, *head.parameters()], lr=learning_rate, weight_decay=5e-4
    )
    yl = y.to(torch.long)
    w = _class_weights(yl, train_idx, dtype)

    def _unit(t: torch.Tensor) -> torch.Tensor:
        return t / t.norm(dim=0, keepdim=True).clamp_min(eps)

    lam_min = torch.zeros((), dtype=dtype)
    for _ in range(epochs):
        head.train()
        Z = _filtered_bank(propagated, _unit(raw))
        gamma = _collective_gamma(a_hat, Z, m_vhat, ridge, tau)
        lam_min = torch.linalg.eigvalsh(gamma)[0]
        ce = F.cross_entropy(head(Z[train_idx]), yl[train_idx], weight=w.to(dtype))
        loss = -capture_weight * lam_min + label_weight * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        theta = _unit(raw).detach()
        Z = _filtered_bank(propagated, theta).detach()
        prob = torch.softmax(head(Z), dim=1)[:, 1]
    metrics = _binary_metrics(yl[test_idx].cpu().numpy(), prob[test_idx].cpu().numpy())
    return theta, Z, metrics, float(lam_min.detach())


def fit_gnn_encoder(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_patterns: list,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    embed_dim: int,
    epochs: int,
    learning_rate: float,
    ridge: float,
    tau: float,
    label_weight: float,
    seed: int,
) -> tuple:
    """2-layer GCN trained on the **same two-term** objective as the joint bank.

    ``loss = -lambda_min(Gamma(H)) + label_weight * CE(logits, y)`` where ``H`` is
    the GCN's hidden embedding.  The collective term shapes ``H`` into a coarsening
    target (so the GNN's embedding detects gangs), while the label term trains the
    GNN's classifier head.  Returns ``(H, metrics, lam_min_final)``.
    """

    torch.manual_seed(int(seed))
    dtype = X.dtype
    m_vhat = _train_gang_m_vhat(a_hat, adjacency, train_patterns, tau)
    enc = GCN2Encoder(in_dim=X.shape[1], embed_dim=embed_dim, num_classes=2).to(
        dtype=dtype
    )
    opt = torch.optim.Adam(enc.parameters(), lr=learning_rate, weight_decay=5e-4)
    yl = y.to(torch.long)
    w = _class_weights(yl, train_idx, dtype)

    lam_min = torch.zeros((), dtype=dtype)
    for _ in range(epochs):
        enc.train()
        H, logits = enc(a_hat, X)
        gamma = _collective_gamma(a_hat, H, m_vhat, ridge, tau)
        lam_min = torch.linalg.eigvalsh(gamma)[0]
        ce = F.cross_entropy(logits[train_idx], yl[train_idx], weight=w.to(dtype))
        loss = -lam_min + label_weight * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    enc.eval()
    with torch.no_grad():
        H, logits = enc(a_hat, X)
        prob = torch.softmax(logits, dim=1)[:, 1]
    metrics = _binary_metrics(yl[test_idx].cpu().numpy(), prob[test_idx].cpu().numpy())
    return H.detach(), metrics, float(lam_min.detach())


def run_classification_comparison(
    normalized: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    theta: torch.Tensor,
    graph,
    train_patterns: list,
    test_patterns: list,
    splits: dict,
    collective_report: dict,
    args,
    tau: float,
) -> dict:
    """Compare three encoders on BOTH tasks -- coarsening detection and labels.

    Each supervised encoder produces one embedding ``Z`` that is *both* handed to
    the same RSA coarsener (detection) *and* read by a label head (classification):

    * ``collective`` -- the unsupervised lambda_min bank (detection already done in
      :func:`run_for_tau`); a frozen linear probe supplies its label scores.
    * ``joint``      -- filter bank + head trained end-to-end on the label loss
      (Update 1): the labels shape ``Z``, and that same ``Z`` drives coarsening.
    * ``gnn``        -- a 2-layer GCN whose hidden ``H`` is the coarsening embedding
      and whose second layer predicts labels (Update 2).

    All see the same ``X`` and the same train/test node split, so the comparison is
    apples-to-apples on held-out gangs.
    """

    y, train_idx, test_idx = build_node_split(
        train_patterns,
        test_patterns,
        graph.num_nodes,
        graph.y,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
    )

    # --- collective bank: frozen linear probe for labels; detection reused -----
    propagated = _basis_stack(normalized, X, theta_degree(theta), args.basis, tau)
    Z_coll = _filtered_bank(propagated, theta)
    coll_cls = train_linear_head(
        Z_coll,
        y,
        train_idx,
        test_idx,
        epochs=args.head_epochs,
        learning_rate=args.head_lr,
        seed=args.seed,
    )

    # --- joint bank: -lambda_min(Gamma) + label CE (both shape Z) --------------
    _, Z_joint, joint_cls, joint_lam = fit_joint_bank_head(
        normalized,
        adjacency,
        X,
        y,
        train_patterns,
        train_idx,
        test_idx,
        degree=args.degree,
        epochs=args.head_epochs,
        learning_rate=args.head_lr,
        ridge=args.ridge,
        tau=tau,
        label_weight=args.label_reg_weight,
        seed=args.seed,
        basis=args.basis,
    )
    _, joint_det = _coarsen_and_detect(
        normalized, adjacency, Z_joint, splits, graph.y, args, tau
    )

    # --- GNN: -lambda_min(Gamma(H)) + label CE (both shape H) ------------------
    H_gnn, gnn_cls, gnn_lam = fit_gnn_encoder(
        normalized,
        adjacency,
        X,
        y,
        train_patterns,
        train_idx,
        test_idx,
        embed_dim=args.gnn_embed_dim,
        epochs=args.gnn_epochs,
        learning_rate=args.gnn_lr,
        ridge=args.ridge,
        tau=tau,
        label_weight=args.label_reg_weight,
        seed=args.seed,
    )
    _, gnn_det = _coarsen_and_detect(
        normalized, adjacency, H_gnn, splits, graph.y, args, tau
    )

    rows = {
        "collective": {
            "detection": collective_report,
            "classification": coll_cls,
            "lambda_min": None,
        },
        "joint": {
            "detection": joint_det,
            "classification": joint_cls,
            "lambda_min": joint_lam,
        },
        "gnn": {"detection": gnn_det, "classification": gnn_cls, "lambda_min": gnn_lam},
    }

    LOGGER.info(
        f"  [tau={tau:g}] encoder comparison -- coarsening detection + node labels "
        f"(held-out: {len(test_patterns)} gangs, "
        f"{int((graph.y[test_idx] == 1).sum())}/{len(test_idx)} test nodes)"
    )
    hdr = (
        f"  {'encoder':<12}{'det_all':>8}{'det_test':>9}{'ret_E':>7}"
        f"{'acc':>7}{'prec':>7}{'recall':>8}{'auc':>7}"
    )
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for name, r in rows.items():
        det, cls = r["detection"], r["classification"]
        da = det["all"]["detection_rate"] or 0.0
        dt = det["test"]["detection_rate"] or 0.0
        re = det["all"]["retained_energy"] or 0.0
        LOGGER.info(
            f"  {name:<12}{da:>8.1%}{dt:>9.1%}{re:>7.3f}"
            f"{cls['accuracy']:>7.3f}{cls['precision']:>7.3f}"
            f"{cls['recall']:>8.3f}{cls['auc']:>7.3f}"
        )
    return rows


def run_for_tau(
    tau: float,
    *,
    normalized,
    adjacency,
    X,
    graph,
    patterns,
    train_patterns,
    test_patterns,
    neg_sampler,
    args,
    out_dir: Path,
    tag: str = "",
) -> dict:
    """Steps 4-7 for one screening level ``tau``: learn, coarsen, detect, persist.

    Everything is measured in the screened metric ``M_tau = L + tau*I``.  The
    target handed to the coarsener is the ``M_tau``-projected gang indicators
    ``z_hat_j = Pi^{M_tau}_{span Z} v_hat_{S_j}`` (Remark C.18), which *do* depend
    on ``tau`` -- both through the learned filter ``theta*`` and through the
    screened projector -- and the coarsening RSA is itself measured in ``M_tau``,
    so the whole detection path is consistently screened.
    """

    suffix = f"_tau{tag}" if tag else ""

    # 4. learn the filter bank on the training motifs -------------------------
    fit = fit_collective_bank(
        [("train", normalized, adjacency, train_patterns, X)],
        degree=args.degree,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge=args.ridge,
        fit_seed=args.seed,
        tau=tau,
        neg_sampler=neg_sampler,
        neg_weight=args.neg_weight,
        neg_temperature=args.neg_temperature,
        neg_project=args.neg_project,
        neg_sharpen=args.neg_sharpen,
        softmin_temperature=args.softmin_temperature,
        basis=args.basis,
        conf_weight=args.conf_weight,
        conf_reduce=args.conf_reduce,
        conf_delta=args.conf_delta,
        conf_halo_hops=args.conf_halo_hops,
        optimizer_kind=args.optimizer,
    )
    theta = fit["theta"]
    LOGGER.info(
        f"  [tau={tau:g}] lambda_min(Gamma) train: {fit['init_objective']:.6g} -> "
        f"{fit['objective']:.6g}  (basis={args.basis})"
    )
    if args.conf_weight > 0.0:
        _chi_kind = (
            f"chi^(tau,delta={args.conf_delta:g})"
            if args.conf_delta > 0.0
            else "chi^tau"
        )
        LOGGER.info(
            f"  [tau={tau:g}] confusability {args.conf_reduce}_j {_chi_kind}(S_j) "
            f"(beta={args.conf_weight:g}): {fit['confusability_init']:.6g} -> "
            f"{fit['confusability']:.6g} (want down)  "
            f"margin lambda_min-beta*chi = {fit['margin']:.6g}"
        )
    # Prop 6.3 diagnostic: conditioning of the screened channel Gram Z^T M_tau Z in
    # each basis at the learned filter -- the monomial Hankel Gram is exponentially
    # worse in K, which is exactly what the Chebyshev switch cures.
    kappa_cheb = channel_gram_cond(normalized, X, theta, tau, "chebyshev")
    kappa_mono = channel_gram_cond(normalized, X, theta, tau, "monomial")
    LOGGER.info(
        f"  [tau={tau:g}] channel-Gram cond(Z^T M Z)  chebyshev={kappa_cheb:.3e}  "
        f"monomial={kappa_mono:.3e}  (ratio {kappa_mono / max(kappa_cheb, 1e-300):.1f}x)"
    )
    if neg_sampler is not None:
        LOGGER.info(
            f"  [tau={tau:g}] neg softmax-lambda_max ({args.num_neg_motifs} sets/epoch, "
            f"beta={args.neg_weight:g}): {fit['neg_objective_init']:.6g} -> "
            f"{fit['neg_objective_mean']:.6g} mean (want down)"
        )
    train_cap = retained_energy(
        normalized,
        adjacency,
        train_patterns,
        X,
        theta,
        args.ridge,
        tau,
        indicator=args.indicator,
        basis=args.basis,
    )
    test_cap = retained_energy(
        normalized,
        adjacency,
        test_patterns,
        X,
        theta,
        args.ridge,
        tau,
        indicator=args.indicator,
        basis=args.basis,
    )
    LOGGER.info(
        f"  [tau={tau:g}] retained M_tau-energy  train: min={train_cap['min_capture']:.3f} "
        f"mean={train_cap['mean_capture']:.3f}   "
        f"test: min={test_cap['min_capture']:.3f} "
        f"mean={test_cap['mean_capture']:.3f}"
    )

    # 5. embedding + target R = M_tau-projected indicators (tau-dependent) ----
    basis = build_bank_subspace(
        normalized,
        adjacency,
        X,
        theta,
        args.ridge,
        train_patterns,
        tau,
        structural_width=args.structural_width,
        seed=args.seed,
        coarsen_target=args.coarsen_target,
        basis=args.basis,
    )

    # 6. Loukas RSA coarsening with the learned target ------------------------
    splits = {"train": train_patterns, "test": test_patterns, "all": patterns}
    coarsening, report = _coarsen_and_detect(
        normalized, adjacency, basis, splits, graph.y, args, tau
    )
    if getattr(coarsening, "ward_trajectory", None) is not None:
        traj = coarsening.ward_trajectory
        best_f1 = max((t["train_f1"] for t in traj), default=0.0)
        LOGGER.info(
            f"  [tau={tau:g}] ward-tree ({args.ward_stop}-stop): "
            f"N={coarsening.n_original} -> n_coarse={coarsening.n_coarse}  "
            f"epsilon={coarsening.epsilon:.4g} "
            f"(budget {args.epsilon if args.epsilon is not None else float('inf'):.4g})  "
            f"train_f1={next(t['train_f1'] for t in traj if t['n_coarse'] == coarsening.n_coarse):.3f}  "
            f"(swept {len(traj)} cuts, best train_f1={best_f1:.3f})"
        )
    else:
        LOGGER.info(
            f"  [tau={tau:g}] coarsening: N={coarsening.n_original} -> n_coarse="
            f"{coarsening.n_coarse}  levels={len(coarsening.sigmas)}  "
            f"epsilon={coarsening.epsilon:.4g} (RSA exact; bound "
            f"{coarsening.epsilon_bound:.4g})"
        )

    # negative separation diagnostic on a fresh batch of repellers
    neg_share_mean = None
    if neg_sampler is not None:
        n2s = coarsening.node_to_supernode
        shares = []
        for nodes in neg_sampler():
            sup = n2s[torch.as_tensor(nodes, dtype=torch.long)]
            _, counts = torch.unique(sup, return_counts=True)
            shares.append(float(counts.max()) / len(nodes))
        if shares:
            neg_share_mean = float(np.mean(shares))
            LOGGER.info(
                f"  [tau={tau:g}] neg co-coarsen: mean dominant-supernode share "
                f"{neg_share_mean:.3f} (lower = better separated)"
            )

    # 7. recall / precision / detection rate ----------------------------------
    header = (
        f"  {'split':<6} {'recall':>8} {'precision':>10} {'ret_energy':>11} "
        f"{'detection':>10} {'det/tot':>9}"
    )
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 2))
    for name in ("train", "test", "all"):
        r = report[name]
        LOGGER.info(
            f"  {name:<6} {(r['mean_recall'] or 0):>8.3f} "
            f"{(r['mean_precision'] or 0):>10.3f} "
            f"{(r['retained_energy'] or 0):>11.3f} "
            f"{(r['detection_rate'] or 0):>10.1%} "
            f"{r['detected']:>4}/{r['total']:<4}"
        )

    # --- baseline encoders (laplacian / structural) on the SAME coarsener -----
    baseline_reports = _baseline_encoder_reports(
        normalized, adjacency, train_patterns, splits, graph.y, args, tau
    )
    encoder_reports = {"collective-bank": report, **baseline_reports}
    if baseline_reports:
        LOGGER.info(f"  [tau={tau:g}] encoder comparison (all motifs, same coarsener):")
        cmp_header = (
            f"    {'encoder':<16} {'recall':>8} {'precision':>10} "
            f"{'ret_energy':>11} {'detection':>10} {'det/tot':>9}"
        )
        LOGGER.info(cmp_header)
        LOGGER.info("    " + "-" * (len(cmp_header) - 4))
        for enc_name, rep in encoder_reports.items():
            r = rep["all"]
            LOGGER.info(
                f"    {enc_name:<16} {(r['mean_recall'] or 0):>8.3f} "
                f"{(r['mean_precision'] or 0):>10.3f} "
                f"{(r['retained_energy'] or 0):>11.3f} "
                f"{(r['detection_rate'] or 0):>10.1%} "
                f"{r['detected']:>4}/{r['total']:<4}"
            )

    # 8. supervised node classification: collective Z + head  vs.  GNN --------
    classification = None
    if args.classify:
        classification = run_classification_comparison(
            normalized,
            adjacency,
            X,
            theta,
            graph,
            train_patterns,
            test_patterns,
            splits,
            report,
            args,
            tau,
        )

    # --- persist JSON + plot --------------------------------------------------
    json_out = out_dir / f"collective_bank_detection{suffix}.json"
    plot_out = out_dir / f"collective_bank_detection{suffix}.png"
    json_out.write_text(
        json.dumps(
            {
                "config": vars(args) | {"output": str(out_dir), "tau": tau},
                "learning": {
                    "tau": tau,
                    "basis": args.basis,
                    "channel_gram_cond_chebyshev": kappa_cheb,
                    "channel_gram_cond_monomial": kappa_mono,
                    "lambda_min_gamma_init": fit["init_objective"],
                    "lambda_min_gamma_final": fit["objective"],
                    "conf_weight": args.conf_weight,
                    "conf_delta": args.conf_delta,
                    "confusability_init": fit.get("confusability_init"),
                    "confusability_final": fit.get("confusability"),
                    "margin_final": fit.get("margin"),
                    "neg_softmax_lambda_max_init": fit.get("neg_objective_init"),
                    "neg_softmax_lambda_max_mean": fit.get("neg_objective_mean"),
                    "theta": theta.detach().cpu().tolist(),
                    "train_capture": train_cap,
                    "test_capture": test_cap,
                },
                "coarsening": {
                    "n_original": coarsening.n_original,
                    "n_coarse": coarsening.n_coarse,
                    "reduction": coarsening.reduction,
                    "epsilon": coarsening.epsilon,
                    "n_levels": len(coarsening.sigmas),
                    "method": args.coarsening_method,
                    "ward_stop": (
                        args.ward_stop
                        if args.coarsening_method == "ward-tree"
                        else None
                    ),
                    "ward_trajectory": (
                        [
                            {k: v for k, v in t.items() if k != "labels"}
                            for t in coarsening.ward_trajectory
                        ]
                        if getattr(coarsening, "ward_trajectory", None) is not None
                        else None
                    ),
                },
                "detection": report,
                "encoder_comparison": encoder_reports,
                "classification": classification,
                "negatives": {
                    "num_sets_per_epoch": args.num_neg_motifs,
                    "resampled_each_epoch": neg_sampler is not None,
                    "weight": args.neg_weight,
                    "temperature": args.neg_temperature,
                    "co_coarsen_mean_share": neg_share_mean,
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    _save_plot(report, args, fit, plot_out, tau=tau)
    try:
        save_rich_plots(
            normalized=normalized,
            adjacency=adjacency,
            X=X,
            graph=graph,
            patterns=patterns,
            train_patterns=train_patterns,
            test_patterns=test_patterns,
            theta=theta,
            fit=fit,
            coarsening=coarsening,
            basis=basis,
            report=report,
            train_cap=train_cap,
            test_cap=test_cap,
            args=args,
            out_dir=out_dir,
            tau=tau,
            suffix=suffix,
        )
    except Exception as _viz_exc:
        LOGGER.warning(f"  [viz] rich plots failed: {_viz_exc}")
    LOGGER.info(f"  [tau={tau:g}] JSON: {json_out}")

    return {
        "tau": tau,
        "lambda_min_init": fit["init_objective"],
        "lambda_min_final": fit["objective"],
        "train_min_capture": train_cap["min_capture"],
        "train_mean_capture": train_cap["mean_capture"],
        "test_min_capture": test_cap["min_capture"],
        "test_mean_capture": test_cap["mean_capture"],
        "n_coarse": coarsening.n_coarse,
        "epsilon": coarsening.epsilon,
        "neg_share_mean": neg_share_mean,
        "report": report,
        "encoder_comparison": encoder_reports,
    }


def _log_tau_sweep(summaries: list) -> None:
    """Print a compact comparison table across the swept ``tau`` values."""

    LOGGER.info("\nM_tau sweep (comparison across tau)")
    header = (
        f"  {'tau':>8} {'lam_min':>9} {'train_cap':>10} {'test_cap':>9} "
        f"{'det_all':>8} {'n_coarse':>9}"
    )
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 2))
    for s in summaries:
        det = s["report"]["all"]["detection_rate"] or 0.0
        LOGGER.info(
            f"  {s['tau']:>8.3g} {s['lambda_min_final']:>9.4f} "
            f"{s['train_min_capture']:>10.3f} {s['test_min_capture']:>9.3f} "
            f"{det:>8.1%} {s['n_coarse']:>9}"
        )


def _save_tau_sweep_plot(summaries: list, args, output: Path) -> None:
    """Two-panel figure: collective margin/capture and detection rate vs tau."""

    taus = [s["tau"] for s in summaries]
    lam = [s["lambda_min_final"] for s in summaries]
    train_cap = [s["train_min_capture"] for s in summaries]
    test_cap = [s["test_min_capture"] for s in summaries]
    det = [(s["report"]["all"]["detection_rate"] or 0.0) for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(taus, lam, "o-", label=r"$\lambda_{\min}(\Gamma)$")
    ax.plot(taus, train_cap, "s--", label="min capture (train)")
    ax.plot(taus, test_cap, "^--", label="min capture (test)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("energy / margin")
    ax.set_title(r"Collective margin vs screening $\tau$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(taus, det, "o-", color="tab:green")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("detection rate (all)")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Detection vs screening $\tau$")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"{args.num_motifs}x {args.motif_type} (size {args.motif_size})  "
        f"reduction {args.reduction:.0%}  {args.coarsening_laplacian}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # graph / motifs
    parser.add_argument("--num-nodes", type=int, default=4000)
    parser.add_argument("--num-motifs", type=int, default=50)
    parser.add_argument(
        "--motif-type",
        choices=["clique", "cycle", "star", "random"],
        default="random",
    )
    parser.add_argument(
        "--motif-density",
        type=float,
        default=0.5,
        help="edge density for --motif-type random (fraction of s(s-1)/2 possible "
        "edges; a spanning path is always added so the motif stays connected)",
    )
    parser.add_argument(
        "--motif-conductance",
        type=float,
        default=-1.0,
        help="target conductance Phi=cut/vol per planted motif; adds random "
        "motif->host edges to reach it (Phi in [0,1)). Negative = no extra edges "
        "(leave motifs disjoint / minimal conductance).",
    )
    parser.add_argument("--motif-size", type=int, default=10)
    parser.add_argument("--avg-degree", type=float, default=2.0)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.4)
    parser.add_argument("--degree", type=int, default=15, help="polynomial degree K")
    parser.add_argument(
        "--basis",
        choices=["chebyshev", "monomial", "lanczos"],
        default="chebyshev",
        help="polynomial basis for the learnable filter bank Z = g_theta(A_hat) X: "
        "'chebyshev' (default) builds Z on the Chebyshev dictionary [T_k(A_hat) X] "
        "of the paper (Section 6.2, eq. 30) -- the screened Gram is uniformly "
        "well-conditioned (Prop 6.3), so the filter trains stably at high K with no "
        "ridge tuning; 'monomial' is the legacy dictionary [A_hat^k X] (Hankel Gram, "
        "exponentially ill-conditioned in K). Both span the same subspace (Lemma "
        "6.1), so the optimum is identical in exact arithmetic.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--optimizer",
        choices=["projected", "riemannian"],
        default="riemannian",
        help="how the per-channel constraint ||theta^(a)||=1 is enforced when "
        "learning the bank: 'projected' (default) runs Adam on an unconstrained raw "
        "and normalizes in the forward pass; 'riemannian' runs Adam on the product "
        "of unit spheres (exact norm each step, geometry-aware) -- usually reaches a "
        "higher margin and reduces confusability more reliably under --conf-weight.",
    )
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--tau",
        default="0.4",
        help="screened metric M_tau = L + tau*I; energy/objective is measured in "
        "||x||^2_{M_tau} = ||x||_L^2 + tau*||x||_2^2 (tau=0 -> L_sym seminorm, "
        "tau->inf -> l2). Pass one value, or a comma-separated list "
        "(e.g. 0,0.1,0.3,1.0) to sweep and compare across tau.",
    )
    parser.add_argument(
        "--structural-width",
        type=int,
        default=0,
        help="if >0, concatenate a class-agnostic low-frequency structural channel "
        "g_theta_bar(A_hat) Omega (random Omega of this width, shared filter) to the "
        "M_tau-projected gang indicators handed to the coarsener. Preserves the "
        "bottom-Laplacian subspace where sparse motifs (cycles/stars/fans) live, so "
        "held-out gangs survive coarsening even if the learned filter missed them "
        "(0 = target unchanged: train-gang projected indicators only).",
    )
    parser.add_argument(
        "--coarsen-target",
        choices=["bank", "indicators"],
        default="bank",
        help="which R to hand the coarsener (and to score retained energy against): "
        "'bank' = the full learned filter-bank subspace span(Z) (d columns; treats "
        "train/test symmetrically, energy == the capture C_S the learner optimizes); "
        "'indicators' = the M_tau-projected TRAIN-gang indicators of Remark C.18 "
        "(m columns; cheaper RSA target but held-out energy is ~0 by construction).",
    )
    # gang-aligned features (Logic 1): class-consistent block-aligned structure
    parser.add_argument(
        "--gang-feat-shared",
        type=float,
        default=0.0,
        help="strength of a single class-consistent 'gangness' direction added "
        "(block-constant) to every gang's nodes; generalizes train->test and drives "
        "union detection / the node head (0 = isotropic features).",
    )
    parser.add_argument(
        "--gang-feat-signature",
        type=float,
        default=0.0,
        help="strength of a fresh per-gang block-constant signature; gives the "
        "collective objective linearly independent per-gang feature directions "
        "(band resolution, Prop 6.8).",
    )
    # supervised node classification (Logic 2/3): linear head on Z vs. a GNN
    parser.add_argument("--classify", action="store_true", default=True)
    parser.add_argument("--no-classify", dest="classify", action="store_false")
    parser.add_argument(
        "--neg-per-pos",
        type=float,
        default=1.0,
        help="host (non-gang) nodes sampled per gang node as the negative class "
        "for the node-classification head/GNN (Definition 8.4 benign negatives).",
    )
    parser.add_argument(
        "--label-reg-weight",
        type=float,
        default=100.0,
        help="beta: weight of the label cross-entropy regularizer added to the "
        "collective -lambda_min(Gamma) objective when training the joint bank and "
        "the GNN (both encoders optimize coarsening capture + label prediction).",
    )
    parser.add_argument("--head-epochs", type=int, default=500)
    parser.add_argument("--head-lr", type=float, default=0.01)
    parser.add_argument("--gnn-epochs", type=int, default=300)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--gnn-embed-dim", type=int, default=64)
    # baseline encoders ported from run_joint_encoder_comparison
    parser.add_argument(
        "--baselines",
        action="store_true",
        default=True,
        help="also coarsen with the 'structural' (theta on the structural Gram, "
        "R = span(g_theta(A_hat) Omega)) and 'laplacian' (R = span(U_K), bottom-K "
        "combinatorial Laplacian eigenvectors) baseline targets on the SAME "
        "coarsener, and print a per-encoder comparison",
    )
    parser.add_argument("--no-baselines", dest="baselines", action="store_false")
    parser.add_argument(
        "--baseline-width",
        type=int,
        default=64,
        help="target subspace width for the laplacian / structural baselines",
    )
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=200,
        help="Adam epochs for the structural-baseline theta fit",
    )
    parser.add_argument(
        "--baseline-laplacian-max-nodes",
        type=int,
        default=4000,
        help="skip the laplacian baseline above this N (its dense NxN eigh is "
        "O(N^3); raise to force it on larger graphs)",
    )
    # negative "repeller" sets (random background neighborhoods)
    parser.add_argument(
        "--num-neg-motifs",
        type=int,
        default=0,
        help="number of negative repeller sets: random connected background "
        "neighborhoods that should NOT coarsen together; the fit also minimizes a "
        "softmax lambda_max of their Gamma (0 disables the negative term)",
    )
    parser.add_argument(
        "--neg-size-min",
        type=int,
        default=3,
        help="minimum size of a sampled negative neighborhood",
    )
    parser.add_argument(
        "--neg-size-max",
        type=int,
        default=10,
        help="maximum size of a sampled negative neighborhood (random per set)",
    )
    parser.add_argument(
        "--neg-weight",
        type=float,
        default=0.0,
        help="beta: weight of the negative softmax-lambda_max penalty in the fit",
    )
    parser.add_argument(
        "--neg-temperature",
        type=float,
        default=0.1,
        help="softmax temperature for the negative lambda_max surrogate (->0 = hard max)",
    )
    parser.add_argument(
        "--neg-project",
        action="store_true",
        help="gradient-surgery: drop the part of the negative step that would lower "
        "lambda_min (guarantees no positive loss, but largely neuters the negatives)",
    )
    parser.add_argument(
        "--neg-sharpen",
        action="store_true",
        help="scale-relative softmax so the negative penalty tracks the single worst "
        "blob (true lambda_max) instead of the mean retained energy",
    )
    parser.add_argument(
        "--softmin-temperature",
        type=float,
        default=0.2,
        help="temperature for a differentiable soft-min objective over the Gamma "
        "spectrum instead of the hard lambda_min (0 = hard min; larger -> closer to "
        "the mean retained energy, spreading ascent over all weak gang directions)",
    )
    # confusability regularizer (eq. 40 margin objective; eq. 36 confusability)
    parser.add_argument(
        "--conf-weight",
        type=float,
        default=17.5,
        help="beta: weight of the confusability penalty in the eq.-40 margin "
        "objective lambda_min(Gamma) - beta*max_j chi^tau_{R_Theta}(S_j). chi(S_j) "
        "(eq. 36) is the largest fraction of a within-gang fluctuation's M_tau-energy "
        "retained by the learned target span(Z); penalizing it stops a gang's *parts* "
        "from surviving as a separate supernode (Theorem 4.10 necessity branch). "
        "0 = pure detect-all objective (no penalty).",
    )
    parser.add_argument(
        "--conf-reduce",
        choices=["max", "mean"],
        default="mean",
        help="how the per-gang confusabilities are pooled into the penalty: 'max' "
        "(eq. 40, the worst gang, Danskin gradient through it) or 'mean' (smoother, "
        "spreads pressure over all training gangs).",
    )
    parser.add_argument(
        "--conf-delta",
        type=float,
        default=0.0,
        help="leakage level delta in [0,1) for the confusability cone (Def. 4.4). "
        "0 = hard confusability chi^tau (eq. 36, confuser supported inside S); >0 = "
        "the delta-leaky variant chi^{tau,delta} (Def. 4.6) that also penalizes "
        "confusers placing up to a delta fraction of their l2 mass in the one-hop "
        "halo of S -- excludes 'arc + a bit of boundary' splits, not just internal "
        "ones (Remark 4.11: reach). Monotone in delta.",
    )
    parser.add_argument(
        "--conf-halo-hops",
        type=int,
        default=1,
        help="radius of the halo the leaky confuser may reach into (Remark 4.12 "
        "r-hop confuser; 1 = one-hop halo, the paper's default). Only used when "
        "--conf-delta > 0.",
    )
    parser.add_argument(
        "--indicator",
        choices=["degree_weighted", "plain"],
        # default="plain",
        default="degree_weighted",
        help="gang indicator used for retained-energy reporting: "
        "'degree_weighted' (default) uses v_S = D_tilde^{1/2} 1_S / sqrt(vol(S)) "
        "(conductance-normalized, matches the training objective); "
        "'plain' uses the raw 0/1 membership vector 1_S normalized to unit "
        "M_tau-energy (||1_S||_{M_tau}^2 = 1_S^T L 1_S + tau*|S|).",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=1000,
        help="option 2 collapses gangs hierarchically over many levels "
        "(a size-k gang needs ~log2(k) edge-matching levels)",
    )
    # coarsening
    parser.add_argument(
        "--reduction",
        type=float,
        default=0.6,
        help="stop coarsening when n_coarse/n_original <= this fraction",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        # default=None,
        default=0.5,
        help="target RSA distortion budget prod_l(1+sigma_l)-1; when set it drives "
        "the coarsening (contract as much as possible until this bound is hit) and "
        "OVERRIDES --reduction",
    )
    parser.add_argument(
        "--epsilon-ramp-levels",
        type=int,
        default=5,
        help="optional: ration the --epsilon budget as a linear ramp over this many "
        "levels instead of offering it all at level 0 (only used with --epsilon)",
    )
    parser.add_argument(
        "--coarsening-method",
        choices=[
            "edges",
            "neighborhood",
            "capped",
            "star",
            "kmeans",
            "linkage",
            "ward",
            "ward-tree",
        ],
        default="ward-tree",
        help="coarsening family. 'ward-tree' builds the contiguity-constrained "
        "Ward merge tree once (like Coarsening_test), then cuts it fine->coarse "
        "and stops per --ward-stop; the others run the Loukas local-variation "
        "greedy under the --reduction/--epsilon budget.",
    )
    parser.add_argument(
        "--ward-stop",
        choices=["epsilon", "f1"],
        default="f1",
        help="stop rule for --coarsening-method ward-tree: 'epsilon' keeps "
        "coarsening while the exact RSA distortion stays <= --epsilon (coarsest "
        "feasible cut); 'f1' returns the cut with the best mean TRAINING F1 "
        "(peak of the recall/precision trade-off on training gangs only).",
    )
    parser.add_argument(
        "--ward-num-cuts",
        type=int,
        default=200,
        help="number of tree cuts sampled (geometric, fine->coarse) when walking "
        "the ward-tree; higher = finer resolution of the epsilon crossing / F1 peak.",
    )
    parser.add_argument(
        "--coarsening-laplacian",
        choices=["symmetric", "combinatorial"],
        default="symmetric",
        help="RSA metric for coarsening: 'symmetric' (L = I - A_hat, matches the "
        "algorithm, default) or 'combinatorial' (L = D - W, the legacy metric)",
    )
    parser.add_argument("--threshold", type=float, default=0.51)
    parser.add_argument("--seed", type=int, default=seed)
    path = f"results/collective_bank_detection/{now}/"
    parser.add_argument("--output", type=Path, default=path)
    args = parser.parse_args()

    # 1-2. build the controlled graph -----------------------------------------
    graph, patterns = build_synthetic_graph(
        num_nodes=args.num_nodes,
        num_motifs=args.num_motifs,
        motif_type=args.motif_type,
        motif_size=args.motif_size,
        avg_degree=args.avg_degree,
        feature_dim=args.feature_dim,
        rng_seed=args.seed,
        motif_density=args.motif_density,
        motif_conductance=args.motif_conductance,
    )
    normalized, adjacency = graph_operators(graph)  # A_hat (sym-norm) and raw W
    X = graph.x.to(device=normalized.device, dtype=normalized.dtype)
    # Logic 1: inject class-consistent block-aligned gang features (no-op if 0).
    X = inject_gang_features(
        X,
        patterns,
        shared=args.gang_feat_shared,
        signature=args.gang_feat_signature,
        seed=args.seed,
    )
    graph.x = X  # the GNN baseline reads graph features through X too

    # 3. train / test split of the motifs -------------------------------------
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(patterns))
    n_train = max(1, int(round(args.train_ratio * len(patterns))))
    train_patterns = [patterns[i] for i in order[:n_train]]
    test_patterns = [patterns[i] for i in order[n_train:]]

    # optional negative "repeller" sets (random background neighborhoods) ------
    # resampled every epoch so the bank can't overfit one fixed batch.
    neg_sampler = None
    if args.num_neg_motifs > 0:
        avoid = torch.nonzero(graph.y == 1, as_tuple=False).flatten().tolist()
        neg_sampler = make_negative_sampler(
            graph.edge_index,
            args.num_nodes,
            num_sets=args.num_neg_motifs,
            size_min=args.neg_size_min,
            size_max=args.neg_size_max,
            avoid=avoid,
            rng=np.random.default_rng(args.seed + 1),
        )

    LOGGER.info("\nCollective learnable filter-bank detection (M_tau metric)")
    _motif_attrs = f"size {args.motif_size}"
    if args.motif_type == "random":
        _motif_attrs += f", density {args.motif_density:.2f}"
    if args.motif_conductance is not None and args.motif_conductance >= 0.0:
        _motif_attrs += f", phi~{args.motif_conductance:.2f}"
    _motif_desc = f"{args.motif_type}({_motif_attrs})"
    # epsilon (RSA distortion budget) overrides the reduction-rate stopping rule
    if args.epsilon is not None:
        _budget_desc = f"epsilon<={args.epsilon:g}"
    else:
        _budget_desc = f"reduction={args.reduction:.0%}"
    LOGGER.info(
        f"  graph: N={args.num_nodes}  motifs={args.num_motifs}x{_motif_desc}"
        f"  avg_degree={args.avg_degree}  d={args.feature_dim}"
    )
    LOGGER.info(
        f"  split: train={len(train_patterns)}  test={len(test_patterns)}  "
        f"K={args.degree}  {_budget_desc}  "
        f"coarsening={args.coarsening_method}/{args.coarsening_laplacian}"
    )

    # --- switch metric to M_tau = L + tau*I; optionally sweep several tau -----
    taus = _parse_taus(args.tau)
    out_dir = Path(args.output) if args.output else Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = len(taus) > 1
    LOGGER.info(
        f"  metric: M_tau = L + tau*I   tau="
        + ", ".join(f"{t:g}" for t in taus)
        + ("  (sweep)" if sweep else "")
    )

    summaries = []
    for tau in taus:
        tag = f"{tau:g}".replace(".", "p").replace("-", "m") if sweep else ""
        if sweep:
            LOGGER.info(f"\n=== tau = {tau:g} ===")
        summaries.append(
            run_for_tau(
                tau,
                normalized=normalized,
                adjacency=adjacency,
                X=X,
                graph=graph,
                patterns=patterns,
                train_patterns=train_patterns,
                test_patterns=test_patterns,
                neg_sampler=neg_sampler,
                args=args,
                out_dir=out_dir,
                tag=tag,
            )
        )

    if sweep:
        _log_tau_sweep(summaries)
        sweep_png = out_dir / "collective_bank_detection_tausweep.png"
        sweep_json = out_dir / "collective_bank_detection_tausweep.json"
        _save_tau_sweep_plot(summaries, args, sweep_png)
        sweep_json.write_text(
            json.dumps(
                {
                    "config": vars(args) | {"output": str(out_dir)},
                    "sweep": [
                        {k: v for k, v in s.items() if k != "report"}
                        | {"detection": s["report"]}
                        for s in summaries
                    ],
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        LOGGER.info(f"\nSweep JSON: {sweep_json}")
        LOGGER.info(f"Sweep plot: {sweep_png}")
    else:
        LOGGER.info(f"\nOutput dir: {out_dir}")


def _save_plot(report: dict, args, fit: dict, output: Path, tau: float = 0.0) -> None:
    """Two-panel figure: detection metrics per split and the training curve."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["train", "test", "all"]
    x = np.arange(len(names))
    w = 0.25
    metrics = ["mean_recall", "mean_precision", "detection_rate"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["recall", "precision", "detection rate"]

    ax = axes[0]
    for offset, metric, color, label in zip([-w, 0, w], metrics, colors, labels):
        vals = [report[n][metric] or 0.0 for n in names]
        bars = ax.bar(x + offset, vals, w * 0.95, color=color, label=label)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title(
        f"{args.num_motifs}x {args.motif_type} (size {args.motif_size})  "
        f"reduction {args.reduction:.0%}  {args.coarsening_laplacian}  "
        rf"$\tau$={tau:g}"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(fit["history"], color="tab:red")
    ax.axhline(fit["init_objective"], color="grey", ls="--", lw=1, label="init")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\lambda_{\min}(\Gamma)$")
    ax.set_title("Collective objective (train motifs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
