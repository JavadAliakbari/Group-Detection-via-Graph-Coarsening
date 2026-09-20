from __future__ import annotations

import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from typing import List

from src.utils.utils import _degrees


def propagation_stack(
    adjacency: torch.Tensor, signals: torch.Tensor, degree: int
) -> List[torch.Tensor]:
    """Compute ``[V, A_hat V, ..., A_hat**degree V]`` by sparse matvecs.

    This is the *monomial* dictionary ``col T = [X, A_hat X, ..., A_hat^K X]`` of
    the paper (eq. 30).  Its screened Gram is a Hankel matrix and therefore
    exponentially ill-conditioned in ``K`` (Beckermann, 2000); use
    :func:`chebyshev_stack` for the well-conditioned Chebyshev dictionary that
    spans the *same* subspace (Lemma 6.1).
    """

    propagated = [signals]
    for _ in range(degree):
        propagated.append(torch.sparse.mm(adjacency, propagated[-1]))
    return propagated


def chebyshev_stack(
    adjacency: torch.Tensor, signals: torch.Tensor, degree: int
) -> List[torch.Tensor]:
    """Chebyshev dictionary ``[T_0(A_hat)V, ..., T_K(A_hat)V]`` (paper eq. 30).

    Built by the three-term recurrence of the first-kind Chebyshev polynomials
    evaluated at the *normalized adjacency* ``A_hat`` (whose spectrum lies in
    ``[-1, 1]``, the natural Chebyshev domain, so **no rescaling is needed** --
    unlike ChebNet, which rescales the Laplacian)::

        T_0(A_hat) = I,   T_1(A_hat) = A_hat,
        T_k(A_hat) = 2 A_hat T_{k-1}(A_hat) - T_{k-2}(A_hat).

    Each new term costs one sparse matvec, so the whole stack is ``O(K|E|d)`` --
    the same cost as :func:`propagation_stack`.  ``span(T_0..T_K) = span(I..A^K)``
    (Lemma 6.1), so swapping this in leaves the learned subspace, the collective
    Gram ``Gamma``, and the objective unchanged in exact arithmetic; what changes
    is the *conditioning* of the coefficient solve: the Chebyshev Gram is
    uniformly well-conditioned ``kappa(G) = O(C/c)`` (Prop 6.3) where the monomial
    Hankel Gram grows like ``e^{c K}`` -- so the filter trains stably at high ``K``
    and needs no Tikhonov ridge to stay solvable.
    """

    stack = [signals]  # T_0(A_hat) V = V
    if degree >= 1:
        stack.append(torch.sparse.mm(adjacency, signals))  # T_1(A_hat) V = A_hat V
    for _ in range(2, degree + 1):
        # T_k V = 2 A_hat (T_{k-1} V) - T_{k-2} V
        stack.append(2.0 * torch.sparse.mm(adjacency, stack[-1]) - stack[-2])
    return stack


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
    from screened_geometry import symmetric_geometry

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
