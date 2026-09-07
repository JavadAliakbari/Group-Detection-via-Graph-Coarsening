r"""The certified-margin training objective (sec:margin-objective).

The collective objective of :mod:`src.run_collective_bank_detection` rewards
capture and penalizes confusability through a hand-tuned weight ``beta``.  This
module implements the sharper principle: train the target to maximize the
*certified recovery margin itself*,

.. math::

    \Theta^\star = \arg\max_\Theta \operatorname{softmin}_\alpha
        \bigl\{ \underline s_\partial(S_j;\Theta)
                - \chi^\tau_{R_\Theta}(S_j) \bigr\}_{j=1}^m ,

with the boundary functional built from the group's capture ``C_j`` and a
*graph-only* adjacency floor,

.. math::

    \underline s_\partial(S_j;\Theta)
      = \frac{1}{C_j}\Bigl[\bigl(\underline\kappa(S_j)
                                 - \sqrt{1-C_j}\bigr)_+\Bigr]^2 .

Why this is different
---------------------
Capture and confusability are *irreconcilable* (Prop. "full capture forces
confusability"): if ``v_S`` lies in the target then

    chi >= nu(S)^2 / ((lambda_max + tau)(Phi(S) + tau)),

where ``nu(S)^2`` is the degree-weighted variance of the boundary rates
``rho_i = d_boundary(i)/d~_i`` about their mean ``Phi(S)``.  So the two terms of
the old objective fight each other for any boundary-irregular group and the
outcome hangs on ``beta``.  The margin objective removes ``beta`` entirely:
capture enters only through the boundary functional, confusability only through
the internal one, and cross-group separation is subsumed by the per-group floors
-- so the ``lambda_min(Gamma)`` machinery is not needed for the guarantee.  Its
value *is* the guarantee: positive at every group implies raw-score Ward recovers
every training group exactly.

The floor is graph-only
-----------------------
``kappa`` bounds the alignment of any *adjacent* boundary contrast with the group
direction.  Ward only ever proposes adjacent merges, and an adjacent boundary
pair shares at least one edge, which feeds the Laplacian part of the alignment.
For blocks ``A`` inside and ``B`` outside ``S``:

    kappa_S(A,B) >= [ tau*sqrt(vol(A)vol(B)/((vol A+vol B) vol S))
                      + w(A,B)*sqrt((vol A+vol B)/(vol A vol B vol S)) ]
                    / [ sqrt(Phi(S)+tau) * ||g_{A,B}||_{M_tau} ]
                 >= 2^{3/4} sqrt(tau w_min) / sqrt(vol(S)(Phi(S)+tau)(lambda_max+tau)).

The uniform form carries no small-fragment penalty: a tiny fragment has weak
``tau``-alignment and a huge one dilutes the edge term, and AM-GM trades the two
off for a constant proportional to ``sqrt(tau w_min / vol S)``.  Crucially it
does **not** depend on ``Theta``, so it is precomputed once per graph and the
objective's only learned inputs are ``C_j`` and ``chi_j``.

A note on reachability
----------------------
``s_boundary`` is positive only when ``C_j > 1 - kappa_j^2``.  On a large
financial graph ``kappa`` is small (it scales as ``sqrt(tau w_min / vol S)``), so
the *hard* functional is identically zero at any realistically achievable
capture and the exact objective degenerates to ``-chi``.  That is a true
statement about the instance, not a bug, and
:func:`certified_margin_terms` reports it (``feasible_capture``); but it also
leaves the capture term with no gradient.  ``softplus_beta > 0`` therefore
replaces ``(x)_+`` with ``softplus(beta x)/beta`` -- the same function wherever
the margin is comfortably positive, but with gradient everywhere, so capture is
still pushed up from below.  The reported certificate always uses the exact
hard clamp regardless of what the optimizer ascended.

Numerical verification: :func:`run_validation_suite` (also
``python -m src.certified_margin``) checks the conflict bound, both forms of the
floor against directly computed alignments over random adjacent boundary
candidates, the reachability index against a direct projection, and the
class-channel proposition.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

__all__ = [
    "boundary_irregularity",
    "adjacency_floor",
    "degree_floor",
    "candidate_alignment_floor",
    "group_margin_stats",
    "boundary_functional",
    "certified_margin_terms",
    "soft_min",
    "reachability_index",
    "resolvent_energy",
    "dictionary_reachability",
    "run_validation_suite",
]


def soft_min(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Boltzmann soft-min; ``alpha <= 0`` is the hard minimum.

    Same convention as the spectral soft-min of
    :mod:`src.run_collective_bank_detection`: weights
    ``softmax((min - v)/alpha)`` put most mass on the smallest entries, so the
    surrogate tends to ``min`` as ``alpha -> 0`` and to the mean as
    ``alpha -> inf``.  Spreading the pressure matters here because the binding
    group changes from step to step.
    """

    if values.numel() == 0:
        return values.new_zeros(())
    if alpha <= 0.0:
        return values.min()
    t = max(float(alpha), 1e-8)
    weights = torch.softmax((values.min().detach() - values) / t, dim=0)
    return (weights * values).sum()


# ---------------------------------------------------------------------------
# graph-only per-group quantities (computed once, no dependence on Theta)
# ---------------------------------------------------------------------------


def _edge_stats(adjacency: torch.Tensor):
    """``(rows, cols, vals, degrees, w_min)`` of the raw weight matrix ``W``."""

    coalesced = adjacency.coalesce()
    idx = coalesced.indices()
    vals = coalesced.values()
    degrees = torch.zeros(
        adjacency.shape[0], dtype=vals.dtype, device=vals.device
    ).index_add_(0, idx[0], vals)
    positive = vals[vals > 0]
    w_min = float(positive.min()) if positive.numel() else 1.0
    return idx[0], idx[1], vals, degrees, w_min


def boundary_irregularity(
    adjacency: torch.Tensor, nodes: torch.Tensor, *, uniform: bool = False
) -> "tuple[float, float, float]":
    r"""``(nu_squared, phi, vol)`` for one group (eq. boundary-irregularity).

    ``rho_i = d_boundary(i)/d~_i`` is node ``i``'s boundary rate and
    ``Phi(S) = cut(S)/vol(S)`` their degree-weighted mean, so

        nu(S)^2 = (1/vol S) sum_{i in S} d~_i (rho_i - Phi)^2

    is the *variance* of the boundary rates.  It is the exact internal cost of
    the indicator end of the seed family: a group whose nodes leak at different
    rates cannot be fully captured without also becoming confusable, at level
    ``nu^2/((lambda_max+tau)(Phi+tau))``.  ``uniform=True`` selects the
    combinatorial convention (``d~_i = 1``, ``vol = |S|``).
    """

    rows, cols, vals, degrees, _ = _edge_stats(adjacency)
    n = adjacency.shape[0]
    device = adjacency.device
    nodes = torch.as_tensor(nodes, dtype=torch.long, device=device)
    inside = torch.zeros(n, dtype=torch.bool, device=device)
    inside[nodes] = True

    # boundary degree of every node: edges leaving S
    leaving = inside[rows] & ~inside[cols]
    d_boundary = torch.zeros(n, dtype=vals.dtype, device=device)
    if leaving.any():
        d_boundary.index_add_(0, rows[leaving], vals[leaving])

    d_tilde = (
        torch.ones(n, dtype=vals.dtype, device=device) if uniform else degrees + 1.0
    )
    w = d_tilde[nodes]
    vol = float(w.sum())
    cut = float(d_boundary[nodes].sum())
    phi = cut / max(vol, 1e-300)
    rho = d_boundary[nodes] / w.clamp_min(1e-300)
    nu2 = float((w * (rho - phi) ** 2).sum()) / max(vol, 1e-300)
    return nu2, phi, vol


def degree_floor(
    vol_s: float, phi: float, tau: float, lambda_max: float, d_min: float
) -> float:
    r"""The ``d~_min`` boundary floor, the ``tau``-only ancestor of the lemma.

    From ``vol(A)vol(B)/(vol A + vol B) >= d~_min/2`` and
    ``||g||_{M_tau}^2 <= lambda_max + tau``:

        kappa >= tau sqrt(d~_min / (2 vol S)) / sqrt((Phi+tau)(lambda_max+tau)).

    It drops the edge term entirely, so it pays the small-fragment penalty the
    AM-GM form avoids -- but it is a valid bound, and the lemma says to use the
    maximum of the two.
    """

    if tau <= 0.0 or vol_s <= 0.0 or d_min <= 0.0:
        return 0.0
    denom = (phi + tau) * (lambda_max + tau)
    if denom <= 0.0:
        return 0.0
    return tau * math.sqrt(d_min / (2.0 * vol_s)) / math.sqrt(denom)


def adjacency_floor(
    vol_s: float, phi: float, tau: float, lambda_max: float, w_min: float
) -> float:
    r"""The uniform boundary floor ``kappa_lower(S)`` (Lemma strengthened-floor).

    ``2^{3/4} sqrt(tau w_min) / sqrt(vol(S)(Phi(S)+tau)(lambda_max+tau))``: the
    smallest normalized screened alignment any *adjacent* boundary candidate of
    ``S`` can have.  Zero when ``tau = 0`` -- without screening the floor rests
    entirely on the edge term, which a large enough fragment dilutes away.
    """

    if tau <= 0.0 or w_min <= 0.0 or vol_s <= 0.0:
        return 0.0
    denom = vol_s * (phi + tau) * (lambda_max + tau)
    if denom <= 0.0:
        return 0.0
    return (2.0**0.75) * math.sqrt(tau * w_min) / math.sqrt(denom)


def candidate_alignment_floor(
    vol_a: float,
    vol_b: float,
    w_ab: float,
    vol_s: float,
    phi: float,
    tau: float,
    g_m_norm: float,
) -> float:
    r"""The *per-candidate* floor for one adjacent boundary pair ``(A, B)``.

    Tighter than :func:`adjacency_floor` but candidate-specific, so it is a
    diagnostic and a validation reference rather than something the objective can
    use.  Both floors are valid; the lemma says to use their maximum.
    """

    if g_m_norm <= 0.0 or vol_s <= 0.0 or vol_a <= 0.0 or vol_b <= 0.0:
        return 0.0
    total = vol_a + vol_b
    tau_term = tau * math.sqrt(vol_a * vol_b / (total * vol_s))
    edge_term = w_ab * math.sqrt(total / (vol_a * vol_b * vol_s))
    return (tau_term + edge_term) / (math.sqrt(phi + tau) * g_m_norm)


def group_margin_stats(
    geometry,
    patterns: Sequence,
    tau: float,
    *,
    lambda_max: "float | None" = None,
) -> dict:
    r"""Per-group graph-only inputs to the margin objective.

    Returns ``kappa`` (the floors, one per group), plus the diagnostics ``vol``,
    ``phi``, ``nu2`` and ``capture_threshold = 1 - kappa^2`` -- the capture a
    group must exceed before its boundary functional is even positive.  Nothing
    here depends on the learned filter, so it is computed once per graph.
    """

    uniform = getattr(geometry, "kind", "symmetric") == "combinatorial"
    adjacency = geometry.adjacency
    _r, _c, _v, degrees, w_min = _edge_stats(adjacency)
    lam = float(lambda_max if lambda_max is not None else geometry.lambda_max)
    # d~_min is global: block B lies outside S, so its volume is bounded only by
    # the graph's smallest augmented degree
    d_min = 1.0 if uniform else float((degrees + 1.0).min())

    vols, phis, nus, kappas = [], [], [], []
    for pattern in patterns:
        nodes = getattr(pattern, "node_indices", pattern)
        nu2, phi, vol = boundary_irregularity(adjacency, nodes, uniform=uniform)
        vols.append(vol)
        phis.append(phi)
        nus.append(nu2)
        # both floors are valid; the lemma says to use their maximum
        kappas.append(
            max(
                adjacency_floor(vol, phi, tau, lam, w_min),
                degree_floor(vol, phi, tau, lam, d_min),
            )
        )

    dtype = adjacency.dtype
    device = adjacency.device
    to = lambda xs: torch.tensor(xs, dtype=dtype, device=device)  # noqa: E731
    kappa = to(kappas)
    return {
        "kappa": kappa,
        "vol": to(vols),
        "phi": to(phis),
        "nu2": to(nus),
        "capture_threshold": (1.0 - kappa**2).clamp_min(0.0),
        "w_min": float(w_min),
        "d_min": float(d_min),
        "lambda_max": lam,
        # the conflict bound of Prop. full-capture-forces-confusability: the
        # confusability any FULLY captured target is forced to carry
        "chi_forced": to(
            [nu2 / ((lam + tau) * (phi + tau)) for nu2, phi in zip(nus, phis)]
        ),
    }


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------


def boundary_functional(
    capture: torch.Tensor,
    kappa: torch.Tensor,
    *,
    softplus_beta: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""``s_boundary = (1/C)[(kappa - sqrt(1-C))_+]^2``, per group.

    ``softplus_beta > 0`` selects a *surrogate* that keeps a capture gradient
    below the certificate's feasibility threshold ``C > 1 - kappa^2``.  Without
    one, a group under that threshold contributes exactly zero and the objective
    has no capture signal at all -- the normal state of affairs on a large graph,
    where ``kappa ~ sqrt(tau w_min / vol S)`` is small.

    Two things have to happen together for the surrogate to be usable:

    * the hinge is smoothed to ``softplus(beta x)/beta``;
    * the ``1/C`` amplification is frozen at the feasibility threshold below it,
      ``1/max(C, 1 - kappa^2)``.

    The second is not cosmetic.  With a smoothed hinge and a live ``1/C``, the
    functional is *decreasing* in capture wherever the hinge is inactive (the
    ``1/C`` factor swamps the tiny hinge), so the surrogate would reward exactly
    the wrong thing -- measured at ``beta = 1``, ``kappa = 0.20``: 7.06 at
    ``C = 0.02`` against 0.45 at ``C = 0.90``.  Freezing the denominator makes the
    surrogate monotone in ``C``.  The frozen denominator is also exactly the
    right one in the certified regime -- ``max(C, 1 - kappa^2) = C`` whenever the
    hinge is active -- so the surrogate converges to the exact functional there
    at rate ``O(1/beta)`` (measured: 1e-4 at ``beta = 100``, 6e-17 at
    ``beta = 5000``).  Choose ``beta`` on the scale of the slack: much larger
    than ``1/|slack|`` and the hard hinge, with its zero gradient, is back to
    within machine precision.
    """

    c = capture.clamp(eps, 1.0)
    slack = kappa - torch.sqrt((1.0 - c).clamp_min(0.0))
    if softplus_beta <= 0.0:
        return slack.clamp_min(0.0) ** 2 / c
    hinge = torch.nn.functional.softplus(slack * softplus_beta) / softplus_beta
    threshold = (1.0 - kappa**2).clamp(eps, 1.0)
    return hinge**2 / torch.maximum(c, threshold)


def certified_margin_terms(
    capture: torch.Tensor,
    chi: torch.Tensor,
    kappa: torch.Tensor,
    *,
    alpha: float = 0.0,
    softplus_beta: float = 0.0,
) -> dict:
    """Everything the training loop and the report need, in one call.

    ``objective`` is what the optimizer ascends (soft-min of the smoothed
    per-group margins).  ``certificate`` is the *exact* hard-clamped minimum
    margin: by the objective-certificate theorem, if it is positive then
    raw-score Ward recovers every training group exactly.  The two differ only
    through ``softplus_beta``.
    """

    s_soft = boundary_functional(capture, kappa, softplus_beta=softplus_beta)
    margins_soft = s_soft - chi
    s_hard = boundary_functional(capture, kappa, softplus_beta=0.0)
    margins_hard = (s_hard - chi).detach()
    return {
        "objective": soft_min(margins_soft, alpha),
        "margins": margins_soft,
        "certificate": float(margins_hard.min()) if margins_hard.numel() else 0.0,
        "margins_hard": margins_hard,
        "s_boundary": s_hard.detach(),
        "n_certified": int((margins_hard > 0).sum()),
        # how many groups are even eligible: s_boundary > 0 needs C > 1 - kappa^2
        "feasible_capture": int((s_hard.detach() > 0).sum()),
    }


# ---------------------------------------------------------------------------
# feature-side diagnostic: what the dictionary can reach at all
# ---------------------------------------------------------------------------


def reachability_index(
    dictionary_gram: torch.Tensor,
    b: torch.Tensor,
    v_resolvent_energy: torch.Tensor,
    *,
    rank_tol: float = 1e-10,
) -> torch.Tensor:
    r"""``R_K(S) = b^T G_K^{-1} b / (v_S^T M_tau^{-1} v_S)`` with ``b = T_K^T v_S``.

    The exact fraction of the group's resolvent direction the propagated features
    can express, computable per group *before* any training -- the feature-side
    complement of the graph-side floor.  ``delta_min(S)^2 = 1 - R_K(S)`` is the
    best resolvent proximity any target inside the dictionary can achieve, so
    ``R_K`` near 1 means the filter bank could in principle do the job and
    ``R_K`` near 0 means no amount of training will.

    The identity needs no resolvent solve on the dictionary side: by the master
    identity ``T_K^T M_tau z_S^* = T_K^T v_S``, the labels enter as raw
    indicators.  Only the scalar normalizer ``v_S^T M_tau^{-1} v_S`` needs one
    sparse solve per group.

    ``G_K`` is inverted by a rank-revealing eigendecomposition rather than a
    ridge: a Chebyshev dictionary is badly conditioned, and a ridge turns the
    ``M_tau``-orthogonal projector onto its range into a contraction, which
    breaks the identity ``delta_min^2 = 1 - R_K`` at the third decimal.
    Truncating instead keeps the projector exact on the numerically meaningful
    part of the span, which is also the only part any filter can use.
    """

    g = 0.5 * (dictionary_gram + dictionary_gram.T)
    evals, evecs = torch.linalg.eigh(g)
    keep = evals > rank_tol * evals.max().clamp_min(0.0)
    if b.ndim == 1:
        b = b.unsqueeze(1)
    coeff = (evecs[:, keep].T @ b) / evals[keep].unsqueeze(1).sqrt()
    num = (coeff * coeff).sum(dim=0)
    return (num / v_resolvent_energy.clamp_min(1e-300)).clamp(0.0, 1.0)


def resolvent_energy(
    geometry, columns: torch.Tensor, tau: float, *, tol: float = 1e-10,
    max_iter: int = 2000,
) -> torch.Tensor:
    r"""``v^T M_tau^{-1} v`` for each column of ``columns``, by conjugate gradients.

    The one genuinely non-local quantity the reachability index needs.  ``M_tau``
    is SPD with condition number at most ``(lambda_max + tau)/tau``, so CG
    converges in a number of iterations that does not grow with ``N``; all
    columns are advanced together.
    """

    b = columns
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum(dim=0)
    target = tol * (b * b).sum(dim=0).sqrt().clamp_min(1e-300)
    for _ in range(max_iter):
        ap = geometry.m_apply(p, tau)
        denom = (p * ap).sum(dim=0).clamp_min(1e-300)
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).sum(dim=0)
        if bool((rs_new.sqrt() <= target).all()):
            break
        p = r + (rs_new / rs.clamp_min(1e-300)) * p
        rs = rs_new
    return (b * x).sum(dim=0).clamp_min(0.0)


def dictionary_reachability(
    geometry,
    patterns: Sequence,
    dictionary_gram: torch.Tensor,
    raw_rhs: torch.Tensor,
    tau: float,
    *,
    rank_tol: float = 1e-8,
) -> torch.Tensor:
    """``R_K(S_j)`` for every training group, from the precomputed kernels.

    ``dictionary_gram`` is ``G_K = <<T_K, T_K>>_{M_tau}`` (the fit's Gram kernel,
    flattened) and ``raw_rhs`` is ``b_j = T_K^T v_j`` (the raw-indicator RHS the
    resolvent regression uses).  Everything but the scalar normalizer is already
    on hand, so this costs one batched CG solve.
    """

    V = geometry.indicators(list(patterns))
    energy = resolvent_energy(geometry, V, tau)
    return reachability_index(dictionary_gram, raw_rhs, energy, rank_tol=rank_tol)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def run_validation_suite(seed: int = 11, verbose: bool = True) -> None:
    """Check every claim this module implements, against direct computation.

    Mirrors the reference experiment: a planted-group graph under the normalized
    Laplacian with self-loops and degree-weighted indicators.
    """

    import scipy.linalg as sla

    rng = np.random.default_rng(seed)
    n = 240
    groups = [list(range(0, 12)), list(range(30, 42)), list(range(60, 72))]
    tau = 0.5

    W = (rng.random((n, n)) < 0.02).astype(float)
    W = np.triu(W, 1)
    W = W + W.T
    for S in groups:
        for i in S:
            for j in S:
                if i < j and rng.random() < 0.9:
                    W[i, j] = W[j, i] = 1.0
    for a in groups:  # keep the groups pairwise non-adjacent (class-channel test)
        for b in groups:
            if a is not b:
                W[np.ix_(a, b)] = 0.0
    deg = W.sum(1)
    for i in np.where(deg == 0)[0]:
        j = 100 + int(rng.integers(0, 100))
        W[i, j] = W[j, i] = 1.0

    w_t = W + np.eye(n)
    d_t = w_t.sum(1)
    a_hat = np.diag(d_t**-0.5) @ w_t @ np.diag(d_t**-0.5)
    lap = np.eye(n) - a_hat
    lam_max = float(sla.eigvalsh(lap)[-1])
    M = lap + tau * np.eye(n)
    m_inv = np.linalg.inv(M)

    def v_of(block):
        v = np.zeros(n)
        v[block] = np.sqrt(d_t[block])
        return v / np.linalg.norm(v)

    def m_norm(x):
        return math.sqrt(float(x @ M @ x))

    adjacency = torch.sparse_coo_tensor(
        torch.tensor(np.array(np.nonzero(W))),
        torch.tensor(W[np.nonzero(W)], dtype=torch.float64),
        (n, n),
    ).coalesce()

    # (1) boundary irregularity matches the direct definition
    S = groups[0]
    nu2, phi, vol = boundary_irregularity(adjacency, S)
    Sc = [i for i in range(n) if i not in S]
    cut = W[np.ix_(S, Sc)].sum()
    vol_ref = d_t[S].sum()
    phi_ref = cut / vol_ref
    rho = W[np.ix_(S, Sc)].sum(1) / d_t[S]
    nu2_ref = float((d_t[S] * (rho - phi_ref) ** 2).sum() / vol_ref)
    assert abs(vol - vol_ref) < 1e-9, f"vol {vol} != {vol_ref}"
    assert abs(phi - phi_ref) < 1e-12, f"phi {phi} != {phi_ref}"
    assert abs(nu2 - nu2_ref) < 1e-12, f"nu2 {nu2} != {nu2_ref}"

    # (2) the conflict bound: a fully captured target carries at least chi_forced
    def chi_of(Z, block):
        Z = np.atleast_2d(np.asarray(Z).T).T
        gram = Z.T @ M @ Z
        U = Z @ np.linalg.inv(sla.sqrtm(gram).real)
        s = len(block)
        e_s = np.zeros((n, s))
        e_s[block, np.arange(s)] = 1.0
        q, _ = np.linalg.qr(
            np.column_stack([e_s.T @ v_of(block), rng.standard_normal((s, s - 1))])
        )
        f_b = e_s @ q[:, 1:]
        f_hat = f_b @ np.linalg.inv(np.linalg.cholesky(f_b.T @ M @ f_b)).T
        t = U.T @ M @ f_hat
        return float(np.linalg.eigvalsh(t.T @ t)[-1]), U

    chi_indicator, _ = chi_of(v_of(S).reshape(-1, 1), S)
    chi_forced = nu2 / ((lam_max + tau) * (phi + tau))
    assert chi_indicator >= chi_forced - 1e-12, (
        f"conflict bound violated: chi(span v_S)={chi_indicator} < {chi_forced}"
    )

    # the seed family M^{-s} v_S trades capture for confusability, ending at chi=0
    evals, evecs = np.linalg.eigh(lap)
    path = []
    for t_s in (0.0, 0.5, 1.0):
        z = evecs @ (((evals + tau) ** (-t_s)) * (evecs.T @ v_of(S)))
        chi_t, u_t = chi_of(z.reshape(-1, 1), S)
        v_hat = v_of(S) / math.sqrt(phi + tau)
        y = u_t @ (u_t.T @ M @ v_hat)
        path.append((t_s, float(y @ M @ y), chi_t))
    assert path[0][1] > 0.999, "indicator end should have capture 1"
    assert path[-1][2] < 1e-10, f"resolvent end should have chi 0, got {path[-1][2]}"
    assert path[-1][2] < path[0][2], "chi should fall along the seed family"

    # (3) both floors lower-bound the true alignment on adjacent boundary pairs
    stats = group_margin_stats(
        _FakeGeometry(adjacency, lam_max), [_FakePattern(S)], tau
    )
    kappa_uniform = float(stats["kappa"][0])
    v_s = v_of(S)
    v_hat = v_s / math.sqrt(phi + tau)
    violations_pc = violations_am = tested = 0
    for _ in range(1500):
        k = int(rng.integers(1, len(S)))
        block_a = list(rng.choice(S, size=k, replace=False))
        seed_node = int(rng.integers(0, n))
        while seed_node in S:
            seed_node = int(rng.integers(0, n))
        block_b = [seed_node]
        if rng.random() < 0.6:
            nbrs = [
                j
                for j in range(n)
                if j not in S and j not in block_b and W[block_b, j].sum() > 0
            ]
            if nbrs:
                block_b += list(
                    rng.choice(
                        nbrs, size=min(len(nbrs), int(rng.integers(1, 5))), replace=False
                    )
                )
        w_ab = W[np.ix_(block_a, block_b)].sum()
        if w_ab == 0:
            continue
        tested += 1
        vol_a, vol_b = d_t[block_a].sum(), d_t[block_b].sum()
        total = vol_a + vol_b
        g = math.sqrt(vol_b / total) * v_of(block_a) - math.sqrt(
            vol_a / total
        ) * v_of(block_b)
        g_m = m_norm(g)
        alignment = float(g @ M @ v_hat) / g_m
        per_candidate = candidate_alignment_floor(
            vol_a, vol_b, w_ab, vol, phi, tau, g_m
        )
        if alignment < per_candidate - 1e-10:
            violations_pc += 1
        if alignment < kappa_uniform - 1e-10:
            violations_am += 1  # kappa_uniform is already the max of both floors
    assert tested > 100, f"too few adjacent boundary candidates tested ({tested})"
    assert violations_pc == 0, f"{violations_pc}/{tested} per-candidate violations"
    assert violations_am == 0, f"{violations_am}/{tested} uniform floor violations"

    # (4) reachability index equals the direct projection distance
    degree = 8
    sigma = 0.3
    all_s = sorted(set().union(*[set(g) for g in groups]))
    x_class = np.zeros(n)
    x_class[all_s] = 1.0
    x_class = x_class + sigma * rng.standard_normal(n)
    X = np.column_stack([x_class] + [rng.standard_normal(n) for _ in range(3)])
    cols = []
    for a in range(X.shape[1]):
        p0, p1 = X[:, a], a_hat @ X[:, a]
        cols += [p0, p1]
        for _k in range(2, degree + 1):
            p0, p1 = p1, 2 * a_hat @ p1 - p0
            cols.append(p1)
    t_k = np.column_stack(cols)
    g_k = t_k.T @ M @ t_k
    b = t_k.T @ v_s
    energy = float(v_s @ m_inv @ v_s)
    # A Chebyshev dictionary Gram is severely ill-conditioned, so the truncation
    # level is what makes the projector -- and hence the identity -- meaningful.
    rank_tol = 1e-8
    r_k = float(
        reachability_index(
            torch.tensor(g_k),
            torch.tensor(b),
            torch.tensor([energy], dtype=torch.float64),
            rank_tol=rank_tol,
        )[0]
    )
    # direct: M_tau-orthonormalize the dictionary with the SAME rank truncation
    # and measure the residual of the normalized resolvent direction
    z_star = m_inv @ v_s
    z_hat = z_star / m_norm(z_star)
    ev, evec = np.linalg.eigh(0.5 * (g_k + g_k.T))
    keep = ev > rank_tol * ev.max()
    u_full = t_k @ (evec[:, keep] / np.sqrt(ev[keep]))
    orth_err = float(
        np.abs(u_full.T @ M @ u_full - np.eye(int(keep.sum()))).max()
    )
    direct = m_norm(z_hat - u_full @ (u_full.T @ M @ z_hat)) ** 2
    assert abs((1.0 - r_k) - direct) < 1e-6, (
        f"reachability index {1 - r_k} != direct projection {direct}"
    )

    # informative features must reach further than random ones
    x_rand = np.column_stack([rng.standard_normal(n) for _ in range(4)])
    cols = []
    for a in range(4):
        p0, p1 = x_rand[:, a], a_hat @ x_rand[:, a]
        cols += [p0, p1]
        for _k in range(2, degree + 1):
            p0, p1 = p1, 2 * a_hat @ p1 - p0
            cols.append(p1)
    t_rand = np.column_stack(cols)
    g_rand = t_rand.T @ M @ t_rand
    r_rand = float(
        reachability_index(
            torch.tensor(g_rand),
            torch.tensor(t_rand.T @ v_s),
            torch.tensor([energy], dtype=torch.float64),
            rank_tol=rank_tol,
        )[0]
    )
    assert r_k > r_rand, f"informative features did not help: {r_k} <= {r_rand}"

    # (5) class-channel sufficiency: ONE resolvent channel zeroes chi on every
    # group of the class
    z_class = m_inv @ (v_of(groups[0]) + v_of(groups[1]))
    for gi in (0, 1):
        chi_c, _ = chi_of(z_class.reshape(-1, 1), groups[gi])
        assert chi_c < 1e-12, f"class channel chi(S{gi + 1}) = {chi_c} != 0"

    # (6) the objective's own algebra
    cap = torch.tensor([0.999, 0.5, 1.0], dtype=torch.float64)
    chi_v = torch.tensor([0.001, 0.02, 0.0], dtype=torch.float64)
    kap = torch.tensor([0.2, 0.05, 0.3], dtype=torch.float64)
    hard = boundary_functional(cap, kap, softplus_beta=0.0)
    ref = torch.tensor(
        [
            max(0.2 - math.sqrt(1 - 0.999), 0.0) ** 2 / 0.999,
            max(0.05 - math.sqrt(0.5), 0.0) ** 2 / 0.5,
            0.3**2 / 1.0,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(hard, ref, atol=1e-12), f"{hard} != {ref}"
    smooth = boundary_functional(cap, kap, softplus_beta=400.0)
    assert torch.allclose(smooth, hard, atol=1e-3), "softplus should track the hinge"
    # it must keep a gradient where the hard clamp has none ...
    cap_g = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    boundary_functional(
        cap_g, torch.tensor([0.05], dtype=torch.float64), softplus_beta=2.0
    ).sum().backward()
    assert float(cap_g.grad) > 0.0, "surrogate has no capture gradient"
    # ... and it must be MONOTONE in capture, which the naive smoothing is not
    sweep = torch.linspace(0.01, 0.999, 60, dtype=torch.float64)
    for beta in (0.5, 1.0, 2.0, 5.0):
        vals = boundary_functional(
            sweep, torch.full_like(sweep, 0.2047), softplus_beta=beta
        )
        assert bool((vals[1:] >= vals[:-1] - 1e-15).all()), (
            f"surrogate is not monotone in capture at beta={beta}"
        )
    # and agree with the exact functional wherever the hinge is live
    kap_hi = torch.full_like(sweep, 0.6)
    live = sweep > 1.0 - 0.6**2
    exact_live = boundary_functional(sweep, kap_hi, softplus_beta=0.0)[live]
    errs = [
        float(
            (boundary_functional(sweep, kap_hi, softplus_beta=b)[live] - exact_live)
            .abs()
            .max()
        )
        for b in (100.0, 1000.0, 10000.0)
    ]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-12, (
        f"surrogate must converge to the exact functional in the certified "
        f"regime; errors {errs}"
    )

    terms = certified_margin_terms(cap, chi_v, kap, alpha=0.0, softplus_beta=0.0)
    assert abs(terms["certificate"] - float((hard - chi_v).min())) < 1e-12
    assert terms["n_certified"] == int(((hard - chi_v) > 0).sum())

    if verbose:
        print(f"  conflict bound:      chi(span v_S) = {chi_indicator:.6f} >= "
              f"nu^2/((lmax+tau)(Phi+tau)) = {chi_forced:.6f}   (nu^2 = {nu2:.4f})")
        print("  seed family (s, capture, chi): "
              + "  ".join(f"({s:.2f}, {c:.4f}, {x:.2e})" for s, c, x in path))
        print(f"  boundary floors:     0 per-candidate and 0 uniform violations on "
              f"{tested} adjacent boundary candidates (kappa = {kappa_uniform:.4f})")
        print(f"  reachability:        R_K = {r_k:.4f} informative vs {r_rand:.4f} "
              f"random-only;  |1-R_K - direct| = {abs((1 - r_k) - direct):.2e} "
              f"(dictionary Gram orthonormality {orth_err:.1e})")
        print("  class channel:       chi = 0 (machine precision) on both groups")
        print(f"  objective algebra:   hinge and certificate exact; surrogate "
              f"monotone in capture and -> exact in the certified regime "
              f"(err {errs[0]:.1e} -> {errs[2]:.1e} as beta 1e2 -> 1e4)")
        print("certified_margin validation suite: all checks passed")


class _FakeGeometry:
    """Minimal stand-in so the suite can exercise :func:`group_margin_stats`."""

    kind = "symmetric"

    def __init__(self, adjacency, lambda_max):
        self.adjacency = adjacency
        self.lambda_max = lambda_max


class _FakePattern:
    def __init__(self, nodes):
        self.node_indices = nodes


if __name__ == "__main__":  # pragma: no cover
    run_validation_suite()
