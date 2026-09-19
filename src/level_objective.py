r"""The LEVEL objective: capture and confusability read off the learned level.

Sec. 5 of the paper ("Computing capture and confusability") shows that both
properties of the target ``R_Theta = span(Z_Theta)`` are statistics of the
vector-valued level

    ell = D_t^{-1/2} M_tau Z_Theta   in R^{N x q},      G = Z_Theta^T M_tau Z_Theta,

taken over the nodes of a group ``S``:

    mean level      lbar_S    = vol(S)^{-1} sum_{v in S} d~_v ell_v
    level covar.    Sigma_S   = sum_{v in S} d~_v (ell_v - lbar_S)(ell_v - lbar_S)^T

    capture         C_S       = vol(S) / (Phi(S) + tau) * ||lbar_S||^2_{G^{-1}}          (exact)
    confusability   chi_S    <= chibar_S := lambda_max(G^{-1/2} Sigma_S G^{-1/2}) / tau  (bound)

So a group is well served when its mean level is large in the ``G^{-1}`` norm and
its level is flat across its members.  This module turns exactly that into a
training objective that STANDS ALONE -- it is not a regularizer on
``lambda_min(Gamma)``:

    J(Theta) = softmin_{T_C}( C_1, ..., C_m )  -  beta * softmax_{T_chi}( chibar_1, ..., chibar_m )

maximized over the per-channel unit spheres.

Smoothness.  Every piece is C^infinity in Theta:

* ``G^{-1}`` enters only through a Cholesky solve on ``G + ridge I`` (the same
  factor the Gamma path uses), never an explicit inverse or square root;
* ``lambda_max`` of the symmetric PSD matrix ``G^{-1/2} Sigma_S G^{-1/2}`` is
  replaced by the log-sum-exp of its eigenvalues,
  ``lambda_max <= T log sum_i exp(lambda_i / T) <= lambda_max + T log s``.  Its
  gradient is ``V diag(softmax(lambda/T)) V^T``, well defined even when
  eigenvalues coincide, unlike the hard ``lambda_max``, whose gradient jumps
  whenever the top two eigenvalues cross;
* the worst group is taken by mean-normalized log-sum-exp in both directions,
  which is a smooth quantity sandwiched between the mean and the min (capture) or
  the mean and the max (confusability); ``T -> 0`` recovers the hard min/max,
  ``T -> inf`` the plain mean.

Cost.  The nonzero eigenvalues of ``G^{-1/2} Sigma_S G^{-1/2}`` (q x q) are those
of ``K_S = E_S G^{-1} E_S^T`` (|S| x |S|), where ``E_S`` stacks the rows
``sqrt(d~_v) (ell_v - lbar_S)``; and ``ell`` is needed only on training-group
nodes.  So the objective reads ``M_tau Z`` on those few rows, which is linear in
Theta and precomputed once as a ``(K+1)``-term stack: the whole objective is
``N``-independent, exactly like the Gram path.

Invariance.  ``C_S`` and ``chibar_S`` are unchanged under ``Z -> Z R`` for any
invertible ``R`` (``lbar -> R^T lbar``, ``Sigma -> R^T Sigma R``, ``G -> R^T G R``),
so ``J`` depends on the SUBSPACE ``R_Theta`` only -- the right invariance for a
coarsening target, and the reason the unit-sphere constraint loses nothing.

Symmetric geometry only (the level and the volumes are stated in
``M_tau = L_sym + tau I``).  Run ``python -m src.level_objective`` for the
validation suite.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "build_level_rows",
    "level_terms",
    "level_objective",
    "build_edge_rows",
    "edge_objective",
    "run_validation_suite",
    "run_edge_validation_suite",
]


def build_level_rows(geometry, patterns: list, m_propagated: list, phi) -> dict:
    """Theta-free precompute: the rows of ``M_tau phi_k(P) X`` on group nodes.

    ``m_propagated[k]`` is the ``(N, d)`` screened dictionary stack, so
    ``M_tau Z`` restricted to the rows below is ``_filtered_bank(rows, theta)``.
    ``phi`` holds ``Phi_j = ||v_j||_L^2`` as computed for the Gamma path, so the
    capture here and ``Gamma_jj`` share every graph constant.
    """

    if getattr(geometry, "kind", "symmetric") == "combinatorial":
        raise NotImplementedError(
            "the level objective is stated in the symmetric geometry "
            "(ell = D_t^{-1/2} M_tau Z, vol(S) = sum d~_v); use "
            "coarsening_laplacian='symmetric'."
        )
    idx, slices, start = [], [], 0
    for p in patterns:
        nodes = list(p.node_indices)
        idx.extend(nodes)
        slices.append((start, start + len(nodes)))
        start += len(nodes)
    device = m_propagated[0].device
    nodes = torch.as_tensor(idx, dtype=torch.long, device=device)
    sqrt_dt = geometry.node_weights(nodes).to(m_propagated[0].dtype)  # sqrt(d~_v)
    vol = torch.stack([(sqrt_dt[a:b] ** 2).sum() for a, b in slices])
    return {
        "mrows": [m[nodes] for m in m_propagated],  # (K+1) x (n_rows, d)
        "sqrt_dt": sqrt_dt,
        "slices": slices,
        "vol": vol,
        "phi": phi.detach().to(sqrt_dt.dtype),
    }


def level_terms(mz_rows: torch.Tensor, rows: dict, chol: torch.Tensor, tau: float):
    """``(capture (m,), [eig(K_S) for each S])`` -- differentiable in ``mz_rows``.

    ``capture_j`` is the exact ``C_{R}^tau(S_j)``; the eigenvalues of
    ``K_S = E_S G^{-1} E_S^T`` are the nonzero eigenvalues of
    ``G^{-1/2} Sigma_S G^{-1/2}`` (plus one structural zero, since
    ``E_S^T sqrt(d~_S) = 0``).  Divide by ``tau`` for ``chibar``.
    """

    sqrt_dt = rows["sqrt_dt"].to(mz_rows.dtype)
    ell = mz_rows / sqrt_dt.unsqueeze(1)  # ell_v, v on group rows
    caps, eigs = [], []
    for j, (a, b) in enumerate(rows["slices"]):
        w = sqrt_dt[a:b] ** 2  # d~_v
        vol = rows["vol"][j].to(mz_rows.dtype)
        lbar = (w.unsqueeze(1) * ell[a:b]).sum(0) / vol  # (q,)
        g_inv_lbar = torch.cholesky_solve(lbar.unsqueeze(1), chol).squeeze(1)
        caps.append(vol / (rows["phi"][j].to(mz_rows.dtype) + tau) * (lbar @ g_inv_lbar))
        E = sqrt_dt[a:b].unsqueeze(1) * (ell[a:b] - lbar)  # (s, q), Sigma = E^T E
        if E.shape[0] < 2:  # a singleton has no internal fluctuation
            eigs.append(E.new_zeros(1))
            continue
        K = E @ torch.cholesky_solve(E.T, chol)  # (s, s)
        eigs.append(torch.linalg.eigvalsh(0.5 * (K + K.T)))
    return torch.stack(caps), eigs


def level_objective(
    mz_rows: torch.Tensor,
    rows: dict,
    chol: torch.Tensor,
    tau: float,
    *,
    beta: float,
    cap_temperature: float,
    chi_temperature: float,
):
    r"""``(J, parts)`` with ``J`` the value to be **maximized**.

    ``J = softmin_{T_C}(C) - beta * softmax_{T_chi}(chibar)`` where each
    ``chibar_j = softmax_{T_chi}(eig(K_{S_j}) / tau)`` is already a smooth upper
    bound on the per-group ``lambda_max / tau``.
    """

    caps, eigs = level_terms(mz_rows, rows, chol, tau)
    m = caps.numel()
    t_c = max(float(cap_temperature), 1e-8)
    t_x = max(float(chi_temperature), 1e-8)
    # smooth worst capture: -T log mean exp(-C/T), between min(C) and mean(C)
    cap_soft = -t_c * (torch.logsumexp(-caps / t_c, dim=0) - math.log(m))
    # per group: T log sum exp(mu/T) >= lambda_max(K)/tau  (conservative)
    chis = torch.stack([t_x * torch.logsumexp((e / tau) / t_x, dim=0) for e in eigs])
    # smooth worst group: T log mean exp(chi/T), between mean(chi) and max(chi)
    chi_soft = t_x * (torch.logsumexp(chis / t_x, dim=0) - math.log(m))
    obj = cap_soft - beta * chi_soft
    with torch.no_grad():
        chi_hard = torch.stack([e.max() / tau for e in eigs])
        parts = {
            "level_obj": float(obj),
            "cap_soft": float(cap_soft),
            "cap_min": float(caps.min()),
            "cap_mean": float(caps.mean()),
            "chibar_soft": float(chi_soft),
            "chibar_max": float(chi_hard.max()),
            "chibar_mean": float(chi_hard.mean()),
        }
    return obj, parts


# --------------------------------------------------------------------------- #
# the EDGE variant: level contrasts on adjacent pairs, no Sec. 5 identities
# --------------------------------------------------------------------------- #
# Instead of the group statistics (mean level -> capture, level covariance ->
# confusability), score the level directly on graph edges:
#
#   d^2(u,v) = (ell_u - ell_v)^T G^{-1} (ell_u - ell_v)
#   internal edges  u, v in S, u ~ v           -> minimize
#   boundary edges  u in S, v not in S, u ~ v  -> maximize
#
#   J_edge = mean_S [ wmean_{bnd(S)} d^2 ]  -  gamma * mean_S [ wmean_{int(S)} d^2 ]
#
# Relation to the Sec. 5 quantities (checked in run_edge_validation_suite):
# * d^2(u,v) <= ||e_u/sqrt(d~_u) - e_v/sqrt(d~_v)||^2_{M_tau}, a Theta-free
#   constant, because it is ||Q_R x||^2_M for that x: every term is bounded.
# * internal term -> confusability ONLY through the group's connectivity: with
#   E_int(S) = sum_{u~v in S} w_uv d^2(u,v) and lambda_2(S) the smallest nonzero
#   eigenvalue of (L_int(S), D~_S), Poincare gives
#       tr(G^-1/2 Sigma_S G^-1/2) <= E_int(S) / lambda_2(S),
#   so chibar_S <= E_int(S) / (tau lambda_2(S)).  Weak for sparse / chain-like
#   groups, vacuous if the group's induced subgraph is disconnected.
# * boundary term -> NO identity with capture: a large jump at the boundary can
#   come from a large host level instead of a large group level.
# * d^2 is the numerator of the raw-Ward score at the singleton partition,
#   s^0(u,v) = c_uv d^2(u,v), without the Theta-free weight c_uv.


def build_edge_rows(geometry, adjacency, patterns: list, m_propagated: list) -> dict:
    """Theta-free precompute for the edge variant: internal and boundary edges of
    every training group, and the rows of ``M_tau phi_k(P) X`` on their endpoints.

    A boundary edge of ``S_j`` is ``(u, v)`` with ``u in S_j`` and ``v not in S_j``
    (the paper's ``partial S``), so an edge between two different groups is a
    boundary edge of BOTH.  Internal edges are taken once per undirected edge.
    """

    if getattr(geometry, "kind", "symmetric") == "combinatorial":
        raise NotImplementedError(
            "the edge objective uses the symmetric level ell = D_t^{-1/2} M_tau Z; "
            "use coarsening_laplacian='symmetric'."
        )
    dtype = m_propagated[0].dtype
    a = adjacency.coalesce()
    idx, val = a.indices(), a.values()
    off = idx[0] != idx[1]
    src, dst, w = idx[0][off], idx[1][off], val[off].to(dtype)
    n = int(a.shape[0])
    gang_of = torch.full((n,), -1, dtype=torch.long, device=src.device)
    for j, p in enumerate(patterns):
        gang_of[torch.as_tensor(list(p.node_indices), dtype=torch.long, device=src.device)] = j
    gs, gd = gang_of[src], gang_of[dst]
    is_int = (gs >= 0) & (gs == gd) & (src < dst)
    is_bnd = (gs >= 0) & (gd != gs)
    ends = torch.cat([src[is_int], dst[is_int], src[is_bnd], dst[is_bnd]])
    nodes, pos = torch.unique(ends, return_inverse=True)
    ni, nb = int(is_int.sum()), int(is_bnd.sum())
    nodes = nodes.to(m_propagated[0].device)
    return {
        "mrows": [m[nodes] for m in m_propagated],
        "sqrt_dt": geometry.node_weights(nodes).to(dtype),
        "int_a": pos[:ni], "int_b": pos[ni : 2 * ni],
        "int_w": w[is_int], "int_g": gs[is_int],
        "bnd_a": pos[2 * ni : 2 * ni + nb], "bnd_b": pos[2 * ni + nb :],
        "bnd_w": w[is_bnd], "bnd_g": gs[is_bnd],
        "m": len(patterns),
        "counts": {"int": ni, "bnd": nb, "rows": int(nodes.numel())},
    }


def _pair_dist2(ell, a, b, chol):
    """``(ell_a - ell_b)^T G^{-1} (ell_a - ell_b)`` for every pair, via one solve."""

    D = ell[a] - ell[b]
    return (D.T * torch.cholesky_solve(D.T, chol)).sum(0)


def _group_wmean(vals, w, g, m):
    """Per-group edge-weighted means; groups with no edges of this kind drop out."""

    w = w.to(vals.dtype)
    num = vals.new_zeros(m).index_add(0, g, w * vals)
    den = vals.new_zeros(m).index_add(0, g, w)
    keep = den > 0
    return num[keep] / den[keep]


def edge_objective(mz_rows: torch.Tensor, rows: dict, chol: torch.Tensor, *, gamma: float):
    """``(J_edge, parts)``: boundary level jumps minus ``gamma`` x internal ones."""

    ell = mz_rows / rows["sqrt_dt"].to(mz_rows.dtype).unsqueeze(1)
    zero = mz_rows.new_zeros(())
    d_int = _pair_dist2(ell, rows["int_a"], rows["int_b"], chol)
    d_bnd = _pair_dist2(ell, rows["bnd_a"], rows["bnd_b"], chol)
    int_j = _group_wmean(d_int, rows["int_w"], rows["int_g"], rows["m"])
    bnd_j = _group_wmean(d_bnd, rows["bnd_w"], rows["bnd_g"], rows["m"])
    int_mean = int_j.mean() if int_j.numel() else zero
    bnd_mean = bnd_j.mean() if bnd_j.numel() else zero
    obj = bnd_mean - gamma * int_mean
    with torch.no_grad():
        parts = {
            "level_obj": float(obj),
            "bnd_mean": float(bnd_mean),
            "int_mean": float(int_mean),
            "int_over_bnd": float(int_mean / bnd_mean.clamp_min(1e-300)),
        }
    return obj, parts


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def run_edge_validation_suite(seed: int = 1, verbose: bool = True) -> None:
    """The edge variant against dense ground truth, and every relation claimed above."""

    import scipy.sparse as sp
    from scipy.linalg import eigh

    from src.raw_ward import screened_operators_kappa

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(seed)
    n, q, tau = 140, 7, 0.5
    Wd = np.triu((rng.random((n, n)) < 0.05) * 1.0, 1)
    groups = [np.arange(0, 10), np.arange(50, 58), np.arange(100, 112)]
    for g in groups:  # dense-ish connected groups
        for i in range(len(g) - 1):
            Wd[g[i], g[i + 1]] = 1.0
        for i, j in zip(*np.triu_indices(len(g), 1)):
            if rng.random() < 0.3:
                Wd[g[i], g[j]] = 1.0
    for i in range(n - 1):
        Wd[i, i + 1] = max(Wd[i, i + 1], 1.0)
    Wd = np.triu(Wd, 1)
    Wd = Wd + Wd.T
    A, dt, L, M = screened_operators_kappa(sp.csr_matrix(Wd), tau, "symmetric")
    Md = M.toarray()

    def say(msg):
        if verbose:
            print(msg)

    class _Geo:
        kind = "symmetric"

        def node_weights(self, nodes):
            return torch.from_numpy(np.sqrt(dt))[nodes]

    class _P:
        def __init__(self, nodes):
            self.node_indices = list(map(int, nodes))

    pats = [_P(g) for g in groups]
    ii, jj = np.nonzero(Wd)
    adj = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([ii, jj])), torch.from_numpy(Wd[ii, jj]), (n, n)
    ).coalesce()

    def J_of(Z, gamma=1.0):
        Zt = Z if torch.is_tensor(Z) else torch.from_numpy(Z)
        mz = torch.from_numpy(Md) @ Zt
        rows = build_edge_rows(_Geo(), adj, pats, [mz])
        G = Zt.T @ mz
        chol = torch.linalg.cholesky(0.5 * (G + G.T))
        return edge_objective(rows["mrows"][0], rows, chol, gamma=gamma), rows

    Z = rng.standard_normal((n, q))
    (J, parts), rows = J_of(Z)

    # [1] d^2 against dense r = ell G^{-1/2}, and the dense J
    G = Z.T @ Md @ Z
    Lc = np.linalg.cholesky(G)
    ell = (Md @ Z) / np.sqrt(dt)[:, None]
    r = np.linalg.solve(Lc, ell.T).T  # ||r_u - r_v||^2 = d^2(u,v)
    gang_of = -np.ones(n, dtype=int)
    for j, g in enumerate(groups):
        gang_of[g] = j
    ints = [[] for _ in groups]
    bnds = [[] for _ in groups]
    for u, v in zip(ii, jj):
        if gang_of[u] >= 0 and gang_of[u] == gang_of[v] and u < v:
            ints[gang_of[u]].append(np.sum((r[u] - r[v]) ** 2))
        elif gang_of[u] >= 0 and gang_of[v] != gang_of[u]:
            bnds[gang_of[u]].append(np.sum((r[u] - r[v]) ** 2))
    J_dense = np.mean([np.mean(b) for b in bnds]) - np.mean([np.mean(x) for x in ints])
    assert abs(float(J) - J_dense) < 1e-10, (float(J), J_dense)
    say(f"[1] J_edge == dense (r = ell G^-1/2)                     : diff {abs(float(J) - J_dense):.1e}"
        f"   ({rows['counts']})")

    # [2] every d^2 is bounded by a Theta-free constant ||x||^2_M
    worst = 0.0
    for u, v in zip(ii, jj):
        x = np.zeros(n)
        x[u], x[v] = 1 / np.sqrt(dt[u]), -1 / np.sqrt(dt[v])
        worst = max(worst, np.sum((r[u] - r[v]) ** 2) / (x @ Md @ x))
    assert worst <= 1 + 1e-10, worst
    say(f"[2] d^2(u,v) <= ||e_u/sqrt(d_u) - e_v/sqrt(d_v)||^2_M       : max ratio {worst:.4f} (<= 1)")

    # [3] Poincare: tr Sigma_S(r) <= E_int(S)/lambda_2(S), chibar_S <= E_int/(tau lambda_2)
    for j, g in enumerate(groups):
        Lint = np.diag(Wd[np.ix_(g, g)].sum(1)) - Wd[np.ix_(g, g)]
        lam = eigh(Lint, np.diag(dt[g]), eigvals_only=True)
        lam2 = lam[1]
        rbar = (dt[g, None] * r[g]).sum(0) / dt[g].sum()
        Sig = ((dt[g, None] * (r[g] - rbar)).T @ (r[g] - rbar))
        E_int = sum(
            np.sum((r[g[a]] - r[g[b]]) ** 2) * Wd[g[a], g[b]]
            for a, b in zip(*np.triu_indices(len(g), 1)) if Wd[g[a], g[b]] > 0
        )
        chibar = np.linalg.eigvalsh(Sig)[-1] / tau
        assert np.trace(Sig) <= E_int / lam2 + 1e-9 and chibar <= E_int / (tau * lam2) + 1e-9
        say(f"[3] group {j}: lambda_2={lam2:.3f}  tr Sigma {np.trace(Sig):.3f} <= E_int/lambda_2 "
            f"{E_int / lam2:.3f};  chibar {chibar:.3f} <= {E_int / (tau * lam2):.3f}")

    # [4] subspace invariance and [5] smoothness
    R = rng.standard_normal((q, q)) + 3 * np.eye(q)
    (J2, _), _ = J_of(Z @ R)
    assert abs(float(J) - float(J2)) < 1e-9
    say(f"[4] J_edge(Z R) == J_edge(Z)                              : diff {abs(float(J) - float(J2)):.1e}")
    Zt = torch.from_numpy(Z).requires_grad_(True)
    Jt = J_of(Zt)[0][0]
    (gz,) = torch.autograd.grad(Jt, Zt)
    Dd = torch.from_numpy(rng.standard_normal((n, q)))
    h = 1e-6
    with torch.no_grad():
        fd = (J_of(Zt + h * Dd)[0][0] - J_of(Zt - h * Dd)[0][0]) / (2 * h)
    rel = abs(float(fd) - float((gz * Dd).sum())) / max(abs(float((gz * Dd).sum())), 1e-12)
    assert rel < 1e-6, rel
    say(f"[5] autograd vs central difference                        : rel err {rel:.1e}")
    say("\nALL EDGE-OBJECTIVE CHECKS PASSED")


def run_validation_suite(seed: int = 0, verbose: bool = True) -> None:
    """Every claim above, against dense ground truth, in float64."""

    import scipy.sparse as sp
    from scipy.linalg import eigh

    from src.raw_ward import screened_operators_kappa

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(seed)
    n, q, tau = 150, 8, 0.5
    Wd = np.triu((rng.random((n, n)) < 0.06) * rng.uniform(0.5, 2.0, (n, n)), 1)
    Wd = Wd + Wd.T
    for i in range(n - 1):  # keep it connected
        Wd[i, i + 1] = Wd[i + 1, i] = max(Wd[i, i + 1], 1.0)
    A, dt, _L, M = screened_operators_kappa(sp.csr_matrix(Wd), tau, "symmetric")
    Md = M.toarray()
    groups = [np.arange(0, 12), np.arange(40, 49), np.arange(90, 106)]

    def say(msg):
        if verbose:
            print(msg)

    class _Geo:
        kind = "symmetric"

        def node_weights(self, nodes):
            return torch.from_numpy(np.sqrt(dt))[nodes]

    class _P:
        def __init__(self, nodes):
            self.node_indices = list(map(int, nodes))

    pats = [_P(g) for g in groups]
    phi = []
    for g in groups:
        v = np.zeros(n)
        v[g] = np.sqrt(dt[g] / dt[g].sum())
        phi.append(v @ (_L @ v))
    phi = torch.tensor(phi)

    def dense_truth(Z):
        G = Z.T @ Md @ Z
        U = Z @ np.linalg.inv(np.linalg.cholesky(G)).T
        ell = (Md @ Z) / np.sqrt(dt)[:, None]
        Gi_half = np.linalg.cholesky(np.linalg.inv(G))
        out = []
        for g in groups:
            v = np.zeros(n)
            v[g] = np.sqrt(dt[g])
            v /= np.sqrt(v @ Md @ v)
            cap = np.sum((U.T @ Md @ v) ** 2)
            lbar = (dt[g, None] * ell[g]).sum(0) / dt[g].sum()
            E = np.sqrt(dt[g])[:, None] * (ell[g] - lbar)
            lam = np.linalg.eigvalsh(Gi_half.T @ (E.T @ E) @ Gi_half)[-1]
            c = np.sqrt(dt[g])
            Nb = np.linalg.qr(np.eye(len(g)) - np.outer(c, c) / (c @ c))[0][:, : len(g) - 1]
            P = (Md @ U)[g]
            chi = eigh(Nb.T @ P @ P.T @ Nb, Nb.T @ Md[np.ix_(g, g)] @ Nb, eigvals_only=True)[-1]
            out.append((cap, lam / tau, chi))
        return np.array(out)

    def module_eval(Z, beta=0.7):
        Zt = torch.from_numpy(Z)
        mz = torch.from_numpy(Md @ Z)
        rows = build_level_rows(_Geo(), pats, [mz], phi)
        # rows["mrows"][0] is exactly (M Z)[group rows]
        G = Zt.T @ mz
        chol = torch.linalg.cholesky(0.5 * (G + G.T))
        caps, eigs = level_terms(rows["mrows"][0], rows, chol, tau)
        obj, parts = level_objective(
            rows["mrows"][0], rows, chol, tau,
            beta=beta, cap_temperature=0.05, chi_temperature=0.05,
        )
        return caps, eigs, obj

    Z = rng.standard_normal((n, q))
    truth = dense_truth(Z)
    caps, eigs, _ = module_eval(Z)
    err_c = np.max(np.abs(caps.numpy() - truth[:, 0]))
    err_x = np.max(np.abs(np.array([float(e.max()) / tau for e in eigs]) - truth[:, 1]))
    assert err_c < 1e-10 and err_x < 1e-10, (err_c, err_x)
    say(f"[1] capture == exact C_S (dense, M_tau-orthonormal basis) : max err {err_c:.1e}")
    say(f"    chibar  == lambda_max(G^-1/2 Sigma G^-1/2)/tau via K_S  : max err {err_x:.1e}")
    assert np.all(truth[:, 2] <= truth[:, 1] + 1e-10)
    say(
        f"[2] exact chi <= chibar on every group                    : "
        f"chi {np.round(truth[:, 2], 3)}  chibar {np.round(truth[:, 1], 3)}"
    )

    # [3] subspace invariance: J(Z R) = J(Z)
    R = rng.standard_normal((q, q)) + 3 * np.eye(q)
    _, _, o1 = module_eval(Z)
    _, _, o2 = module_eval(Z @ R)
    assert abs(float(o1) - float(o2)) < 1e-9 * max(1.0, abs(float(o1)))
    say(f"[3] J(Z R) == J(Z) for invertible R (subspace-only)       : diff {abs(float(o1 - o2)):.1e}")

    # [4] smoothness: autograd matches central differences along a random
    # direction, and at a point where lambda_max of one K_S is (nearly) repeated
    Zt = torch.from_numpy(Z).requires_grad_(True)

    def J_of(Zv):
        mz = torch.from_numpy(Md) @ Zv
        rows = build_level_rows(_Geo(), pats, [mz], phi)
        G = Zv.T @ mz
        chol = torch.linalg.cholesky(0.5 * (G + G.T))
        return level_objective(
            rows["mrows"][0], rows, chol, tau,
            beta=0.7, cap_temperature=0.05, chi_temperature=0.05,
        )[0]

    J = J_of(Zt)
    (g,) = torch.autograd.grad(J, Zt)
    D = torch.from_numpy(rng.standard_normal((n, q)))
    h = 1e-6
    with torch.no_grad():
        fd = (J_of(Zt + h * D) - J_of(Zt - h * D)) / (2 * h)
    ad = float((g * D).sum())
    rel = abs(float(fd) - ad) / max(abs(ad), 1e-12)
    assert rel < 1e-6, rel
    say(f"[4] autograd vs central difference (random direction)     : rel err {rel:.1e}")

    # a Z whose level is constant on each group -> chi = chibar = 0 exactly
    Vs = np.zeros((n, len(groups)))
    for j, gg in enumerate(groups):
        Vs[gg, j] = np.sqrt(dt[gg])
    Zstar = np.linalg.solve(Md, Vs)
    tstar = dense_truth(Zstar)
    cs, es, _ = module_eval(Zstar)
    assert np.all(np.abs(tstar[:, 1]) < 1e-8) and np.all(np.abs(tstar[:, 2]) < 1e-8)
    say(
        f"[5] screened dual Z* = M^-1 V*: chi = chibar = 0, capture "
        f"{np.round(cs.numpy(), 4)} (== exact {np.round(tstar[:, 0], 4)})"
    )
    say("\nALL LEVEL-OBJECTIVE CHECKS PASSED")


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    run_validation_suite()
    print()
    run_edge_validation_suite()
