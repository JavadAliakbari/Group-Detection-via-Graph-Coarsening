r"""Closed-form targets for the EDGE objective and the LEVEL objective's trace surrogate.

Every term these objectives use is linear in the M_tau-orthogonal projector onto
the target R = span(Z):

    d^2(u,v) = ||ell_u - ell_v||^2_{G^-1} = ||Q_R x_uv||^2_M,  x_uv = e_u/sqrt(d~_u) - e_v/sqrt(d~_v)
    C_S      = ||Q_R v_hat_S||^2_M
    tr(G^-1/2 Sigma_S G^-1/2) = ||Q_R F_S||^2_{M,F}   (F_S: internal fluctuations of S)

Write R = span(D W), with D = [T_0 X | ... | T_K X] the Chebyshev dictionary
(N x P, P = (K+1) d) and W in R^{P x r}.  Then any such objective is

    J(W) = tr( (W^T S_M W)^{-1} W^T A W ),     S_M = D^T M_tau D,

for ONE fixed P x P matrix A.  Over all W that is a generalized Rayleigh trace,
maximized (Ky Fan) by the leading generalized eigenvectors of (A, S_M); the
optimal rank is the number of positive eigenvalues.

* EDGE objective:  A = sum_bnd c_p y_p y_p^T - gamma * sum_int c_p y_p y_p^T, where
  y_p is the level row-difference of the dictionary and c_p the per-group
  normalized edge weight -- the SAME J_edge, no approximation.
* LEVEL trace surrogate:  A = mean_j a_j a_j^T - (beta/tau) mean_j B_j, with
  a_j = D^T M v_hat_j (capture) and B_j = (MD)_S^T P_perp (MD)_S (within-group level
  variance).  It replaces the level objective's soft-min by a mean and lambda_max
  by the trace (an upper bound), so it is a surrogate, not the same objective.

Two solvers:
  full     -- the global optimum over ALL rank-r subspaces of the dictionary.  That
              is a Chebyshev layer whose outputs MIX input channels
              (z_r = sum_a g_{r,a}(A_hat) x_a), not the per-channel bank.
  channel  -- the same pencil solved independently per channel (the diagonal
              (K+1) x (K+1) blocks), top-H eigenvectors -> theta of shape (H, K+1, d):
              EXACTLY the paper's parameterization, but ignoring cross-channel coupling.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch

def _sym(M):
    return 0.5 * (M + M.T)


def _ridge_whitener(M, rel_ridge=1e-10):
    """W = L^-T with L L^T = M + rel_ridge * mean(diag M) I, so W^T M W ~ I.

    The historical metric of solve_channel: it keeps M's numerically null directions,
    amplified by up to rel_ridge^-1/2."""

    L = torch.linalg.cholesky(
        M + rel_ridge * M.diagonal().mean() * torch.eye(M.shape[0], dtype=M.dtype)
    )
    return torch.linalg.inv(L).T


def _support_whitener(M, rtol=1e-10):
    """W = U_+ diag(s_+^-1/2) on M's positive support (s > rtol * s_max), so W^T M W = I.

    A channel's 33 Chebyshev columns span only ~25 numerically independent
    functions; the dropped directions carry < rtol of the energy.  Ratios of traces
    along them are 0/0 roundoff, which is what made Dinkelbach cycle under the ridge."""

    s, U = torch.linalg.eigh(_sym(M))
    keep = s > rtol * s.max()
    return U[:, keep] / s[keep].sqrt()


def _gen_eig(A, W):
    """Generalized eigenpairs of (A, M) for a whitener W of M (W^T M W = I):
    descending values, M-orthonormal vectors W Q."""

    ev, Q = torch.linalg.eigh(_sym(W.T @ A @ W))
    order = torch.argsort(ev, descending=True)
    return ev[order], W @ Q[:, order]


def _solve_block(
    S, N, P, C, *, penalty, host_weight, heads, scale, form, ratio_iters=200
):
    """One (K+1)x(K+1) block pencil -> (vecs (K+1, heads) with unit columns, info).

    N, P are the reward and the penalty (edge: B_bnd, B_int; level: Cap, Pen/tau),
    C the host matrix or None.  penalty_star = lambda_max(N, P) is the CLIFF: for
    penalty >= penalty_star no direction has a positive objective.

    scale="relative"  penalty    -> penalty * penalty_star
                      host_weight -> host_weight * lambda_max(A_penalty, S) / lambda_max(C, S)
                      (ratio form: host_weight * lambda_max(P, S) / lambda_max(C, S))
    form="difference" top generalized eigenvectors of (N - penalty P - host C, S) in the
                      ridge metric of solve_channel; positive modes only, padded with
                      the top one.
    form="ratio"      Dinkelbach on max tr(T^T N T) / tr(T^T (P + host C) T), T^T S T = I,
                      on S's positive support: rho <- ratio(T), T <- top-heads eigvecs
                      of (N - rho D, S).  rho increases monotonically; stop when it
                      does not, keep the best T.  The penalty is an OUTPUT (rho*); at the
                      fixed point the top-heads eigenvalues of N - rho* D sum to 0.
    """

    star = float(_gen_eig(N, _support_whitener(P))[0][0])
    relative = scale == "relative"
    host = C is not None and host_weight > 0.0
    info = {"penalty_star": star}
    if form == "difference":
        W = _ridge_whitener(S)
        top_C = float(_gen_eig(C, W)[0][0]) if host else 0.0
        pen_abs = penalty * star if relative else penalty
        A = N - pen_abs * P
        host_abs = 0.0
        if host:
            if not relative:
                host_abs = host_weight
            elif top_C > 0.0:
                host_abs = host_weight * max(float(_gen_eig(A, W)[0][0]), 0.0) / top_C
            A = A - host_abs * C
        ev, V = _gen_eig(A, W)
        n_raw = int((ev > 1e-10 * ev.abs().max()).sum())
        n_pos = max(1, min(heads, n_raw))
        vecs = V[:, [i if i < n_pos else 0 for i in range(heads)]]
        info.update(penalty_abs=pen_abs, host_abs=host_abs, n_raw=n_raw)
    else:
        W = _support_whitener(S)
        host_abs = 0.0
        if host:
            top_C = float(_gen_eig(C, W)[0][0])
            if not relative:
                host_abs = host_weight
            elif top_C > 0.0:
                host_abs = host_weight * float(_gen_eig(P, W)[0][0]) / top_C
        D = P + host_abs * C if host else P
        rho, best, T_best, it, converged = 0.0, -float("inf"), None, 0, False
        while it < ratio_iters:
            it += 1
            T = _gen_eig(N - rho * D, W)[1][:, :heads]
            new = float(torch.trace(T.T @ N @ T) / torch.trace(T.T @ D @ T))
            if new > best:
                best, T_best = new, T
            if new <= rho * (1.0 + 1e-13):  # no strict increase: fixed point reached
                converged = True
                break
            rho = new
        ev, _V = _gen_eig(N - best * D, W)
        vecs = T_best
        info.update(
            penalty_abs=best,
            host_abs=host_abs,
            n_raw=int((ev > 1e-10 * ev.abs().max()).sum()),
            ratio_iters=it,
            converged=converged,
            support_dim=int(W.shape[1]),
            fixed_point_topH_sum=float(ev[:heads].sum() / ev.abs().max()),
        )
    return vecs / vecs.norm(dim=0, keepdim=True), info


def host_nodes(adjacency, patterns, mode, *, count=0, seed=0):
    """The known-host set H_t (uniform weights nu_v = 1/|H_t| are applied in host_C).

    mode="neighbours"  every NON-group endpoint of an edge leaving a training
                       group: the groups' immediate surroundings.
    mode="random"      a uniform sample of non-group nodes drawn from the whole
                       graph; ``count`` of them (0 = as many as "neighbours" gives,
                       so the two masks are the same size).

    Both masks exclude the nodes of the TRAINING groups and nothing else.  The
    labels of held-out groups are not available at fit time, so their nodes CAN
    land in H_t -- the spec's "unknown" case, here penalized as background.
    """

    a = adjacency.coalesce()
    idx = a.indices()
    off = idx[0] != idx[1]
    src, dst = idx[0][off], idx[1][off]
    gang_of = torch.full((int(a.shape[0]),), -1, dtype=torch.long, device=src.device)
    for j, p in enumerate(patterns):
        gang_of[
            torch.as_tensor(list(p.node_indices), dtype=torch.long, device=src.device)
        ] = j
    nbr = torch.unique(dst[(gang_of[src] >= 0) & (gang_of[dst] < 0)])
    if mode == "neighbours":
        return nbr
    if mode != "random":
        raise ValueError(
            f"host mode must be 'none', 'neighbours' or 'random', got {mode!r}"
        )
    pool = torch.nonzero(gang_of < 0, as_tuple=False).flatten()
    k = min(int(count) if count > 0 else int(nbr.numel()), int(pool.numel()))
    g = torch.Generator().manual_seed(int(seed))
    return pool[torch.randperm(int(pool.numel()), generator=g)[:k]].sort().values


def host_C(geo, mprop, hosts):
    """C_H = Psi_H^T W_H Psi_H  (P x P, PSD), Psi = D~^{-1/2} M_tau D on the host rows.

    Uniform host weights nu_v = 1/|H| enter as sqrt(nu) row scaling, so W_H is
    never materialized.  theta^T C_H theta = sum_v nu_v (Psi theta)_v^2 is the
    mean squared screened response on the hosts.
    """

    if hosts.numel() == 0:  # no known hosts -> no penalty
        p = sum(m.shape[1] for m in mprop)
        return torch.zeros(p, p, dtype=mprop[0].dtype)
    psi = torch.cat([m[hosts] for m in mprop], 1)
    psi = psi / geo.node_weights(hosts).to(psi.dtype).unsqueeze(1)
    R = psi / float(hosts.numel()) ** 0.5
    C = R.T @ R
    return 0.5 * (C + C.T)

