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

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch

from src.collective_detector import CollectiveBankDetector
from src.level_objective import (
    build_edge_rows,
    build_level_rows,
    edge_objective,
    level_objective,
)
from src.run_collective_bank_detection import _basis_stack, _filtered_bank


# --------------------------------------------------------------------------- #
# pencils
# --------------------------------------------------------------------------- #
def dictionary(geo, X, K, basis, tau):
    prop = _basis_stack(geo.prop, X, K, basis, tau, geometry=geo)
    mprop = [geo.m_apply(p, tau) for p in prop]
    D, MD = torch.cat(prop, 1), torch.cat(mprop, 1)  # column k*d + a
    S_M = D.T @ MD
    return prop, mprop, D, 0.5 * (S_M + S_M.T)


def edge_parts(geo, adjacency, patterns, mprop):
    """(B_bnd, B_int), unsymmetrized: J_edge's matrix is B_bnd - gamma * B_int."""

    rows = build_edge_rows(geo, adjacency, patterns, mprop)
    lev = torch.cat(rows["mrows"], 1) / rows["sqrt_dt"].unsqueeze(1)  # dictionary level rows

    def part(a, b, w, g):
        Y = lev[a] - lev[b]
        den = torch.zeros(rows["m"], dtype=Y.dtype).index_add(0, g, w.to(Y.dtype))
        c = w.to(Y.dtype) / den[g] / int((den > 0).sum())  # per-group mean, then mean over groups
        return Y.T @ (c.unsqueeze(1) * Y)

    return (part(rows["bnd_a"], rows["bnd_b"], rows["bnd_w"], rows["bnd_g"]),
            part(rows["int_a"], rows["int_b"], rows["int_w"], rows["int_g"]))


def edge_A(geo, adjacency, patterns, mprop, gamma):
    """The exact J_edge matrix: J_edge(W) = tr((W^T S_M W)^-1 W^T A W)."""

    B_bnd, B_int = edge_parts(geo, adjacency, patterns, mprop)
    A = B_bnd - gamma * B_int
    return 0.5 * (A + A.T)


def _level_group_terms(geo, patterns, mprop, tau):
    """Per training group: a_j = D^T M v_hat_j (capture) and E_j = P_perp (M D)_S
    (within-group level fluctuation rows); returns (terms, m, P, dtype)."""

    V = geo.indicators(patterns)
    phi = (V * geo.l_apply(V)).sum(0)
    rows = build_level_rows(geo, patterns, mprop, phi)
    MDs = torch.cat(rows["mrows"], 1)  # (M D) on group rows
    terms = []
    for j, (a, b) in enumerate(rows["slices"]):
        s, X = rows["sqrt_dt"][a:b].to(MDs.dtype), MDs[a:b]
        vol = (s * s).sum()
        aj = (s.unsqueeze(1) * X).sum(0) / torch.sqrt(vol * (phi[j] + tau))  # D^T M v_hat_j
        E = X - s.unsqueeze(1) * ((s @ X) / (s @ s)).unsqueeze(0)  # P_perp (M D)_S
        terms.append((aj, E))
    return terms, len(rows["slices"]), MDs.shape[1], MDs.dtype


def level_trace_A(geo, patterns, mprop, beta, tau):
    """Trace surrogate of the level objective: mean capture - (beta/tau) mean tr(level variance)."""

    terms, m, p, dtype = _level_group_terms(geo, patterns, mprop, tau)
    A = torch.zeros(p, p, dtype=dtype)
    for aj, E in terms:
        A += torch.outer(aj, aj) / m - (beta / (m * tau)) * (E.T @ E)
    return 0.5 * (A + A.T)


def level_parts(geo, patterns, mprop, tau):
    """(Cap, Pen): the level trace surrogate's matrix is Cap - beta * Pen."""

    terms, m, p, dtype = _level_group_terms(geo, patterns, mprop, tau)
    Cap = torch.zeros(p, p, dtype=dtype)
    Pen = torch.zeros(p, p, dtype=dtype)
    for aj, E in terms:
        Cap += torch.outer(aj, aj) / m
        Pen += (E.T @ E) / (m * tau)
    return Cap, Pen


def solve_full(S_M, A, r_max, rel_ridge=1e-4):
    """Global max of tr((W^T (S_M + rho I) W)^-1 W^T A W) over rank <= r_max (Ky Fan).

    ``rho = rel_ridge * mean(diag S_M)`` is the pencil analogue of the relative
    ridge used in training.  Without it, S_M (rank-deficient once P > N) is
    whitened by near-zero eigenvalues and roundoff in A becomes spurious positive
    eigen-directions -- the same knife edge the ridge removes from the fit.
    """

    S = S_M + rel_ridge * S_M.diagonal().mean() * torch.eye(S_M.shape[0], dtype=S_M.dtype)
    ev, V = torch.linalg.eigh(0.5 * (S + S.T))
    T = V / ev.sqrt()
    ea, Q = torch.linalg.eigh(0.5 * (T.T @ A @ T + (T.T @ A @ T).T))
    order = torch.argsort(ea, descending=True)
    # A has a huge exact null space (no training pair / group sees it); its zero
    # eigenvalues come out as +-1e-15, so "positive" must be RELATIVE or the target
    # gets padded with arbitrary null-space directions the coarsener then sees.
    n_pos = int((ea > 1e-10 * ea.abs().max()).sum())
    r = max(1, min(r_max, n_pos))
    return T @ Q[:, order[:r]], ea[order], r, n_pos


def solve_channel(S_M, A, K, d, H, rel_ridge=1e-10, counts=None):
    """Per-channel pencils -> theta (H, K+1, d): the paper's parameterization.

    ``counts`` (a list), if given, receives each channel's number of positive
    generalized eigenvalues (before the pad to H heads)."""

    theta = torch.zeros(H, K + 1, d, dtype=S_M.dtype)
    for a in range(d):
        idx = torch.arange(K + 1) * d + a
        Sa, Aa = S_M[idx][:, idx], A[idx][:, idx]
        L = torch.linalg.cholesky(Sa + rel_ridge * Sa.diagonal().mean() * torch.eye(K + 1, dtype=Sa.dtype))
        Li = torch.linalg.inv(L)
        ev, Q = torch.linalg.eigh(Li @ Aa @ Li.T)
        order = torch.argsort(ev, descending=True)
        # only this channel's POSITIVE directions raise J; pad the remaining heads
        # with copies of the top one (duplicates add nothing to the span)
        n_raw = int((ev > 1e-10 * ev.abs().max()).sum())
        if counts is not None:
            counts.append(n_raw)
        n_pos = max(1, min(H, n_raw))
        picks = [int(order[i]) if i < n_pos else int(order[0]) for i in range(H)]
        vecs = Li.T @ Q[:, picks]  # (K+1, H)
        theta[:, :, a] = (vecs / vecs.norm(dim=0, keepdim=True)).T
    return theta


def solve_svd(W, K, d, H):
    """theta (H, K+1, d) from the FULL closed form: per channel, the top-H left
    singular vectors of that channel's coefficient block W[a-rows, :] -- the
    per-channel bank whose span best contains the mixing solution."""

    theta = torch.zeros(H, K + 1, d, dtype=W.dtype)
    for a in range(d):
        idx = torch.arange(K + 1) * d + a
        U, _s, _v = torch.linalg.svd(W[idx], full_matrices=False)  # (K+1, min(K+1, r))
        # a rank-r < H solution has only r directions per channel: pad the extra
        # heads with the top one (duplicates add nothing to the span)
        picks = [i if i < U.shape[1] else 0 for i in range(H)]
        theta[:, :, a] = U[:, picks].T
    return theta


def _sym(M):
    return 0.5 * (M + M.T)


def _ridge_whitener(M, rel_ridge=1e-10):
    """W = L^-T with L L^T = M + rel_ridge * mean(diag M) I, so W^T M W ~ I.

    The historical metric of solve_channel: it keeps M's numerically null directions,
    amplified by up to rel_ridge^-1/2."""

    L = torch.linalg.cholesky(M + rel_ridge * M.diagonal().mean() * torch.eye(M.shape[0], dtype=M.dtype))
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


def _solve_block(S, N, P, C, *, penalty, host_weight, heads, scale, form, ratio_iters=200):
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
        info.update(penalty_abs=best, host_abs=host_abs,
                    n_raw=int((ev > 1e-10 * ev.abs().max()).sum()), ratio_iters=it,
                    converged=converged, support_dim=int(W.shape[1]),
                    fixed_point_topH_sum=float(ev[:heads].sum() / ev.abs().max()))
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
        gang_of[torch.as_tensor(list(p.node_indices), dtype=torch.long, device=src.device)] = j
    nbr = torch.unique(dst[(gang_of[src] >= 0) & (gang_of[dst] < 0)])
    if mode == "neighbours":
        return nbr
    if mode != "random":
        raise ValueError(f"host mode must be 'none', 'neighbours' or 'random', got {mode!r}")
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


def _fresh_features(X, seed):
    """A fresh draw from the synthetic harness's feature law (standardized N(0,1) columns)."""

    g = torch.Generator().manual_seed(int(seed))
    Xr = torch.randn(tuple(X.shape), dtype=torch.float64, generator=g)
    Xr = (Xr - Xr.mean(0, keepdim=True)) / Xr.std(0, keepdim=True).clamp_min(1e-8)
    return Xr.to(device=X.device, dtype=X.dtype)


def fit_channel_closed_form(graphs, *, objective, degree, heads, tau, basis,
                            level_beta=1.0, edge_gamma=10.0, share_filters=False,
                            feature_draws=0, seed=0, host_mode="none", host_weight=0.0,
                            host_count=0, param_scale="absolute", solver_form="difference"):
    """Closed-form bank on one or more training instances.

    ``graphs`` is ``[(label, geometry, GraphData, train_patterns), ...]``.  A
    training INSTANCE t is one graph with one feature realization: the graph's own
    X plus ``feature_draws`` fresh draws from the same (pure-noise) law.  Every
    instance contributes its pencil (A_t, S_t) and they are POOLED with uniform
    weights, solving (mean_t A_t, mean_t S_t) -- the ratio of the pooled numerator
    to the pooled metric, a surrogate, not the mean of the per-instance objectives
    (inversion and averaging do not commute).

    share_filters=False  per channel c: the (K+1)x(K+1) block pencil
                         (Abar_cc, Sbar_cc) -> theta[:, :, c]  (H(K+1)d coefficients)
    share_filters=True   one pencil averaged over channels too,
                         (mean_c Abar_cc, mean_c Sbar_cc) -> theta_h shared by all
                         channels (H(K+1) coefficients); range Z(XO) = range Z(X)
                         for any orthogonal O.
    ``theta`` lives in the graph-independent dictionary coefficients, so one
    solve serves every graph.

    param_scale="absolute"  edge_gamma / level_beta / host_weight are raw weights
                            (the historical path, bitwise unchanged).
    param_scale="relative"  they are fractions of each block's OWN scale (see
                            _solve_block): the penalty as a fraction of the cliff
                            lambda_max(N, P) (keep < 1), the host weight as a multiple
                            of lambda_max(A, S) / lambda_max(C_H, S).
    solver_form="ratio"     maximize tr(N) / tr(P + host C_H) by Dinkelbach iteration:
                            edge_gamma / level_beta are unused, the optimal penalty
                            rho* is solved for and reported.
    (N, P) = (B_bnd, B_int) for the edge pencil, (Cap, Pen/tau) for the level one.
    """

    if objective not in ("level", "edge"):
        raise ValueError(
            f"channel-closed-form needs capture_objective 'level' (trace surrogate) "
            f"or 'edge', got {objective!r}"
        )
    if param_scale not in ("absolute", "relative"):
        raise ValueError(f"param_scale must be 'absolute' or 'relative', got {param_scale!r}")
    if solver_form not in ("difference", "ratio"):
        raise ValueError(f"solver_form must be 'difference' or 'ratio', got {solver_form!r}")
    if objective == "level" and param_scale == "relative":
        raise ValueError(
            "param_scale='relative' is undefined for the level pencil: its within-group "
            "variance matrix is rank-deficient on the dictionary while the capture matrix "
            "has mass on that null space, so the cliff lambda_max(Cap, Pen) is effectively "
            "infinite (measured 1e6 under the ridge, 3e2 / 8e1 / 3 at support tolerances "
            "1e-10 / 1e-8 / 1e-6) -- not a scale.  Use param_scale='absolute'."
        )
    parts = param_scale == "relative" or solver_form == "ratio"
    t0 = time.time()
    A_sum = S_sum = N_sum = P_sum = C_sum = None
    n_inst = 0
    host_on = host_weight > 0.0 and host_mode != "none"
    host_counts: list = []
    for gi, (_lbl, geo, data, pats) in enumerate(graphs):
        hosts = (host_nodes(data.adjacency, pats, host_mode, count=host_count,
                            seed=7919 * (seed + 1) + gi) if host_on else None)
        for r in range(1 + feature_draws):
            # seeds disjoint from the harness's graph/transfer feature seeds
            X = data.X if r == 0 else _fresh_features(data.X, 7919 * (seed + 1) + 101 * gi + r)
            _prop, mprop, _D, S_M = dictionary(geo, X, degree, basis, tau)
            if parts:
                # keep reward, penalty and host matrices apart: the scales and the
                # ratio are only defined on the pooled pieces
                N, P = (level_parts(geo, pats, mprop, tau) if objective == "level"
                        else edge_parts(geo, data.adjacency, pats, mprop))
                N_sum = _sym(N) if N_sum is None else N_sum + _sym(N)
                P_sum = _sym(P) if P_sum is None else P_sum + _sym(P)
                if host_on:
                    C = host_C(geo, mprop, hosts)
                    C_sum = C if C_sum is None else C_sum + C
                    host_counts.append(int(hosts.numel()))
            else:
                A = (level_trace_A(geo, pats, mprop, level_beta, tau) if objective == "level"
                     else edge_A(geo, data.adjacency, pats, mprop, edge_gamma))
                if host_on:
                    # A_new = B_bnd - gamma B_int - lambda_H C_H (the host set is fixed
                    # per graph; C_H still moves with the feature draw through Psi)
                    A = A - host_weight * host_C(geo, mprop, hosts)
                    host_counts.append(int(hosts.numel()))
                A_sum = A if A_sum is None else A_sum + A
            S_sum = S_M if S_sum is None else S_sum + S_M
            n_inst += 1
    S_bar = S_sum / n_inst
    t_pencil = time.time() - t0
    t0 = time.time()
    counts: list = []
    blocks_info: list = []
    d = graphs[0][2].X.shape[1]
    blocks = [torch.arange(degree + 1) * d + a for a in range(d)]  # column k*d + a

    def chan_mean(M):
        return torch.stack([M[i][:, i] for i in blocks]).mean(0)

    if parts:
        N_bar, P_bar = N_sum / n_inst, P_sum / n_inst
        C_bar = C_sum / n_inst if host_on else None
        if share_filters:
            pencils = [(chan_mean(S_bar), chan_mean(N_bar), chan_mean(P_bar),
                        chan_mean(C_bar) if host_on else None)]
        else:
            pencils = [(S_bar[i][:, i], N_bar[i][:, i], P_bar[i][:, i],
                        C_bar[i][:, i] if host_on else None) for i in blocks]
        theta = torch.zeros(heads, degree + 1, d, dtype=S_bar.dtype)
        for c, (Sb, Nb, Pb, Cb) in enumerate(pencils):
            vecs, bi = _solve_block(Sb, Nb, Pb, Cb, heads=heads, scale=param_scale, form=solver_form,
                                    penalty=level_beta if objective == "level" else edge_gamma,
                                    host_weight=host_weight if host_on else 0.0)
            counts.append(bi["n_raw"])
            blocks_info.append(bi)
            if share_filters:
                theta[:] = vecs.T.unsqueeze(-1)
            else:
                theta[:, :, c] = vecs.T
    elif share_filters:
        A_bar = A_sum / n_inst
        shared = solve_channel(chan_mean(S_bar), chan_mean(A_bar), degree, 1, heads,
                               counts=counts)  # (H, K+1, 1)
        theta = shared.expand(-1, -1, d).clone()
    else:
        theta = solve_channel(S_bar, A_sum / n_inst, degree, d, heads, counts=counts)

    def _stat(key):
        v = [b[key] for b in blocks_info if key in b and np.isfinite(b[key])]
        return (float(np.median(v)), float(min(v)), float(max(v))) if v else (float("nan"),) * 3

    return theta, {
        "pencil_seconds": t_pencil,
        "solve_seconds": time.time() - t0,
        "n_positive_per_channel": counts,
        "pencil": "level-trace" if objective == "level" else "edge",
        "share_filters": bool(share_filters),
        "feature_draws": int(feature_draws),
        "n_instances": n_inst,
        "host_mode": host_mode if host_on else "none",
        "host_weight": float(host_weight) if host_on else 0.0,
        "n_hosts": host_counts,
        "param_scale": param_scale,
        "solver_form": solver_form,
        # resolved per block: the cliff lambda_max(N, P), the absolute penalty used
        # (ratio form: the self-tuned rho*), and the absolute host weight
        "penalty_star_median": _stat("penalty_star")[0],
        "penalty_abs_median": _stat("penalty_abs")[0],
        "penalty_abs_range": list(_stat("penalty_abs")[1:]),
        "host_abs_median": _stat("host_abs")[0],
        "ratio_iters_max": max((b["ratio_iters"] for b in blocks_info if "ratio_iters" in b), default=0),
        "ratio_converged": all(b.get("converged", True) for b in blocks_info),
        "ratio_fixed_point_topH_sum_max": max(
            (abs(b["fixed_point_topH_sum"]) for b in blocks_info if "fixed_point_topH_sum" in b),
            default=0.0),
    }


# --------------------------------------------------------------------------- #
# the true objectives, evaluated identically for every solution
# --------------------------------------------------------------------------- #
def true_objectives(geo, data, patterns, Z, tau, beta, gamma, ridge=1e-7, ridge_rel=1e-4):
    MZ = geo.m_apply(Z, tau)
    G = Z.T @ MZ
    G = 0.5 * (G + G.T)
    chol = torch.linalg.cholesky(
        G + (ridge + ridge_rel * G.diagonal().mean()) * torch.eye(G.shape[0], dtype=G.dtype)
    )
    V = geo.indicators(patterns)
    phi = (V * geo.l_apply(V)).sum(0)
    lr = build_level_rows(geo, patterns, [MZ], phi)
    er = build_edge_rows(geo, data.adjacency, patterns, [MZ])
    Jl, pl = level_objective(lr["mrows"][0], lr, chol, tau, beta=beta,
                             cap_temperature=0.02, chi_temperature=0.05)
    Je, pe = edge_objective(er["mrows"][0], er, chol, gamma=gamma)
    return {"J_level": float(Jl), "J_edge": float(Je),
            "cap_min": pl["cap_min"], "chibar_max": pl["chibar_max"],
            "edge_bnd": pe["bnd_mean"], "edge_int": pe["int_mean"]}


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #
def harness_args(seed):
    """The exact synthetic-harness configuration of the level/edge runs."""

    return argparse.Namespace(
        num_nodes=1000, num_motifs=10, motif_types=["random"], motif_size_min=7,
        motif_size_max=20, avg_degree=6.0, feature_dim=32, motif_density=0.4,
        motif_conductance=-1.0, feat_shared=0.0, feat_signature=0.0, train_ratio=0.5,
        max_train_gangs=0, transfer_graphs=5,
        degree=32, basis="chebyshev", tau=1.0, epochs=4000, learning_rate=0.05,
        ridge=1e-7, optimizer="projected", softmin_temperature=0.02,
        warm_start="closed_form", softmin_anneal=1.0, capture_objective="lambda_min",
        margin_alpha=0.0, margin_softplus=0.0, collective_solver="gradient",
        pencil_beta=0.0, trace_ratio_iters=40, day_aggregate="min", label_weight=0.0,
        conf_weight=50.0, conf_reduce="mean", conf_delta=0.0, conf_halo_hops=1,
        structural_width=0, coarsen_target="bank", heads=8, head_diversity=0.0,
        coarsening_method="ward-tree", coarsening_laplacian="symmetric",
        reduction=0.3, epsilon=1.0, max_levels=10, ward_stop="f1", ward_num_cuts=500,
        threshold=0.51, seed=seed, level_beta=1.0, level_chi_temperature=0.05,
        ridge_relative=1e-4, edge_gamma=10.0, compare_num_cuts=100,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--refit", action="store_true",
                    help="also refit the gradient level@1 / edge@10 arms (reproducibility + J)")
    ap.add_argument("--out", type=Path, default=Path("results/closed_form_level/"))
    a = ap.parse_args()
    # lazy: the harness itself imports this module (collective_solver="channel-closed-form")
    from src.run_synthetic_modular import _coarsen_and_score, _load_graph, _make_cfg

    os.makedirs(a.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    args = harness_args(a.seed)
    beta, gamma, tau, K, H = 1.0, 10.0, 1.0, 32, 8

    data, tr, te, _ = _load_graph(a.seed, args)
    transfer = [(f"T{k}", *_load_graph(a.seed + 9000 + 100 * k, args)[::3][:2]) for k in range(5)]
    transfer = [(lbl, d, p) for lbl, d, p in transfer]
    evals = [("train:test", data, te)] + transfer
    det = CollectiveBankDetector(_make_cfg(args))  # coarsening/eval config of the harness
    geo = det.geometry(data)

    t0 = time.time()
    prop, mprop, D, S_M = dictionary(geo, data.X, K, "chebyshev", tau)
    t_dict = time.time() - t0
    d = data.X.shape[1]
    sols, info = {}, {}
    for name, builder in (
        ("edge", lambda: edge_A(geo, data.adjacency, tr, mprop, gamma)),
        ("leveltrace", lambda: level_trace_A(geo, tr, mprop, beta, tau)),
    ):
        t0 = time.time()
        A = builder()
        t_A = time.time() - t0
        t0 = time.time()
        W, ea, r, n_pos = solve_full(S_M, A, r_max=H * d)
        t_full = time.time() - t0
        t0 = time.time()
        th = solve_channel(S_M, A, K, d, H)
        t_chan = time.time() - t0
        sols[f"cf-full/{name}"] = ("W", W)
        sols[f"cf-svd/{name}"] = ("theta", solve_svd(W, K, d, H))
        sols[f"cf-channel/{name}"] = ("theta", th)
        info[name] = {"t_dictionary_s": t_dict, "t_pencil_s": t_A, "t_full_solve_s": t_full,
                      "t_channel_solve_s": t_chan, "full_rank_used": r, "n_positive_eig": n_pos,
                      "top_eig": [float(x) for x in ea[:5]]}
        print(f"[{name}] pencil {t_A:.2f}s | full solve {t_full:.2f}s (rank {r}, {n_pos} positive "
              f"eigenvalues) | per-channel solve {t_chan:.3f}s | dictionary {t_dict:.2f}s")

    if a.refit:
        for arm, over in (("grad/level@1", dict(capture_objective="level", level_beta=beta)),
                          ("grad/edge@10", dict(capture_objective="edge", edge_gamma=gamma))):
            dg = CollectiveBankDetector(_make_cfg(args, **over))
            t0 = time.time()
            dg.fit([("G0", data, tr, te)])
            info[arm] = {"fit_seconds": time.time() - t0, "fit_best_J": dg.fit_info_["margin"]}
            sols[arm] = ("theta", dg.theta_.detach().clone())
            print(f"[{arm}] fit {info[arm]['fit_seconds']:.0f}s  best J {dg.fit_info_['margin']:.5f}")

    def basis_on(sol, dd, pats):
        kind, obj = sol
        if kind == "W":
            g = det.geometry(dd)
            pr = _basis_stack(g.prop, dd.X, K, "chebyshev", tau, geometry=g)
            return torch.cat(pr, 1) @ obj
        det.theta_ = obj
        return det.target_subspace(dd, pats)

    rows, jrows = [], []
    for arm, sol in sols.items():
        Ztr = basis_on(sol, data, tr)
        jrows.append({"seed": a.seed, "arm": arm, "width": int(Ztr.shape[1]),
                      **true_objectives(geo, data, tr, Ztr, tau, beta, gamma)})
        for lbl, dd, pats in evals:
            _c, out = _coarsen_and_score(det, dd, basis_on(sol, dd, pats), pats,
                                         "deflated-dual-ward", SimpleNamespace(compare_num_cuts=100))
            rows.append({"seed": a.seed, "arm": arm, "graph": lbl, **out})
        print(f"  {arm:<22} mean F1* {np.mean([r['f1_star'] for r in rows if r['arm'] == arm]):.4f}")
    pd.DataFrame(rows).to_csv(a.out / f"rows_seed{a.seed}.csv", index=False)
    pd.DataFrame(jrows).to_csv(a.out / f"objectives_seed{a.seed}.csv", index=False)
    (a.out / f"info_seed{a.seed}.json").write_text(json.dumps(info, indent=2))
    print(pd.DataFrame(jrows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
