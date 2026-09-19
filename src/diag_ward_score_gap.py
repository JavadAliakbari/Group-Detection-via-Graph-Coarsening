r"""Why does deflated Ward beat raw Ward?  A single-graph forensic.

The propositions of Sec.~"Adapting Ward Clustering" say raw Ward should have the
*better* guarantee: its internal bound is ``chi`` and its boundary bound depends
only on ``(C, kappa)``, while both deflated bounds are degraded by the leakage
``eta_n`` -- internal upward, boundary downward.  Empirically deflated wins.  This
script settles that on ONE graph, in five parts:

A. **Are the bounds correct?**  Every candidate merge at the singleton partition
   is classified internal / boundary / host and checked against
   eq. (raw-internal), (raw-boundary), (main-internal), (main-boundary) and the
   sandwich (deflated-raw-bounds), for both the learned target and the idealized
   screened-dual target ``Z* = M_tau^{-1} V*`` (for which the theory predicts
   ``s^0 = 0`` exactly on internal and host merges).

B. **How big is eta actually?**  The leakage penalty can only explain the gap if
   eta is large.  Reported separately for internal and boundary candidates.

C. **The separation that actually drives recovery.**  A bound is not a decision
   rule: what the greedy needs is that internal candidates *rank below* boundary
   candidates.  Reported as the realized gap ``min boundary - max internal`` and
   as ``AUC = P(s_int < s_bnd)``, for both scores.

D. **Score vs engine.**  Comparing ``src.raw_ward`` with ``src.deflated_coarsen``
   confounds the score with the engine (the latter has raw insertion keys, a
   truncated r-hop solve, a lazy rescoring cap and a fanout cap).  Here ONE engine
   (``raw_ward``) is run with ``score="raw"`` and with ``score="deflated"`` (exact
   harmonic deflation), so the only difference is the score.  The production
   coarsener is run too, to size the engine effect.

E. **Cumulative damage.**  Prop. (rank-one) says the deflated score is exactly the
   one-step increment of ``tr(H_P^tau)``, so its greedy descends a real cumulative
   objective; the raw score is not the increment of anything.  ``tr(H)``,
   ``eps_Q`` and detection F1 are tracked against the level for both.

Run::

    python -m src.diag_ward_score_gap --num-nodes 700 --num-motifs 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.collective_detector import CollectiveBankDetector, DetectorConfig
from src.deflated_coarsen import (
    _block_cg,
    deflated_tree_coarsen,
    exact_harmonic_H,
)
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.raw_ward import raw_ward, screened_operators_kappa
from src.run_synthetic_modular import _load_graph
from src.smooth_dual_ward import m_orthonormal_basis
from src.utils.utils import LOGGER

_TINY = np.finfo(np.float64).tiny


# --------------------------------------------------------------------------- #
# the paper's per-group constants
# --------------------------------------------------------------------------- #
def capture_of(U, M, kappa, nodes):
    r"""``C_R^tau(S) = ||Q_R^tau v_hat_S||^2_{M_tau}`` with ``||v_hat_S||_{M_tau} = 1``."""

    v = np.zeros(kappa.size)
    v[nodes] = np.sqrt(kappa[nodes])
    v /= math.sqrt(max(float(v @ (M @ v)), _TINY))
    return float(np.sum((U.T @ (M @ v)) ** 2)), v


def confusability_of(U, M, kappa, nodes):
    r"""``chi_R^tau(S)``: the largest fraction of screened energy of an *internal*
    fluctuation that the target retains.

    The admissible ``w`` are supported on ``S`` and satisfy
    ``<w, K^{1/2} 1_S>_2 = 0`` -- exactly the set every ``h_{A,B}`` with
    ``A, B subset S`` lives in, which is what makes eq. (raw-internal) an upper
    bound on every internal raw score.
    """

    nodes = np.asarray(nodes)
    s = nodes.size
    if s < 2:
        return 0.0
    B = np.asarray((M[nodes][:, nodes]).todense())
    P = np.asarray((M @ U)[nodes])  # (s, q):  w^T P = (U^T M w)^T
    c = np.sqrt(kappa[nodes])
    # orthonormal basis of {w : c^T w = 0}
    Q, _ = np.linalg.qr(np.eye(s) - np.outer(c, c) / (c @ c))
    N = Q[:, : s - 1]
    lhs = N.T @ P @ P.T @ N
    rhs = N.T @ B @ N
    rhs = 0.5 * (rhs + rhs.T)
    from scipy.linalg import eigh

    return float(eigh(0.5 * (lhs + lhs.T), rhs, eigvals_only=True)[-1])


# --------------------------------------------------------------------------- #
# per-candidate quantities at the SINGLETON partition
# --------------------------------------------------------------------------- #
def candidate_table(W, U, M, kappa, tau, gang_of, vhat, capture):
    """One row per graph edge: ``s^0``, ``eta``, deflated ``s``, ``kappa_S``, class."""

    A = sp.triu(sp.csr_matrix(W), k=1).tocoo()
    n = W.shape[0]
    MU = np.asarray(M @ U)
    lev = MU / np.sqrt(kappa)[:, None]  # r_v = ell_v G^{-1/2}
    deg = np.asarray(sp.csr_matrix(W).sum(axis=1)).ravel()
    diag0 = deg + tau * kappa  # (A_P)_{vv} at the singleton partition
    rows, cols, vals = A.row, A.col, A.data

    out = []
    for u, v, w in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        ka, kb = kappa[u], kappa[v]
        sab = ka + kb
        m2 = ka * kb / sab
        ell = (kb * (deg[u] / ka) + ka * (deg[v] / kb) + 2.0 * w) / sab
        h_sq = (ell + tau) / m2  # ||h||^2_{M_tau}
        umh = lev[u] - lev[v]  # U^T M_tau h
        s0 = float(umh @ umh) / h_sq

        # --- exact deflation against the POST-merge partition ----------------
        # blocks of P': every node except v, with u standing for {u, v}
        lab = np.arange(n)
        lab[v] = u
        _, lab = np.unique(lab, return_inverse=True)
        k = int(lab.max()) + 1
        V = sp.csr_matrix(
            (np.sqrt(kappa), (np.arange(n), lab)), shape=(n, k)
        )
        A_P = (V.T @ (M @ V)).tocsr()
        h = np.zeros(n)
        h[u] = math.sqrt(ka) / ka
        h[v] = -math.sqrt(kb) / kb
        rhs = np.asarray(V.T @ (M @ h)).ravel()
        x = _block_cg(A_P, rhs[:, None]).ravel()
        eta_sq = float(np.clip((rhs @ x) / max(h_sq, _TINY), 0.0, 1.0))
        proj = np.asarray(MU.T @ (V @ x)).ravel()
        sdef = float(np.sum((umh - proj) ** 2) / max(h_sq * (1 - eta_sq), _TINY))
        # The sandwich is a triangle inequality on  a = (umh - proj)/||(I-Q)h||.
        # Exactly:  s (1 - eta^2) = s0 - 2 sqrt(s0) zeta cos(theta) + zeta^2, with
        #   zeta       = ||Q_R Q_P' h||_M / ||h||_M   (the leaked part the TARGET sees)
        #   cos(theta) = angle between U^T M h and U^T M Q_P' h.
        # The published bound takes cos = -+1 AND replaces zeta by its upper bound
        # eta = ||Q_P' h||_M / ||h||_M >= zeta.  Those are its only two slacks.
        n_umh = float(np.linalg.norm(umh))
        n_proj = float(np.linalg.norm(proj))
        zeta = n_proj / math.sqrt(h_sq)
        cos_t = (
            float(umh @ proj) / max(n_umh * n_proj, _TINY)
            if n_umh > 0 and n_proj > 0
            else 0.0
        )

        gu, gv = gang_of[u], gang_of[v]
        if gu >= 0 and gu == gv:
            cls, gid = "internal", gu
        elif gu >= 0 and gv >= 0:
            cls, gid = "gang-gang", gu
        elif gu >= 0 or gv >= 0:
            cls, gid = "boundary", (gu if gu >= 0 else gv)
        else:
            cls, gid = "host", -1
        kap = (
            float(h @ (M @ vhat[gid])) / math.sqrt(h_sq) if gid >= 0 else float("nan")
        )
        out.append(
            {
                "u": u, "v": v, "w": w, "class": cls, "gang": gid,
                "s0": s0, "eta": math.sqrt(eta_sq), "s_def": sdef,
                "zeta": zeta, "cos_theta": cos_t,
                "kappa_S": kap, "C_S": capture.get(gid, float("nan")),
                "h_sq": h_sq,
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# level-indexed damage + detection
# --------------------------------------------------------------------------- #
def sweep_damage(res, U, M, kappa, patterns, y, threshold, levels):
    """``tr(H)``, ``eps_Q`` and detection F1 at each level of a hierarchy."""

    out = []
    for k in levels:
        lab = res.labels_at(int(k))
        nc = int(lab.max()) + 1
        H = exact_harmonic_H(U, M, kappa, lab)
        n2s = torch.from_numpy(lab).to(y.device)
        results, by_label = evaluate_loukas_patterns(
            patterns, n2s, y, threshold=threshold
        )
        alert = by_label.get("alert", {})
        out.append(
            {
                "n_coarse": nc,
                "kept": nc / lab.size,
                "trace_H": float(np.trace(H)),
                "eps_Q": math.sqrt(max(float(np.linalg.eigvalsh(H)[-1]), 0.0)),
                "f1": float(np.mean([r.f1 for r in results])) if results else 0.0,
                "recall": float(alert.get("mean_recall", 0.0) or 0.0),
                "precision": float(alert.get("mean_precision", 0.0) or 0.0),
                "detection": float(alert.get("detection_rate", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(out).drop_duplicates("n_coarse")


def first_leak(res, gang_of, n):
    """Merge index at which each gang first absorbs a non-member (or is split off)."""

    parent = list(range(n + res.children_.shape[0]))
    tag = {i: (gang_of[i] if i < n else -2) for i in range(n)}
    leak = {}
    for t, (a, b) in enumerate(res.children_.tolist()):
        ta, tb = tag[a], tag[b]
        new = ta if ta == tb else (-1 if -1 in (ta, tb) or ta != tb else ta)
        if ta != tb:
            for g in (ta, tb):
                if g >= 0 and g not in leak:
                    leak[g] = n - t - 1  # n_coarse right after the contaminating merge
            new = -1
        tag[n + t] = new
    return leak


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _report_tightness(T):
    """Is the sandwich (deflated-raw-bounds) vacuous, or tight?

    ``s`` lives in ``[0, 1]`` (it is a squared projection of a unit vector), so an
    upper bound at or above 1 carries no information and a lower bound at 0 carries
    none either.  Between those extremes the question is *how wide* the bracket is
    relative to the value it brackets, and -- for the recovery argument -- whether
    the brackets of internal and boundary candidates are disjoint.
    """

    s0, eta, sd = T.s0.to_numpy(), T.eta.to_numpy(), T.s_def.to_numpy()
    zeta, cos_t = T.zeta.to_numpy(), T.cos_theta.to_numpy()
    lo = np.maximum(np.sqrt(s0) - eta, 0.0) ** 2 / (1 - eta**2)
    hi = (np.sqrt(s0) + eta) ** 2 / (1 - eta**2)
    # the same bound with the exact leaked-and-visible mass zeta in place of eta
    lo_z = np.maximum(np.sqrt(s0) - zeta, 0.0) ** 2 / (1 - eta**2)
    hi_z = (np.sqrt(s0) + zeta) ** 2 / (1 - eta**2)

    LOGGER.info("\n  [A2] IS THE SANDWICH VACUOUS OR TIGHT?   (s always lies in [0, 1])")
    LOGGER.info(
        f"      lower bound = 0 (no information) on {100 * (lo <= 1e-12).mean():.1f}% "
        f"of candidates;  upper bound >= 1 (no information) on "
        f"{100 * (hi >= 1.0).mean():.1f}%"
    )
    LOGGER.info(
        f"      bracket width hi-lo : median {np.median(hi - lo):.4f}   "
        f"= {np.median((hi - lo) / np.maximum(sd, 1e-12)):.2f} x the value it brackets"
    )
    LOGGER.info(
        f"      upper / actual      : median {np.median(hi / np.maximum(sd, 1e-12)):.2f} x"
        f"      actual / lower : median "
        f"{np.median(sd / np.maximum(lo, 1e-12)):.2f} x"
    )
    LOGGER.info(
        f"      where s sits in the bracket: median "
        f"{np.median((sd - lo) / np.maximum(hi - lo, 1e-12)):.3f} "
        "(0 = at the lower bound, 1 = at the upper)"
    )
    LOGGER.info(
        "\n      the two slacks, both exact "
        "(s(1-eta^2) = s0 - 2 sqrt(s0) zeta cos(theta) + zeta^2):"
    )
    LOGGER.info(
        f"        (i)  eta vs the leaked mass the TARGET sees, zeta <= eta:  "
        f"zeta/eta median {np.median(zeta / np.maximum(eta, 1e-12)):.3f}"
    )
    LOGGER.info(
        f"        (ii) |cos(theta)| vs the bound's assumed 1:                "
        f"|cos| median {np.median(np.abs(cos_t)):.3f}"
    )
    LOGGER.info(
        f"      replacing eta by zeta alone shrinks the bracket to "
        f"{np.median((hi_z - lo_z) / np.maximum(hi - lo, 1e-12)):.3f} of its width "
        f"({np.median((hi_z - lo_z) / np.maximum(sd, 1e-12)):.2f} x the value)"
    )

    # does the bracket certify the internal-vs-boundary ordering the greedy needs?
    for name, L, H in (("published (eta)", lo, hi), ("sharpened (zeta)", lo_z, hi_z)):
        rows = []
        for g in sorted(set(T[T["class"] == "internal"].gang)):
            hi_i = H[(T["class"] == "internal").to_numpy() & (T.gang == g).to_numpy()]
            lo_b = L[(T["class"] == "boundary").to_numpy() & (T.gang == g).to_numpy()]
            if hi_i.size and lo_b.size:
                rows.append((hi_i[:, None] < lo_b[None, :]).mean())
        if rows:
            LOGGER.info(
                f"      certified ordering, {name:<17}: "
                f"{100 * float(np.mean(rows)):.1f}% of (internal, boundary) pairs have "
                f"hi_int < lo_bnd"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-nodes", type=int, default=700)
    ap.add_argument("--num-motifs", type=int, default=8)
    ap.add_argument("--motif-size-min", type=int, default=10)
    ap.add_argument("--motif-size-max", type=int, default=20)
    ap.add_argument("--motif-types", type=str, default="random")
    ap.add_argument("--motif-density", type=float, default=0.4)
    ap.add_argument("--motif-conductance", type=float, default=-1.0)
    ap.add_argument("--avg-degree", type=float, default=6.0)
    ap.add_argument("--feature-dim", type=int, default=32)
    ap.add_argument("--feat-shared", type=float, default=0.0)
    ap.add_argument("--feat-signature", type=float, default=0.0)
    ap.add_argument("--train-ratio", type=float, default=1.0)
    ap.add_argument("--max-train-gangs", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--targets", type=str, default="ideal,bank",
        help="ideal = the screened-dual Z* = M_tau^-1 V* (theory's clean case); "
        "bank = the learned filter-bank subspace",
    )
    ap.add_argument("--out", type=Path, default=Path("results/ward_score_gap/"))
    args = ap.parse_args()
    args.motif_types = [t.strip() for t in args.motif_types.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- the one test graph -------------------------------------------------
    data, gang_train, gang_test, gangs = _load_graph(args.seed, args)
    n = int(data.num_nodes)
    idx = data.adjacency.coalesce().indices().cpu().numpy()
    val = data.adjacency.coalesce().values().cpu().numpy().astype(np.float64)
    W = sp.coo_matrix((val, (idx[0], idx[1])), shape=(n, n)).tocsr()
    A_off, kappa, L, M = screened_operators_kappa(W, args.tau, "symmetric")
    gang_of = -np.ones(n, dtype=np.int64)
    for j, p in enumerate(gangs):
        gang_of[np.asarray(p.node_indices)] = j

    LOGGER.info("=" * 92)
    LOGGER.info(
        f"ONE TEST GRAPH: N={n:,}  |E|={int(A_off.nnz // 2):,}  "
        f"mean deg {A_off.nnz / n:.2f}  {len(gangs)} gangs "
        f"sizes {sorted(len(p.node_indices) for p in gangs)}  tau={args.tau}"
    )
    LOGGER.info("=" * 92)

    # ---- the two targets ----------------------------------------------------
    targets = {}
    want = [t.strip() for t in args.targets.split(",") if t.strip()]
    if "ideal" in want:
        # Z* = M_tau^{-1} V*, V* = [v_S1 ... v_Sm]:  the screened-dual target for
        # which the theory predicts s^0 = 0 on every internal and host merge.
        Vs = np.zeros((n, len(gangs)))
        for j, p in enumerate(gangs):
            nodes = np.asarray(p.node_indices)
            Vs[nodes, j] = np.sqrt(kappa[nodes])
            Vs[:, j] /= math.sqrt(float(Vs[:, j] @ (M @ Vs[:, j])))
        targets["ideal"] = _block_cg(M.tocsr(), Vs, tol=1e-13)
    if "bank" in want:
        # Only the knobs that shape the target subspace matter here; everything
        # else stays at DetectorConfig's defaults so the fit is the standard one.
        cfg = DetectorConfig(
            degree=args.degree,
            tau=args.tau,
            epochs=args.epochs,
            heads=args.heads,
            capture_objective="lambda_min",
            coarsening_method="raw-ward",
            coarsening_laplacian="symmetric",
            ward_stop="f1",
            threshold=args.threshold,
            seed=args.seed,
        )
        det = CollectiveBankDetector(cfg)
        det.fit([("G0", data, gang_train, gang_test)])
        targets["bank"] = (
            det.target_subspace(data, gang_train).detach().cpu().double().numpy()
        )

    rows_bounds, rows_sep, rows_sweep, rows_ablate = [], [], [], []
    for tname, Z in targets.items():
        U, q_eff = m_orthonormal_basis(Z, M)
        LOGGER.info("\n" + "=" * 92)
        LOGGER.info(f"TARGET = {tname}   (Z is {Z.shape[0]}x{Z.shape[1]}, rank {q_eff})")
        LOGGER.info("=" * 92)

        capture, vhat = {}, {}
        chi = {}
        for j, p in enumerate(gangs):
            nodes = np.asarray(p.node_indices)
            capture[j], vhat[j] = capture_of(U, M, kappa, nodes)
            chi[j] = confusability_of(U, M, kappa, nodes)
        LOGGER.info(
            f"  capture C_S: mean {np.mean(list(capture.values())):.4f} "
            f"min {np.min(list(capture.values())):.4f}   |   "
            f"confusability chi_S: mean {np.mean(list(chi.values())):.4f} "
            f"max {np.max(list(chi.values())):.4f}"
        )

        T = candidate_table(W, U, M, kappa, args.tau, gang_of, vhat, capture)
        T["chi_S"] = T.gang.map(chi)
        T["target"] = tname
        T.to_csv(args.out / f"candidates_{tname}.csv", index=False)

        # ---------------- A. bound verification ------------------------------
        LOGGER.info("\n  [A] BOUND VERIFICATION over every edge at the singleton partition")
        LOGGER.info(f"      candidates: {T['class'].value_counts().to_dict()}")
        eta = T.eta.to_numpy()
        s0 = T.s0.to_numpy()
        sd = T.s_def.to_numpy()
        lo = np.maximum(np.sqrt(s0) - eta, 0.0) ** 2 / (1 - eta**2)
        hi = (np.sqrt(s0) + eta) ** 2 / (1 - eta**2)
        LOGGER.info(
            f"      sandwich (deflated-raw-bounds): violations "
            f"{int((sd < lo - 1e-9).sum())} low / {int((sd > hi + 1e-9).sum())} high "
            f"of {len(T)}   worst slack {max((lo - sd).max(), (sd - hi).max()):.2e}"
        )
        _report_tightness(T)

        I = T[T["class"] == "internal"]
        B = T[T["class"] == "boundary"]
        if len(I):
            v_raw = int((I.s0 > I.chi_S + 1e-9).sum())
            ub = (np.sqrt(I.chi_S) + I.eta) ** 2 / (1 - I.eta**2)
            v_def = int((I.s_def > ub + 1e-9).sum())
            LOGGER.info(
                f"      (raw-internal)  s0 <= chi_S            : {v_raw} violations "
                f"of {len(I)}   max s0/chi = {(I.s0 / I.chi_S.clip(1e-300)).max():.3f}"
            )
            LOGGER.info(
                f"      (main-internal) s <= (sqrt(chi)+eta)^2/(1-eta^2): "
                f"{v_def} violations of {len(I)}   max ratio "
                f"{(I.s_def / ub).max():.3f}"
            )
        if len(B):
            lb_raw = (np.maximum(B.kappa_S - np.sqrt(1 - B.C_S), 0.0) ** 2) / B.C_S
            v_raw = int((B.s0 < lb_raw - 1e-9).sum())
            inner = np.maximum(B.kappa_S - B.eta, 0.0) / np.sqrt(1 - B.eta**2)
            lb_def = (np.maximum(inner - np.sqrt(1 - B.C_S), 0.0) ** 2) / B.C_S
            v_def = int((B.s_def < lb_def - 1e-9).sum())
            LOGGER.info(
                f"      (raw-boundary)  s0 >= (kappa-sqrt(1-C))^2/C : {v_raw} "
                f"violations of {len(B)}   bound active on "
                f"{int((lb_raw > 1e-12).sum())} of them"
            )
            LOGGER.info(
                f"      (main-boundary) deflated lower bound        : {v_def} "
                f"violations of {len(B)}   bound active on "
                f"{int((lb_def > 1e-12).sum())} of them"
            )
        for cls in ("internal", "host"):
            sub = T[T["class"] == cls]
            if len(sub):
                LOGGER.info(
                    f"      {cls:<9} s0: median {sub.s0.median():.3e} "
                    f"max {sub.s0.max():.3e}"
                    + (
                        "   (theory: exactly 0 for the ideal target)"
                        if tname == "ideal"
                        else ""
                    )
                )
        rows_bounds.append(T)

        # ---------------- B. leakage ------------------------------------------
        LOGGER.info("\n  [B] LEAKAGE eta_n  (the only term separating the two guarantees)")
        for cls in ("internal", "boundary", "host"):
            sub = T[T["class"] == cls]
            if len(sub):
                LOGGER.info(
                    f"      {cls:<9} eta: median {sub.eta.median():.4f}  "
                    f"p90 {sub.eta.quantile(0.9):.4f}  max {sub.eta.max():.4f}   "
                    f"=> inflation 1/(1-eta^2) median "
                    f"{1 / (1 - sub.eta.median() ** 2):.4f}"
                )

        # ---------------- C. separation ---------------------------------------
        LOGGER.info(
            "\n  [C] SEPARATION internal vs boundary  (what the greedy actually needs)"
        )
        LOGGER.info(
            f"      {'score':<10}{'max s_int':>12}{'min s_bnd':>12}{'gap':>12}"
            f"{'AUC P(int<bnd)':>16}"
        )
        for lbl, col in (("raw s0", "s0"), ("deflated", "s_def")):
            per_gang = []
            for j in range(len(gangs)):
                si = I[I.gang == j][col].to_numpy()
                sb = B[B.gang == j][col].to_numpy()
                if not si.size or not sb.size:
                    continue
                auc = float((si[:, None] < sb[None, :]).mean())
                per_gang.append((si.max(), sb.min(), sb.min() - si.max(), auc))
            if per_gang:
                a = np.array(per_gang)
                LOGGER.info(
                    f"      {lbl:<10}{a[:, 0].mean():>12.4f}{a[:, 1].mean():>12.4f}"
                    f"{a[:, 2].mean():>12.4f}{a[:, 3].mean():>16.4f}"
                )
                rows_sep.append(
                    {
                        "target": tname, "score": lbl,
                        "max_internal": a[:, 0].mean(), "min_boundary": a[:, 1].mean(),
                        "gap": a[:, 2].mean(), "auc": a[:, 3].mean(),
                        "gangs_ordered": int((a[:, 2] > 0).sum()), "n_gangs": len(a),
                    }
                )

        # ---------------- D. score vs engine ----------------------------------
        LOGGER.info("\n  [D] SAME ENGINE, TWO SCORES  (the confound-free comparison)")
        levels = np.unique(
            np.round(np.geomspace(max(2, n // 40), n - 1, 40)).astype(int)
        )
        runs = {}
        for sname in ("raw", "deflated"):
            runs[sname] = raw_ward(W, Z, args.tau, score=sname, build_full_tree=True)
        for sname, res in runs.items():
            S = sweep_damage(
                res, U, M, kappa, gangs, data.y, args.threshold, levels
            )
            S["target"], S["variant"] = tname, f"same-engine/{sname}"
            rows_sweep.append(S)
            best = S.loc[S.f1.idxmax()]
            LOGGER.info(
                f"      {sname:<10} best F1={best.f1:.4f} at n={int(best.n_coarse)} "
                f"(kept {best.kept:.2f})  det={best.detection:.1%}  "
                f"eps_Q={best.eps_Q:.4f}  tr(H)={best.trace_H:.3f}"
            )
            rows_ablate.append(
                {
                    "target": tname, "variant": f"same-engine/{sname}",
                    "best_f1": float(best.f1), "detection": float(best.detection),
                    "n_coarse": int(best.n_coarse), "eps_Q": float(best.eps_Q),
                    "trace_H": float(best.trace_H),
                }
            )
        # how often do the two scores disagree on the SAME partition?
        agree = sum(
            1
            for x, y_ in zip(
                runs["raw"].children_.tolist(), runs["deflated"].children_.tolist()
            )
            if tuple(x) == tuple(y_)
        )
        LOGGER.info(
            f"      merge orders share {agree}/{len(runs['raw'].children_)} steps"
        )

        # the production coarsener, to size the ENGINE effect
        basis_t = torch.from_numpy(Z)
        co, traj = deflated_tree_coarsen(
            data.adjacency, basis_t, gangs, data.y, tau=args.tau,
            rule="dual-ward", stop="f1", num_cuts=60,
        )
        bt = max(traj, key=lambda e: e["train_f1"])
        n2s = torch.from_numpy(bt["labels"]).to(data.y.device)
        _r, bl = evaluate_loukas_patterns(
            gangs, n2s, data.y, threshold=args.threshold
        )
        H = exact_harmonic_H(U, M, kappa, bt["labels"])
        LOGGER.info(
            f"      {'production':<10} best F1={bt['train_f1']:.4f} at "
            f"n={bt['n_coarse']} (kept {bt['n_coarse'] / n:.2f})  "
            f"det={bl.get('alert', {}).get('detection_rate', 0):.1%}  "
            f"eps_Q={math.sqrt(max(float(np.linalg.eigvalsh(H)[-1]), 0)):.4f}  "
            f"tr(H)={float(np.trace(H)):.3f}   "
            "(src.deflated_coarsen: raw keys + truncated solve + lazy queue)"
        )
        rows_ablate.append(
            {
                "target": tname, "variant": "production/deflated-dual-ward",
                "best_f1": float(bt["train_f1"]),
                "detection": float(bl.get("alert", {}).get("detection_rate", 0.0)),
                "n_coarse": int(bt["n_coarse"]),
                "eps_Q": math.sqrt(max(float(np.linalg.eigvalsh(H)[-1]), 0.0)),
                "trace_H": float(np.trace(H)),
            }
        )

        # ---------------- E. cumulative damage --------------------------------
        LOGGER.info("\n  [E] CUMULATIVE DAMAGE vs LEVEL  (tr(H) is what deflated descends)")
        LOGGER.info(
            f"      {'kept':>7}" + "".join(
                f"{h:>12}" for h in ("tr(H) raw", "tr(H) defl", "F1 raw", "F1 defl")
            )
        )
        Sr = rows_sweep[-2].set_index("n_coarse")
        Sd = rows_sweep[-1].set_index("n_coarse")
        for k in sorted(set(Sr.index) & set(Sd.index), reverse=True)[::4]:
            LOGGER.info(
                f"      {k / n:>7.2f}{Sr.loc[k].trace_H:>12.3f}"
                f"{Sd.loc[k].trace_H:>12.3f}{Sr.loc[k].f1:>12.3f}"
                f"{Sd.loc[k].f1:>12.3f}"
            )

        # first contamination of each gang
        lr = first_leak(runs["raw"], gang_of, n)
        ld = first_leak(runs["deflated"], gang_of, n)
        common = sorted(set(lr) & set(ld))
        if common:
            LOGGER.info(
                f"\n      first contamination of a gang, as MERGES SURVIVED "
                f"(n - n_coarse at the merge that first pulls a non-member in; "
                f"HIGHER = the gang stays pure longer):\n"
                f"        raw       median "
                f"{np.median([n - lr[g] for g in common]):.0f}"
                f"   ({sorted(n - lr[g] for g in common)})\n"
                f"        deflated  median "
                f"{np.median([n - ld[g] for g in common]):.0f}"
                f"   ({sorted(n - ld[g] for g in common)})"
            )

    pd.concat(rows_bounds).to_csv(args.out / "candidates_all.csv", index=False)
    pd.DataFrame(rows_sep).to_csv(args.out / "separation.csv", index=False)
    pd.concat(rows_sweep).to_csv(args.out / "damage_sweep.csv", index=False)
    pd.DataFrame(rows_ablate).to_csv(args.out / "ablation.csv", index=False)
    LOGGER.info(f"\n  CSVs -> {args.out}")


if __name__ == "__main__":
    main()
