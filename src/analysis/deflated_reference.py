r"""Brute-force reference for the deflated merge score, and an exact greedy.

Nothing here is fast: every quantity is assembled from its definition on the
*original* ``N``-node graph, so a claim made with this module is independent of
the block algebra, the incremental updates, the truncated solve, the lazy queue
and the fanout cap that the production coarsener uses.  It exists to answer two
questions on small graphs:

* does :func:`src.deflated_coarsen.deflated_coarsen` select the true
  minimum-score admissible pair at every step?
* is Prop. ``cumulative-score-error`` satisfied by the *exact* quantities,
  ``(eps_Q)^2 <= sum_k s_k <= q (eps_Q)^2``?

Definitions used (paper, Sec. "Ward score"), with ``M = L_sym + tau I``:

    v_C     = D~^{1/2} 1_C / sqrt(vol C)                 (Euclidean-orthonormal)
    g_{A,B} = sqrt(vol B / s) v_A - sqrt(vol A / s) v_B,  s = vol A + vol B
    g_bar   = g / ||g||_M
    Q_P     = V_P (V_P^T M V_P)^{-1} V_P^T M              (M-orthogonal projector)
    g_hat   = (I - Q_{P'}) g_bar / ||(I - Q_{P'}) g_bar||_M
    s(A,B)  = ||Q_R g_hat||_M^2 = || U^T M g_hat ||^2     (U is M-orthonormal)

``g_{A,B}`` is proportional to the paper's ``v_A/sqrt(vol A) - v_B/sqrt(vol B)``
(same coefficient ratio ``-sqrt(vol B/vol A)``), and the score normalizes the
direction, so the two conventions give the same number.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from smooth_dual_ward import m_orthonormal_basis, screened_operators

__all__ = [
    "ExactDeflation",
    "exact_greedy_hierarchy",
]


class ExactDeflation:
    """Exact deflated scores over an evolving partition, from the definitions."""

    def __init__(self, W, Z: np.ndarray, tau: float, rank_tol: float = 1e-10):
        self.A_off, self.d_tilde, self.L, self.M = screened_operators(W, tau)
        self.n = int(self.A_off.shape[0])
        self.U, self.q = m_orthonormal_basis(np.asarray(Z, float), self.M, rank_tol)
        self.MU = np.asarray(self.M @ self.U)
        self.tau = float(tau)
        self.root = np.sqrt(self.d_tilde)

    # -- partition algebra ------------------------------------------------- #
    def block_basis(self, labels: np.ndarray) -> np.ndarray:
        """``V_P``: column ``c`` is ``D~^{1/2} 1_c / sqrt(vol c)``."""

        labels = np.asarray(labels, np.int64)
        k = int(labels.max()) + 1
        V = np.zeros((self.n, k))
        V[np.arange(self.n), labels] = self.root
        vol = np.bincount(labels, weights=self.d_tilde, minlength=k)
        return V / np.sqrt(vol)[None, :]

    def harmonic_H(self, labels: np.ndarray) -> np.ndarray:
        """``H_P = U^T M (I - Q_P) U``, assembled without any incremental update."""

        V = self.block_basis(labels)
        MV = np.asarray(self.M @ V)
        G = V.T @ MV
        B = MV.T @ self.U  # V^T M U
        return self.U.T @ self.MU - B.T @ np.linalg.solve(G, B)

    def epsilon_q(self, labels: np.ndarray) -> float:
        H = self.harmonic_H(labels)
        return float(np.sqrt(max(np.linalg.eigvalsh(0.5 * (H + H.T))[-1], 0.0)))

    # -- the score ---------------------------------------------------------- #
    def contrast(self, labels: np.ndarray, a: int, b: int) -> np.ndarray:
        labels = np.asarray(labels, np.int64)
        mask_a, mask_b = labels == a, labels == b
        vol_a = float(self.d_tilde[mask_a].sum())
        vol_b = float(self.d_tilde[mask_b].sum())
        s = vol_a + vol_b
        v_a = np.where(mask_a, self.root, 0.0) / np.sqrt(vol_a)
        v_b = np.where(mask_b, self.root, 0.0) / np.sqrt(vol_b)
        return np.sqrt(vol_b / s) * v_a - np.sqrt(vol_a / s) * v_b

    def score(self, labels: np.ndarray, a: int, b: int) -> dict:
        """Exact ``s_{P_n}(A, B)`` plus the raw score and the leakage."""

        labels = np.asarray(labels, np.int64)
        g = self.contrast(labels, a, b)
        Mg = np.asarray(self.M @ g)
        m_tau = float(g @ Mg)

        merged = labels.copy()
        merged[merged == b] = a
        _, merged = np.unique(merged, return_inverse=True)
        V = self.block_basis(merged)
        MV = np.asarray(self.M @ V)
        G = V.T @ MV
        bvec = MV.T @ g  # V_{P'}^T M g
        c = np.linalg.solve(G, bvec)
        leak = float(bvec @ c)
        gt_sq = max(m_tau - leak, np.finfo(float).tiny)

        umg = self.U.T @ Mg  # U^T M g
        a_vec = (umg - (MV.T @ self.U).T @ c) / np.sqrt(gt_sq)
        return {
            "score": float(a_vec @ a_vec),
            "raw_score": float(umg @ umg) / m_tau,
            "eta0": float(np.sqrt(max(leak / m_tau, 0.0))),
            "m_tau": m_tau,
            "a": a_vec,
        }

    # -- candidates --------------------------------------------------------- #
    def adjacent_pairs(self, labels: np.ndarray) -> list:
        labels = np.asarray(labels, np.int64)
        coo = sp.triu(self.A_off, k=1).tocoo()
        pairs = set()
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if w <= 0.0:
                continue
            ci, cj = int(labels[i]), int(labels[j])
            if ci != cj:
                pairs.add((min(ci, cj), max(ci, cj)))
        return sorted(pairs)

    def all_scores(self, labels: np.ndarray) -> dict:
        return {p: self.score(labels, *p)["score"] for p in self.adjacent_pairs(labels)}


def exact_greedy_hierarchy(W, Z, tau, *, max_merges=None, verbose=False):
    """Agglomerate by the exact minimum deflated score over **all** eligible pairs.

    Returns ``(labels_history, records)``; ``records[t]`` carries the chosen
    pair, its exact score, the number of eligible candidates and the full score
    table at that step.
    """

    ex = ExactDeflation(W, Z, tau)
    labels = np.arange(ex.n, dtype=np.int64)
    history, records = [labels.copy()], []
    limit = ex.n - 1 if max_merges is None else int(max_merges)
    for t in range(limit):
        table = ex.all_scores(labels)
        if not table:
            break
        (a, b), best = min(table.items(), key=lambda kv: (kv[1], kv[0]))
        merged = labels.copy()
        merged[merged == b] = a
        _, merged = np.unique(merged, return_inverse=True)
        records.append(
            {
                "step": t,
                "pair": (a, b),
                "score": best,
                "n_candidates": len(table),
                "table": dict(table),
                "members": (
                    np.flatnonzero(labels == a).tolist(),
                    np.flatnonzero(labels == b).tolist(),
                ),
            }
        )
        labels = merged
        history.append(labels.copy())
        if verbose:
            print(f"  step {t}: merge {a},{b} s={best:.6g} of {len(table)} cands")
    return history, records


class CoarseExactDeflation(ExactDeflation):
    r"""The same exact scores, assembled on the **coarse** system instead.

    ``A_{P'} = V_{P'}^T M_tau V_{P'}`` is the volume-normalized coarse screened
    Laplacian, closed form in the block cut statistics:

        (A_P)_{CC} = (cut(C) + tau vol(C)) / vol(C),
        (A_P)_{CD} = -w(C, D) / sqrt(vol C vol D),

    so scoring every admissible pair at a ``k``-block partition costs ``k``
    sparse solves of size ``k`` rather than ``k`` dense solves of size ``N``.
    Verified against :class:`ExactDeflation` (the from-definition route) to
    machine precision by :func:`check_routes_agree`; it exists only so the
    selection audit can run on graphs of a few thousand nodes.
    """

    def _blocks(self, labels: np.ndarray):
        labels = np.asarray(labels, np.int64)
        k = int(labels.max()) + 1
        vol = np.bincount(labels, weights=self.d_tilde, minlength=k)
        # coarse adjacency (off-diagonal weights) and cut per block
        S = sp.csr_matrix(
            (np.ones(self.n), (np.arange(self.n), labels)), shape=(self.n, k)
        )
        Wc = (S.T @ self.A_off @ S).tocsr()
        cut = np.asarray(Wc.sum(axis=1)).ravel() - Wc.diagonal()
        Wc.setdiag(0.0)
        Wc.eliminate_zeros()
        MU_block = np.zeros((k, self.U.shape[1]))  # U^T M_tau v_C
        np.add.at(MU_block, labels, self.MU * self.root[:, None])
        MU_block /= np.sqrt(vol)[:, None]
        return k, vol, cut, Wc.tocoo(), MU_block

    def _system(self, k, vol, cut, Wco, keep):
        """``A_P`` of the partition whose block ids are remapped by ``keep``."""

        size = int(keep.max()) + 1
        diag = np.zeros(size)
        np.add.at(diag, keep, cut + self.tau * vol)
        # the merged block's cut double-counts the internal edge; corrected by
        # the off-diagonal entries that collapse onto its diagonal below
        rows, cols, vals = keep[Wco.row], keep[Wco.col], -Wco.data
        vnorm = np.zeros(size)
        np.add.at(vnorm, keep, vol)
        scale = 1.0 / np.sqrt(vnorm)
        A = sp.coo_matrix(
            (
                np.concatenate([vals, diag]),
                (np.concatenate([rows, np.arange(size)]),
                 np.concatenate([cols, np.arange(size)])),
            ),
            shape=(size, size),
        ).tocsr()
        D = sp.diags(scale)
        return (D @ A @ D).tocsr(), vnorm

    def all_scores(self, labels: np.ndarray) -> dict:
        """Override: the coarse route, so the audit can run past a few hundred nodes."""

        return self.all_scores_coarse(labels)

    def all_scores_coarse(self, labels: np.ndarray) -> dict:
        """Exact ``s_{P_n}(A, B)`` for every adjacent pair, via the coarse route."""

        from src.deflated_coarsen import _block_cg

        k, vol, cut, Wco, MUb = self._blocks(labels)
        nbr = [dict() for _ in range(k)]
        for i, j, w in zip(Wco.row, Wco.col, Wco.data):
            if w > 0.0 and i != j:
                nbr[int(i)][int(j)] = float(w)
        gd = (cut + self.tau * vol) / vol

        out = {}
        for a in range(k):
            for b, w_ab in nbr[a].items():
                if b <= a:
                    continue
                va, vb = vol[a], vol[b]
                s = va + vb
                al, be = np.sqrt(va / s), np.sqrt(vb / s)
                s_ab = -w_ab / np.sqrt(va * vb)
                m_tau = be * be * gd[a] - 2.0 * al * be * s_ab + al * al * gd[b]

                keep = np.arange(k, dtype=np.int64)
                keep[keep > b] -= 1
                keep[b] = a - (1 if a > b else 0)
                A_P, _ = self._system(k, vol, cut, Wco, keep)

                # b_vec = V_{P'}^T M_tau g,  g = beta v_a - alpha v_b
                rhs = np.zeros(k - 1)
                for c, w in nbr[a].items():
                    if c != b:
                        rhs[keep[c]] += be * (-w / np.sqrt(vol[c] * va))
                for c, w in nbr[b].items():
                    if c != a:
                        rhs[keep[c]] -= al * (-w / np.sqrt(vol[c] * vb))
                rhs[keep[a]] += (
                    al * be * gd[a] + (be * be - al * al) * s_ab - al * be * gd[b]
                )

                x = _block_cg(A_P, rhs[:, None]).ravel()
                gt_sq = max(m_tau - float(rhs @ x), np.finfo(float).tiny)
                # U^T M_tau V_{P'} x, with the merged column alpha m_a + beta m_b
                wts = np.ones(k)
                wts[a], wts[b] = al, be
                proj = (x[keep] * wts) @ MUb
                a_vec = (be * MUb[a] - al * MUb[b] - proj) / np.sqrt(gt_sq)
                out[(a, b)] = float(a_vec @ a_vec)
        return out


def check_routes_agree(W, Z, tau, labels=None, atol=1e-8) -> float:
    """Max |coarse - fine| over every adjacent pair; raises if it exceeds ``atol``."""

    fine = ExactDeflation(W, Z, tau)
    coarse = CoarseExactDeflation(W, Z, tau)
    if labels is None:
        labels = np.arange(fine.n, dtype=np.int64)
    a = fine.all_scores(labels)
    b = coarse.all_scores_coarse(labels)
    worst = max(abs(a[p] - b[p]) for p in a)
    if worst > atol:
        raise AssertionError(f"coarse and fine deflated scores differ by {worst:.3e}")
    return worst
