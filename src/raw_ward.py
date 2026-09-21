r"""Rank-``q`` **raw-Ward** coarsening on the vector-valued screened level.

This is the coarsener of the "Rank-$q$ level representation and raw-Ward
coarsening" paragraph, implemented exactly as stated -- including the local
volume / level / neighbour / cut updates and the ``O(q)`` prewhitened score --
rather than by delegating to a general agglomeration driver.

Setting
-------
With ``M_tau = L_sym + tau I``, ``L_sym = I - D_t^{-1/2}(W + I)D_t^{-1/2}``,
``d~ = (W + I)1`` and a target ``R = span(Z)``, ``Z = [z_1 ... z_q]``:

    G    = Z^T M_tau Z                                   (q x q Gram)
    ell  = D_t^{-1/2} M_tau Z          in R^{N x q}       (the *level*)

Each block ``A`` carries its degree-weighted mean level

    ell_bar_A = (1/vol(A)) sum_{v in A} d~_v ell_{v,:},   vol(A) = sum_{v in A} d~_v,

and adjacent blocks are scored by

    m(A,B)^2      = vol(A) vol(B) / (vol(A) + vol(B)),
    g_{A,B}       = m(A,B) ( u_A/vol(A) - u_B/vol(B) ),   u_A = D_t^{1/2} 1_A,
    s^0_R(A,B)    = m(A,B)^2 || ell_bar_A - ell_bar_B ||^2_{G^-1} / || g_{A,B} ||^2_{M_tau}.

The denominator is *local*: with the cached cut ``d(A) = w~(A, V\A)`` and
``Phi(A) = d(A)/vol(A)``,

    || g_{A,B} ||^2_{M_tau} = tau + [ vol(B) Phi(A) + vol(A) Phi(B) + 2 w~(A,B) ]
                                    / ( vol(A) + vol(B) ),

which is why no graph-wide quantity is ever recomputed during the agglomeration.

Prewhitening
------------
Writing ``G = V Lam V^T`` and ``r_A := G^{-1/2} ell_bar_A`` (numerically:
``r = ell V_r Lam_r^{-1/2}``, dropping directions below ``rank_tol``), the
numerator collapses to ``m(A,B)^2 || r_A - r_B ||_2^2`` -- an ``O(q)`` evaluation.
Because ``r = D_t^{-1/2} M_tau Z G^{-1/2} = D_t^{-1/2} M_tau U_tau`` with
``U_tau`` an ``M_tau``-orthonormal basis of ``R``, the score is also exactly the
*normalized dual* score ``||U_tau^T M_tau g||^2 / ||g||^2_{M_tau}`` in ``[0, 1]``:
the fraction of the merge direction's screened energy that the target sees.
:func:`run_validation_suite` checks that identity, and also checks the merge
order against :func:`src.smooth_dual_ward.smooth_dual_ward` at ``alpha = 0``.

Algorithm
---------
Start from the singleton partition (``vol({v}) = d~_v``, ``ell_bar_{v} = ell_v``),
score every *edge* once into a min-heap, and repeatedly pop the cheapest entry
whose two blocks are both still alive and still adjacent (stale entries are
discarded, exactly as specified).  A merge ``C = A u B`` updates

    vol(C)      = vol(A) + vol(B)
    ell_bar_C   = (vol(A) ell_bar_A + vol(B) ell_bar_B) / vol(C)
    w~(C,D)     = w~(A,D) + w~(B,D)          for D in N(A) u N(B) \ {A,B}
    d(C)        = d(A) + d(B) - 2 w~(A,B)

and only the scores ``s^0_R(C,D)``, ``D in N(C)``, are recomputed and pushed;
every other cached score is still exact.  Only adjacent blocks are ever merged,
so each block induces a connected subgraph.

Run ``python -m src.raw_ward`` to execute the validation suite.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp

from smooth_dual_ward import (
    _sanitize_adjacency,
    m_orthonormal_basis,
    screened_operators,
)

_SCORES = ("raw", "ward", "ward_vol", "deflated", "deflated_unnorm")

__all__ = [
    "RawWardResult",
    "rank_q_level",
    "raw_ward",
    "raw_ward_tree_coarsen",
    "screened_operators_kappa",
    "exact_rsa_epsilon_kappa",
    "run_validation_suite",
]

# The two geometries, and the ONE substitution that separates them.  Everything
# below is written in the mass vector ``kappa``:
#
#   symmetric      M_tau = L_sym + tau I,  L_sym = I - D_t^{-1/2}(W+I)D_t^{-1/2}
#                  kappa_v = d~_v = (W+I)1,   u_A = D_t^{1/2} 1_A,
#                  kappa(A) = vol(A),         Phi(A) = cut(A)/vol(A)   (conductance)
#                  ell = D_t^{-1/2} M_tau Z
#
#   combinatorial  M_tau = (D - W) + tau I
#                  kappa_v = 1,               u_A = 1_A,
#                  kappa(A) = |A|,            Phi(A) = cut(A)/|A|      (Section 3)
#                  ell = M_tau Z              <- no D^{-1/2}: THERE IS NO VOLUME
#
# In both, ``ell = diag(kappa)^{-1/2} M_tau Z``, the block mean is the
# kappa-weighted mean ``ell_bar_A = (1/kappa(A)) sum_{v in A} kappa_v ell_v``
# (a plain unweighted mean when kappa = 1), ``m(A,B)^2 = kappa_A kappa_B /
# (kappa_A + kappa_B)``, and
#
#   ||g_{A,B}||^2_{M_tau} = tau + [kappa_B Phi(A) + kappa_A Phi(B) + 2 w~(A,B)]
#                                 / (kappa_A + kappa_B),
#
# because ``g^T L g`` reduces to the same edge sum in both metrics.  The cut
# ``cut(A) = w~(A, V \ A)`` is the raw off-diagonal weight either way.
_SYMMETRIC = ("symmetric", "normalized", "sym", "norm")
_COMBINATORIAL = ("combinatorial", "comb")

_TINY = np.finfo(np.float64).tiny


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class RawWardResult:
    """Full agglomeration output.

    * ``labels_`` -- ``(N,)`` contiguous block label per original node.
    * ``children_`` -- ``(m, 2)`` merged block ids; step ``t`` creates id ``N+t``
      (sklearn's ``children_`` convention, so the tree can be cut with the same
      union-find helper the Ward tree uses).
    * ``merge_records_`` -- per-merge diagnostics: ``children``, ``new_id``,
      ``n_clusters``, ``score`` (``s^0_R``), ``numerator``, ``m_tau``, ``ell``,
      ``volume``, ``cut`` (``w~(A,B)``).
    * ``level_`` -- the prewhitened level ``r = ell G^{-1/2}`` (``(N, q_eff)``).
    * ``gram_`` -- ``G = Z^T M_tau Z``.
    * ``n_effective_target_dims_`` -- retained rank ``q_eff`` of ``G``.
    """

    labels_: np.ndarray
    children_: np.ndarray
    merge_records_: List[dict]
    level_: np.ndarray
    gram_: np.ndarray
    n_effective_target_dims_: int
    target_basis_: np.ndarray
    rsa_curve_: List[tuple] = field(default_factory=list)
    n_leaves_: int = 0
    n_clusters_: int = 0

    def labels_at(self, n_clusters: int) -> np.ndarray:
        """Re-cut the stored hierarchy to (at most) ``n_clusters`` blocks."""

        n_merges = max(0, self.n_leaves_ - int(n_clusters))
        return self._cut(min(n_merges, int(self.children_.shape[0])))

    def _cut(self, n_merges_to_apply: int) -> np.ndarray:
        n = self.n_leaves_
        m = int(self.children_.shape[0])
        apply = max(0, min(int(n_merges_to_apply), m))
        parent = list(range(n + m))

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # path compression
                parent[x], x = root, parent[x]
            return root

        pairs = self.children_.tolist()
        for t in range(apply):
            a, b = pairs[t]
            parent[find(int(a))] = n + t
            parent[find(int(b))] = n + t

        seen: Dict[int, int] = {}
        labels = np.empty(n, dtype=np.int64)
        for leaf in range(n):
            root = find(leaf)
            labels[leaf] = seen.setdefault(root, len(seen))
        return labels


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def screened_operators_kappa(W, tau: float, geometry: str = "symmetric"):
    r"""``(W_offdiag, kappa, L, M_tau)`` for either geometry.

    ``W_offdiag`` keeps the original off-diagonal weights (self-loops never
    contribute to a cut).  ``kappa`` is the per-node mass the whole construction
    is written in: the augmented degree ``d~`` under ``symmetric``, and the
    constant ``1`` under ``combinatorial`` -- which is exactly why the
    combinatorial score has no volumes, only cardinalities.
    """

    if tau <= 0.0:
        raise ValueError("tau must be strictly positive (M_tau must be PD)")
    if geometry in _SYMMETRIC:
        return screened_operators(W, tau)
    if geometry not in _COMBINATORIAL:
        raise ValueError(
            f"geometry must be symmetric or combinatorial, got {geometry!r}"
        )
    A = _sanitize_adjacency(W)
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = (sp.diags(deg) - A).tocsr()
    M = (L + tau * sp.identity(n, format="csr", dtype=np.float64)).tocsr()
    return A, np.ones(n, dtype=np.float64), L, M


def exact_rsa_epsilon_kappa(
    U: np.ndarray, M: sp.spmatrix, kappa: np.ndarray, labels: np.ndarray
) -> float:
    r"""Exact RSA constant ``sqrt(lambda_max(E^T M_tau E))`` in either geometry.

    ``E = (I - Pi_P) U`` with the ``kappa``-weighted block-averaging projector
    ``(Pi_P U)_i = sqrt(kappa_i) r_{C(i)}``, ``r_C = (1/kappa(C)) sum_{j in C}
    sqrt(kappa_j) U_j``.  Under ``combinatorial`` (``kappa = 1``) this is the
    plain uniform block average, i.e. the same projector
    :func:`src.loukas_sgc_detection._exact_rsa_epsilon` uses, so the coarsener's
    own constant and the pipeline's reported one coincide there.  Under
    ``symmetric`` they differ (degree-weighted vs uniform) and both are reported.
    """

    labels = np.asarray(labels, dtype=np.int64)
    k = int(labels.max()) + 1
    root = np.sqrt(kappa)
    mass = np.bincount(labels, weights=kappa, minlength=k)
    numer = np.zeros((k, U.shape[1]), dtype=np.float64)
    np.add.at(numer, labels, root[:, None] * U)
    r = numer / np.maximum(mass, _TINY)[:, None]
    E = U - root[:, None] * r[labels]
    H = E.T @ (M @ E)
    H = 0.5 * (H + H.T)
    return math.sqrt(max(float(np.linalg.eigvalsh(H)[-1]), 0.0))


# --------------------------------------------------------------------------- #
# the rank-q level
# --------------------------------------------------------------------------- #
def rank_q_level(
    Z: np.ndarray, M: sp.spmatrix, kappa: np.ndarray, rank_tol: float = 1e-10
):
    r"""``(ell, G, r, q_eff)`` -- the level, its Gram, and the prewhitened level.

    ``ell = diag(kappa)^{-1/2} M_tau Z`` (so ``D_t^{-1/2} M_tau Z`` under the
    symmetric geometry and plain ``M_tau Z`` under the combinatorial one, where
    ``kappa = 1``), ``G = Z^T M_tau Z`` and ``r = ell G^{-1/2}``
    with the ``G``-eigendirections below ``rank_tol * lam_max`` dropped (``Z`` may
    be rank deficient -- a wide filter bank usually is).  ``r`` is what the score
    is evaluated on, so ``|| . ||_{G^-1}`` never needs a solve.
    """

    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    MZ = M @ Z
    G = Z.T @ MZ
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(G)
    lam_max = float(evals[-1]) if evals.size else 0.0
    if lam_max <= 0.0:
        raise ValueError("G = Z^T M_tau Z is not positive definite; empty target")
    keep = evals > rank_tol * lam_max
    if not keep.any():
        raise ValueError("no target direction survives the rank tolerance")
    inv_root = 1.0 / np.sqrt(kappa)
    ell = inv_root[:, None] * MZ
    r = ell @ (evecs[:, keep] / np.sqrt(evals[keep]))
    return ell, G, r, int(keep.sum())


# --------------------------------------------------------------------------- #
# the agglomeration
# --------------------------------------------------------------------------- #
def raw_ward(
    W,
    Z: np.ndarray,
    tau: float,
    *,
    geometry: str = "symmetric",
    score: str = "raw",
    n_clusters: int | None = None,
    build_full_tree: bool = True,
    rank_tol: float = 1e-10,
    max_cluster_size: int = 0,
    evaluate_rsa: bool = False,
    rsa_levels: Sequence[int] | None = None,
) -> RawWardResult:
    r"""Adjacency-constrained raw-Ward agglomeration minimizing ``s^0_R``.

    Parameters
    ----------
    W:
        Symmetric nonnegative (sparse) adjacency; self-loops are ignored for cuts
        and the ``+I`` renormalization is applied internally.
    Z:
        ``(N, q)`` target basis; columns need not be orthogonal or independent.
    tau:
        Screening level, strictly positive (so ``M_tau`` is PD and the score's
        denominator is bounded below by ``tau``).
    score:
        ``"raw"`` -- the paper's ``s^0_R(A,B)``, the undeflated normalized score.
        ``"ward"`` -- the classical Ward increment
        ``|A||B|/(|A|+|B|) * ||U_bar_A - U_bar_B||_2^2`` on the *primal*
        ``M_tau``-orthonormal target basis with **cardinality** block means: the
        quantity ``src.ward_pr_sweep.ward_order`` (sklearn, connectivity-
        constrained) minimizes.  Running it here puts it in the same engine as
        the other four, so a difference between them is a difference of score
        alone.
        ``"ward_vol"`` -- the *unnormalized* raw score
        ``m(A,B)^2 ||ell_bar_A - ell_bar_B||^2_{G^-1}``, i.e. ``"raw"`` with the
        ``||g||^2_{M_tau}`` denominator removed (equivalently
        ``||Q_R^tau g_{A,B}||^2_{M_tau}`` with ``g`` left at its natural scale).
        It sits exactly between ``"ward"`` and ``"raw"``: it keeps the dual level
        and the volume mass of ``"raw"`` and drops only the normalization.
        ``"deflated_unnorm"`` -- the same removal applied to the deflated score,
        ``||Q_R^tau (I - Q_{P'}^tau) g_{A,B}||^2_{M_tau}
        = ||g||^2_{M_tau} (1 - eta^2) s_{P_n}(A,B)``.  With ``"deflated"`` this
        completes a 2x2 (raw vs deflated) x (normalized vs not) design.
        ``"deflated"`` -- the EXACT deflated score
        ``s_{P_n}(A,B) = ||Q_R^tau h_hat_{A,B}||^2_{M_tau}`` of
        eq. (deflated score), where ``h_hat`` is ``h_{A,B}`` made ``M_tau``-orthogonal
        to the post-merge block-constant subspace and renormalized.  This exists so
        the two scores can be compared *in one engine*: identical queue, identical
        stale-entry policy, identical tie-breaking, identical local updates -- the
        ONLY difference is the score.  Comparing ``raw_ward`` against
        :mod:`src.deflated_coarsen` instead confounds the score with the engine
        (that one uses raw insertion keys, a truncated r-hop solve, a lazy
        rescoring cap and a fanout cap).  ``"deflated"`` pays one exact sparse CG
        solve on the coarse system ``A_P = V_P^T M_tau V_P`` per candidate, so it
        is a diagnostic for graphs of a few thousand nodes, not a production path.
    geometry:
        ``"symmetric"`` (mass ``kappa = d~``, volumes and conductance) or
        ``"combinatorial"`` (mass ``kappa = 1``: **no volumes at all** -- block
        mass is cardinality, ``Phi = cut/|A|``, the block mean level is the plain
        unweighted mean, and the level drops its ``D_t^{-1/2}`` factor).
    n_clusters / build_full_tree:
        Stop early at ``n_clusters`` blocks, or build the whole hierarchy and cut
        afterwards (the default -- the tree is then reusable at every level).
    max_cluster_size:
        Optional cap on a block's *cardinality* (``0`` = uncapped).  A cost bound
        only: ``m(A,B)^2`` saturates at ``min(vol A, vol B)``, so nothing in the
        score itself discourages a large block from absorbing one more node.
    evaluate_rsa / rsa_levels:
        Record the exact (degree-weighted) RSA constant at the listed block
        counts into ``rsa_curve_``.
    """

    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")

    A, kappa, _L, M = screened_operators_kappa(W, tau, geometry)
    n = A.shape[0]
    ell, G, level, q_eff = rank_q_level(Z, M, kappa, rank_tol=rank_tol)
    U = None
    if evaluate_rsa or score == "ward":
        U, _ = m_orthonormal_basis(Z, M, rank_tol=rank_tol)

    # ---- block state, indexed by immutable block id -------------------------
    max_ids = 2 * n
    lev = np.zeros((max_ids, q_eff), dtype=np.float64)
    lev[:n] = level  # ell_bar_{v} = ell_v for singletons (kappa({v}) = kappa_v)
    # "vol" is the block's kappa-mass: vol(A) under symmetric, |A| under
    # combinatorial.  Every update below is stated in it, so the two geometries
    # share one implementation.
    vol = np.zeros(max_ids, dtype=np.float64)
    vol[:n] = kappa
    ocut = np.zeros(max_ids, dtype=np.float64)
    ocut[:n] = np.asarray(A.sum(axis=1)).ravel()  # d(A) = w~(A, V \ A)
    active = np.zeros(max_ids, dtype=bool)
    active[:n] = True
    size = np.zeros(max_ids, dtype=np.int64)
    size[:n] = 1
    # ``score="ward"`` is the only variant whose block representative is the
    # *cardinality* mean of the *primal* basis; it gets its own state so the
    # volume-weighted dual level every other score uses is untouched.
    prim = (
        np.zeros((max_ids, U.shape[1]), dtype=np.float64)
        if score == "ward"
        else None
    )
    if prim is not None:
        prim[:n] = U

    coo = sp.triu(A, k=1).tocoo()
    nbr: List[Dict[int, float]] = [dict() for _ in range(max_ids)]
    for i, j, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if w <= 0.0:
            continue
        nbr[i][j] = w
        nbr[j][i] = w

    # ---- exact-deflation state (only used by score="deflated") -------------
    # A_P = V_P^T M_tau V_P in the u_C = K^{1/2} 1_C basis has a closed form in
    # exactly the quantities the engine already tracks:
    #     (A_P)_{CC} = cut(C) + tau * kappa(C),   (A_P)_{CD} = -w~(C, D).
    # (u_C^T L u_D = 1_C^T (K - W~) 1_D, whose diagonal collapses to the cut.)
    # So no fine-grained N x N algebra is ever needed -- and neither is any
    # N-vector, because U^T M_tau u_C = kappa(C) * ell_bar_C = vol[C] * lev[C].
    if score not in _SCORES:
        raise ValueError(f"score must be one of {_SCORES}, got {score!r}")
    deflated = score in ("deflated", "deflated_unnorm")
    slot = np.full(max_ids, -1, dtype=np.int64)
    alive: List[int] = list(range(n))
    slot[:n] = np.arange(n)
    step_cache: dict = {}

    def _refresh_block_matrix():
        """Coarse edge list + diagonal of the CURRENT partition, in slot order."""

        k = len(alive)
        ids = np.asarray(alive, dtype=np.int64)
        diag = ocut[ids] + tau * vol[ids]
        rows, cols, vals = [], [], []
        for i_id in alive:
            si = slot[i_id]
            for j_id, w in nbr[i_id].items():
                sj = slot[j_id]
                if si < sj:
                    rows.append(si)
                    cols.append(sj)
                    vals.append(w)
        step_cache["k"] = k
        step_cache["diag"] = diag
        step_cache["rows"] = np.asarray(rows, dtype=np.int64)
        step_cache["cols"] = np.asarray(cols, dtype=np.int64)
        step_cache["vals"] = np.asarray(vals, dtype=np.float64)
        step_cache["VL"] = vol[ids][:, None] * lev[ids]  # (k, q): U^T M_tau u_C

    unnormalized = score == "deflated_unnorm"

    def deflated_scores_many(c: int, ks: np.ndarray, ws: np.ndarray):
        """Exact ``s_{P_n}(c, k)`` and leakage ``eta_n(c, k)`` for each ``k``.

        With ``score="deflated_unnorm"`` the returned score is instead
        ``||Q_R (I - Q_{P'}) g||^2_{M_tau}`` -- the same eliminated direction, at
        the natural scale of ``g`` rather than renormalized to a unit vector.
        """

        from src.deflated_coarsen import _block_cg

        k_now = step_cache["k"]
        diag = step_cache["diag"]
        E_r, E_c, E_v = step_cache["rows"], step_cache["cols"], step_cache["vals"]
        VL = step_cache["VL"]
        pos_c = slot[c]
        out_s = np.empty(ks.size)
        out_eta = np.empty(ks.size)
        for t in range(ks.size):
            b = int(ks[t])
            wab = float(ws[t])
            pos_b = slot[b]
            va, vb = vol[c], vol[b]
            sab = va + vb
            m2 = va * vb / sab
            # ||h||^2_{M_tau} = ||g||^2_{M_tau} / m^2
            ell_ab = max(
                (vb * (ocut[c] / va) + va * (ocut[b] / vb) + 2.0 * wab) / sab, 0.0
            )
            h_sq = (ell_ab + tau) / m2

            # --- index map of the POST-MERGE partition P' ---------------------
            newpos = np.arange(k_now, dtype=np.int64)
            newpos[newpos > pos_b] -= 1
            newpos[pos_b] = pos_c - (1 if pos_c > pos_b else 0)

            # --- A_{P'} : duplicate (i, i) entries are summed by coo, which is
            #     exactly the cut(C) = cut(A)+cut(B)-2w~(A,B) bookkeeping.
            r2 = np.concatenate([newpos[E_r], newpos[E_c], newpos])
            c2 = np.concatenate([newpos[E_c], newpos[E_r], newpos])
            v2 = np.concatenate([-E_v, -E_v, diag])
            A_P = sp.coo_matrix((v2, (r2, c2)), shape=(k_now - 1, k_now - 1)).tocsr()

            # --- c = V_{P'}^T M_tau h,  h = u_c/vol_c - u_b/vol_b -------------
            rhs = np.zeros(k_now - 1)
            np.add.at(
                rhs,
                newpos[E_r],
                -E_v * ((E_c == pos_c) / va - (E_c == pos_b) / vb),
            )
            np.add.at(
                rhs,
                newpos[E_c],
                -E_v * ((E_r == pos_c) / va - (E_r == pos_b) / vb),
            )
            rhs[newpos[pos_c]] += diag[pos_c] / va
            rhs[newpos[pos_b]] -= diag[pos_b] / vb

            x = _block_cg(A_P, rhs[:, None]).ravel()
            eta_sq = float(np.clip((rhs @ x) / max(h_sq, _TINY), 0.0, 1.0))
            # a = U^T M_tau h_hat = (U^T M_tau h - W_P x) / ||(I - Q)h||_{M_tau}
            umh = lev[c] - lev[b]  # U^T M_tau h  (see the note above)
            proj = x[newpos] @ VL
            denom = math.sqrt(max(h_sq * (1.0 - eta_sq), _TINY))
            a_vec = (umh - proj) / denom
            out_s[t] = float(a_vec @ a_vec)
            out_eta[t] = math.sqrt(eta_sq)
            if unnormalized:
                # ||Q_R (I - Q_{P'}) g||^2 = ||g||^2_{M_tau} (1 - eta^2) s,
                # and ||g||^2_{M_tau} = m^2 ||h||^2_{M_tau}
                out_s[t] *= m2 * h_sq * (1.0 - eta_sq)
        return out_s, out_eta

    def scores_many(c: int, ks: np.ndarray, ws: np.ndarray):
        r"""Vectorized ``(s^0, numerator, m_tau, ell)`` for the pairs ``(c, ks)``.

        Numerator ``m(c,k)^2 ||r_c - r_k||^2`` (the ``G^-1`` norm, prewhitened);
        denominator the local ``||g||^2_{M_tau} = tau + [v_k Phi_c + v_c Phi_k +
        2 w~]/(v_c + v_k)``.
        """

        va = vol[c]
        vb = vol[ks]
        s = va + vb
        diff = lev[ks] - lev[c]
        numer = (va * vb / s) * np.einsum("ij,ij->i", diff, diff)
        np.maximum(numer, 0.0, out=numer)
        cut_term = vb * (ocut[c] / va) + va * (ocut[ks] / vb) + 2.0 * ws
        ell_ab = np.maximum(cut_term / s, 0.0)
        m_tau = ell_ab + tau
        if score == "ward":
            ca = float(size[c])
            cb = size[ks].astype(np.float64)
            dp = prim[ks] - prim[c]
            ward = (ca * cb / (ca + cb)) * np.einsum("ij,ij->i", dp, dp)
            return np.maximum(ward, 0.0), numer, m_tau, ell_ab
        if score == "ward_vol":
            return numer, numer, m_tau, ell_ab
        return numer / m_tau, numer, m_tau, ell_ab

    # ---- initial queue: one entry per edge ----------------------------------
    heap: List[tuple] = []
    if coo.nnz:
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        vals = coo.data.astype(np.float64)
        keep = vals > 0.0
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        va0, vb0 = kappa[rows], kappa[cols]
        s0 = va0 + vb0
        diff0 = level[rows] - level[cols]
        numer0 = (va0 * vb0 / s0) * np.einsum("ij,ij->i", diff0, diff0)
        cut0 = (vb0 * (ocut[rows] / va0) + va0 * (ocut[cols] / vb0) + 2.0 * vals) / s0
        keys0 = np.maximum(numer0, 0.0) / (np.maximum(cut0, 0.0) + tau)
        if score == "ward":
            dp0 = U[rows] - U[cols]
            keys0 = 0.5 * np.einsum("ij,ij->i", dp0, dp0)  # |A||B|/(|A|+|B|) = 1/2
        elif score == "ward_vol":
            keys0 = np.maximum(numer0, 0.0)
        if deflated:
            # one edge at a time: every candidate needs its own A_{P'}
            _refresh_block_matrix()
            keys0 = np.empty(rows.size)
            for _t in range(rows.size):
                sc_, _e_ = deflated_scores_many(
                    int(rows[_t]),
                    np.array([cols[_t]], dtype=np.int64),
                    np.array([vals[_t]], dtype=np.float64),
                )
                keys0[_t] = sc_[0]
        heap = list(zip(keys0.tolist(), rows.tolist(), cols.tolist()))
        heapq.heapify(heap)
    n_edges = len(heap)

    n_active = n
    children: List[tuple] = []
    records: List[dict] = []
    rsa_curve: List[tuple] = []
    next_id = n
    cap = int(max_cluster_size) if max_cluster_size else 0
    target = int(n_clusters) if n_clusters is not None else 1

    rsa_targets: set = set()
    if evaluate_rsa:
        if rsa_levels is not None:
            rsa_targets = {int(x) for x in rsa_levels}
        else:
            rsa_targets = {
                int(round(x))
                for x in np.geomspace(max(target, 1), max(n, 1), num=min(12, n))
                if 1 <= round(x) <= n
            }

    def stop_now() -> bool:
        if build_full_tree:
            return False
        return n_clusters is not None and n_active <= target

    # The score is scale free in block mass (it is a *ratio*), so a block can
    # chain: pushes per merge are bounded by its degree, which grows.  Compact
    # the heap when it outgrows the live candidate count by a wide margin --
    # every dropped entry is one lazy invalidation would have rejected anyway.
    compact_at = max(4 * (n_edges + n), 1 << 20)

    while heap and n_active > 1 and not stop_now():
        _score, a, b = heapq.heappop(heap)
        # ---- discard stale entries (merged away, or no longer adjacent) -----
        if not active[a] or not active[b]:
            continue
        wab = nbr[a].get(b, 0.0)
        if wab <= 0.0:
            continue
        if cap and size[a] + size[b] > cap:
            continue
        # the entry is current, so re-deriving is exact and returns the parts
        kb = np.array([b], dtype=np.int64)
        wb = np.array([wab], dtype=np.float64)
        sc, numer, m_tau, ell_ab = scores_many(a, kb, wb)
        eta_ab = float("nan")
        if deflated:
            _refresh_block_matrix()
            sc_d, eta_d = deflated_scores_many(a, kb, wb)
            sc, eta_ab = sc_d, float(eta_d[0])

        # ---- block update ---------------------------------------------------
        new = next_id
        next_id += 1
        va, vb = vol[a], vol[b]
        vol[new] = va + vb
        lev[new] = (va * lev[a] + vb * lev[b]) / (va + vb)  # volume-weighted mean
        if prim is not None:  # cardinality-weighted mean of the primal basis
            ca, cb = float(size[a]), float(size[b])
            prim[new] = (ca * prim[a] + cb * prim[b]) / (ca + cb)
        ocut[new] = ocut[a] + ocut[b] - 2.0 * wab
        size[new] = size[a] + size[b]
        active[a] = active[b] = False
        active[new] = True
        if deflated:  # swap-remove b, then let `new` take a's slot
            j = slot[b]
            last = alive[-1]
            alive[j] = last
            slot[last] = j
            alive.pop()
            slot[b] = -1
            slot[new] = slot[a]
            alive[slot[a]] = new
            slot[a] = -1

        merged: Dict[int, float] = {}
        for k, w in nbr[a].items():
            if k != b and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k, w in nbr[b].items():
            if k != a and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k in nbr[a]:
            nbr[k].pop(a, None)
        for k in nbr[b]:
            nbr[k].pop(b, None)
        nbr[a] = {}
        nbr[b] = {}
        for k, w in merged.items():
            if w <= 0.0:
                continue
            nbr[new][k] = w
            nbr[k][new] = w
        # Surviving neighbours ``k`` keep their volume, level and cut, and the
        # new pair carries a *fresh* id, so every cached score that passes the
        # alive/adjacent test above is still exact -- "scores between unaffected
        # blocks remain unchanged".

        n_active -= 1
        children.append((a, b))
        records.append(
            {
                "children": (int(a), int(b)),
                "new_id": int(new),
                "n_clusters": int(n_active),
                "score": float(sc[0]),
                "score_raw": float(numer[0] / m_tau[0]),
                "eta": eta_ab,
                "numerator": float(numer[0]),
                "m_tau": float(m_tau[0]),
                "ell": float(ell_ab[0]),
                "volume": float(vol[new]),
                "cut": float(wab),
            }
        )

        # ---- recompute only s^0(C, D) for D in N(C) -------------------------
        if nbr[new]:
            ks = np.fromiter(nbr[new].keys(), dtype=np.int64, count=len(nbr[new]))
            ws = np.fromiter(nbr[new].values(), dtype=np.float64, count=len(nbr[new]))
            if cap:
                admissible = size[ks] + size[new] <= cap
                ks, ws = ks[admissible], ws[admissible]
            if ks.size:
                if deflated:
                    _refresh_block_matrix()
                    s_k, _eta_k = deflated_scores_many(new, ks, ws)
                else:
                    s_k, _, _, _ = scores_many(new, ks, ws)
                for t in range(ks.size):
                    k = int(ks[t])
                    lo, hi = (k, new) if k < new else (new, k)
                    heapq.heappush(heap, (float(s_k[t]), lo, hi))

        if len(heap) > compact_at:
            heap = [e for e in heap if active[e[1]] and active[e[2]]]
            heapq.heapify(heap)
            compact_at = max(compact_at, 2 * len(heap))

        if rsa_targets and n_active in rsa_targets:
            labels_now = _current_labels(n, children)
            rsa_curve.append(
                (int(n_active), exact_rsa_epsilon_kappa(U, M, kappa, labels_now))
            )

    children_arr = (
        np.asarray(children, dtype=np.int64)
        if children
        else np.empty((0, 2), dtype=np.int64)
    )
    result = RawWardResult(
        labels_=np.zeros(n, dtype=np.int64),
        children_=children_arr,
        merge_records_=records,
        level_=level,
        gram_=G,
        n_effective_target_dims_=q_eff,
        target_basis_=(U if U is not None else np.empty((n, 0))),
        rsa_curve_=rsa_curve,
        n_leaves_=n,
    )
    if n_clusters is not None and build_full_tree:
        result.labels_ = result.labels_at(int(n_clusters))
    else:
        result.labels_ = result._cut(len(children))
    result.n_clusters_ = int(result.labels_.max()) + 1 if n else 0
    return result


def _current_labels(n: int, children: Sequence[tuple]) -> np.ndarray:
    """Contiguous leaf labels after applying every merge in ``children``."""

    parent = list(range(n + len(children)))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for t, (a, b) in enumerate(children):
        parent[find(int(a))] = n + t
        parent[find(int(b))] = n + t
    seen: Dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for leaf in range(n):
        labels[leaf] = seen.setdefault(find(leaf), len(seen))
    return labels


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _random_connected_graph(rng, n: int):
    Wd = np.zeros((n, n))
    for i in range(n - 1):
        Wd[i, i + 1] = rng.uniform(0.5, 2.0)
    extra = np.triu(rng.random((n, n)) < 0.18, 1)
    Wd = np.where(extra, rng.uniform(0.5, 2.0, size=(n, n)), Wd)
    Wd = Wd + Wd.T
    np.fill_diagonal(Wd, 0.0)
    return sp.csr_matrix(Wd)


def _brute_pair(level_r, ell, G, M, kappa, A, labels, ca, cb, tau):
    """Score of blocks ``ca``/``cb`` computed from scratch (no incremental state).

    Written in ``kappa`` so it validates BOTH geometries: with ``kappa = d~`` the
    contrast vector is ``m (D_t^{1/2}1_A/vol A - ...)`` and the block mean is
    volume weighted; with ``kappa = 1`` it is ``m (1_A/|A| - ...)`` and the block
    mean is the plain average.
    """

    n = A.shape[0]
    ia = np.flatnonzero(labels == ca)
    ib = np.flatnonzero(labels == cb)
    va = float(kappa[ia].sum())
    vb = float(kappa[ib].sum())
    s = va + vb
    m2 = va * vb / s
    # explicit contrast vector g = m (u_A/kappa_A - u_B/kappa_B), u_A = K^{1/2}1_A
    x = np.zeros(n)
    x[ia] = 1.0 / va
    x[ib] = -1.0 / vb
    g = math.sqrt(m2) * np.sqrt(kappa) * x
    m_tau = float(g @ (M @ g))
    # numerator in the G^{-1} norm on the RAW (un-prewhitened) level
    ellbar_a = (kappa[ia, None] * ell[ia]).sum(0) / va
    ellbar_b = (kappa[ib, None] * ell[ib]).sum(0) / vb
    diff = ellbar_a - ellbar_b
    numer_G = m2 * float(diff @ np.linalg.pinv(G) @ diff)
    # numerator on the prewhitened level
    ra = (kappa[ia, None] * level_r[ia]).sum(0) / va
    rb = (kappa[ib, None] * level_r[ib]).sum(0) / vb
    numer_r = m2 * float((ra - rb) @ (ra - rb))
    return numer_G / m_tau, numer_r / m_tau, m_tau, numer_r, m2


def _check_geometry(W, Z, tau, geometry, verbose):
    """Every identity of the paragraph, in one geometry, to machine precision."""

    def say(msg):
        if verbose:
            print(msg)

    say(f"\n--- geometry = {geometry} " + "-" * (46 - len(geometry)))
    A, kappa, L, M = screened_operators_kappa(W, tau, geometry)
    n = A.shape[0]
    ell, G, level_r, q_eff = rank_q_level(Z, M, kappa)
    res = raw_ward(W, Z, tau, geometry=geometry, build_full_tree=True)

    # --- 1. prewhitening: ||.||_{G^-1} on ell  ==  ||.||_2 on r -------------
    worst_pre = worst_hi = worst_lo = 0.0
    Acoo = sp.triu(A, k=1).tocoo()
    for k in (n, 40, 12, 5):
        lab = res.labels_at(k)
        pairs = set()
        for i, j in zip(Acoo.row.tolist(), Acoo.col.tolist()):
            if lab[i] != lab[j]:
                pairs.add((min(lab[i], lab[j]), max(lab[i], lab[j])))
        for ca, cb in list(pairs)[:60]:
            s_G, s_r, _m, _num, _m2 = _brute_pair(
                level_r, ell, G, M, kappa, A, lab, ca, cb, tau
            )
            worst_pre = max(worst_pre, abs(s_G - s_r) / max(abs(s_G), 1e-12))
            worst_hi = max(worst_hi, s_r - 1.0)
            worst_lo = max(worst_lo, -s_r)
    assert worst_pre < 1e-9, worst_pre
    assert worst_hi < 1e-9 and worst_lo < 1e-9, (worst_hi, worst_lo)
    say(f"[1] prewhitening  ||.||_G^-1 == ||.||_2 on r : rel err {worst_pre:.2e}")
    say(
        f"    s^0 in [0, 1]                             : slack {max(worst_hi, worst_lo):.2e}"
    )

    # --- 2. local denominator / numerator / cut vs brute force --------------
    labels_now = np.arange(n)
    worst_m_tau = worst_num = worst_cut = 0.0
    block_of = {i: i for i in range(n)}
    for rec in res.merge_records_:
        a, b = rec["children"]
        ca, cb = block_of[a], block_of[b]
        _sg, _sr, m_tau, numer, _m2 = _brute_pair(
            level_r, ell, G, M, kappa, A, labels_now, ca, cb, tau
        )
        worst_m_tau = max(worst_m_tau, abs(m_tau - rec["m_tau"]) / max(m_tau, 1e-12))
        worst_num = max(worst_num, abs(numer - rec["numerator"]) / max(numer, 1e-12))
        ia = np.flatnonzero(labels_now == ca)
        ib = np.flatnonzero(labels_now == cb)
        wab = float(A[ia][:, ib].sum())
        worst_cut = max(worst_cut, abs(wab - rec["cut"]) / max(wab, 1e-12))
        labels_now[labels_now == cb] = ca
        block_of[rec["new_id"]] = ca
    assert worst_m_tau < 1e-9 and worst_num < 1e-9 and worst_cut < 1e-9
    say(f"[2] local ||g||^2_M_tau vs brute force        : rel err {worst_m_tau:.2e}")
    say(f"    kappa-weighted level update (numerator)   : rel err {worst_num:.2e}")
    say(f"    cut update d(C)=d(A)+d(B)-2w~(A,B)        : rel err {worst_cut:.2e}")

    # --- 3. the mean-level identity of the paragraph ------------------------
    worst_id = 0.0
    vol = {i: float(kappa[i]) for i in range(n)}
    lvl = {i: level_r[i].copy() for i in range(n)}
    for rec in res.merge_records_:
        a, b = rec["children"]
        va, vb = vol[a], vol[b]
        lc = (va * lvl[a] + vb * lvl[b]) / (va + vb)
        lhs = float(np.linalg.norm(lc - lvl[a]))
        m = math.sqrt(va * vb / (va + vb))
        rhs = (vb / (va + vb)) * math.sqrt(rec["m_tau"]) / m * math.sqrt(rec["score"])
        worst_id = max(worst_id, abs(lhs - rhs) / max(lhs, 1e-12))
        vol[rec["new_id"]] = va + vb
        lvl[rec["new_id"]] = lc
    assert worst_id < 1e-9, worst_id
    say(f"[3] mean-level identity (displayed equation)  : rel err {worst_id:.2e}")

    # --- 4. connectivity ----------------------------------------------------
    import scipy.sparse.csgraph as csg

    for k in (30, 10, 4):
        lab = res.labels_at(k)
        for c in np.unique(lab):
            members = np.flatnonzero(lab == c)
            ncomp, _ = csg.connected_components(A[members][:, members], directed=False)
            assert ncomp == 1, f"block {c} at k={k} has {ncomp} components"
    say("[4] every block induces a connected subgraph  : ok")

    # --- 5. the coarsener's own RSA projector vs the pipeline's -------------
    #  kappa = 1 makes the kappa-weighted block average the UNIFORM one, so under
    #  the combinatorial geometry the two constants must agree exactly.
    U, _ = m_orthonormal_basis(Z, M)
    lab = res.labels_at(12)
    eps_kappa = exact_rsa_epsilon_kappa(U, M, kappa, lab)
    kk = int(lab.max()) + 1
    sums = np.zeros((kk, U.shape[1]))
    np.add.at(sums, lab, U)
    counts = np.bincount(lab, minlength=kk).clip(1)
    E = U - (sums / counts[:, None])[lab]
    H = E.T @ (M @ E)
    eps_uniform = math.sqrt(max(float(np.linalg.eigvalsh(0.5 * (H + H.T))[-1]), 0.0))
    if geometry in _COMBINATORIAL:
        assert abs(eps_kappa - eps_uniform) < 1e-9, (eps_kappa, eps_uniform)
        say(
            f"[5] kappa-weighted RSA == uniform RSA         : {eps_kappa:.6f} (must match)"
        )
    else:
        say(
            f"[5] RSA  degree-weighted {eps_kappa:.4f}  vs uniform "
            f"{eps_uniform:.4f}  (differ by construction)"
        )
    return res


    """Every identity of the paragraph, in BOTH geometries, to machine precision."""

    rng = np.random.default_rng(seed)
    n, q, tau = 90, 4, 0.35
    W = _random_connected_graph(rng, n)
    Z = rng.standard_normal((n, q))
    Z = np.hstack([Z, Z[:, :1] * 2.0])  # deliberately rank deficient

    res_sym = _check_geometry(W, Z, tau, "symmetric", verbose)
    _check_geometry(W, Z, tau, "combinatorial", verbose)

    # --- symmetric raw Ward IS smooth_dual_ward(alpha=0, dual) --------------
    from smooth_dual_ward import smooth_dual_ward

    sdw = smooth_dual_ward(W, Z, tau, alpha=0.0, build_full_tree=True)
    same = all(
        tuple(x) == tuple(y)
        for x, y in zip(res_sym.children_.tolist(), sdw.children_.tolist())
    )
    for k in (40, 20, 8, 3):
        la, lb = res_sym.labels_at(k), sdw.labels_at(k)
        assert len({(int(x), int(y)) for x, y in zip(la, lb)}) == len(np.unique(la))
    if verbose:
        print(
            f"\n[6] symmetric raw Ward == smooth_dual_ward(alpha=0): "
            f"merge order {'identical' if same else 'ties differ'}, partitions equal"
        )

    # --- the two geometries must genuinely disagree -------------------------
    #  (otherwise the combinatorial switch would be a no-op and the comparison
    #  below would be measuring nothing)
    rc = raw_ward(W, Z, tau, geometry="combinatorial", build_full_tree=True)
    la, lb = res_sym.labels_at(12), rc.labels_at(12)
    identical = len({(int(x), int(y)) for x, y in zip(la, lb)}) == len(np.unique(la))
    assert not identical, "symmetric and combinatorial produced the same partition"
    if verbose:
        print("[7] symmetric and combinatorial partitions differ  : ok")
        print("\nALL RAW-WARD CHECKS PASSED (both geometries)")