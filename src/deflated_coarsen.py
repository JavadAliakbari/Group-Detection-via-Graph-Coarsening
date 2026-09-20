r"""Screened-consistent agglomeration: the deflated dual score.

This module implements the coarsener of the paper's *deflated dual* section.  It
replaces the Euclidean block average ``Pi_P`` used to *score* merges by the
``M_tau``-orthogonal (harmonic) block projector

    Q_P^tau = V_P (V_P^T M_tau V_P)^{-1} V_P^T M_tau ,       M_tau = L + tau I,

which is the ``M_tau``-orthogonal projector onto the same block-constant
subspace ``A_P = range(V_P)``.  The realized coarsening is still the Euclidean
block averaging used everywhere else in the pipeline; the two are tied by the
RSA sandwich (Lemma "RSA sandwich")

    eps_Q(P)  <=  eps_Pi(P)  <=  mu_P^tau * eps_Q(P),
    mu_P^tau  = ||I - Pi_P||_{M_tau -> M_tau} ,

so the algorithm optimizes and certifies ``eps_Q`` and the realized coarsening
pays one a-posteriori factor ``mu_P^tau``.  Both constants are returned.

Why deflate
-----------
Merging ``A, B`` removes exactly one direction from ``A_P``: the normalized
contrast

    g_{A,B} = sqrt(vol B / s) v_A - sqrt(vol A / s) v_B ,   s = vol A + vol B.

The *raw* contrast is not the direction the merge destroys, because part of it
is still representable by the coarser partition.  Deflating against ``P'``,

    g~ = (I - Q_{P'}^tau) g,      g^ = g~ / ||g~||_{M_tau},

isolates the part that genuinely disappears, and then the deflation identity

    Q_P^tau = Q_{P'}^tau + g^ g^T M_tau                                (exact)

makes the target-residual matrix

    H_P^tau := U_tau^T M_tau (I - Q_P^tau) U_tau       (d x d, PSD)

obey an *exact rank-one* merge calculus

    H_{P'}^tau = H_P^tau + a a^T,    a := U_tau^T M_tau g^,
    ||a||^2 = ||Q_R^tau g^||^2_{M_tau}   (the screened visibility of g^).

That is the whole point of the section: unlike the Euclidean minimax update
(rank two and indefinite, :mod:`src.minimax_coarsen`), this update is PSD and
rank one, so

* ``tr H_P^tau = sum_t ||a_t||^2``  telescopes and is path-independent;
* ``tr H_P`` and ``lambda_max(H_P)`` are *monotone* along the hierarchy;
* ``eps_Q(P) = sqrt(lambda_max(H_P^tau))`` exactly, and stale candidate scores
  are valid lower bounds forever (so lazy-greedy is sound);
* every increment is a screened visibility, so the ``chi^tau`` recovery
  certificate applies verbatim.

Two merge rules
---------------
``rule="dual-ward"``  score ``s(A,B) = ||a||^2``            -- greedy steepest
    descent on the exact state function ``tr H_P^tau``.
``rule="minimax"``    score ``s(A,B) = lambda_max(H_P + a a^T)`` -- greedy
    descent on the exact screened RSA constant ``eps_Q``.  Evaluated by the
    rank-one secular equation in ``O(d)`` (``score_mode="secular"``) or a plain
    ``eigvalsh`` (``O(d^3)``); both agree to machine precision, and the free
    Weyl bracket ``lam1 + (u1^T a)^2 <= lam_max' <= lam1 + ||a||^2`` is exposed
    for pruning.

Geometry and block algebra
--------------------------
Identical conventions to :mod:`src.smooth_dual_ward` /
:mod:`src.minimax_coarsen`: ``d~ = (W + I) 1``, ``L = I - D~^{-1/2}(W+I)D~^{-1/2}``,
block vectors ``v_C = D~^{1/2} 1_C / sqrt(vol C)`` (Euclidean-orthonormal,
``v_i = e_i`` at singletons).  Because ``D~^{1/2} L D~^{1/2} = D_W - W`` the
block Gram is closed form in the cut statistics,

    v_C^T M_tau v_C = (o_C + tau vol C)/vol C ,
    v_C^T M_tau v_K = -w_CK / sqrt(vol C vol K)   (C != K),

so ``A_P^tau := V_P^T M_tau V_P`` is the (volume-normalized) coarse screened
Laplacian ``L_c + tau I``, maintained incrementally from the block adjacency.
No ``N x N`` matrix is ever formed; per-block state is ``vol``, ``o`` (cut),
the neighbour weights, and the two ``d``-vectors

    p_C = U_tau^T v_C        (Euclidean, for the realized Pi-tracking)
    m_C = U_tau^T M_tau v_C  (screened, for the deflation)

both of which merge exactly as ``x_new = alpha x_A + beta x_B``.

Candidate evaluation (Lemma "leakage is local and computable")
-------------------------------------------------------------
``b := V_{P'}^T M_tau g`` is supported on ``A u B`` and its coarse neighbours
only (``g`` is Euclidean-orthogonal to every ``P'`` block indicator, so
``b = V_{P'}^T L g`` is a pure boundary quantity).  With ``A_{P'}^tau c = b``,

    ||g~||^2_{M_tau} = ||g||^2_{M_tau} - b^T c ,
    eta_0^2 = (b^T c) / ||g||^2_{M_tau}                (deflation leakage)
    a = (U^T M_tau g - W_{P'} c) / ||g~||_{M_tau}, W_{P'} = [m_C]_C .

``solve="local"`` truncates that solve to an ``hops``-hop coarse ball around
``A u B`` (Dirichlet), which is exact up to ``O(q^r)`` by the exponential decay
of ``(L_c + tau I)^{-1}``; ``solve="exact"`` assembles and solves the full
coarse system.  Screening is what makes the harmonic geometry local: the
condition number is at most ``(lambda_max + tau)/tau``.

``commit_solve`` selects the mode used for the merge that is actually committed
(default: same as ``solve``).  Setting ``commit_solve="exact"`` keeps ``H`` --
and hence the reported certificate -- exact while candidate *ranking* stays
cheap.

Stopping
--------
``build_full_tree=True`` merges all the way down to one block per connected
component and records ``eps_Q``, ``eps_Pi`` and ``tr H`` after every merge, so
any level can be re-cut afterwards (:meth:`labels_at`, :meth:`epsilon_q_at`,
:meth:`epsilon_pi_at`).  ``eps_Q`` is monotone by construction; ``eps_Pi`` need
not be (``Pi_P`` is Euclidean), so :meth:`coarsest_within` scans the trajectory.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import scipy.sparse as sp

from smooth_dual_ward import (
    exact_rsa_epsilon,
    m_orthonormal_basis,
    screened_operators,
)

__all__ = [
    "DeflatedCoarseningResult",
    "deflated_coarsen",
    "deflated_tree_coarsen",
    "exact_harmonic_H",
    "harmonic_rsa_epsilon",
    "block_distortion_mu",
    "rank_one_lambda_max",
    "weyl_bracket",
    "run_validation_suite",
]

_TINY = np.finfo(np.float64).tiny


# ---------------------------------------------------------------------------
# reference (brute-force) quantities -- used for validation and for the exact
# certificate reported at the returned partition
# ---------------------------------------------------------------------------


def _block_matrices(M: sp.spmatrix, d_tilde: np.ndarray, labels: np.ndarray):
    """``(V, A_P)``: block-indicator matrix and ``A_P^tau = V^T M_tau V``.

    ``A_P`` is returned **sparse**: it is the coarse screened Laplacian
    ``L_c + tau I``, as sparse as the coarse graph, and a fine cut of a large
    graph has tens of thousands of blocks -- densifying it is a quadratic-memory
    trap (14k blocks is a 1.6 GB array and an ``O(k^3)`` solve).
    """

    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    k = int(labels.max()) + 1
    root = np.sqrt(d_tilde)
    vol = np.bincount(labels, weights=d_tilde, minlength=k)
    V = sp.csr_matrix(
        (root / np.sqrt(vol[labels]), (np.arange(n), labels)), shape=(n, k)
    )
    A_P = (V.T @ (M @ V)).tocsc()
    A_P = (0.5 * (A_P + A_P.T)).tocsc()
    return V, A_P


def _block_cg(A: sp.spmatrix, B: np.ndarray, tol: float = 1e-11, maxiter: int = 1000):
    """Solve ``A X = B`` for a sparse SPD ``A`` and many right-hand sides.

    Conjugate gradients, one independent solve per column, all columns advanced
    together in numpy.  A direct factorization is the wrong tool here: ``A`` is a
    coarse *graph* Laplacian, and on a scale-free graph (Elliptic++ day 25 has a
    degree-4,960 node) sparse LU fills in catastrophically.  CG has no fill and
    the screened system is well conditioned by construction --
    ``cond(L_c + tau I) <= (lambda_max + tau)/tau`` -- so it converges in a few
    dozen iterations regardless of the graph's size.
    """

    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()
    rs = np.einsum("ij,ij->j", R, R)
    target = tol * np.maximum(np.sqrt(np.einsum("ij,ij->j", B, B)), _TINY)
    for _ in range(maxiter):
        AP = A @ P
        denom = np.maximum(np.einsum("ij,ij->j", P, AP), _TINY)
        alpha = rs / denom
        X += alpha * P
        R -= alpha * AP
        rs_new = np.einsum("ij,ij->j", R, R)
        if np.all(np.sqrt(rs_new) <= target):
            break
        P = R + (rs_new / np.maximum(rs, _TINY)) * P
        rs = rs_new
    return X


def exact_harmonic_H(
    U: np.ndarray, M: sp.spmatrix, d_tilde: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    r"""Brute-force ``H_P^tau = U^T M_tau (I - Q_P^tau) U`` for an ``M_tau``-orthonormal ``U``.

    Uses ``H = I_d - W_P A_P^{-1} W_P^T`` with ``W_P = U^T M_tau V_P`` and
    ``A_P = V_P^T M_tau V_P``, which never forms ``Q_P^tau``.
    """

    V, A_P = _block_matrices(M, d_tilde, labels)
    W_P = np.asarray(U.T @ (M @ V))  # d x k
    # A_P = L_c + tau I is SPD and as sparse as the coarse graph: solved
    # iteratively with d right-hand sides, never densified and never factorized.
    H = np.eye(U.shape[1]) - W_P @ _block_cg(A_P.tocsr(), np.ascontiguousarray(W_P.T))
    return 0.5 * (H + H.T)


def harmonic_rsa_epsilon(
    U: np.ndarray, M: sp.spmatrix, d_tilde: np.ndarray, labels: np.ndarray
) -> float:
    """``eps_Q(P) = sqrt(lambda_max(H_P^tau))`` -- the exact screened RSA of the
    harmonic reconstruction (Theorem "exact identities", part 1)."""

    lam = float(np.linalg.eigvalsh(exact_harmonic_H(U, M, d_tilde, labels))[-1])
    return math.sqrt(max(lam, 0.0))


def block_distortion_mu(
    M: sp.spmatrix,
    d_tilde: np.ndarray,
    labels: np.ndarray,
    *,
    n_iter: int = 60,
    tol: float = 1e-9,
    seed: int = 0,
) -> float:
    r"""``mu_P^tau = ||I - Pi_P||_{M_tau -> M_tau}`` by ``M_tau``-power iteration.

    ``mu^2`` is the largest eigenvalue of ``M^{-1} R^T M R`` with
    ``R = I - Pi_P`` the Euclidean block-averaging residual; ``M^{-1}`` is applied
    by conjugate gradients (the screened system is well conditioned,
    ``cond <= (lambda_max + tau)/tau``).  This is the single a-posteriori factor
    of the RSA sandwich, so the certified interval for the realized Euclidean RSA
    error is ``[eps_Q, mu * eps_Q]``.
    """

    from scipy.sparse.linalg import cg

    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    k = int(labels.max()) + 1
    root = np.sqrt(d_tilde)
    vol = np.maximum(np.bincount(labels, weights=d_tilde, minlength=k), _TINY)

    def residual(x: np.ndarray) -> np.ndarray:
        acc = np.bincount(labels, weights=root * x, minlength=k)
        return x - root * (acc / vol)[labels]

    rng = np.random.default_rng(seed)
    x = residual(rng.normal(size=n))
    nrm = math.sqrt(max(float(x @ (M @ x)), _TINY))
    x /= nrm
    lam = 0.0
    for _ in range(n_iter):
        y = residual(M @ residual(x))
        z, _info = cg(M, y, rtol=1e-10, atol=0.0, maxiter=500)
        new = math.sqrt(max(float(z @ (M @ z)), 0.0))
        if new <= 0.0:
            break
        x = z / new
        lam_new = float(x @ residual(M @ residual(x)))
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            lam = lam_new
            break
        lam = lam_new
    return math.sqrt(max(lam, 0.0))


# ---------------------------------------------------------------------------
# rank-one spectral updates
# ---------------------------------------------------------------------------


def rank_one_lambda_max(
    evals: np.ndarray, z: np.ndarray, *, tol: float = 1e-13, max_iter: int = 100
) -> float:
    r"""Largest root of the secular equation ``1 + sum_i z_i^2/(lam_i - lam) = 0``.

    ``evals`` are the eigenvalues of the symmetric ``H`` and ``z = Q^T a`` the
    update in its eigenbasis, so this returns ``lambda_max(H + a a^T)`` exactly
    in ``O(d)`` per iteration.  On ``(lam_max(H), lam_max(H) + ||a||^2]`` the
    secular function is increasing from ``-inf`` to ``1``, so a safeguarded
    bisection/Newton is unconditionally convergent.
    """

    if evals.size == 0:
        return 0.0
    lam1 = float(evals[-1])
    nz = float(z @ z)
    if nz <= 0.0:
        return max(lam1, 0.0)
    lo, hi = lam1, lam1 + nz
    # exclude the pole at lam1 (present only if the top eigenvector is hit)
    lo = lam1 + max(1e-16, 1e-14 * max(1.0, abs(lam1)))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        denom = evals - mid
        # every denominator is negative on (lam1, inf)
        f = 1.0 + float(np.sum(z * z / denom))
        if f > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, abs(hi)):
            break
    return max(0.5 * (lo + hi), 0.0)


def weyl_bracket(lam1: float, u1: np.ndarray, a: np.ndarray):
    """``(lower, upper)`` of ``lambda_max(H + a a^T)``: the PSD rank-one bracket
    ``lam1 + (u1^T a)^2 <= . <= lam1 + ||a||^2``."""

    ua = float(u1 @ a)
    return lam1 + ua * ua, lam1 + float(a @ a)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class DeflatedCoarseningResult:
    """Merge hierarchy plus *both* RSA constants at every level.

    ``epsilon_q_`` is the harmonic (intrinsic, monotone) constant the algorithm
    optimizes; ``epsilon_pi_`` is the realized Euclidean block-averaging constant
    the rest of the pipeline reports (identical to
    :func:`src.smooth_dual_ward.exact_rsa_epsilon`).  ``mu_`` is the sandwich
    factor, so ``epsilon_pi_`` is certified to lie in
    ``[epsilon_q_, mu_ * epsilon_q_]``.
    """

    labels_: np.ndarray
    children_: np.ndarray
    merge_records_: List[dict]
    H_: np.ndarray
    epsilon_q_: float
    epsilon_pi_: float
    trace_h_: float
    rule_: str
    n_effective_target_dims_: int
    target_basis_: np.ndarray
    n_leaves_: int = 0
    n_clusters_: int = 0
    mu_: float = float("nan")
    curve_: List[dict] = field(default_factory=list)

    # -- level lookups ------------------------------------------------------
    def _level(self, n_clusters: int) -> dict:
        k = int(n_clusters)
        for entry in self.curve_:
            if entry["n_clusters"] == k:
                return entry
        raise ValueError(f"level {k} not present in the recorded curve")

    def epsilon_q_at(self, n_clusters: int) -> float:
        if int(n_clusters) >= self.n_leaves_:
            return 0.0
        return float(self._level(n_clusters)["epsilon_q"])

    def epsilon_pi_at(self, n_clusters: int) -> float:
        if int(n_clusters) >= self.n_leaves_:
            return 0.0
        return float(self._level(n_clusters)["epsilon_pi"])

    def trace_at(self, n_clusters: int) -> float:
        if int(n_clusters) >= self.n_leaves_:
            return 0.0
        return float(self._level(n_clusters)["trace_h"])

    def coarsest_within(self, epsilon_max: float, key: str = "epsilon_pi") -> int:
        """Fewest blocks still within the budget.

        Scans the whole trajectory: ``epsilon_q`` is monotone (so the scan finds
        the crossing), but ``epsilon_pi`` need not be.
        """

        ok = [e["n_clusters"] for e in self.curve_ if e[key] <= epsilon_max]
        return min(ok) if ok else self.n_leaves_

    def labels_at(self, n_clusters: int) -> np.ndarray:
        n_merges = max(0, self.n_leaves_ - int(n_clusters))
        return self._cut(min(n_merges, int(self.children_.shape[0])))

    def labels_at_many(self, n_clusters_list) -> Dict[int, np.ndarray]:
        """``{n_clusters: labels}`` for many cuts in one pass.

        Cutting a hierarchy at ``c`` levels one at a time costs ``O(c (n + m))``
        *Python* operations -- on a 17k-node tree with 500 cuts that is ~17M
        interpreted union-find steps and dominates everything the coarsener
        itself does.  Here the merges are replayed once, in order, and the roots
        are resolved with a vectorized pointer-jumping pass that also compresses
        the paths, so each additional cut costs a couple of numpy passes over
        ``n``.  Returns the same partitions as :meth:`labels_at` (the label
        *numbering* may differ; the blocks do not).
        """

        n = self.n_leaves_
        m = int(self.children_.shape[0])
        ts = sorted({min(max(0, n - int(k)), m) for k in n_clusters_list})
        parent = np.arange(n + m, dtype=np.int64)
        leaves = np.arange(n, dtype=np.int64)
        pairs = self.children_
        out: Dict[int, np.ndarray] = {}
        done = 0
        for t in ts:
            # ids are created in merge order, so both children are roots already
            for step in range(done, t):
                parent[pairs[step, 0]] = n + step
                parent[pairs[step, 1]] = n + step
            done = t
            root = parent[leaves]
            while True:
                nxt = parent[root]
                if np.array_equal(nxt, root):
                    break
                root = nxt
            parent[leaves] = root  # path compression: later cuts converge at once
            _, labels = np.unique(root, return_inverse=True)
            out[n - t] = labels.astype(np.int64)
        return out

    def _cut(self, n_merges_to_apply: int) -> np.ndarray:
        n = self.n_leaves_
        m = int(self.children_.shape[0])
        apply = max(0, min(int(n_merges_to_apply), m))
        parent = list(range(n + m))

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
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
            labels[leaf] = seen.setdefault(find(leaf), len(seen))
        return labels


# ---------------------------------------------------------------------------
# the coarsener
# ---------------------------------------------------------------------------


def deflated_coarsen(
    W,
    Z: np.ndarray,
    tau: float,
    *,
    rule: str = "dual-ward",
    n_clusters: int | None = None,
    build_full_tree: bool = False,
    hops: int = 2,
    max_ball: int = 32,
    solve: str = "local",
    commit_solve: str | None = None,
    score_mode: str = "auto",
    max_cluster_size: int = 0,
    max_rescore: int = 8,
    fanout: int = 32,
    strict: bool = False,
    epsilon_max: float | None = None,
    epsilon_key: str = "epsilon_q",
    track_euclidean: bool = True,
    record_curve: bool = True,
    curve_levels: set | None = None,
    compute_mu: bool = False,
    rank_tol: float = 1e-10,
) -> DeflatedCoarseningResult:
    r"""Screened-consistent agglomeration (deflated dual-Ward / deflated minimax).

    Implements the pseudocode of the deflated-dual section verbatim: candidate
    merges are the graph-*adjacent* block pairs, each is scored by the part of
    its contrast that the post-merge partition can no longer represent
    (deflation), the score is how much of that eliminated direction lies in the
    target subspace, and the pair causing the least target damage is merged.

    Parameters
    ----------
    W, Z, tau
        Graph adjacency, target-subspace generator (``R = span(Z)``) and
        screening level.  ``Z`` is ``M_tau``-orthonormalized internally.
    rule
        ``"dual-ward"`` (score ``||a||^2``) or ``"minimax"``
        (score ``lambda_max(H + a a^T)``).
    hops, max_ball, solve, commit_solve
        Deflation solve: ``"local"`` truncates ``(L_c + tau I) c = b`` to an
        ``hops``-hop coarse ball (capped at ``max_ball`` blocks), ``"exact"``
        solves the full coarse system.  ``commit_solve`` overrides the mode for
        the merge that is actually committed -- use ``"exact"`` to keep ``H``
        (and therefore the certificate) exact while ranking stays cheap.
    fanout
        How many queue entries a merge pushes for the new block: the ``fanout``
        neighbours with the best insertion key, or all of them if ``fanout <= 0``.
        A block that has absorbed a hub can have thousands of neighbours, and
        pushing one entry each makes the queue -- not the linear algebra --
        superlinear.  The excluded pairs are the ones whose raw score is already
        worse than ``fanout`` siblings of the same block, and they are
        re-considered every time that block merges again.
    epsilon_max, epsilon_key
        Refuse merges that would push ``epsilon_key`` past the budget.
    track_euclidean
        Also maintain the realized Euclidean ``H_Pi`` by its exact rank-two
        update, giving ``epsilon_pi`` at every recorded level.  It is exact (it
        reproduces :func:`src.smooth_dual_ward.exact_rsa_epsilon` to machine
        precision at every level, including under the truncated deflation solve),
        but it is *not* free: the neighbour refresh ``g_K += (q^T M v_K) a`` is
        ``O(deg * d)`` per merge, and on a scale-free graph with a wide target
        (Elliptic++ day 25: a degree-4,960 node, a 440-column target) that alone
        outweighs evaluating a few hundred cuts outright.  So
        :func:`deflated_tree_coarsen` leaves it off and computes ``eps_Pi``
        directly at its cut levels; keep it on when the whole curve is wanted, or
        when ``d`` and the degrees are small.
    curve_levels
        Block counts at which the curve is recorded.  ``None`` records every
        level.  Both ``H`` and ``H_Pi`` are accumulated exactly at every merge
        regardless -- what this controls is the *spectral* work: turning
        ``H -> lambda_max(H)`` costs ``O(d^3)``, and with a width-440 target on a
        17k-node graph that single ``eigh`` per merge is the whole running time.
        The deflated dual-Ward rule never needs the spectrum to choose a merge
        (its score is ``||a||^2``), and neither rule needs the Euclidean spectrum,
        so restricting both to the levels a caller will actually read is exact.
        The minimax rule does need ``H``'s eigenpairs at every step and computes
        them regardless.  The final level is always recorded.
    """

    if rule not in ("dual-ward", "minimax"):
        raise ValueError("rule must be 'dual-ward' or 'minimax'")
    if solve not in ("local", "exact"):
        raise ValueError("solve must be 'local' or 'exact'")
    commit_solve = solve if commit_solve is None else commit_solve
    if commit_solve not in ("local", "exact"):
        raise ValueError("commit_solve must be 'local' or 'exact'")
    if epsilon_key not in ("epsilon_q", "epsilon_pi"):
        raise ValueError("epsilon_key must be 'epsilon_q' or 'epsilon_pi'")

    A_off, d_tilde, _, M = screened_operators(W, tau)
    n = A_off.shape[0]
    U, n_dims = m_orthonormal_basis(Z, M, rank_tol=rank_tol)
    d = int(n_dims)
    if score_mode == "auto":
        score_mode = "exact" if d <= 96 else "secular"
    if score_mode not in ("exact", "secular"):
        raise ValueError("score_mode must be 'auto', 'exact' or 'secular'")

    MU = np.asarray(M @ U)  # rows are m_i = U^T M_tau e_i at singletons

    # ---- per-block state --------------------------------------------------
    max_ids = 2 * n
    vol = np.zeros(max_ids)
    vol[:n] = d_tilde
    ocut = np.zeros(max_ids)
    ocut[:n] = np.asarray(A_off.sum(axis=1)).ravel()
    size = np.zeros(max_ids, dtype=np.int64)
    size[:n] = 1
    active = np.zeros(max_ids, dtype=bool)
    active[:n] = True
    Pm = np.zeros((max_ids, d))  # m_C = U^T M_tau v_C   (deflation)
    Pm[:n] = MU
    Pe = np.zeros((max_ids, d))  # p_C = U^T v_C         (Euclidean tracking)
    Pe[:n] = U
    Ge = np.zeros((max_ids, d))  # g_C = E^T M_tau v_C, E = (I - Pi_P) U

    coo = sp.triu(A_off, k=1).tocoo()
    nbr: List[Dict[int, float]] = [dict() for _ in range(max_ids)]
    for i, j, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if w > 0.0:
            nbr[i][j] = w
            nbr[j][i] = w

    H = np.zeros((d, d))  # harmonic residual H_P^tau
    Hpi = np.zeros((d, d))  # Euclidean residual (realized coarsening)
    # Scratch for the merge updates.  With H = 8 heads x 55 channels the target is
    # 440 columns wide, so every `H + np.outer(a, a)` is a 1.5 MB allocation; at
    # one per merge on a 17k-node graph that allocation churn (macOS returns
    # medium blocks to the kernel with madvise on free) costs more than the
    # arithmetic.  Everything below is written into these buffers instead.
    Hc = np.zeros((d, d))
    Hpic = np.zeros((d, d))
    ob = np.zeros((d, d))
    ob2 = np.zeros((d, d))
    evals = np.zeros(d)
    evecs = np.eye(d)
    lam1 = 0.0
    u1 = evecs[:, -1] if d else np.zeros(0)
    trace_h = 0.0

    def gram_diag(c: int) -> float:
        return (ocut[c] + tau * vol[c]) / vol[c]

    # ---- candidate evaluation (steps 2-6 of the pseudocode) ---------------

    def _ball(a_id: int, b_id: int, radius: int):
        """``hops``-hop coarse ball around the hypothetical merged block.

        Returned as a list of *current* block ids, nearest first.  The merged
        block itself is not in the list -- it is implicit at index 0 of the
        assembled local system.
        """

        order: List[int] = []
        seen = {a_id, b_id}
        level: List[int] = []
        for src in (a_id, b_id):
            for k in nbr[src]:
                if k in seen or not active[k]:
                    continue
                seen.add(k)
                order.append(k)
                level.append(k)
                if len(order) >= max_ball:
                    return order  # early exit: never walks a hub's full adjacency
        for _ in range(max(0, radius - 1)):
            nxt: List[int] = []
            for c in level:
                for k in nbr[c]:
                    if k in seen or not active[k]:
                        continue
                    seen.add(k)
                    order.append(k)
                    nxt.append(k)
                    if len(order) >= max_ball:
                        return order
            if not nxt:
                break
            level = nxt
        return order

    def _all_active_blocks() -> List[int]:
        return [c for c in range(max_ids) if active[c]]

    def deflate(a_id: int, b_id: int, mode: str):
        """Return the full candidate record for merging ``a_id, b_id``."""

        va, vb = vol[a_id], vol[b_id]
        s = va + vb
        alpha = math.sqrt(va / s)
        beta = math.sqrt(vb / s)
        w_ab = nbr[a_id].get(b_id, 0.0)
        s_ab = -w_ab / math.sqrt(va * vb)
        gd_a, gd_b = gram_diag(a_id), gram_diag(b_id)
        # ||g||^2_{M_tau}
        m_tau = beta * beta * gd_a - 2.0 * alpha * beta * s_ab + alpha * alpha * gd_b
        m_tau = max(float(m_tau), _TINY)

        # merged-block statistics of P'
        ocut_ab = ocut[a_id] + ocut[b_id] - 2.0 * w_ab
        gd_ab = (ocut_ab + tau * s) / s

        # b = V_{P'}^T M_tau g : merged block entry (the boundary entries are
        # gathered below, on the ball only -- outside it they are discarded by the
        # Dirichlet truncation anyway, and a hub block can have thousands of
        # neighbours).
        b_ab = (
            alpha * beta * gd_a + (beta * beta - alpha * alpha) * s_ab - alpha * beta * gd_b
        )

        if mode == "exact":
            ball = [c for c in _all_active_blocks() if c not in (a_id, b_id)]
            if len(ball) > 4000:
                raise MemoryError(
                    f"solve='exact' assembles a dense {len(ball) + 1}^2 coarse "
                    "system per candidate; use solve='local' on graphs this size "
                    "(the r-hop truncation is exact to O(q^r) by the exponential "
                    "decay of (L_c + tau I)^{-1}) and certify the returned "
                    "partition with exact_harmonic_H()."
                )
        else:
            ball = _ball(a_id, b_id, hops)

        nbr_a, nbr_b = nbr[a_id], nbr[b_id]
        b_nb: Dict[int, float] = {}
        nb_ab: Dict[int, float] = {}
        for k in ball:
            wa = nbr_a.get(k, 0.0)
            wb = nbr_b.get(k, 0.0)
            if wa or wb:
                nb_ab[k] = wa + wb
                b_nb[k] = beta * (-wa / math.sqrt(va * vol[k])) - alpha * (
                    -wb / math.sqrt(vb * vol[k])
                )

        L = len(ball) + 1
        A_loc = np.zeros((L, L))
        b_loc = np.zeros(L)
        A_loc[0, 0] = gd_ab
        b_loc[0] = b_ab
        # Off-diagonals are read by *pair lookup* over the ball rather than by
        # iterating each block's adjacency: a hub block can have thousands of
        # neighbours while the ball is capped at ``max_ball``, so scanning
        # ``nbr[c]`` would dominate the whole coarsening.
        for p, c in enumerate(ball, start=1):
            A_loc[p, p] = gram_diag(c)
            b_loc[p] = b_nb.get(c, 0.0)
            w = nb_ab.get(c)
            if w:
                val = -w / math.sqrt(s * vol[c])
                A_loc[0, p] = val
                A_loc[p, 0] = val
            nbr_c = nbr[c]
            vol_c = vol[c]
            for q in range(p + 1, L):
                k = ball[q - 1]
                wk = nbr_c.get(k)
                if wk:
                    val = -wk / math.sqrt(vol_c * vol[k])
                    A_loc[p, q] = val
                    A_loc[q, p] = val
        c_loc = np.linalg.solve(A_loc, b_loc)

        bc = float(b_loc @ c_loc)
        # bc = b^T A^{-1} b is the squared leakage energy, so it lies in
        # [0, m_tau).  Restricting the solve to the ball can only *lower* it
        # (Dirichlet truncation shrinks the trial space of the same variational
        # problem), so the local mode under-estimates the leakage and never
        # produces a negative ||g~||^2; the clamp is purely defensive.
        bc = min(max(bc, 0.0), m_tau * (1.0 - 1e-14))
        gt_sq = max(m_tau - bc, _TINY)
        eta0 = math.sqrt(max(bc / m_tau, 0.0))

        m_ab = alpha * Pm[a_id] + beta * Pm[b_id]  # m_{A u B}
        umg = beta * Pm[a_id] - alpha * Pm[b_id]  # U^T M_tau g
        Vc = c_loc[0] * m_ab
        if ball:
            Vc = Vc + c_loc[1:] @ Pm[ball]
        a_vec = (umg - Vc) / math.sqrt(gt_sq)

        return {
            "a": a_vec,
            "alpha": alpha,
            "beta": beta,
            "m_tau": m_tau,
            "w_ab": w_ab,
            "s_ab": s_ab,
            "gd_a": gd_a,
            "gd_b": gd_b,
            "eta0": eta0,
            # Lemma "leakage is local and computable": eta_0 <= ||b|| / (sqrt(tau)
            # ||g||_{M_tau}), available without the solve (uses A_{P'} >= tau I).
            "eta0_bound": float(np.linalg.norm(b_loc)) / math.sqrt(tau * m_tau),
            "ball": len(ball),
        }

    def raw_keys(c_id: int, ks: np.ndarray, ws: np.ndarray) -> np.ndarray:
        r"""``O(d)`` insertion keys for all neighbours ``ks`` of ``c_id`` at once.

        The key is the *raw* normalized dual score
        ``||U^T M_tau g||^2 / ||g||^2_{M_tau}`` -- the deflated score with the
        deflation left out.  The two coincide to ``O(eta_0)`` (the leakage is a
        boundary quantity, small whenever the pair has a weak external boundary
        relative to ``tau``), which makes this an excellent *hint* but not a
        decision rule: every popped entry is rescored with the real deflated
        score before it can be chosen.  Using it at insertion is what keeps the
        queue near-linear -- a merge pushes one entry per neighbour of the new
        block, and paying a local solve for each of those would dominate the
        whole coarsening on a heavy-tailed graph.  Vectorized for the same
        reason: Elliptic++ day 25 has a degree-4,960 node.
        """

        va = vol[c_id]
        vb = vol[ks]
        s = va + vb
        alpha = np.sqrt(va / s)
        beta = np.sqrt(vb / s)
        s_ab = -ws / np.sqrt(va * vb)
        gd_a = (ocut[c_id] + tau * va) / va
        gd_b = (ocut[ks] + tau * vb) / vb
        m_tau = beta * beta * gd_a - 2.0 * alpha * beta * s_ab + alpha * alpha * gd_b
        umg = beta[:, None] * Pm[c_id] - alpha[:, None] * Pm[ks]
        return np.einsum("ij,ij->i", umg, umg) / np.maximum(m_tau, _TINY)

    def score_of(a_vec: np.ndarray) -> float:
        if rule == "dual-ward":
            return float(a_vec @ a_vec)
        if score_mode == "secular":
            return rank_one_lambda_max(evals, evecs.T @ a_vec)
        cand = H + np.outer(a_vec, a_vec)
        return max(float(np.linalg.eigvalsh(0.5 * (cand + cand.T))[-1]), 0.0)

    # ---- Euclidean (realized) rank-two tracking ---------------------------

    def euclid_pair(a_id: int, b_id: int, alpha: float, beta: float):
        ae = beta * Pe[a_id] - alpha * Pe[b_id]
        ce = beta * Ge[a_id] - alpha * Ge[b_id]
        return ae, ce

    # ---- priority queue ---------------------------------------------------

    keep = coo.data > 0.0
    rows = coo.row[keep].astype(np.int64)
    cols = coo.col[keep].astype(np.int64)
    vals = coo.data[keep].astype(np.float64)
    va0, vb0 = d_tilde[rows], d_tilde[cols]
    s0 = va0 + vb0
    al0, be0 = np.sqrt(va0 / s0), np.sqrt(vb0 / s0)
    sab0 = -vals / np.sqrt(va0 * vb0)
    gd0 = (ocut[:n] + tau * d_tilde) / d_tilde
    mt0 = be0**2 * gd0[rows] - 2.0 * al0 * be0 * sab0 + al0**2 * gd0[cols]
    umg0 = be0[:, None] * MU[rows] - al0[:, None] * MU[cols]
    keys0 = np.einsum("ij,ij->i", umg0, umg0) / np.maximum(mt0, _TINY)
    heap: List[tuple] = list(zip(keys0.tolist(), rows.tolist(), cols.tolist()))
    heapq.heapify(heap)

    n_active = n
    children: List[tuple] = []
    records: List[dict] = []
    curve: List[dict] = []
    next_id = n
    cap = int(max_cluster_size) if max_cluster_size else 0
    target = int(n_clusters) if n_clusters is not None else 1

    def stop_now() -> bool:
        if build_full_tree:
            return False
        return n_clusters is not None and n_active <= target

    while n_active > 1 and not stop_now():
        chosen = None
        if strict:
            best = None
            seen_pairs = set()
            for _sc, a_id, b_id in heap:
                if not (active[a_id] and active[b_id]):
                    continue
                if (a_id, b_id) in seen_pairs or nbr[a_id].get(b_id, 0.0) <= 0.0:
                    continue
                seen_pairs.add((a_id, b_id))
                if cap and size[a_id] + size[b_id] > cap:
                    continue
                rec = deflate(a_id, b_id, solve)
                sc = score_of(rec["a"])
                if best is None or sc < best[0]:
                    best = (sc, a_id, b_id, rec)
            chosen = best
        else:
            # Lazy greedy.  For the minimax rule this is *exact* whenever the
            # scores are non-decreasing, which the PSD rank-one update
            # guarantees for the H-part (stale scores are valid lower bounds);
            # the deflation part moves only inside the r-hop ball of the last
            # merge, by O(q^r).
            batch: List[tuple] = []
            best = None
            refreshed = 0
            while heap and refreshed < max_rescore:
                _sc_old, a_id, b_id = heapq.heappop(heap)
                if not (active[a_id] and active[b_id]):
                    continue
                if nbr[a_id].get(b_id, 0.0) <= 0.0:
                    continue
                if cap and size[a_id] + size[b_id] > cap:
                    continue
                refreshed += 1
                rec = deflate(a_id, b_id, solve)
                sc = score_of(rec["a"])
                if best is None or sc < best[0]:
                    if best is not None:
                        batch.append((best[0], best[1], best[2]))
                    best = (sc, a_id, b_id, rec)
                else:
                    batch.append((sc, a_id, b_id))
                if heap and best[0] <= heap[0][0] + 1e-12:
                    break
            for entry in batch:
                heapq.heappush(heap, entry)
            chosen = best
        if chosen is None:
            break

        sc, a_id, b_id, rec = chosen
        if commit_solve != solve:
            rec = deflate(a_id, b_id, commit_solve)
        a_vec = rec["a"]
        alpha, beta = rec["alpha"], rec["beta"]
        w_ab, s_ab = rec["w_ab"], rec["s_ab"]
        m_tau = rec["m_tau"]

        level_next = n_active - 1
        want_level = curve_levels is None or level_next in curve_levels or level_next <= 1
        need_spectrum = (
            rule == "minimax" or epsilon_max is not None or (record_curve and want_level)
        )

        # H_{P'} = H_P + a a^T : exactly symmetric, so no re-symmetrization
        np.multiply.outer(a_vec, a_vec, out=ob)
        np.add(H, ob, out=Hc)
        if need_spectrum:
            cand_evals, cand_evecs = np.linalg.eigh(Hc)
            cand_lam1 = max(float(cand_evals[-1]), 0.0)
            cand_eps_q = math.sqrt(cand_lam1)
        else:
            cand_evals, cand_evecs, cand_lam1 = evals, evecs, lam1
            cand_eps_q = float("nan")

        cand_eps_pi = float("nan")
        ae = ce = None
        if track_euclidean:
            ae, ce = euclid_pair(a_id, b_id, alpha, beta)
            # Pi_{P'} = Pi_P - q q^T gives the rank-TWO indefinite update
            # H_Pi' = H_Pi + c a^T + a c^T + m_tau a a^T.  Built symmetrically by
            # construction (ob + ob^T) rather than symmetrized afterwards.
            np.multiply.outer(ce, ae, out=ob)
            np.add(ob, ob.T, out=ob2)
            np.add(Hpi, ob2, out=Hpic)
            np.multiply.outer(ae, ae, out=ob)
            ob *= m_tau
            Hpic += ob
            if need_spectrum or (record_curve and want_level):
                cand_eps_pi = math.sqrt(
                    max(float(np.linalg.eigvalsh(Hpic)[-1]), 0.0)
                )

        if epsilon_max is not None:
            budget_val = cand_eps_q if epsilon_key == "epsilon_q" else cand_eps_pi
            if budget_val > epsilon_max:
                continue  # refuse this merge, try the next candidate

        # ---- commit --------------------------------------------------------
        new = next_id
        next_id += 1
        va, vb = vol[a_id], vol[b_id]
        vol[new] = va + vb
        ocut[new] = ocut[a_id] + ocut[b_id] - 2.0 * w_ab
        size[new] = size[a_id] + size[b_id]
        Pm[new] = alpha * Pm[a_id] + beta * Pm[b_id]
        Pe[new] = alpha * Pe[a_id] + beta * Pe[b_id]

        if track_euclidean:
            q_m_new = (
                alpha * beta * rec["gd_a"]
                + (beta * beta - alpha * alpha) * s_ab
                - alpha * beta * rec["gd_b"]
            )
            Ge[new] = alpha * Ge[a_id] + beta * Ge[b_id] + q_m_new * ae
            touched: Dict[int, float] = {}
            for k, w in nbr[a_id].items():
                if k != b_id and active[k]:
                    touched[k] = touched.get(k, 0.0) + beta * (
                        -w / math.sqrt(va * vol[k])
                    )
            for k, w in nbr[b_id].items():
                if k != a_id and active[k]:
                    touched[k] = touched.get(k, 0.0) - alpha * (
                        -w / math.sqrt(vb * vol[k])
                    )
            if touched:
                # one BLAS rank-one update instead of a Python loop of AXPYs:
                # a grown block can have thousands of neighbours, and this loop
                # runs once per merge.
                ks = np.fromiter(touched.keys(), dtype=np.int64, count=len(touched))
                cs = np.fromiter(touched.values(), dtype=np.float64, count=len(touched))
                Ge[ks] += cs[:, None] * ae

        merged: Dict[int, float] = {}
        for k, w in nbr[a_id].items():
            if k != b_id and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k, w in nbr[b_id].items():
            if k != a_id and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k in nbr[a_id]:
            nbr[k].pop(a_id, None)
        for k in nbr[b_id]:
            nbr[k].pop(b_id, None)
        nbr[a_id] = {}
        nbr[b_id] = {}
        for k, w in merged.items():
            if w > 0.0:
                nbr[new][k] = w
                nbr[k][new] = w

        active[a_id] = active[b_id] = False
        active[new] = True

        H, Hc = Hc, H  # commit by swapping the buffers; no copy, no allocation
        if need_spectrum:
            evals, evecs = cand_evals, cand_evecs
            lam1 = cand_lam1
            u1 = evecs[:, -1] if d else u1
        trace_h += float(a_vec @ a_vec)
        if track_euclidean:
            Hpi, Hpic = Hpic, Hpi

        n_active -= 1
        children.append((a_id, b_id))
        records.append(
            {
                "children": (int(a_id), int(b_id)),
                "new_id": int(new),
                "n_clusters": int(n_active),
                "score": float(sc),
                "a_sq": float(a_vec @ a_vec),
                "lambda_max": float(cand_lam1) if need_spectrum else float("nan"),
                "epsilon_q": float(cand_eps_q),
                "epsilon_pi": float(cand_eps_pi),
                "trace_h": float(trace_h),
                "leakage": float(rec["eta0"]),
                "leakage_bound": float(rec["eta0_bound"]),
                "m_tau": float(m_tau),
                "cut": float(w_ab),
                "volume": float(vol[new]),
                "ball": int(rec["ball"]),
            }
        )
        if record_curve and want_level:
            curve.append(
                {
                    "n_clusters": int(n_active),
                    "epsilon_q": float(cand_eps_q),
                    "epsilon_pi": float(cand_eps_pi),
                    "trace_h": float(trace_h),
                    "leakage": float(rec["eta0"]),
                    "score": float(sc),
                }
            )

        if nbr[new]:
            ks = np.fromiter(nbr[new].keys(), dtype=np.int64, count=len(nbr[new]))
            ws = np.fromiter(nbr[new].values(), dtype=np.float64, count=len(nbr[new]))
            keys = raw_keys(new, ks, ws)
            if fanout > 0 and keys.size > fanout:
                sel = np.argpartition(keys, fanout)[:fanout]
                keys, ks = keys[sel], ks[sel]
            for key, k in zip(keys.tolist(), ks.tolist()):
                lo, hi = (k, new) if k < new else (new, k)
                heapq.heappush(heap, (key, lo, hi))
        # NOTE: entries for pairs whose deflation moved (inside the r-hop ball of
        # this merge) keep their stale key.  That is safe: every popped entry is
        # rescored exactly against the current state, and the PSD rank-one update
        # makes stale minimax scores valid lower bounds forever.

    # The stopping level is not known until the loop exits, so if the spectra
    # were skipped there, compute them once now: every consumer reads the final
    # level.
    if children and (not curve or curve[-1]["n_clusters"] != n_active):
        evals, evecs = np.linalg.eigh(H)
        lam1 = max(float(evals[-1]), 0.0)
        u1 = evecs[:, -1] if d else u1
        eps_pi_final = (
            math.sqrt(max(float(np.linalg.eigvalsh(Hpi)[-1]), 0.0))
            if track_euclidean
            else float("nan")
        )
        if record_curve:
            curve.append(
                {
                    "n_clusters": int(n_active),
                    "epsilon_q": math.sqrt(lam1),
                    "epsilon_pi": eps_pi_final,
                    "trace_h": float(trace_h),
                    "leakage": float(records[-1]["leakage"]) if records else 0.0,
                    "score": float(records[-1]["score"]) if records else 0.0,
                }
            )

    children_arr = (
        np.asarray(children, dtype=np.int64)
        if children
        else np.empty((0, 2), dtype=np.int64)
    )
    result = DeflatedCoarseningResult(
        labels_=np.zeros(n, dtype=np.int64),
        children_=children_arr,
        merge_records_=records,
        H_=H,
        epsilon_q_=math.sqrt(max(lam1, 0.0)),
        epsilon_pi_=float("nan"),
        trace_h_=trace_h,
        rule_=rule,
        n_effective_target_dims_=d,
        target_basis_=U,
        n_leaves_=n,
        curve_=curve,
    )
    if n_clusters is not None and build_full_tree:
        result.labels_ = result.labels_at(int(n_clusters))
    else:
        result.labels_ = result._cut(len(children))
    result.n_clusters_ = int(result.labels_.max()) + 1 if n else 0
    if result.n_clusters_ >= n:
        result.epsilon_q_ = 0.0
        result.epsilon_pi_ = 0.0
        result.trace_h_ = 0.0
    else:
        if record_curve:
            try:
                lvl = result._level(result.n_clusters_)
                result.epsilon_q_ = float(lvl["epsilon_q"])
                result.epsilon_pi_ = float(lvl["epsilon_pi"])
                result.trace_h_ = float(lvl["trace_h"])
            except ValueError:
                pass
        if not np.isfinite(result.epsilon_pi_):
            result.epsilon_pi_ = float(
                exact_rsa_epsilon(U, M, d_tilde, result.labels_)
            )
    if compute_mu and n:
        result.mu_ = block_distortion_mu(M, d_tilde, result.labels_)
    return result


# ---------------------------------------------------------------------------
# pipeline driver: build the tree once, cut it fine -> coarse
# ---------------------------------------------------------------------------


def deflated_tree_coarsen(
    adjacency,
    basis,
    train_patterns: list,
    node_labels,
    *,
    tau: float,
    rule: str = "dual-ward",
    laplacian: str = "symmetric",
    threshold: float = 0.51,
    stop: str = "epsilon",
    epsilon_budget: float = float("inf"),
    epsilon_key: str = "epsilon_pi",
    num_cuts: int = 200,
    hops: int = 2,
    max_ball: int = 32,
    max_rescore: int = 8,
    fanout: int = 32,
    max_cluster_size: int = 0,
    certify: bool = True,
) -> tuple:
    """Deflated agglomeration over ``R = span(basis)``, cut fine->coarse.

    Drop-in replacement for
    :func:`src.run_collective_bank_detection.ward_tree_coarsen`: the whole merge
    hierarchy is built once and then walked from the finest combination toward
    the root, recording at each cut the *two* RSA constants (the harmonic
    ``epsilon_q`` the algorithm optimizes and the realized Euclidean
    ``epsilon_pi`` the rest of the pipeline reports, both exact and both free
    from the incremental trackers) and the mean training F1.  Stop rules match
    ``ward_tree_coarsen``:

    * ``stop="epsilon"`` -- coarsest cut whose ``epsilon_key`` is within budget;
    * ``stop="f1"``      -- cut with the best mean *training* F1.

    ``certify=True`` recomputes ``eps_Q`` at the chosen cut by the brute-force
    ``H = I - W_P A_P^{-1} W_P^T`` and the block-distortion factor ``mu_P^tau``,
    so the returned trajectory carries an exact, a-posteriori certified interval
    ``eps_Pi in [eps_Q, mu eps_Q]`` (RSA sandwich) alongside the incremental
    numbers.  The difference between the certified and incremental ``eps_Q`` is
    the accumulated ``O(q^r)`` truncation error of the local deflation solves and
    is reported as ``eps_q_drift``.
    """

    import torch
    from scipy.sparse import coo_matrix

    from loukas_sgc_detection import (
        LoukasCoarseningResult,
        evaluate_loukas_patterns,
    )
    from src.utils.utils import LOGGER

    if laplacian not in ("symmetric", "normalized", "sym", "norm"):
        raise NotImplementedError(
            "deflated coarsening is symmetric-only: the harmonic projector, the "
            "coarse system L_c + tau I and the block Gram closed form are all "
            "stated in M_tau = L_sym + tau I.  Use coarsening_laplacian='symmetric'."
        )

    adjacency = adjacency.coalesce()
    n = int(adjacency.shape[0])
    idx = adjacency.indices().cpu().numpy()
    val = adjacency.values().cpu().numpy().astype(np.float64)
    W = coo_matrix((val, (idx[0], idx[1])), shape=(n, n)).tocsr()
    Z = basis.detach().cpu().to(torch.float64).numpy()

    # The cut schedule is known up front, so the O(d^3) spectral work is done at
    # those levels only (see ``curve_levels``): with H = 8 heads x 55 channels the
    # target is 440 columns wide, and one eigh per merge would otherwise be ~90%
    # of the running time on a 17k-node graph.
    ks = np.unique(
        np.round(np.geomspace(2, max(n - 1, 2), max(num_cuts, 2))).astype(int)
    )
    ks = ks[(ks >= 2) & (ks <= n - 1)][::-1]
    res = deflated_coarsen(
        W,
        Z,
        float(tau),
        rule=rule,
        build_full_tree=True,
        curve_levels=set(ks.tolist()),
        solve="local",
        hops=hops,
        max_ball=max_ball,
        max_rescore=max_rescore,
        fanout=fanout,
        max_cluster_size=max_cluster_size,
        # eps_Pi is computed below, directly at the cut levels: the incremental
        # rank-two tracker is exact but costs three d x d outer products plus an
        # O(deg * d) neighbour refresh per merge, which is more than evaluating a
        # few hundred cuts outright.
        track_euclidean=False,
        record_curve=True,
    )

    A_off, d_tilde, _, M = screened_operators(W, float(tau))
    U, _ = m_orthonormal_basis(Z, M)
    by_level = {e["n_clusters"]: e for e in res.curve_}

    trajectory: list = []
    best = None
    cuts = res.labels_at_many(ks.tolist())
    for k in ks.tolist():
        # On a disconnected graph the agglomeration runs out of adjacent pairs
        # before k blocks are reached, so labels_at_many has no entry for that k
        # (its key is the coarsest reachable count, n - m).  Skip those levels.
        labels = cuts.get(int(k))
        if labels is None:
            continue
        n_coarse = int(labels.max()) + 1
        lvl = by_level.get(n_coarse)
        if lvl is None:  # k below the number of connected components
            continue
        n2s = torch.from_numpy(labels).to(node_labels.device)
        results, by_label = evaluate_loukas_patterns(
            train_patterns, n2s, node_labels, threshold=threshold
        )
        f1 = float(np.mean([r.f1 for r in results])) if results else 0.0
        alert = by_label.get("alert", {})
        entry = {
            "n_coarse": n_coarse,
            # the realized Euclidean RSA constant -- the same quantity, computed
            # the same way, that ward_tree_coarsen reports, so budgets and tables
            # stay comparable across coarseners
            "epsilon": float(exact_rsa_epsilon(U, M, d_tilde, labels)),
            "epsilon_q": float(lvl["epsilon_q"]),
            "trace_h": float(lvl["trace_h"]),
            "leakage": float(lvl["leakage"]),
            "train_f1": f1,
            "recall": float(alert.get("mean_recall", 0.0) or 0.0),
            "precision": float(alert.get("mean_precision", 0.0) or 0.0),
            "labels": labels,
        }
        trajectory.append(entry)
        if stop == "epsilon":
            key = "epsilon_q" if epsilon_key == "epsilon_q" else "epsilon"
            if entry[key] <= epsilon_budget:
                best = entry
            elif epsilon_key == "epsilon_q":
                break  # eps_Q is monotone: coarser cuts can only be worse
        else:
            if best is None or f1 > best["train_f1"]:
                best = entry

    if best is None:
        if trajectory:
            LOGGER.warning(
                f"  deflated-{rule}: the FINEST cut "
                f"(n_coarse={trajectory[0]['n_coarse']:,}) already has "
                f"{epsilon_key}={trajectory[0]['epsilon_q' if epsilon_key == 'epsilon_q' else 'epsilon']:.4g}"
                f" > budget {epsilon_budget:g}; returning it near-uncoarsened."
            )
        best = trajectory[0] if trajectory else None
    if best is None:
        raise ValueError("deflated tree produced no valid cut")

    labels = best["labels"]
    if certify:
        eps_q_exact = harmonic_rsa_epsilon(U, M, d_tilde, labels)
        mu = block_distortion_mu(M, d_tilde, labels)
        best["epsilon_q_exact"] = float(eps_q_exact)
        best["eps_q_drift"] = float(abs(eps_q_exact - best["epsilon_q"]))
        best["mu"] = float(mu)
        best["sandwich_upper"] = float(mu * eps_q_exact)
        best["sandwich_ok"] = bool(
            eps_q_exact <= best["epsilon"] + 1e-8
            and best["epsilon"] <= mu * eps_q_exact + 1e-8
        )

    result = LoukasCoarseningResult(
        node_to_supernode=torch.from_numpy(labels).to(node_labels.device),
        n_original=n,
        n_coarse=best["n_coarse"],
        epsilon=best["epsilon"],
        epsilon_bound=float("nan"),
        sigmas=[],
        sizes=[n, best["n_coarse"]],
    )
    # The certificate does not fit the shared LoukasCoarseningResult schema (every
    # other coarsener reports one epsilon), so it rides along as an attribute and
    # is also kept on the chosen trajectory entry.
    result.deflated_certificate = {
        "rule": rule,
        "n_coarse": int(best["n_coarse"]),
        "epsilon_pi": float(best["epsilon"]),
        "epsilon_q": float(best["epsilon_q"]),
        "trace_h": float(best["trace_h"]),
        "leakage_last": float(best["leakage"]),
        "leakage_mean": float(
            np.mean([r["leakage"] for r in res.merge_records_])
            if res.merge_records_
            else float("nan")
        ),
        # How much of the hierarchy the target subspace actually *sees*: a merge
        # with ||a||^2 ~ 0 eliminates a direction the target is blind to, so its
        # position in the merge order is decided by tie-breaking, not by the
        # score.  A narrow target on a large graph leaves most of the graph in
        # that regime -- worth reporting next to the certificate.
        "a_sq_zero_frac": float(
            np.mean([r["a_sq"] < 1e-12 for r in res.merge_records_])
            if res.merge_records_
            else float("nan")
        ),
        "a_sq_median": float(
            np.median([r["a_sq"] for r in res.merge_records_])
            if res.merge_records_
            else float("nan")
        ),
        "a_sq_p99": float(
            np.percentile([r["a_sq"] for r in res.merge_records_], 99)
            if res.merge_records_
            else float("nan")
        ),
        "epsilon_q_exact": best.get("epsilon_q_exact", float("nan")),
        "eps_q_drift": best.get("eps_q_drift", float("nan")),
        "mu": best.get("mu", float("nan")),
        "sandwich_upper": best.get("sandwich_upper", float("nan")),
        "sandwich_ok": best.get("sandwich_ok", None),
    }
    return result, trajectory

# ---------------------------------------------------------------------------
# validation: every identity of the section, to machine precision
# ---------------------------------------------------------------------------


def _random_connected_graph(rng, n: int):
    Wd = np.zeros((n, n))
    for i in range(n - 1):
        Wd[i, i + 1] = rng.uniform(0.5, 2.0)
    extra = np.triu(rng.random((n, n)) < 0.15, 1)
    Wd = np.where(extra, rng.uniform(0.5, 2.0, size=(n, n)), Wd)
    Wd = np.triu(Wd, 1)
    for i in range(n - 1):
        Wd[i, i + 1] = max(Wd[i, i + 1], 0.5)
    return sp.csr_matrix(Wd + Wd.T)


def run_validation_suite(seed: int = 0, verbose: bool = True) -> None:
    """Check every exact identity of the deflated-dual section numerically."""

    rng = np.random.default_rng(seed)
    n, tau = 40, 0.4
    W = _random_connected_graph(rng, n)
    Z = rng.normal(size=(n, 4))

    A_off, d_tilde, _, M = screened_operators(W, tau)
    U, d = m_orthonormal_basis(Z, M)
    assert np.allclose(U.T @ (M @ U), np.eye(d), atol=1e-10), "U is not M-orthonormal"

    def labels_of(children, t):
        parent = list(range(n + len(children)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for s in range(t):
            a, b = children[s]
            parent[find(int(a))] = n + s
            parent[find(int(b))] = n + s
        seen: Dict[int, int] = {}
        out = np.empty(n, dtype=np.int64)
        for leaf in range(n):
            out[leaf] = seen.setdefault(find(leaf), len(seen))
        return out

    for rule in ("dual-ward", "minimax"):
        res = deflated_coarsen(
            W,
            Z,
            tau,
            rule=rule,
            n_clusters=6,
            build_full_tree=True,
            solve="exact",
            compute_mu=True,
        )
        children = res.children_.tolist()

        # (1) incremental H equals brute-force harmonic H at every level
        worst_H = 0.0
        worst_eps = 0.0
        worst_pi = 0.0
        for t in range(1, len(children) + 1):
            lab = labels_of(children, t)
            Hb = exact_harmonic_H(U, M, d_tilde, lab)
            k = int(lab.max()) + 1
            entry = res._level(k)
            worst_eps = max(
                worst_eps,
                abs(entry["epsilon_q"] - math.sqrt(max(np.linalg.eigvalsh(Hb)[-1], 0))),
            )
            worst_H = max(worst_H, abs(np.trace(Hb) - entry["trace_h"]))
            worst_pi = max(
                worst_pi,
                abs(entry["epsilon_pi"] - exact_rsa_epsilon(U, M, d_tilde, lab)),
            )
        assert worst_eps < 1e-8, f"[{rule}] eps_Q != brute force ({worst_eps:.2e})"
        # (2) telescoping trace: tr H = sum_t ||a_t||^2 (path independent)
        assert worst_H < 1e-8, f"[{rule}] trace telescoping broken ({worst_H:.2e})"
        # (3) the Euclidean tracker equals exact_rsa_epsilon at every level
        assert worst_pi < 1e-8, f"[{rule}] eps_Pi != exact_rsa_epsilon ({worst_pi:.2e})"
        cum = np.cumsum([r["a_sq"] for r in res.merge_records_])
        assert np.allclose(
            cum, [r["trace_h"] for r in res.merge_records_], atol=1e-10
        ), f"[{rule}] cumulative ||a||^2 != trace H"

        # (4) monotonicity of trace and lambda_max along the hierarchy
        eq = [e["epsilon_q"] for e in res.curve_]
        tr = [e["trace_h"] for e in res.curve_]
        assert all(
            eq[i] >= eq[i - 1] - 1e-12 for i in range(1, len(eq))
        ), f"[{rule}] eps_Q is not monotone"
        assert all(
            tr[i] >= tr[i - 1] - 1e-12 for i in range(1, len(tr))
        ), f"[{rule}] trace H is not monotone"
        # tr H / d <= eps_Q^2 <= tr H
        for e in res.curve_:
            assert (
                e["trace_h"] / d - 1e-10 <= e["epsilon_q"] ** 2 <= e["trace_h"] + 1e-10
            ), f"[{rule}] trace/eps bracket violated"

        # (4b) the computable leakage bound holds at every committed merge
        assert all(
            r["leakage"] <= r["leakage_bound"] + 1e-9 for r in res.merge_records_
        ), f"[{rule}] eta_0 exceeds its ||b||/sqrt(tau ||g||^2) bound"

        # (5) capacity: |P| < d  =>  eps_Q >= 1
        for e in res.curve_:
            if e["n_clusters"] < d:
                assert (
                    e["epsilon_q"] >= 1.0 - 1e-8
                ), f"[{rule}] capacity obstruction violated at k={e['n_clusters']}"

        # (6) RSA sandwich eps_Q <= eps_Pi <= mu eps_Q at the returned partition
        mu = res.mu_
        assert (
            res.epsilon_q_ <= res.epsilon_pi_ + 1e-9
        ), f"[{rule}] sandwich lower bound violated"
        assert (
            res.epsilon_pi_ <= mu * res.epsilon_q_ + 1e-9
        ), f"[{rule}] sandwich upper bound violated"
        mu_bound = math.sqrt((float(np.linalg.eigvalsh(M.toarray())[-1])) / tau)
        assert 1.0 - 1e-9 <= mu <= mu_bound + 1e-6, f"[{rule}] mu out of range: {mu}"

        # (7) every block is connected
        for cid in range(int(res.labels_.max()) + 1):
            members = np.where(res.labels_ == cid)[0]
            sub = A_off[members][:, members]
            assert (
                sp.csgraph.connected_components(sub, directed=False)[0] == 1
            ), f"[{rule}] cluster {cid} disconnected"

        if verbose:
            print(
                f"  rule={rule:<10} eps_Q={res.epsilon_q_:.6f} <= "
                f"eps_Pi={res.epsilon_pi_:.6f} <= mu*eps_Q="
                f"{mu * res.epsilon_q_:.6f}   (mu={mu:.4f}, tr H={res.trace_h_:.6f})"
            )

    # ---- deflation identity, leakage formula, Weyl bracket, secular solve ---
    res = deflated_coarsen(W, Z, tau, rule="minimax", n_clusters=12, solve="exact")
    lab = res.labels_
    k = int(lab.max()) + 1
    V, A_P = _block_matrices(M, d_tilde, lab)
    A_P = A_P.toarray()  # tiny test graph: dense is fine and clearer
    Vd = V.toarray()
    Md = M.toarray()
    Q_P = Vd @ np.linalg.solve(A_P, Vd.T @ Md)

    # merge two adjacent blocks of this partition by hand and check
    # Q_P = Q_P' + g^ g^T M  and  the leakage formula
    coarse = np.zeros((k, k))
    for i, j in zip(*sp.triu(A_off, 1).nonzero()):
        if lab[i] != lab[j]:
            coarse[lab[i], lab[j]] = coarse[lab[j], lab[i]] = 1.0
    pair = None
    for a in range(k):
        for b in range(a + 1, k):
            if coarse[a, b]:
                pair = (a, b)
                break
        if pair:
            break
    a_b, b_b = pair
    volc = np.bincount(lab, weights=d_tilde, minlength=k)
    s = volc[a_b] + volc[b_b]
    alpha, beta = math.sqrt(volc[a_b] / s), math.sqrt(volc[b_b] / s)
    g = beta * Vd[:, a_b] - alpha * Vd[:, b_b]
    lab2 = lab.copy()
    lab2[lab2 == b_b] = a_b
    _, lab2 = np.unique(lab2, return_inverse=True)
    V2, A_P2 = _block_matrices(M, d_tilde, lab2)
    A_P2 = A_P2.toarray()
    V2d = V2.toarray()
    Q_P2 = V2d @ np.linalg.solve(A_P2, V2d.T @ Md)
    g_t = g - Q_P2 @ g
    g_hat = g_t / math.sqrt(float(g_t @ (Md @ g_t)))
    err = np.abs(Q_P - (Q_P2 + np.outer(g_hat, g_hat) @ Md)).max()
    assert err < 1e-8, f"deflation identity violated: {err:.2e}"

    b_vec = V2d.T @ (Md @ g)
    c_vec = np.linalg.solve(A_P2, b_vec)
    g_m2 = float(g @ (Md @ g))
    eta0_sq = float(b_vec @ c_vec) / g_m2
    eta0_direct = float(Q_P2 @ g @ (Md @ (Q_P2 @ g))) / g_m2
    assert (
        abs(eta0_sq - eta0_direct) < 1e-10
    ), f"leakage formula mismatch: {eta0_sq} vs {eta0_direct}"
    assert eta0_sq <= (float(np.linalg.norm(b_vec)) ** 2) / (tau * g_m2) + 1e-10, (
        "leakage upper bound violated"
    )
    # b is a pure boundary quantity: V'^T M g == V'^T L g
    Ld = Md - tau * np.eye(n)
    assert np.abs(b_vec - V2d.T @ (Ld @ g)).max() < 1e-10, "b is not V'^T L g"

    # rank-one spectral tools
    H0 = exact_harmonic_H(U, M, d_tilde, lab2)
    ev, evec = np.linalg.eigh(H0)
    a_vec = U.T @ (Md @ g_hat)
    lam_exact = float(np.linalg.eigvalsh(H0 + np.outer(a_vec, a_vec))[-1])
    lam_sec = rank_one_lambda_max(ev, evec.T @ a_vec)
    assert (
        abs(lam_exact - lam_sec) < 1e-9
    ), f"secular equation != eigvalsh: {lam_sec} vs {lam_exact}"
    lo, hi = weyl_bracket(float(ev[-1]), evec[:, -1], a_vec)
    assert lo - 1e-10 <= lam_exact <= hi + 1e-10, "Weyl bracket violated"
    # the merged partition's H must equal H_fine + a a^T (rank-one merge calculus)
    H_fine = exact_harmonic_H(U, M, d_tilde, lab)
    assert (
        np.abs(H0 - (H_fine + np.outer(a_vec, a_vec))).max() < 1e-8
    ), "rank-one merge calculus violated"

    # The Euclidean tracker is independent of the deflation solve, so eps_Pi --
    # the constant the rest of the pipeline stops on and reports -- stays EXACT
    # even in the truncated local mode.  (eps_Q does not: it accumulates the
    # O(q^r) truncation error, which is why the tree driver re-certifies it.)
    loc = deflated_coarsen(
        W, Z, tau, rule="minimax", n_clusters=9, build_full_tree=True, solve="local"
    )
    for k in (9, 15, 20):
        assert (
            abs(
                loc.epsilon_pi_at(k)
                - exact_rsa_epsilon(U, M, d_tilde, loc.labels_at(k))
            )
            < 1e-8
        ), f"eps_Pi drifted under the local solve at k={k}"

    # local vs exact deflation solve: the r-hop truncation must be accurate
    ex = deflated_coarsen(W, Z, tau, rule="dual-ward", n_clusters=8, solve="exact")
    for hops_r in (1, 2, 3):
        lc = deflated_coarsen(
            W, Z, tau, rule="dual-ward", n_clusters=8, solve="local", hops=hops_r
        )
        if verbose:
            print(
                f"  local solve hops={hops_r}: eps_Q={lc.epsilon_q_:.6f} "
                f"(exact {ex.epsilon_q_:.6f})"
            )

    if verbose:
        print("  deflation identity, leakage formula, secular + Weyl: OK")
        print("deflated_coarsen validation suite: all checks passed")


if __name__ == "__main__":  # pragma: no cover
    run_validation_suite()
