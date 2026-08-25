"""The screened geometry ``M_tau`` the whole detector is measured in.

Everything in the method -- the group indicator ``v_S``, the screened norm the
capture ``C_S`` and the confusability ``chi_S`` are fractions *of*, the Gram the
filter bank ascends, and the RSA distortion the coarsener is budgeted by -- lives
in ONE inner product.  This module is the single place that inner product is
defined, so the two Laplacian conventions of the paper cannot drift apart across
the ~40 call sites that used to hard-code ``L = I - A_hat``:

``kind="symmetric"`` (the historical default)
    ``L = I - A_hat`` with ``A_hat = D_tilde^{-1/2} (W + I) D_tilde^{-1/2}``,
    ``v_S = D_tilde^{1/2} 1_S / sqrt(vol(S))`` and ``||v_S||_L^2 = Phi(S) =
    cut(S)/vol(S)`` -- the *conductance*-normalized geometry.

``kind="combinatorial"``
    ``L = D - W``, ``v_S = 1_S / sqrt(|S|)`` and ``||v_S||_L^2 = Phi(S) =
    cut(S)/|S|`` -- the *cardinality*-normalized geometry of the paper's
    Section 3 (eq. 1-2), where ``M_tau = L + tau I`` and ``||v_S||^2_{M_tau} =
    Phi(S) + tau``.

The two differ in three coupled places, which is exactly why they need to travel
together in one object:

1. **the metric** ``M_tau = L + tau I`` (:meth:`m_apply`),
2. **the indicator** ``v_S`` (:meth:`indicator_columns`) -- degree-weighted vs
   uniform, so ``Phi(S)`` is conductance vs cardinality-normalized,
3. **the propagation operator** the Chebyshev bank filters on
   (:attr:`prop`) -- ``A_hat`` vs ``S = I - (2/lambda_max) L``.

Point 3 is the paper's ``L_tilde = (2/lambda_max) L - I`` up to the sign flip
``S = -L_tilde``: ``T_k(-x) = (-1)^k T_k(x)``, so the two span the identical
Krylov space and the learned ``theta`` merely absorbs the signs -- but ``S``
keeps *low* frequency at ``+1`` like ``A_hat`` does, so the ``warm_start="ones"``
flat low-pass initialization means the same thing in both geometries.

``lambda_max`` enters the *propagation operator only*: ``L = scale * (I - prop)``
holds by construction with ``scale = lambda_max/2``, so :meth:`l_apply` and
:meth:`m_apply` return the exact ``L`` and ``M_tau`` no matter how loose the
estimate is.  A loose estimate costs Chebyshev conditioning, never correctness.

A note on ``tau`` across the two: ``M_tau = L + tau I`` screens *relative to the
Laplacian's own scale*.  ``lambda_max(L_sym) <= 2`` always, while
``lambda_max(D - W)`` is of order the maximum degree, so the same numeric ``tau``
is a far weaker screening in the combinatorial geometry -- and the block
distortion bound ``mu_P^tau <= sqrt((lambda_max + tau)/tau)`` is correspondingly
larger.  :attr:`tau_equivalent` converts between the two on a like-for-like
basis; comparisons that hold ``tau`` numerically fixed are comparing different
screening strengths, which is a real effect and not a bug, but it should be said
out loud rather than discovered in the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ScreenedGeometry", "build_geometry", "symmetric_geometry"]


def _degrees(adjacency: torch.Tensor) -> torch.Tensor:
    """Weighted degree ``d = W 1`` of a sparse ``W`` (self-loops excluded upstream)."""

    degree = torch.zeros(
        adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    coalesced = adjacency.coalesce()
    degree.scatter_add_(0, coalesced.indices()[0], coalesced.values())
    return degree


def _lambda_max_combinatorial(
    adjacency: torch.Tensor, *, iters: int = 300, tol: float = 1e-7
) -> float:
    """Largest eigenvalue of ``L = D - W``, Lanczos with a power-iteration fallback.

    ``L`` is positive semidefinite, so its dominant eigenvalue *is* ``lambda_max``
    and no shift is needed.  Both estimators converge from *below*, and an
    underestimate is the one direction that hurts: ``S = I - (2/lambda_max) L``
    would then have spectrum outside ``[-1, 1]``, where the Chebyshev recurrence
    grows exponentially instead of staying bounded.  So the estimate is inflated
    by ``1e-3`` and clipped to the Gershgorin bound ``lambda_max <= 2 max_i d_i``,
    which is a true upper bound -- the result is therefore never below
    ``lambda_max`` and never gratuitously above it.  Only the Chebyshev
    conditioning depends on this number: ``L`` and ``M_tau`` are reconstructed
    exactly from it either way (see the module docstring), so erring high is free.
    """

    deg = _degrees(adjacency)
    gershgorin = float(2.0 * deg.max().item()) if deg.numel() else 0.0
    if gershgorin <= 0.0:
        return 1.0  # empty / edgeless graph: any positive scale will do

    n = adjacency.shape[0]
    dtype, device = adjacency.dtype, adjacency.device
    estimate = None

    try:  # Lanczos: far tighter than power iteration on a clustered spectrum
        import numpy as np
        from scipy.sparse import coo_matrix, diags
        from scipy.sparse.linalg import eigsh

        coalesced = adjacency.coalesce()
        idx = coalesced.indices().cpu().numpy()
        val = coalesced.values().detach().cpu().numpy().astype(np.float64)
        w_sp = coo_matrix((val, (idx[0], idx[1])), shape=(n, n)).tocsr()
        l_sp = diags(deg.detach().cpu().numpy().astype(np.float64)) - w_sp
        estimate = float(eigsh(l_sp, k=1, which="LA", return_eigenvectors=False)[0])
    except Exception:  # scipy missing, or ARPACK failed to converge
        estimate = None

    if estimate is None or not (estimate > 0.0):
        generator = torch.Generator(device="cpu").manual_seed(0)
        v = torch.randn(n, 1, dtype=dtype, generator=generator).to(device)
        v = v / v.norm().clamp_min(torch.finfo(dtype).eps)
        d_col = deg.unsqueeze(1)
        previous = 0.0
        for _ in range(iters):
            w = d_col * v - torch.sparse.mm(adjacency, v)  # L v
            norm = float(w.norm().item())
            if norm <= 0.0:
                break
            v = w / norm
            if abs(norm - previous) <= tol * max(norm, 1.0):
                break
            previous = norm
        estimate = float(
            (v * (d_col * v - torch.sparse.mm(adjacency, v))).sum().item()
        )

    return min(max(estimate, 1e-12) * (1.0 + 1e-3), gershgorin)


def _shifted_operator(adjacency: torch.Tensor, lambda_max: float) -> torch.Tensor:
    """``S = I - (2/lambda_max) (D - W)``, sparse, with ``spec(S) subset [-1, 1]``."""

    n = adjacency.shape[0]
    dtype, device = adjacency.dtype, adjacency.device
    scale = 2.0 / lambda_max
    deg = _degrees(adjacency)
    coalesced = adjacency.coalesce()

    loop = torch.arange(n, device=device)
    indices = torch.cat((torch.stack((loop, loop)), coalesced.indices()), dim=1)
    values = torch.cat((1.0 - scale * deg, scale * coalesced.values().to(dtype)))
    return torch.sparse_coo_tensor(
        indices, values, (n, n), dtype=dtype, device=device
    ).coalesce()


@dataclass(frozen=True)
class ScreenedGeometry:
    """The metric, the indicator, and the propagation operator, as one object.

    ``L = scale * (I - prop)`` by construction, so :meth:`l_apply` is exact for
    both conventions and the propagation operator is always the one whose
    spectrum sits in ``[-1, 1]`` (what :func:`~src.sgc_detection.chebyshev_stack`
    requires).
    """

    kind: str  # "symmetric" | "combinatorial"
    prop: torch.Tensor  # sparse; spectrum in [-1, 1]; the bank filters on this
    adjacency: torch.Tensor  # raw W (no self-loops)
    scale: float = 1.0  # L = scale * (I - prop)
    lambda_max: float = 2.0  # lambda_max(L) (exact bound for L_sym, estimated for L)

    # ---------------------------------------------------------------- metric --
    def l_apply(self, signals: torch.Tensor) -> torch.Tensor:
        """``L x`` for dense ``signals`` whose columns are graph signals."""

        out = signals - torch.sparse.mm(self.prop, signals)
        return out * self.scale if self.scale != 1.0 else out

    def m_apply(self, signals: torch.Tensor, tau: float) -> torch.Tensor:
        """``M_tau x = (L + tau I) x``; ``tau = 0`` is the plain ``L`` seminorm."""

        out = self.l_apply(signals)
        return out + tau * signals if tau else out

    # ------------------------------------------------------------- indicator --
    def indicator_columns(self, node_sets: "list") -> torch.Tensor:
        """Unit-``l2`` group indicators ``v_S``, one column per node set.

        ``symmetric``     -> ``v_S = D_tilde^{1/2} 1_S / sqrt(vol_tilde(S))``,
        for which ``||v_S||_L^2 = cut(S)/vol(S)`` (conductance).
        ``combinatorial`` -> ``v_S = 1_S / sqrt(|S|)``, for which
        ``||v_S||_L^2 = cut(S)/|S|`` (the paper's ``Phi(S)``, eq. 1-2).

        Both are ``l2``-normalized, so the caller's shared
        ``vhat_S = v_S / sqrt(Phi(S) + tau)`` has unit ``M_tau`` norm either way.
        """

        n = self.adjacency.shape[0]
        dtype, device = self.adjacency.dtype, self.adjacency.device
        eps = torch.finfo(dtype).eps
        uniform = self.kind == "combinatorial"
        weight = None if uniform else _degrees(self.adjacency) + 1.0  # D_tilde = D + I

        columns = []
        for nodes in node_sets:
            nodes = torch.as_tensor(nodes, dtype=torch.long, device=device)
            column = torch.zeros(n, dtype=dtype, device=device)
            if uniform:
                column[nodes] = 1.0
                mass = torch.tensor(float(nodes.numel()), dtype=dtype, device=device)
            else:
                column[nodes] = weight[nodes].sqrt()
                mass = weight[nodes].sum()
            columns.append(column / mass.clamp_min(eps).sqrt())
        return torch.stack(columns, dim=1)  # (N, m)

    def indicators(self, patterns: "list") -> torch.Tensor:
        """:meth:`indicator_columns` for a list of evaluation ``Pattern`` objects."""

        return self.indicator_columns([p.node_indices for p in patterns])

    # ------------------------------------------------- local (per-group) form --
    def node_weights(self, nodes: torch.Tensor) -> torch.Tensor:
        """The per-node weight ``w_i`` with ``v_S ~ w`` on ``S``.

        ``sqrt(d_tilde_i)`` in the symmetric geometry, ``1`` in the combinatorial
        one.  This is the coordinate change ``w = diag(weights) z`` the
        confusability tables are written in, *and* the ``l2``-orthogonality
        vector of the fluctuation space ``F_S`` (``<w, v_S>_2 = 0`` becomes
        ``sum_i weights_i^2 z_i = 0``), so both must come from here.
        """

        if self.kind == "combinatorial":
            return torch.ones(
                nodes.numel(), dtype=self.adjacency.dtype, device=self.adjacency.device
            )
        return (_degrees(self.adjacency) + 1.0)[nodes].sqrt()

    def screening_diagonal(self, nodes: torch.Tensor) -> torch.Tensor:
        """Diagonal of the ``tau`` term in the local form ``Q_S`` on ``S``.

        ``Q_S = L^int_S + diag(d_boundary) + tau * diag(this)``: the Laplacian
        part is convention-free (``w^T L w`` for ``w`` supported on ``S`` is the
        same combinatorial quadratic form either way), and only the screening
        term carries the geometry -- ``tau * D_tilde_S`` in ``z``-coordinates for
        the symmetric case, ``tau * I`` for the combinatorial one.
        """

        return self.node_weights(nodes) ** 2

    @property
    def ridge_scale(self) -> float:
        """Factor the absolute Tikhonov ``ridge`` must carry into this geometry.

        The learned objective itself is invariant to a global rescaling of
        ``M_tau``: ``Gamma = (Z^T M vhat)^T (Z^T M Z)^+ (Z^T M vhat)`` with
        ``vhat`` normalized *in that same metric*, so scaling ``M -> cM`` scales
        the two outer factors by ``sqrt(c)`` each and the inverse by ``1/c``.
        The one exception is the ridge added to ``Z^T M_tau Z``, which is an
        *absolute* quantity: leaving it at its symmetric value under a metric
        that is ``lambda_max/2`` times larger silently turns it off, and the
        near-collinear warm-start Gram is exactly where it is needed.

        Scaling it by :attr:`scale` restores the same *relative* regularization.
        For the symmetric geometry ``scale == 1.0``, so this is an exact no-op
        and pre-existing runs reproduce bit-for-bit.
        """

        return self.scale

    @property
    def tau_equivalent(self) -> float:
        """``tau`` multiplier that matches this geometry's screening to ``L_sym``'s.

        ``M_tau = L + tau I`` screens relative to the Laplacian's own scale, and
        ``lambda_max(L) / lambda_max(L_sym) = lambda_max / 2``.  So a symmetric
        run at ``tau`` is screened like a combinatorial run at
        ``tau * tau_equivalent`` -- the like-for-like setting when the point of a
        comparison is the *geometry* rather than the screening strength.
        """

        return self.lambda_max / 2.0

    def describe(self) -> str:
        return (
            f"{self.kind} (L = {'D - W' if self.kind == 'combinatorial' else 'I - A_hat'}"
            f", lambda_max~{self.lambda_max:.4g}, v_S = "
            f"{'1_S/sqrt(|S|)' if self.kind == 'combinatorial' else 'D^1/2 1_S/sqrt(vol)'})"
        )


def symmetric_geometry(a_hat: torch.Tensor, adjacency: torch.Tensor) -> ScreenedGeometry:
    """The historical geometry: ``L = I - A_hat``, degree-weighted ``v_S``.

    ``prop = A_hat`` and ``scale = 1``, so :meth:`ScreenedGeometry.l_apply` is
    literally ``x - A_hat x`` -- bit-for-bit what the call sites computed before
    this object existed.
    """

    return ScreenedGeometry(
        kind="symmetric", prop=a_hat, adjacency=adjacency, scale=1.0, lambda_max=2.0
    )


def build_geometry(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    kind: str = "symmetric",
    *,
    propagation: str = "auto",
) -> ScreenedGeometry:
    """Build the geometry named by ``kind`` (``"symmetric"``/``"combinatorial"``).

    ``propagation`` picks the operator the Chebyshev bank filters on:

    * ``"auto"`` (default) -- follow ``kind``: ``A_hat`` for symmetric,
      ``S = I - (2/lambda_max)(D - W)`` for combinatorial.  This is the paper's
      own pairing (``L_tilde = (2/lambda_max) L - I``, up to the sign flip
      documented at module level).
    * ``"a_hat"`` -- keep filtering on ``A_hat`` while measuring in the requested
      metric.  Not the paper, but it isolates the two effects: a combinatorial
      *metric + indicator* run against a symmetric one differs then only in what
      is measured, not in what is propagated.  Useful precisely because the
      combinatorial ``S`` is near-identity on low-degree nodes (its diffusion per
      hop is ``O(d_i / d_max)``), so a degree-``K`` bank reaches far less of the
      spectrum on a heavy-tailed graph -- an effect that would otherwise be
      indistinguishable from the geometry change itself.
    """

    if kind in ("combinatorial", "comb"):
        lambda_max = _lambda_max_combinatorial(adjacency)
        if propagation == "a_hat":
            # L = D - W can no longer be written as scale * (I - A_hat); this
            # combination is served by a dedicated subclass below.
            return _MixedGeometry(
                kind="combinatorial",
                prop=a_hat,
                adjacency=adjacency,
                scale=1.0,
                lambda_max=lambda_max,
            )
        return ScreenedGeometry(
            kind="combinatorial",
            prop=_shifted_operator(adjacency, lambda_max),
            adjacency=adjacency,
            scale=lambda_max / 2.0,
            lambda_max=lambda_max,
        )
    if kind in ("symmetric", "normalized", "sym", "norm"):
        if propagation not in ("auto", "a_hat"):
            raise ValueError("symmetric geometry only propagates on 'a_hat'")
        return symmetric_geometry(a_hat, adjacency)
    raise ValueError("kind must be 'symmetric' or 'combinatorial'")


@dataclass(frozen=True)
class _MixedGeometry(ScreenedGeometry):
    """Combinatorial metric + indicator, ``A_hat`` propagation (``propagation="a_hat"``).

    Here ``prop`` is no longer the operator ``L`` is built from, so ``l_apply``
    applies ``D - W`` directly instead of going through ``scale * (I - prop)``.
    Everything else -- the indicator, the local forms, ``tau_equivalent`` -- is
    inherited unchanged, which is the point: this isolates the propagation
    operator as the single differing factor.
    """

    def l_apply(self, signals: torch.Tensor) -> torch.Tensor:
        deg = _degrees(self.adjacency).unsqueeze(1)
        return deg * signals - torch.sparse.mm(self.adjacency, signals)
