r"""The screened geometry and the ONE screened level every component reads.

The paper measures everything in the screened inner product
``M_tau = L + tau I``.  The node-wise quantity the edge objective, the
coarsener, the label classifier and the capture/confusability diagnostics are
all functions of is the *Gram-whitened screened level*

    ell = D_tilde^{-1/2} M_tau Z (G^dagger)^{1/2},
    G   = Z^T M_tau Z,

(Sec. 5 of ``iclr2027_gang_detection.tex``).  This module is the single place
that level is computed, so the four consumers cannot drift apart.

Whitening convention
--------------------
``(G^dagger)^{1/2}`` is only defined up to a right orthogonal factor, and *no*
downstream quantity can see that factor: pair distances, group-mean norms,
covariance eigenvalues and a learnable linear head are all invariant under
``ell -> ell O``.  We therefore realize the whitener as the inverse Cholesky
factor of the ridged Gram,

    ell = D_tilde^{-1/2} M_tau Z L^{-T},   L L^T = G + rho I,

which is smooth in ``Z`` (unlike an eigendecomposition at repeated eigenvalues),
needs no pseudo-inverse, and makes

    ||ell_u - ell_v||_2^2 = (l_u - l_v)^T G^{-1} (l_u - l_v)

exactly the ``G^{-1}`` distance the objective and the raw-Ward score are stated
in.  The ridge ``rho = ridge + ridge_relative * mean(diag G)`` is the same
regularization the legacy Gram path uses; a wide filter bank is routinely
rank-deficient, and without it the null directions are amplified by ``rho^-1/2``.

Geometry
--------
Only the symmetric geometry is supported: ``L = I - A_hat``,
``v_S = D_tilde^{1/2} 1_S / sqrt(vol(S))``, ``Phi(S) = cut(S)/vol(S)``.  The
level above, the edge objective and the deflated coarsener are all stated in it.
The combinatorial convention is a one-line extension of
:class:`~src.screened_geometry.ScreenedGeometry` (``kappa = 1``, no volumes) but
is deliberately not exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.screened_geometry import ScreenedGeometry, symmetric_geometry

__all__ = [
    "ScreenedLevel",
    "screened_geometry",
    "screened_level",
    "group_capture",
    "group_confusability_bound",
]


def screened_geometry(a_hat: torch.Tensor, adjacency: torch.Tensor) -> ScreenedGeometry:
    """The symmetric screened geometry ``L = I - A_hat`` for one graph."""

    return symmetric_geometry(a_hat, adjacency)


@dataclass(frozen=True)
class ScreenedLevel:
    r"""The whitened screened level ``ell`` of a target ``R = span(Z)``.

    ``values`` is ``(N, q)``; ``gram`` is ``G = Z^T M_tau Z``; ``cholesky`` is the
    factor ``L`` of ``G + rho I`` used as the whitener.  ``rank`` is the number of
    Gram eigenvalues above the relative tolerance -- the effective target width,
    reported because a wide bank is usually rank-deficient and ``q`` alone is
    misleading.
    """

    values: torch.Tensor
    gram: torch.Tensor
    cholesky: torch.Tensor
    tau: float
    ridge: float
    rank: int

    @property
    def num_nodes(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    def rows(self, nodes: torch.Tensor) -> torch.Tensor:
        """Level rows of ``nodes`` (a long tensor of node indices)."""

        return self.values[nodes]

    def pair_distances(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Squared ``G^{-1}`` level distances ``||ell_u - ell_v||^2`` per pair."""

        return (self.values[left] - self.values[right]).square().sum(dim=1)


def screened_level(
    Z: torch.Tensor,
    geometry: ScreenedGeometry,
    tau: float,
    *,
    ridge: float = 0.0,
    ridge_relative: float = 1e-4,
    rank_tol: float = 1e-10,
) -> ScreenedLevel:
    r"""``ell = D_tilde^{-1/2} M_tau Z L^{-T}`` with ``L L^T = G + rho I``.

    Differentiable in ``Z``: the Cholesky factor and the triangular solve both
    carry gradient, so a gradient learner can back-propagate through the level
    into the representation model.

    The ridge defaults to the *relative* term alone
    (``rho = ridge_relative * mean(diag G)``), which keeps the level exactly
    invariant to ``Z -> cZ`` -- the invariance the objective, the capture and the
    coarsener all rely on.  A nonzero absolute ``ridge`` breaks it: a filter with
    little screened energy would then be driven to a level of zero rather than
    being rescaled.
    """

    if Z.ndim != 2 or Z.shape[1] == 0:
        raise ValueError("Z must be a non-empty (N, q) matrix")
    m_z = geometry.m_apply(Z, tau)
    gram = Z.T @ m_z
    gram = 0.5 * (gram + gram.T)
    rho = float(ridge) + float(ridge_relative) * float(gram.diagonal().mean().detach())
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    chol = torch.linalg.cholesky(gram + rho * eye)
    # X = (M_tau Z) L^{-T}  <=>  L X^T = (M_tau Z)^T
    whitened = torch.linalg.solve_triangular(chol, m_z.T, upper=False).T
    nodes = torch.arange(Z.shape[0], device=Z.device)
    weights = geometry.node_weights(nodes).to(Z.dtype).unsqueeze(1)
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(gram)
        top = float(eigenvalues[-1].clamp_min(0.0))
        rank = int((eigenvalues > rank_tol * max(top, 1e-300)).sum())
    return ScreenedLevel(
        values=whitened / weights,
        gram=gram,
        cholesky=chol,
        tau=float(tau),
        ridge=rho,
        rank=rank,
    )


def _group_mass(geometry: ScreenedGeometry, nodes: torch.Tensor, dtype) -> torch.Tensor:
    """Per-node mass ``d_tilde_v`` on ``nodes`` (``1`` in a uniform geometry)."""

    return geometry.node_weights(nodes).to(dtype) ** 2


def group_capture(
    level: ScreenedLevel,
    geometry: ScreenedGeometry,
    node_sets: "list[torch.Tensor]",
) -> torch.Tensor:
    r"""Exact capture ``C_S = vol(S)/(Phi(S) + tau) ||ell_bar_S||_2^2`` per group.

    Identical to ``Gamma_jj`` of the legacy Gram path (validated in
    :func:`src.level_objective.run_validation_suite`); here it is read off the
    whitened level, so the ``G^{-1}`` norm is a plain Euclidean one.
    """

    if not node_sets:
        return level.values.new_zeros(0)
    indicators = geometry.indicator_columns([s.tolist() for s in node_sets])
    phi = (indicators * geometry.l_apply(indicators)).sum(0).to(level.values.dtype)
    out = []
    for j, nodes in enumerate(node_sets):
        mass = _group_mass(geometry, nodes, level.values.dtype)
        vol = mass.sum()
        mean = (mass.unsqueeze(1) * level.values[nodes]).sum(0) / vol
        out.append(vol / (phi[j] + level.tau) * (mean @ mean))
    return torch.stack(out)


def group_confusability_bound(
    level: ScreenedLevel,
    geometry: ScreenedGeometry,
    node_sets: "list[torch.Tensor]",
) -> torch.Tensor:
    r"""``chibar_S = lambda_max(Sigma_S)/tau >= chi_S``, the Sec. 5 bound.

    ``Sigma_S`` is the mass-weighted level covariance of ``S``.  Its nonzero
    spectrum equals that of the ``|S| x |S|`` matrix ``E_S E_S^T``, which is what
    is diagonalized here (groups are small; the target can be wide).
    """

    out = []
    for nodes in node_sets:
        if nodes.numel() < 2:
            out.append(level.values.new_zeros(()))
            continue
        mass = _group_mass(geometry, nodes, level.values.dtype)
        rows = level.values[nodes]
        mean = (mass.unsqueeze(1) * rows).sum(0) / mass.sum()
        centered = mass.sqrt().unsqueeze(1) * (rows - mean)
        kernel = centered @ centered.T
        out.append(torch.linalg.eigvalsh(0.5 * (kernel + kernel.T))[-1] / level.tau)
    return torch.stack(out) if out else level.values.new_zeros(0)
