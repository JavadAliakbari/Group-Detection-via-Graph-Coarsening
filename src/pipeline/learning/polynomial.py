r"""Polynomial filter-bank learning: ``Z = [g_theta_1(A_hat) X | ... | g_theta_H(A_hat) X]``.

Bases ``chebyshev`` (``T_k(A_hat)``) and ``monomial`` (``A_hat^k``) span the same
Krylov space; only the conditioning differs.  ``num_heads = H`` gives ``H``
filter banks whose concatenated columns are the coarsening target, exactly as in
the paper's multi-head filter bank.  ``shared_filters`` keeps the existing
channel-wise meaning: one coefficient vector per head shared by every input
channel (``True``) or one per (head, channel) pair (``False``).

Both training modes are available:

``gradient``
    handled entirely by :class:`~src.pipeline.learning.base.Learning`.

``closed_form``
    the exact edge pencil.  Every term of ``J_edge`` is a quadratic form in the
    dictionary coefficients, so the optimum over the whole dictionary is the
    leading generalized eigenvector of ``(B_bnd - gamma B_int - w_host C_H, S_M)``
    -- solved per channel by :func:`src.closed_form_level._solve_block`, which
    also carries the Dinkelbach ratio form used when ``gamma is None``.  The
    pencil is assembled here rather than by
    :func:`src.closed_form_level.fit_channel_closed_form` for one reason: the
    latter partitions edges through a single ``gang_of`` array, which is wrong
    for overlapping groups.  The *solver* is reused unchanged.

Note on whitening.  The closed-form pencil necessarily works with the
*unwhitened* dictionary response ``D_tilde^{-1/2} M_tau D``; the Gram ``S_M`` is
the pencil's metric and plays the whitener's role.  Every quantity reported
afterwards -- capture, confusability, the label head, the coarsening -- is read
off the whitened level of :mod:`src.pipeline.geometry`, so the pipeline stays
consistent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.data import Graph
from src.pipeline.geometry import (
    group_capture,
    group_confusability_bound,
    screened_level,
)
from src.pipeline.learning.base import GraphContext, Learning, LearningConfig
from src.pipeline.objective import EdgeIndexSets
from src.run_collective_bank_detection import _basis_stack
from src.utils.utils import LOGGER

__all__ = ["PolynomialFilterLearning", "PolynomialBank"]


class PolynomialBank(nn.Module):
    """The filter bank ``Theta``; columns are normalized in the forward pass."""

    def __init__(self, degree: int, feature_dim: int, heads: int, shared: bool, dtype):
        super().__init__()
        channels = 1 if shared else feature_dim
        self.shared = bool(shared)
        self.feature_dim = int(feature_dim)
        self.raw = nn.Parameter(torch.ones(heads, degree + 1, channels, dtype=dtype))
        with torch.no_grad():  # break head symmetry: duplicate heads add no span
            if heads > 1:
                self.raw.add_(0.05 * torch.randn_like(self.raw))

    @property
    def theta(self) -> torch.Tensor:
        """``(H, K+1, d)`` with unit-norm coefficient columns."""

        unit = self.raw / self.raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return unit.expand(-1, -1, self.feature_dim) if self.shared else unit

    def forward(self, propagated: "list[torch.Tensor]") -> torch.Tensor:
        theta = self.theta
        heads = [
            sum(propagated[k] * theta[h, k].unsqueeze(0) for k in range(theta.shape[1]))
            for h in range(theta.shape[0])
        ]
        return heads[0] if len(heads) == 1 else torch.cat(heads, dim=1)


class PolynomialFilterLearning(Learning):
    """Learned polynomial filter bank (closed-form or gradient)."""

    def __init__(self, config: LearningConfig):
        if config.architecture != "polynomial":
            raise ValueError(
                "PolynomialFilterLearning requires architecture='polynomial'"
            )
        super().__init__(config)
        self._stack_cache: dict = {}

    # -- representation ----------------------------------------------------- #
    def build_model(self, feature_dim: int, dtype: torch.dtype) -> nn.Module:
        c = self.config
        return PolynomialBank(
            c.degree, feature_dim, c.num_heads, c.shared_filters, dtype
        )

    def propagated(self, graph: Graph, geometry) -> "list[torch.Tensor]":
        """``[phi_k(A_hat) X]_{k=0..K}`` -- model-independent, so cached per graph."""

        cached = self._stack_cache.get(graph.graph_id)
        if cached is None:
            c = self.config
            cached = _basis_stack(
                geometry.prop,
                graph.features,
                c.degree,
                c.basis,
                c.tau,
                geometry=geometry,
            )
            self._stack_cache[graph.graph_id] = cached
        return cached

    def encode(self, model: nn.Module, graph: Graph, geometry) -> torch.Tensor:
        return model(self.propagated(graph, geometry))

    # -- closed form -------------------------------------------------------- #
    def fit_closed_form(self, contexts: "list[GraphContext]") -> dict:
        from src.closed_form_level import _solve_block, _support_whitener, host_C

        c = self.config
        gamma = c.objective.gamma
        ratio = c.objective.is_ratio
        before = self._closed_form_diagnostics(contexts)

        n_pencils = 0
        S_sum = N_sum = P_sum = C_sum = None
        for ctx in contexts:
            m_prop = [
                ctx.geometry.m_apply(p, c.tau)
                for p in self.propagated(ctx.graph, ctx.geometry)
            ]
            prop = self.propagated(ctx.graph, ctx.geometry)
            dictionary = torch.cat(prop, 1)
            screened = torch.cat(m_prop, 1)
            gram = _sym(dictionary.T @ screened)
            boundary, internal = _edge_pencil_parts(ctx.geometry, screened, ctx.edges)
            S_sum = gram if S_sum is None else S_sum + gram
            N_sum = boundary if N_sum is None else N_sum + boundary
            P_sum = internal if P_sum is None else P_sum + internal
            if c.objective.host_active and ctx.host_index.numel():
                host = host_C(ctx.geometry, m_prop, ctx.host_index)
                C_sum = host if C_sum is None else C_sum + host
            n_pencils += 1

        S_bar, N_bar, P_bar = S_sum / n_pencils, N_sum / n_pencils, P_sum / n_pencils
        C_bar = C_sum / n_pencils if C_sum is not None else None
        d = self.feature_dim_
        blocks = [torch.arange(c.degree + 1) * d + a for a in range(d)]

        def channel_mean(matrix):
            return torch.stack([matrix[i][:, i] for i in blocks]).mean(0)

        if c.shared_filters:
            pencils = [
                (
                    channel_mean(S_bar),
                    channel_mean(N_bar),
                    channel_mean(P_bar),
                    channel_mean(C_bar) if C_bar is not None else None,
                )
            ]
        else:
            pencils = [
                (
                    S_bar[i][:, i],
                    N_bar[i][:, i],
                    P_bar[i][:, i],
                    C_bar[i][:, i] if C_bar is not None else None,
                )
                for i in blocks
            ]

        theta = torch.zeros(c.num_heads, c.degree + 1, d, dtype=S_bar.dtype)
        infos, conditioning, ranks = [], [], []
        for channel, (S_b, N_b, P_b, C_b) in enumerate(pencils):
            # Solve on the Gram's numerical support.  A degree-K Chebyshev block
            # Gram is routinely rank deficient (measured: 26 of 33 at K=32), and
            # the difference form's ridge metric would otherwise return a filter
            # living in that null space -- a roundoff-sized "positive" eigenvalue
            # whose Z has no screened energy at all.
            support = _support_whitener(_sym(S_b))
            rank = int(support.shape[1])
            if rank < 1:
                raise ValueError(
                    f"channel {channel}: the dictionary Gram has no positive "
                    "direction; lower `degree` or raise `tau`"
                )
            identity = torch.eye(rank, dtype=S_b.dtype)
            vectors, info = _solve_block(
                identity,
                _sym(support.T @ N_b @ support),
                _sym(support.T @ P_b @ support),
                _sym(support.T @ C_b @ support) if C_b is not None else None,
                penalty=0.0 if ratio else float(gamma),
                host_weight=c.objective.host_weight if c.objective.host_active else 0.0,
                heads=c.num_heads,
                scale="absolute",
                form="ratio" if ratio else "difference",
            )
            vectors = support @ vectors
            vectors = vectors / vectors.norm(dim=0, keepdim=True).clamp_min(1e-300)
            infos.append(info)
            ranks.append(rank)
            eigenvalues = torch.linalg.eigvalsh(_sym(S_b))
            positive = eigenvalues[eigenvalues > 0]
            conditioning.append(
                float(eigenvalues[-1] / positive.min())
                if positive.numel()
                else float("inf")
            )
            if c.shared_filters:
                theta[:] = vectors.T.unsqueeze(-1)
            else:
                theta[:, :, channel] = vectors.T

        with torch.no_grad():
            self.model_.raw.copy_(theta[:, :, :1] if c.shared_filters else theta)
        after = self._closed_form_diagnostics(contexts)

        positive_directions = [int(i["n_raw"]) for i in infos]
        cliffs = [float(i["penalty_star"]) for i in infos]
        report = {
            "solver": "edge-pencil (per-channel generalized eigenproblem)",
            "form": "ratio" if ratio else "difference",
            "shared_filters": bool(c.shared_filters),
            "n_pencils": n_pencils,
            "gram_condition_median": float(np.median(conditioning)),
            "gram_condition_max": float(np.max(conditioning)),
            "gram_rank_median": float(np.median(ranks)),
            "gram_dimension": int(c.degree + 1),
            "positive_directions_per_channel": positive_directions,
            "penalty_star_median": float(np.median(cliffs)),
            "coefficients": theta.detach().cpu().tolist(),
            "initial": before,
            "final": after,
        }
        self._warn_about_the_solution(report, gamma, ratio)
        if ratio:
            report["ratio_solver"] = {
                "rho_per_channel": [float(i["penalty_abs"]) for i in infos],
                "iterations_max": max(int(i.get("ratio_iters", 0)) for i in infos),
                "converged": all(bool(i.get("converged", True)) for i in infos),
                "fixed_point_residual_max": max(
                    abs(float(i.get("fixed_point_topH_sum", 0.0))) for i in infos
                ),
            }
        return report

    def _warn_about_the_solution(self, report: dict, gamma, ratio: bool) -> None:
        """Flag the ways the closed-form optimum can be degenerate.

        These are properties of the configuration, not failures of the solve, so
        they are reported rather than raised -- but silently returning a filter
        whose level is numerically flat is exactly what this is here to prevent.
        """

        positive = report["positive_directions_per_channel"]
        if not ratio:
            if max(positive) == 0:
                LOGGER.warning(
                    f"gamma={float(gamma):g} is at or above the pencil cliff "
                    f"lambda_max(N, P) ~ {report['penalty_star_median']:.4g}: no direction "
                    "has a positive edge objective, so the maximizer of "
                    "boundary - gamma*internal is the one that flattens the level "
                    "everywhere.  Lower gamma below the cliff, or use gamma=None "
                    "(the ratio form solves for the penalty)."
                )
            elif self.config.num_heads > max(positive):
                LOGGER.warning(
                    f"num_heads={self.config.num_heads} exceeds the {max(positive)} "
                    "positive direction(s) the pencil offers; the surplus heads are "
                    "duplicates and add nothing to the target's span"
                )
        for graph_id, after in report["final"].items():
            before = report["initial"][graph_id]
            if after["boundary"] < 1e-4 * max(before["boundary"], 1e-300):
                LOGGER.warning(
                    f"[{graph_id}] the solved level is essentially flat (boundary "
                    f"{before['boundary']:.3g} -> {after['boundary']:.3g}, capture "
                    f"{before['train_capture']:.4f} -> {after['train_capture']:.4f}): "
                    "the objective was satisfied by removing contrast rather than by "
                    "separating the groups, so this filter will not detect anything"
                )
        if report["gram_rank_median"] < report["gram_dimension"]:
            LOGGER.info(
                f"    dictionary Gram is rank {report['gram_rank_median']:.0f} of "
                f"{report['gram_dimension']} (condition "
                f"{report['gram_condition_median']:.3g}); the solve is restricted to "
                "that support"
            )

    def _closed_form_diagnostics(self, contexts: "list[GraphContext]") -> dict:
        """Objective, boundary/internal, host loss, capture and chi at the current filter."""

        from src.pipeline.objective import edge_terms

        c = self.config
        rows = {}
        for ctx in contexts:
            with torch.no_grad():
                Z = self.encode(self.model_, ctx.graph, ctx.geometry)
                level = screened_level(
                    Z,
                    ctx.geometry,
                    c.tau,
                    ridge=c.ridge,
                    ridge_relative=c.ridge_relative,
                )
                boundary, internal = edge_terms(level, ctx.edges)
                host = (
                    float(level.values[ctx.host_index].square().sum(1).mean())
                    if ctx.host_index.numel()
                    else 0.0
                )
                capture = group_capture(level, ctx.geometry, ctx.train_sets)
                chi = group_confusability_bound(level, ctx.geometry, ctx.train_sets)
                held = (
                    group_capture(level, ctx.geometry, ctx.heldout_sets)
                    if ctx.heldout_sets
                    else torch.zeros(0, dtype=level.values.dtype)
                )
            internal_value = float(internal)
            ratio = (
                float(boundary) / internal_value if internal_value > 0 else float("inf")
            )
            rows[ctx.graph_id] = {
                "boundary": float(boundary),
                "internal": internal_value,
                "ratio": ratio,
                # the ratio form's objective IS the ratio; the difference form's is J_edge
                "edge_objective": (
                    ratio
                    if c.objective.is_ratio
                    else float(boundary) - float(c.objective.gamma) * internal_value
                ),
                "host_loss": host,
                "train_capture": (
                    float(capture.mean()) if capture.numel() else float("nan")
                ),
                "train_capture_min": (
                    float(capture.min()) if capture.numel() else float("nan")
                ),
                "heldout_capture": float(held.mean()) if held.numel() else float("nan"),
                "confusability": float(chi.max()) if chi.numel() else float("nan"),
                "level_rank": int(level.rank),
            }
        return rows


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def _edge_pencil_parts(
    geometry, screened_dictionary: torch.Tensor, edges: EdgeIndexSets
):
    r"""``(B_bnd, B_int)`` of the edge pencil, assembled group by group.

    ``B = sum_p c_p y_p y_p^T`` with ``y_p`` the dictionary level difference of
    pair ``p`` and ``c_p`` the edge weight normalized by its group's total, then
    averaged over the groups that own an edge of that kind -- i.e. exactly the
    group-averaged weighted mean the objective uses, written as a quadratic form
    in the dictionary coefficients.
    """

    nodes = torch.arange(
        screened_dictionary.shape[0], device=screened_dictionary.device
    )
    weights = geometry.node_weights(nodes).to(screened_dictionary.dtype).unsqueeze(1)
    level = screened_dictionary / weights

    def part(a, b, w, g, num_groups):
        if a.numel() == 0:
            width = level.shape[1]
            return torch.zeros(width, width, dtype=level.dtype)
        Y = level[a] - level[b]
        w = w.to(Y.dtype)
        totals = torch.zeros(num_groups, dtype=Y.dtype).index_add(0, g, w)
        active = int((totals > 0).sum())
        coefficient = w / totals[g] / max(active, 1)
        return _sym(Y.T @ (coefficient.unsqueeze(1) * Y))

    return (
        part(edges.bnd_a, edges.bnd_b, edges.bnd_w, edges.bnd_g, edges.num_groups),
        part(edges.int_a, edges.int_b, edges.int_w, edges.int_g, edges.num_groups),
    )
