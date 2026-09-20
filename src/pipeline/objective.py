r"""The learning objective: edge term, host-response penalty, label loss.

Edge objective (``edge_cost_screened.tex``, ``J_edge`` of the paper)
--------------------------------------------------------------------
With the whitened screened level ``ell`` of :mod:`src.pipeline.geometry`, the
squared level distance of an adjacent pair is
``d^2(u, v) = ||ell_u - ell_v||_2^2``.  For a group ``S_j`` let

    int_j = weighted mean of d^2 over the INTERNAL edges of S_j,
    bnd_j = weighted mean of d^2 over the BOUNDARY edges of S_j,

each weighted by the edge weight and normalized by that group's own total edge
weight.  Averaging over groups gives

    boundary = mean_j bnd_j,   internal = mean_j int_j,
    J_edge   = boundary - gamma * internal          (difference form)
    J_ratio  = boundary / internal                  (ratio form, gamma=None)

Both are **maximized**.  The difference form is what
:func:`src.level_objective.edge_objective` computes (validated against dense
ground truth); it is reproduced here on the whitened level -- algebraically the
same quantity, since ``||ell_u - ell_v||_2^2`` *is* the ``G^{-1}`` distance --
with one correction: groups are handled independently, so overlapping groups no
longer collapse onto a single ``gang_of`` array.  An edge may therefore be
internal to one group and boundary to another, and contributes to both.

Ratio form and Dinkelbach
-------------------------
The closed-form solver already maximizes ``tr(N)/tr(D)`` by Dinkelbach
iteration.  The gradient trainer has no such formulation natively, so
:class:`~src.pipeline.learning.base.Learning` wraps it: an OUTER Dinkelbach loop
fixes ``rho <- boundary/internal`` and an INNER gradient stage maximizes the
linearization ``boundary - rho * internal``.  This module only supplies the
linearized value; the loop lives in the learner.

Host-response penalty
---------------------
``L_host = mean_{v in H} ||ell_v||_2^2`` on the selected host set -- the same
screened response the edge term and the classifier read.  Host modes are the
existing ``none`` / ``neighbours`` / ``random``.

Label loss
----------
``logits_u = W ell_u + b`` (raw logits, softmax only at prediction time), scored
by cross-entropy against the label the *training* groups assign to their member
nodes.  Test groups never enter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.geometry import ScreenedLevel

__all__ = [
    "ObjectiveConfig",
    "EdgeIndexSets",
    "build_edge_sets",
    "select_host_nodes",
    "LabelHead",
    "LossComponents",
    "edge_terms",
    "evaluate_objective",
]

_HOST_MODES = ("none", "neighbours", "random")


@dataclass
class ObjectiveConfig:
    """Validated configuration of the learning objective."""

    gamma: "float | None" = 1.0
    host_mode: str = "none"
    host_weight: float = 0.0
    host_count: int = 0
    label_enabled: bool = False
    label_weight: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.gamma is not None and float(self.gamma) < 0.0:
            raise ValueError("gamma must be non-negative or None (ratio form)")
        if self.host_mode not in _HOST_MODES:
            raise ValueError(
                f"host_mode must be one of {_HOST_MODES}, got {self.host_mode!r}"
            )
        if self.host_weight < 0.0:
            raise ValueError("host_weight must be non-negative")
        if self.label_weight < 0.0:
            raise ValueError("label_weight must be non-negative")
        if self.host_count < 0:
            raise ValueError("host_count must be non-negative")

    @property
    def is_ratio(self) -> bool:
        return self.gamma is None

    @property
    def host_active(self) -> bool:
        return self.host_mode != "none" and self.host_weight > 0.0


# --------------------------------------------------------------------------- #
# edge sets (model-independent; built once per graph)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdgeIndexSets:
    """Per-group internal and boundary edges, flattened with a group index."""

    int_a: torch.Tensor
    int_b: torch.Tensor
    int_w: torch.Tensor
    int_g: torch.Tensor
    bnd_a: torch.Tensor
    bnd_b: torch.Tensor
    bnd_w: torch.Tensor
    bnd_g: torch.Tensor
    num_groups: int

    @property
    def counts(self) -> dict:
        return {
            "internal": int(self.int_a.numel()),
            "boundary": int(self.bnd_a.numel()),
        }


def build_edge_sets(
    adjacency: torch.Tensor, node_sets: "list[torch.Tensor]"
) -> EdgeIndexSets:
    """Internal / boundary edges of every group, evaluated group by group.

    Internal edges are counted once per undirected pair; a boundary edge of
    ``S_j`` is ``(u, v)`` with ``u in S_j`` and ``v not in S_j``.  Groups are
    processed independently, so overlapping groups each get their own -- possibly
    shared -- edges rather than a single owner.
    """

    coalesced = adjacency.coalesce()
    idx, val = coalesced.indices(), coalesced.values()
    off = idx[0] != idx[1]
    src, dst, weight = idx[0][off], idx[1][off], val[off]
    n = int(coalesced.shape[0])

    int_a, int_b, int_w, int_g = [], [], [], []
    bnd_a, bnd_b, bnd_w, bnd_g = [], [], [], []
    for j, nodes in enumerate(node_sets):
        member = torch.zeros(n, dtype=torch.bool, device=src.device)
        member[nodes.to(src.device)] = True
        in_src, in_dst = member[src], member[dst]
        internal = in_src & in_dst & (src < dst)
        boundary = in_src & ~in_dst
        int_a.append(src[internal])
        int_b.append(dst[internal])
        int_w.append(weight[internal])
        int_g.append(torch.full((int(internal.sum()),), j, dtype=torch.long))
        bnd_a.append(src[boundary])
        bnd_b.append(dst[boundary])
        bnd_w.append(weight[boundary])
        bnd_g.append(torch.full((int(boundary.sum()),), j, dtype=torch.long))

    empty_long = torch.zeros(0, dtype=torch.long, device=src.device)
    empty_val = torch.zeros(0, dtype=weight.dtype, device=src.device)
    cat = lambda parts, fallback: torch.cat(parts) if parts else fallback  # noqa: E731
    return EdgeIndexSets(
        int_a=cat(int_a, empty_long),
        int_b=cat(int_b, empty_long),
        int_w=cat(int_w, empty_val),
        int_g=cat(int_g, empty_long),
        bnd_a=cat(bnd_a, empty_long),
        bnd_b=cat(bnd_b, empty_long),
        bnd_w=cat(bnd_w, empty_val),
        bnd_g=cat(bnd_g, empty_long),
        num_groups=len(node_sets),
    )


def select_host_nodes(
    adjacency: torch.Tensor,
    node_sets: "list[torch.Tensor]",
    config: ObjectiveConfig,
    *,
    graph_index: int = 0,
) -> torch.Tensor:
    """Host set ``H``: ``neighbours`` of the training groups, or a random sample.

    Only the *training* groups are excluded, so a held-out group's nodes can land
    in ``H`` -- at fit time their labels are unknown and they are background.
    """

    if not config.host_active or not node_sets:
        return torch.zeros(0, dtype=torch.long)
    from src.closed_form_level import host_nodes

    patterns = [type("_G", (), {"node_indices": s.tolist()})() for s in node_sets]
    return host_nodes(
        adjacency,
        patterns,
        config.host_mode,
        count=config.host_count,
        seed=7919 * (config.seed + 1) + graph_index,
    )


# --------------------------------------------------------------------------- #
# label head
# --------------------------------------------------------------------------- #
class LabelHead(nn.Module):
    """``logits_u = W ell_u + b`` on the screened level; softmax only at predict."""

    def __init__(self, level_width: int, num_classes: int, dtype=torch.float64):
        super().__init__()
        if num_classes < 2:
            raise ValueError("the label head needs at least two classes")
        self.linear = nn.Linear(level_width, num_classes).to(dtype=dtype)

    def forward(self, level_rows: torch.Tensor) -> torch.Tensor:
        return self.linear(level_rows)

    @torch.no_grad()
    def probabilities(self, level_rows: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(level_rows), dim=1)


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #
@dataclass
class LossComponents:
    """Every term reported separately, as the specification requires."""

    total: torch.Tensor
    edge_objective: torch.Tensor
    boundary: torch.Tensor
    internal: torch.Tensor
    ratio: float
    host: torch.Tensor
    label: torch.Tensor
    rho: "float | None" = None

    def to_dict(self) -> dict:
        return {
            "total_loss": float(self.total),
            "edge_objective": float(self.edge_objective),
            "boundary": float(self.boundary),
            "internal": float(self.internal),
            "ratio": float(self.ratio),
            "host_loss": float(self.host),
            "label_loss": float(self.label),
            "rho": None if self.rho is None else float(self.rho),
        }


def _group_weighted_mean(values, weights, groups, num_groups):
    """Per-group edge-weight means; groups with no edge of that kind drop out."""

    weights = weights.to(values.dtype)
    numerator = values.new_zeros(num_groups).index_add(0, groups, weights * values)
    denominator = values.new_zeros(num_groups).index_add(0, groups, weights)
    keep = denominator > 0
    return numerator[keep] / denominator[keep]


def edge_terms(level: ScreenedLevel, edges: EdgeIndexSets):
    """``(boundary, internal)``: the two group-averaged level contrasts."""

    zero = level.values.new_zeros(())
    d_int = level.pair_distances(edges.int_a, edges.int_b)
    d_bnd = level.pair_distances(edges.bnd_a, edges.bnd_b)
    per_int = _group_weighted_mean(d_int, edges.int_w, edges.int_g, edges.num_groups)
    per_bnd = _group_weighted_mean(d_bnd, edges.bnd_w, edges.bnd_g, edges.num_groups)
    internal = per_int.mean() if per_int.numel() else zero
    boundary = per_bnd.mean() if per_bnd.numel() else zero
    return boundary, internal


def evaluate_objective(
    level: ScreenedLevel,
    edges: EdgeIndexSets,
    config: ObjectiveConfig,
    *,
    host_index: "torch.Tensor | None" = None,
    label_head: "LabelHead | None" = None,
    label_targets: "torch.Tensor | None" = None,
    label_nodes: "torch.Tensor | None" = None,
    rho: "float | None" = None,
) -> LossComponents:
    r"""Assemble the loss to be **minimized**.

    Difference form (``gamma`` numeric)::

        loss = -J_edge + w_host * L_host + w_label * L_label,
        J_edge = boundary - gamma * internal

    Ratio form (``gamma=None``): ``J_edge`` is replaced by the Dinkelbach
    linearization ``boundary - rho * internal`` at the outer iterate ``rho``;
    the reported ``ratio`` is always ``boundary/internal`` so the two forms stay
    comparable.
    """

    boundary, internal = edge_terms(level, edges)
    if config.is_ratio:
        if rho is None:
            raise ValueError(
                "the ratio form needs the current Dinkelbach rho; the learner "
                "supplies it once per outer iteration"
            )
        edge_objective = boundary - float(rho) * internal
    else:
        edge_objective = boundary - float(config.gamma) * internal

    zero = level.values.new_zeros(())
    host = zero
    if config.host_active and host_index is not None and host_index.numel():
        host = level.values[host_index].square().sum(dim=1).mean()

    label = zero
    if config.label_enabled and label_head is not None and label_nodes is not None:
        if label_nodes.numel():
            logits = label_head(level.values[label_nodes])
            label = F.cross_entropy(logits, label_targets[label_nodes])

    total = -edge_objective + config.host_weight * host + config.label_weight * label
    denominator = float(internal.detach())
    ratio = float(boundary.detach()) / denominator if denominator > 0 else float("inf")
    return LossComponents(
        total=total,
        edge_objective=edge_objective,
        boundary=boundary,
        internal=internal,
        ratio=ratio,
        host=host,
        label=label,
        rho=None if rho is None else float(rho),
    )
