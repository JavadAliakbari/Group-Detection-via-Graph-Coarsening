r"""Nonlinear message-passing learners: GCN and GraphSAGE.

Multi-head convention (approved design)
---------------------------------------
The paper defines heads only for the polynomial bank.  For the nonlinear models
a head is an **independent encoder**: ``num_heads = H`` instantiates ``H``
encoders with their own parameters, and the node representation is the
concatenation of their outputs,

    Z = [Z^(1) | ... | Z^(H)] in R^{N x (H * hidden_dim)},

so the *concatenated-span* semantics match the filter bank exactly (the target
subspace is the span of the union of the heads' columns) at ``H`` times the
parameter count.  ``H = 1`` is the plain single encoder.

Both encoders are configurable in depth, width, activation and dropout;
GraphSAGE additionally in its aggregation (``mean`` / ``max`` / ``sum``).  The
input dimension is inferred from the data and the output class count from the
training groups, both by :class:`~src.pipeline.learning.base.Learning`.

Neither model has a closed-form solution in this repository, so
``training_mode="closed_form"`` is rejected by ``LearningConfig`` validation
rather than silently switched to gradient descent.

Why GraphSAGE is the fragile one
--------------------------------
The screened level only sees ``span(Z)`` -- it is invariant to any invertible
recombination of the columns -- so what an encoder contributes is the *kind* of
subspace it can reach.  ``GCNEncoder`` moves information only by propagation
(``A_hat h W``), so every subspace it reaches is a spectral object and behaves
the same way on a graph it has never seen.  ``SAGEEncoder`` adds ``W_self h``,
which is not a propagation: it can act on a node without reference to the graph.

Under the ratio form that extra freedom has a degenerate optimum.  The objective
rewards ``internal -> 0`` on the **training** groups, and the cheapest way to
get there is to map exactly those nodes onto the origin of the level, which
costs nothing under the whitening.  Measured on the synthetic benchmark
(2 graphs, 12 groups, half held out): GraphSAGE reaches a boundary/internal
ratio of ~650 while training-group capture falls 0.153 -> 0.023 and the
held-out groups' intra-edge distances stay 356x larger -- it has memorized the
group identities.  Freezing ``W_self`` at zero drops that gap to 5.6x and holds
capture at 0.122, which is what isolates the self path as the cause.

This is a property of the objective, not a bug in the encoder, so it is not
patched here.  :meth:`Learning._warn_about_the_trajectory` watches capture and
says so; the difference form (a numeric ``gamma``), ``weight_decay`` and the
host term are the levers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.data import Graph
from src.pipeline.learning.base import _ACTIVATIONS, Learning, LearningConfig

__all__ = ["GCNLearning", "GraphSAGELearning", "GCNEncoder", "SAGEEncoder"]


class GCNEncoder(nn.Module):
    """``h <- act(A_hat h W)`` repeated ``num_layers`` times."""

    def __init__(self, in_dim, hidden_dim, num_layers, activation, dropout, dtype):
        super().__init__()
        widths = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]).to(dtype=dtype)
            for i in range(num_layers)
        )
        self.activation = activation
        self.dropout = float(dropout)

    def forward(self, a_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = torch.sparse.mm(a_hat, layer(h))
            if i != last:
                h = _ACTIVATIONS[self.activation](h)
                if self.dropout:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class SAGEEncoder(nn.Module):
    """``h <- act(W_self h + W_neigh AGG_{u~v} h_u)`` repeated ``num_layers`` times."""

    def __init__(
        self, in_dim, hidden_dim, num_layers, activation, dropout, aggregation, dtype
    ):
        super().__init__()
        widths = [in_dim] + [hidden_dim] * num_layers
        self.self_layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]).to(dtype=dtype)
            for i in range(num_layers)
        )
        self.neighbour_layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1], bias=False).to(dtype=dtype)
            for i in range(num_layers)
        )
        self.activation = activation
        self.dropout = float(dropout)
        self.aggregation = aggregation

    def _aggregate(self, adjacency, degree, src, dst, h):
        if self.aggregation == "sum":
            return torch.sparse.mm(adjacency, h)
        if self.aggregation == "mean":
            return torch.sparse.mm(adjacency, h) / degree.clamp_min(1e-12).unsqueeze(1)
        pooled = torch.zeros_like(h)
        return pooled.index_reduce_(0, dst, h[src], "amax", include_self=False)

    def forward(self, adjacency, degree, src, dst, x: torch.Tensor) -> torch.Tensor:
        h = x
        last = len(self.self_layers) - 1
        for i, (own, neighbour) in enumerate(
            zip(self.self_layers, self.neighbour_layers)
        ):
            h = own(h) + neighbour(self._aggregate(adjacency, degree, src, dst, h))
            if i != last:
                h = _ACTIVATIONS[self.activation](h)
                if self.dropout:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class _MultiHead(nn.Module):
    """``H`` independent encoders whose outputs are concatenated."""

    def __init__(self, heads: "list[nn.Module]"):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, *args) -> torch.Tensor:
        outputs = [head(*args) for head in self.heads]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


class GCNLearning(Learning):
    """Configurable nonlinear GCN; gradient training only."""

    def __init__(self, config: LearningConfig):
        if config.architecture != "gcn":
            raise ValueError("GCNLearning requires architecture='gcn'")
        super().__init__(config)

    def build_model(self, feature_dim: int, dtype: torch.dtype) -> nn.Module:
        c = self.config
        return _MultiHead(
            [
                GCNEncoder(
                    feature_dim,
                    c.hidden_dim,
                    c.num_layers,
                    c.activation,
                    c.dropout,
                    dtype,
                )
                for _ in range(c.num_heads)
            ]
        )

    def encode(self, model: nn.Module, graph: Graph, geometry) -> torch.Tensor:
        return model(graph.a_hat, graph.features)


class GraphSAGELearning(Learning):
    """Configurable nonlinear GraphSAGE; gradient training only."""

    def __init__(self, config: LearningConfig):
        if config.architecture != "graphsage":
            raise ValueError("GraphSAGELearning requires architecture='graphsage'")
        super().__init__(config)
        self._neighbourhood: dict = {}

    def build_model(self, feature_dim: int, dtype: torch.dtype) -> nn.Module:
        c = self.config
        return _MultiHead(
            [
                SAGEEncoder(
                    feature_dim,
                    c.hidden_dim,
                    c.num_layers,
                    c.activation,
                    c.dropout,
                    c.aggregation,
                    dtype,
                )
                for _ in range(c.num_heads)
            ]
        )

    def _neighbourhood_of(self, graph: Graph):
        cached = self._neighbourhood.get(graph.graph_id)
        if cached is None:
            adjacency = graph.adjacency.coalesce()
            src, dst = adjacency.indices()
            degree = torch.zeros(
                graph.num_nodes, dtype=adjacency.dtype, device=adjacency.device
            ).scatter_add_(0, src, adjacency.values())
            cached = (adjacency, degree, src, dst)
            self._neighbourhood[graph.graph_id] = cached
        return cached

    def encode(self, model: nn.Module, graph: Graph, geometry) -> torch.Tensor:
        adjacency, degree, src, dst = self._neighbourhood_of(graph)
        return model(adjacency, degree, src, dst, graph.features)
