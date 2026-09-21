r"""Learning: the common contract, the shared training procedures, the result.

Every architecture is a :class:`Learning` subclass with the same contract::

    learner = PolynomialFilterLearning(config)
    result  = learner.run(training_graphs)

so the rest of the pipeline never depends on which representation model was
chosen.  The subclass supplies only a *representation model* -- a module mapping
one :class:`~src.pipeline.data.Graph` to ``Z in R^{N x q}`` -- plus, for the
polynomial bank, a closed-form solver.  Everything else (the screened level, the
objective, the gradient loop, the Dinkelbach outer loop, the label head) lives
here and is therefore identical across architectures.

Training modes
--------------
``gradient``
    Adam (or L-BFGS) on the representation parameters.  One training graph is
    drawn uniformly per epoch by default (``multi_graph_mode="sample"``);
    ``"mean"`` and ``"min"`` step on the average / worst graph instead.  Every
    loss and mask is computed from the graph that epoch stepped on.

``closed_form``
    Only the polynomial bank has one.  Requesting it for GCN or GraphSAGE is a
    configuration error, never a silent fallback.

Ratio form (``objective.gamma=None``)
-------------------------------------
Gradient training gains a ratio formulation through an explicit **outer
Dinkelbach loop**: iteration ``t`` fixes ``rho_t = boundary/internal`` measured
at the current iterate and runs an **inner gradient stage** of
``epochs // dinkelbach_iters`` epochs maximizing ``boundary - rho_t * internal``.
At the fixed point the linearized objective is zero and ``rho`` is the optimal
ratio.  Outer iterations and inner epochs are reported separately, and both
convergence signals are returned.

Because that surrogate is measured against a ``rho_t`` that grows every outer
iteration, its value *shrinks toward zero* as the ratio improves and is
therefore not comparable across outer iterations.  Best-iterate selection ranks
the ratio form by ``-ratio`` (plus the auxiliary penalties) instead; see
:meth:`Learning._selection_score`.

Label prediction
----------------
The head is linear on the **screened level**, never on ``Z``.  Under
``gradient`` it is trained jointly (its gradient reaches the representation
unless ``freeze_representation``); under ``closed_form`` the subspace is solved
first, ``Z`` and the level are frozen, and only the head is fitted -- its
trajectory is returned separately.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.pipeline.data import Graph
from src.pipeline.geometry import (
    ScreenedLevel,
    group_capture,
    group_confusability_bound,
    screened_geometry,
    screened_level,
)
from src.pipeline.objective import (
    EdgeIndexSets,
    LabelHead,
    LossComponents,
    ObjectiveConfig,
    build_edge_sets,
    evaluate_objective,
    select_host_nodes,
)
from src.utils.utils import LOGGER

__all__ = [
    "LearningConfig",
    "LearningResult",
    "Learning",
    "GraphContext",
]

_ARCHITECTURES = ("polynomial", "gcn", "graphsage", "static_spectral")
_MODES = ("closed_form", "gradient")
_BASES = ("chebyshev", "monomial")
_ACTIVATIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "elu": torch.nn.functional.elu,
    "gelu": torch.nn.functional.gelu,
    "identity": lambda x: x,
}
_AGGREGATIONS = ("mean", "max", "sum")


@dataclass
class LearningConfig:
    """Every knob of the learning component, validated on construction."""

    architecture: str = "polynomial"
    training_mode: str = "gradient"
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    # --- screened geometry ---
    tau: float = 0.5
    ridge: float = 0.0  # absolute; nonzero breaks the level's scale invariance
    ridge_relative: float = 1e-4

    # --- polynomial bank ---
    degree: int = 16
    basis: str = "chebyshev"
    num_heads: int = 1
    shared_filters: bool = False

    # --- nonlinear encoders ---
    num_layers: int = 2
    hidden_dim: int = 32
    activation: str = "relu"
    dropout: float = 0.0
    aggregation: str = "mean"

    # --- optimization ---
    epochs: int = 300
    learning_rate: float = 0.05
    optimizer: str = "adam"
    weight_decay: float = 0.0
    dinkelbach_iters: int = 5
    dinkelbach_tol: float = 1e-6
    multi_graph_mode: str = "sample"
    # the training figure wants capture/confusability at every epoch; raise this
    # to trade trajectory resolution for the per-epoch eigenproblems it costs
    diagnostic_interval: int = 1

    # --- label head ---
    freeze_representation: bool = False
    label_head_epochs: int = 300
    label_head_learning_rate: float = 0.05

    seed: int = 0

    def __post_init__(self) -> None:
        if self.architecture not in _ARCHITECTURES:
            raise ValueError(f"architecture must be one of {_ARCHITECTURES}")
        if self.training_mode not in _MODES:
            raise ValueError(f"training_mode must be one of {_MODES}")
        if self.training_mode == "closed_form" and self.architecture not in (
            "polynomial",
            "static_spectral",
        ):
            raise ValueError(
                f"training_mode='closed_form' is only defined for the polynomial "
                f"filter bank; {self.architecture!r} has no closed-form solution in "
                "this repository.  Use training_mode='gradient'."
            )
        if self.architecture == "static_spectral" and self.training_mode != "closed_form":
            raise ValueError(
                "architecture='static_spectral' has nothing to train by gradient: "
                "its target is a function of the graph alone.  Use "
                "training_mode='closed_form', which fits only the label head."
            )
        if self.basis not in _BASES:
            raise ValueError(f"basis must be one of {_BASES}, got {self.basis!r}")
        if self.num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {_AGGREGATIONS}")
        if self.tau <= 0.0:
            raise ValueError("tau must be strictly positive (M_tau must be definite)")
        if self.optimizer not in ("adam", "lbfgs"):
            raise ValueError("optimizer must be 'adam' or 'lbfgs'")
        if self.multi_graph_mode not in ("sample", "mean", "min"):
            raise ValueError("multi_graph_mode must be 'sample', 'mean' or 'min'")
        if self.dinkelbach_iters < 1:
            raise ValueError("dinkelbach_iters must be at least 1")
        if self.diagnostic_interval < 1:
            raise ValueError("diagnostic_interval must be at least 1")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["objective"] = asdict(self.objective)
        return out


@dataclass
class GraphContext:
    """Everything the objective needs from one graph, precomputed once."""

    graph: Graph
    geometry: object
    edges: EdgeIndexSets
    host_index: torch.Tensor
    label_targets: torch.Tensor
    label_nodes: torch.Tensor
    train_sets: "list[torch.Tensor]"
    heldout_sets: "list[torch.Tensor]"

    @property
    def graph_id(self) -> str:
        return self.graph.graph_id


@dataclass
class LearningResult:
    """Structured output of :meth:`Learning.run`."""

    model: nn.Module
    representations: dict
    levels: dict
    history: list
    label_head: "LabelHead | None"
    label_history: list
    closed_form: "dict | None"
    dinkelbach: "dict | None"
    config: dict
    diagnostics: dict = field(default_factory=dict)

    def to_serializable(self) -> dict:
        """JSON-ready view (the tensors are dropped; widths are kept)."""

        return {
            "config": self.config,
            "history": self.history,
            "label_history": self.label_history,
            "closed_form": self.closed_form,
            "dinkelbach": self.dinkelbach,
            "diagnostics": self.diagnostics,
            "representation_width": {
                k: int(v.shape[1]) for k, v in self.representations.items()
            },
            "level_rank": {k: int(v.rank) for k, v in self.levels.items()},
        }


class Learning(ABC):
    """Common interface and shared training procedures."""

    def __init__(self, config: LearningConfig):
        self.config = config
        self.model_: "nn.Module | None" = None
        self.label_head_: "LabelHead | None" = None
        self._geometry_cache: dict = {}

    # -- subclass contract -------------------------------------------------- #
    @abstractmethod
    def build_model(self, feature_dim: int, dtype: torch.dtype) -> nn.Module:
        """Construct the representation model for this architecture."""

    @abstractmethod
    def encode(self, model: nn.Module, graph: Graph, geometry) -> torch.Tensor:
        """Map one graph to its node representation ``Z``."""

    def fit_closed_form(self, contexts: "list[GraphContext]") -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form solver; this should have been "
            "rejected by LearningConfig validation"
        )

    # -- shared helpers ----------------------------------------------------- #
    def geometry_of(self, graph: Graph):
        cached = self._geometry_cache.get(graph.graph_id)
        if cached is None:
            cached = screened_geometry(graph.a_hat, graph.adjacency)
            self._geometry_cache[graph.graph_id] = cached
        return cached

    def represent(self, graph: Graph) -> torch.Tensor:
        """``Z`` for any graph under the learned model (inductive transfer)."""

        if self.model_ is None:
            raise RuntimeError("call run(...) before represent(...)")
        if graph.feature_dim != self.feature_dim_:
            raise ValueError(
                f"graph {graph.graph_id!r} has feature dimension {graph.feature_dim} "
                f"but the model was trained on {self.feature_dim_}"
            )
        self.model_.eval()
        with torch.no_grad():
            return self.encode(self.model_, graph, self.geometry_of(graph))

    def level_of(self, graph: Graph, Z: "torch.Tensor | None" = None) -> ScreenedLevel:
        """The whitened screened level of ``graph`` (computing ``Z`` if needed)."""

        c = self.config
        Z = self.represent(graph) if Z is None else Z
        return screened_level(
            Z,
            self.geometry_of(graph),
            c.tau,
            ridge=c.ridge,
            ridge_relative=c.ridge_relative,
        )

    def build_context(self, graph: Graph, index: int) -> GraphContext:
        c = self.config
        train_sets = [g.nodes for g in graph.train_groups]
        if not train_sets:
            raise ValueError(
                f"training graph {graph.graph_id!r} has no training groups"
            )
        targets, mask = graph.node_group_labels("train")
        return GraphContext(
            graph=graph,
            geometry=self.geometry_of(graph),
            edges=build_edge_sets(graph.adjacency, train_sets),
            host_index=select_host_nodes(
                graph.adjacency, train_sets, c.objective, graph_index=index
            ),
            label_targets=targets,
            label_nodes=torch.nonzero(mask, as_tuple=False).flatten(),
            train_sets=train_sets,
            heldout_sets=[g.nodes for g in graph.test_groups],
        )

    def _loss_on(self, ctx: GraphContext, rho: "float | None") -> LossComponents:
        level = self.level_of(
            ctx.graph, self.encode(self.model_, ctx.graph, ctx.geometry)
        )
        return evaluate_objective(
            level,
            ctx.edges,
            self.config.objective,
            host_index=ctx.host_index,
            label_head=self.label_head_,
            label_targets=ctx.label_targets,
            label_nodes=ctx.label_nodes,
            rho=rho,
        )

    # -- the public entry point -------------------------------------------- #
    def run(self, training_graphs: "list[Graph]") -> LearningResult:
        """Fit the representation (and optionally the label head) on ``training_graphs``."""

        if not training_graphs:
            raise ValueError("run() needs at least one training graph")
        dims = {g.feature_dim for g in training_graphs}
        if len(dims) > 1:
            raise ValueError(
                f"training graphs disagree on feature dimension: {sorted(dims)}"
            )
        self.feature_dim_ = training_graphs[0].feature_dim
        classes = {g.label for graph in training_graphs for g in graph.train_groups}
        self.num_classes_ = int(max(classes)) + 1 if classes else 2
        if self.num_classes_ < 2:
            self.num_classes_ = 2

        torch.manual_seed(self.config.seed)
        dtype = training_graphs[0].features.dtype
        self.model_ = self.build_model(self.feature_dim_, dtype)
        contexts = [self.build_context(g, i) for i, g in enumerate(training_graphs)]

        closed_form_report = None
        dinkelbach_report = None
        history: list = []
        label_history: list = []

        if self.config.training_mode == "closed_form":
            closed_form_report = self.fit_closed_form(contexts)
            if self.config.objective.label_enabled:
                label_history = self._fit_frozen_label_head(contexts)
        else:
            history, dinkelbach_report = self._train_gradient(contexts)
            if self.config.objective.label_enabled:
                label_history = [
                    {k: v for k, v in row.items() if k in ("epoch", "label_loss")}
                    for row in history
                ]

        representations, levels = {}, {}
        for ctx in contexts:
            Z = self.represent(ctx.graph)
            representations[ctx.graph_id] = Z
            levels[ctx.graph_id] = self.level_of(ctx.graph, Z)

        return LearningResult(
            model=self.model_,
            representations=representations,
            levels=levels,
            history=history,
            label_head=self.label_head_,
            label_history=label_history,
            closed_form=closed_form_report,
            dinkelbach=dinkelbach_report,
            config=self.config.to_dict(),
            diagnostics={
                "feature_dim": self.feature_dim_,
                "num_classes": self.num_classes_,
                "train_graphs": [c.graph_id for c in contexts],
                "edge_counts": {c.graph_id: c.edges.counts for c in contexts},
                "host_sizes": {c.graph_id: int(c.host_index.numel()) for c in contexts},
            },
        )

    # -- gradient training -------------------------------------------------- #
    def _make_label_head(self, width: int, dtype) -> LabelHead:
        return LabelHead(width, self.num_classes_, dtype=dtype)

    def _diagnostic_row(self, ctx: GraphContext) -> dict:
        """Capture / confusability on the sampled graph (reporting only).

        Held-out capture uses the *node sets* of the held-out groups, never their
        labels, and is never consulted to select an iterate.
        """

        with torch.no_grad():
            level = self.level_of(ctx.graph)
            train = group_capture(level, ctx.geometry, ctx.train_sets)
            chi = group_confusability_bound(level, ctx.geometry, ctx.train_sets)
            held = (
                group_capture(level, ctx.geometry, ctx.heldout_sets)
                if ctx.heldout_sets
                else torch.zeros(0, dtype=level.values.dtype)
            )
        return {
            "train_capture": float(train.mean()) if train.numel() else float("nan"),
            "train_capture_min": float(train.min()) if train.numel() else float("nan"),
            "heldout_capture": float(held.mean()) if held.numel() else float("nan"),
            "confusability": float(chi.max()) if chi.numel() else float("nan"),
        }

    def _selection_score(self, row: dict) -> float:
        """Comparable "lower is better" score for best-iterate tracking.

        Under Dinkelbach ``total_loss`` is the *linearized* surrogate
        ``-(boundary - rho_t * internal)``, and ``rho_t`` is re-measured at every
        outer iteration.  Its minimum therefore drifts upward toward zero even
        while the true ratio improves, so ranking by it would always return an
        iterate from the first outer stage.  Rank the ratio form by the ratio.
        """

        c = self.config
        if not c.objective.is_ratio:
            return float(row["total_loss"])
        ratio = float(row["ratio"])
        if not math.isfinite(ratio):
            return math.inf
        return (
            -ratio
            + c.objective.host_weight * float(row["host_loss"])
            + c.objective.label_weight * float(row["label_loss"])
        )

    def _aggregate_loss(self, contexts, drawn, rho):
        if drawn is not None:
            return self._loss_on(drawn, rho), drawn
        parts = [self._loss_on(ctx, rho) for ctx in contexts]
        if self.config.multi_graph_mode == "min":  # worst graph only (maximin)
            worst = int(np.argmax([float(p.total) for p in parts]))
            return parts[worst], contexts[worst]
        total = torch.stack([p.total for p in parts]).mean()
        worst = int(np.argmax([float(p.total) for p in parts]))
        merged = LossComponents(
            total=total,
            edge_objective=torch.stack([p.edge_objective for p in parts]).mean(),
            boundary=torch.stack([p.boundary for p in parts]).mean(),
            internal=torch.stack([p.internal for p in parts]).mean(),
            ratio=float(np.mean([p.ratio for p in parts])),
            host=torch.stack([p.host for p in parts]).mean(),
            label=torch.stack([p.label for p in parts]).mean(),
            rho=None if rho is None else float(rho),
        )
        return merged, contexts[worst]

    def _train_gradient(self, contexts: "list[GraphContext]"):
        c = self.config
        dtype = contexts[0].graph.features.dtype
        if c.objective.label_enabled:
            width = self.level_of(contexts[0].graph).width
            self.label_head_ = self._make_label_head(width, dtype)

        parameters = list(self.model_.parameters())
        if c.freeze_representation:
            for p in parameters:
                p.requires_grad_(False)
            parameters = []
        if self.label_head_ is not None:
            parameters = parameters + list(self.label_head_.parameters())
        if not parameters:
            raise ValueError(
                "nothing to optimize: freeze_representation=True with no label head"
            )
        if c.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(parameters, lr=c.learning_rate, max_iter=5)
        else:
            optimizer = torch.optim.Adam(
                parameters, lr=c.learning_rate, weight_decay=c.weight_decay
            )

        rng = np.random.default_rng(c.seed)
        ratio_form = c.objective.is_ratio
        outer = c.dinkelbach_iters if ratio_form else 1
        inner = max(1, c.epochs // outer)
        rho = 0.0
        rho_trace: list = []
        history: list = []
        best = {"total": math.inf, "state": None}
        epoch = 0

        bar = tqdm(total=outer * inner, desc=f"fitting {c.architecture}", leave=False)
        for outer_iter in range(outer):
            if ratio_form:
                with torch.no_grad():
                    probe, _ = self._aggregate_loss(contexts, None, rho=0.0)
                internal = float(probe.internal)
                rho = float(probe.boundary) / internal if internal > 0 else 0.0
                rho_trace.append(rho)
            for inner_step in range(inner):
                drawn = (
                    contexts[int(rng.integers(len(contexts)))]
                    if c.multi_graph_mode == "sample"
                    else None
                )
                self.model_.train()

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self._aggregate_loss(
                        contexts, drawn, rho if ratio_form else None
                    )
                    loss.total.backward()
                    return loss.total

                if c.optimizer == "lbfgs":
                    optimizer.step(closure)
                    with torch.no_grad():
                        loss, binding = self._aggregate_loss(
                            contexts, drawn, rho if ratio_form else None
                        )
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss, binding = self._aggregate_loss(
                        contexts, drawn, rho if ratio_form else None
                    )
                    loss.total.backward()
                    optimizer.step()

                self.model_.eval()
                row = {
                    "epoch": epoch,
                    "outer_iteration": outer_iter,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "graph_id": binding.graph_id,
                    **loss.to_dict(),
                }
                measured = epoch % c.diagnostic_interval == 0 or inner_step == inner - 1
                if measured:
                    row.update(self._diagnostic_row(binding))
                else:
                    row.update(
                        train_capture=float("nan"),
                        train_capture_min=float("nan"),
                        heldout_capture=float("nan"),
                        confusability=float("nan"),
                    )
                history.append(row)

                # Iterate selection uses the TRAINING objective only, and never
                # the single sampled graph -- a lucky draw is not an improvement.
                if measured:
                    scored = row
                    if drawn is not None and len(contexts) > 1:
                        with torch.no_grad():
                            full, _ = self._aggregate_loss(
                                contexts, None, rho if ratio_form else None
                            )
                        scored = full.to_dict()
                        row.update({f"all_graphs_{k}": v for k, v in scored.items()})
                    score = self._selection_score(scored)
                else:
                    score = float("nan")
                row["selection_score"] = score
                if score < best["total"]:
                    best = {
                        "total": score,
                        "epoch": epoch,
                        "ratio": float(scored["ratio"]),
                        "state": {
                            k: v.detach().clone()
                            for k, v in self.model_.state_dict().items()
                        },
                        "head": (
                            None
                            if self.label_head_ is None
                            else {
                                k: v.detach().clone()
                                for k, v in self.label_head_.state_dict().items()
                            }
                        ),
                    }
                epoch += 1
                bar.update(1)
        bar.close()

        if best["state"] is not None:
            self.model_.load_state_dict(best["state"])
            if self.label_head_ is not None and best.get("head") is not None:
                self.label_head_.load_state_dict(best["head"])
        self.model_.eval()

        scores = [
            r["selection_score"]
            for r in history
            if math.isfinite(r.get("selection_score", math.nan))
        ]
        report = None
        if ratio_form:
            converged = len(rho_trace) > 1 and abs(
                rho_trace[-1] - rho_trace[-2]
            ) <= c.dinkelbach_tol * max(1.0, abs(rho_trace[-1]))
            report = {
                "outer_iterations": outer,
                "inner_epochs_per_iteration": inner,
                "rho_trace": rho_trace,
                "rho_final": rho_trace[-1] if rho_trace else None,
                "rho_converged": bool(converged),
                "inner_converged": _tail_settled(scores),
                "best_epoch": best.get("epoch"),
                "best_ratio": best.get("ratio"),
            }
        else:
            report = {
                "outer_iterations": 1,
                "inner_epochs_per_iteration": inner,
                "inner_converged": _tail_settled(scores),
                "best_epoch": best.get("epoch"),
            }
        self._warn_about_the_trajectory(history)
        return history, report

    def _warn_about_the_trajectory(self, history: "list[dict]") -> None:
        """Flag the degenerate optimum the edge objective admits.

        ``internal -> 0`` can be bought two ways: by separating the groups from
        their surroundings (what we want) or by shrinking the groups onto the
        origin of the level, which costs nothing under the whitening and detects
        nothing.  Capture is what tells the two apart, so it is checked here.
        """

        measured = [
            r for r in history if math.isfinite(r.get("train_capture", math.nan))
        ]
        if len(measured) < 2:
            return
        first, last = measured[0], measured[-1]
        start, end = first["train_capture"], last["train_capture"]
        if end < 0.5 * start:
            LOGGER.warning(
                f"the training groups are collapsing onto the origin of the level: "
                f"capture {start:.4f} -> {end:.4f} while the boundary/internal ratio "
                f"reached {last['ratio']:.4g}.  The objective was satisfied by "
                "removing the groups' contrast, not by separating them, so this "
                "representation will not detect anything.  Add weight_decay or "
                "dropout, use the difference form (a numeric gamma), or raise "
                "host_weight."
            )
        held = last["heldout_capture"]
        if math.isfinite(held) and end > 0 and held > 1.5 * end:
            LOGGER.warning(
                f"the model is fitting the *identities* of the training groups: "
                f"their capture ({end:.4f}) is well below the held-out groups' "
                f"({held:.4f}), so what was learned is specific to the groups it "
                f"was shown.  Reduce capacity (num_heads, hidden_dim, num_layers) "
                "or regularize; the polynomial bank cannot do this by construction."
            )

    # -- frozen-level label head (closed-form path) ------------------------- #
    def _fit_frozen_label_head(self, contexts: "list[GraphContext]") -> list:
        """Fit only the head on the frozen closed-form level, by gradient descent."""

        c = self.config
        frozen = []
        for ctx in contexts:
            with torch.no_grad():
                level = self.level_of(ctx.graph)
            frozen.append((level.values.detach(), ctx))
        width = frozen[0][0].shape[1]
        dtype = frozen[0][0].dtype
        self.label_head_ = self._make_label_head(width, dtype)
        optimizer = torch.optim.Adam(
            self.label_head_.parameters(), lr=c.label_head_learning_rate
        )
        rng = np.random.default_rng(c.seed + 1)
        history: list = []
        bar = tqdm(range(c.label_head_epochs), desc="fitting label head", leave=False)
        for epoch in bar:
            values, ctx = frozen[int(rng.integers(len(frozen)))]
            optimizer.zero_grad(set_to_none=True)
            logits = self.label_head_(values[ctx.label_nodes])
            loss = torch.nn.functional.cross_entropy(
                logits, ctx.label_targets[ctx.label_nodes]
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy = float(
                    (logits.argmax(1) == ctx.label_targets[ctx.label_nodes])
                    .to(torch.float64)
                    .mean()
                )
            history.append(
                {
                    "epoch": epoch,
                    "label_loss": float(loss),
                    "train_accuracy": accuracy,
                    "graph_id": ctx.graph_id,
                }
            )
        bar.close()
        return history

    # -- prediction --------------------------------------------------------- #
    def predict_proba(self, graph: Graph) -> torch.Tensor:
        """Per-node class probabilities (softmax applied only here)."""

        if self.label_head_ is None:
            raise RuntimeError("label prediction is disabled for this learner")
        return self.label_head_.probabilities(self.level_of(graph).values)


def _tail_settled(
    values: "list[float]", fraction: float = 0.1, tol: float = 0.02
) -> bool:
    """True when the last ``fraction`` of the trace drifts < ``tol`` of its range."""

    trace = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if trace.size < 20:
        return False
    span = max(float(trace.max() - trace.min()), 1e-12)
    k = max(1, trace.size // 10)
    drift = abs(float(trace[-k:].mean() - trace[-2 * k : -k].mean())) / span
    return bool(drift < tol)
