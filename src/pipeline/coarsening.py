r"""Coarsening: one hierarchy, one trajectory, one operating-point selection.

Three merge rules are supported, each reusing its existing implementation:

``ward_tree``      contiguity-constrained Ward on the ``M_tau``-orthonormal
                   target basis (:func:`src.ward_pr_sweep.ward_order`)
``raw_ward``       the rank-``q`` raw-Ward score ``s^0_R`` on the screened level
                   (:func:`src.raw_ward.raw_ward`)
``deflated_ward``  the screened-consistent deflated dual-Ward agglomeration
                   (:func:`src.deflated_coarsen.deflated_coarsen`)

Hierarchy construction and operating-point selection are strictly separated: the
hierarchy always runs from ``N`` singletons down to a single block, and the cut
rule is applied afterwards to the stored tree.  A merge rule that only merges
*adjacent* blocks stalls at the number of connected components; the remaining
blocks are then joined by ``completion merges`` scored by the mass-weighted Ward
increment on the screened level.  Completion merges are counted and reported, so
they are never mistaken for constrained ones.

Epsilon
-------
One axis for all three methods: the Loukas Def. 2 restricted-spectral-approximation
constant of the realized *uniform* block averaging, measured in ``M_tau``
(:func:`src.loukas_sgc_detection._exact_rsa_epsilon`) -- the repository's common
comparison axis, since each coarsener otherwise reports epsilon in its own
convention.  It is computed exactly at at most ``exact_epsilon_budget``
adaptively placed levels and interpolated at the rest; every row is flagged
``epsilon_is_exact``.  The selected operating point's epsilon is always
recomputed exactly before reporting.  Precision, recall, F1 and detection rate
are never interpolated -- they are exact at every level.  ``deflated_ward``
additionally reports its own ``eps_Q``, ``mu`` and the sandwich
``eps_Q <= eps_Pi <= mu eps_Q``, whose ``eps_Pi`` is the *degree-weighted*
constant the theory is stated in.

Cut rules
---------
``f1``         coarsest-F1 level over the *training* groups (training graphs only)
``epsilon``    coarsest level whose common epsilon is within ``epsilon_budget``
``epsilon_q``  the same, on ``eps_Q(P) = sqrt(lambda_max(H_P^tau))`` -- the exact
               screened RSA of the **harmonic** reconstruction
               (:func:`src.deflated_coarsen.harmonic_rsa_epsilon`) rather than of
               the uniform block average.  It is the constant the deflated
               sandwich ``eps_Q <= eps_Pi <= mu eps_Q`` is stated in, and it is
               defined for any partition, so it is available under every merge
               rule.  One eigensolve per level, so its axis is only materialized
               when a rule reads it.  It is *not* comparable term by term with
               ``epsilon``, which scores a different reconstruction.
``score_sum``  the same, on the **cumulative merge cost** ``sum_{s<=t} score_s``
               normalized by the hierarchy's total -- a purely intrinsic stopping
               rule that needs no spectral evaluation at all.  Normalizing puts it
               in ``[0, 1]`` so it shares the ``epsilon_budget`` knob with the two
               epsilon rules; the unnormalized sum is reported as ``score_sum_raw``.
``reduction``  the level closest to a target kept fraction

All three budgeted rules are monotone along nested partitions, so the operating
point is found by bisection on the **exact** axis value, never off the
interpolated trajectory.  ``transfer_cut_rule`` may be any of them: a learned
:class:`CutRule` carries all four coordinates of its level.

Conventions
-----------
``reduction = 1 - n_coarse / n_original`` -- the fraction of nodes **removed**.
The retained fraction is reported separately as ``retained``.

Precision / recall / F1 are reported **both** ways: ``mean_*`` are the
repository's per-group macro averages (the primary numbers), ``micro_*`` pool
node counts across groups.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from src.utils.utils import LOGGER

__all__ = [
    "CoarseningConfig",
    "CutRule",
    "CutEvaluation",
    "Hierarchy",
    "CoarseningResult",
    "Coarsening",
]

_METHODS = ("ward_tree", "raw_ward", "deflated_ward")
_CUT_RULES = ("epsilon", "epsilon_q", "score_sum", "reduction", "f1")
#: rules that stop at the coarsest level whose axis is still within ``epsilon_budget``
_BUDGETED_RULES = ("epsilon", "epsilon_q", "score_sum")
_TRANSFERABLE_RULES = _BUDGETED_RULES + ("reduction",)


@dataclass
class CoarseningConfig:
    """Validated configuration of the coarsening component."""

    method: str = "deflated_ward"
    tau: float = 0.5
    cut_rule: str = "f1"
    epsilon_budget: float = 0.5
    reduction: float = 0.7
    detection_threshold: float = 0.51
    exact_epsilon_budget: int = 200
    max_cluster_size: int = 0
    deflated_hops: int = 2
    deflated_max_ball: int = 32
    deflated_max_rescore: int = 8
    deflated_fanout: int = 32
    transfer_cut_rule: str = "epsilon"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}")
        if self.cut_rule not in _CUT_RULES:
            raise ValueError(
                f"cut_rule must be one of {_CUT_RULES}, got {self.cut_rule!r}"
            )
        if self.transfer_cut_rule not in _TRANSFERABLE_RULES:
            raise ValueError(
                f"transfer_cut_rule must be one of {_TRANSFERABLE_RULES}: an F1 "
                "rule would need the transfer graph's own labels"
            )
        if self.tau <= 0.0:
            raise ValueError("tau must be strictly positive")
        if not 0.0 <= self.reduction < 1.0:
            raise ValueError("reduction must lie in [0, 1) (fraction of nodes removed)")
        if self.epsilon_budget < 0.0:
            raise ValueError("epsilon_budget must be non-negative")
        if self.exact_epsilon_budget < 0:
            raise ValueError("exact_epsilon_budget must be non-negative")


@dataclass
class CutRule:
    """An operating point learned on training data and transferable as-is.

    All four transferable coordinates of the selected level are carried, so
    ``transfer_cut_rule`` can pick whichever axis is still binding on the
    receiving graph.
    """

    epsilon: float
    reduction: float
    n_coarse: int
    source_graph: str
    rule: str = "f1"
    epsilon_q: float = float("nan")
    score_sum: float = float("nan")

    def budget_for(self, transfer_rule: str) -> float:
        """The coordinate ``transfer_rule`` should reproduce on another graph."""

        value = {
            "epsilon": self.epsilon,
            "epsilon_q": self.epsilon_q,
            "score_sum": self.score_sum,
            "reduction": self.reduction,
        }[transfer_rule]
        if not math.isfinite(value):
            raise ValueError(
                f"the learned cut rule carries no {transfer_rule!r} coordinate; it "
                "is only measured when transfer_cut_rule asks for it"
            )
        return float(value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CutEvaluation:
    """One operating point of a hierarchy, fully evaluated.

    ``mode`` is ``"stopping_rule"`` (the deployable cut, chosen without test
    labels) or ``"oracle"`` (the post-hoc upper bound: the level maximizing F1
    over this graph's evaluation groups, which *does* read their labels and must
    never feed back into training or the stopping rule).
    """

    mode: str
    rule: str
    merges: int
    n_coarse: int
    reduction: float
    retained: float
    epsilon: float
    node_to_supernode: torch.Tensor
    metrics: dict
    per_group: list
    row: dict

    def to_dict(self) -> dict:
        out = {
            "mode": self.mode,
            "rule": self.rule,
            "n_coarse": self.n_coarse,
            "reduction": self.reduction,
            "retained": self.retained,
            "epsilon": self.epsilon,
            "metrics": self.metrics,
        }
        for axis in ("epsilon_q", "score_sum", "score_sum_raw"):
            if axis in self.row:
                out[axis] = float(self.row[axis])
        return out


@dataclass
class Hierarchy:
    """A complete merge tree from ``N`` singletons to one block."""

    children: np.ndarray
    scores: np.ndarray
    n_leaves: int
    method: str
    constrained_merges: int

    @property
    def completion_merges(self) -> int:
        return int(self.children.shape[0]) - int(self.constrained_merges)

    def labels_at(self, n_clusters: int) -> np.ndarray:
        """Contiguous supernode labels after cutting the tree to ``n_clusters``."""

        return _labels_after(
            self.children, self.n_leaves, self.n_leaves - int(n_clusters)
        )


@dataclass
class CoarseningResult:
    """Hierarchy, full trajectory, and both evaluated operating points."""

    hierarchy: Hierarchy
    trajectory: list
    stopping_rule: CutEvaluation
    oracle: CutEvaluation
    pr_auc: float
    levels: int
    cut_rule: "CutRule | None"
    target_rank: int = 0
    deflated_certificate: "dict | None" = None
    config: dict = field(default_factory=dict)

    # the stopping rule is the deployable answer, so it is what the bare
    # attributes refer to
    @property
    def selected(self) -> dict:
        return self.stopping_rule.row

    @property
    def metrics(self) -> dict:
        return self.stopping_rule.metrics

    @property
    def per_group(self) -> list:
        return self.stopping_rule.per_group

    @property
    def node_to_supernode(self) -> torch.Tensor:
        return self.stopping_rule.node_to_supernode

    @property
    def n_coarse(self) -> int:
        return int(self.stopping_rule.n_coarse)

    def cut(self, mode: str) -> CutEvaluation:
        return self.oracle if mode == "oracle" else self.stopping_rule

    def to_serializable(self) -> dict:
        return {
            "config": self.config,
            "method": self.hierarchy.method,
            "n_original": self.hierarchy.n_leaves,
            "constrained_merges": self.hierarchy.constrained_merges,
            "completion_merges": self.hierarchy.completion_merges,
            "levels": self.levels,
            "pr_auc": self.pr_auc,
            "target_rank": self.target_rank,
            "stopping_rule": self.stopping_rule.to_dict(),
            "oracle": {
                **self.oracle.to_dict(),
                "note": "post-hoc upper bound; reads this graph's evaluation "
                "labels and never influences training or the stopping rule",
            },
            "selected": self.selected,
            "metrics": self.metrics,
            "per_group": self.per_group,
            "cut_rule": None if self.cut_rule is None else self.cut_rule.to_dict(),
            "deflated_certificate": self.deflated_certificate,
        }


# --------------------------------------------------------------------------- #
# tree utilities
# --------------------------------------------------------------------------- #
def _labels_after(children: np.ndarray, n: int, merges: int) -> np.ndarray:
    merges = max(0, min(int(merges), int(children.shape[0])))
    parent = np.arange(n + merges, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for t in range(merges):
        a, b = int(children[t, 0]), int(children[t, 1])
        parent[find(a)] = n + t
        parent[find(b)] = n + t
    roots = np.fromiter((find(v) for v in range(n)), dtype=np.int64, count=n)
    _, inverse = np.unique(roots, return_inverse=True)
    return inverse.astype(np.int64)


def _to_scipy(adjacency: torch.Tensor) -> sp.csr_matrix:
    coalesced = adjacency.coalesce()
    idx = coalesced.indices().cpu().numpy()
    val = coalesced.values().detach().cpu().numpy().astype(np.float64)
    n = int(coalesced.shape[0])
    return sp.coo_matrix((val, (idx[0], idx[1])), shape=(n, n)).tocsr()


# --------------------------------------------------------------------------- #
# metric replay
# --------------------------------------------------------------------------- #
class _Tracker:
    """Incremental per-group detection metrics along a merge order.

    Each group keeps its own node counts per supernode, so overlapping groups are
    tracked independently.  Updating a merge touches only the groups present in
    the two merged blocks.
    """

    def __init__(self, groups, n: int, threshold: float):
        self.groups = list(groups)
        self.n = int(n)
        self.threshold = float(threshold)
        total = 2 * n - 1
        self.size = np.ones(total, dtype=np.int64)
        self.parent = np.arange(total, dtype=np.int64)
        self.counts = [dict() for _ in self.groups]
        self.best_count = np.ones(len(self.groups), dtype=np.int64)
        self.best_root = np.zeros(len(self.groups), dtype=np.int64)
        self.group_size = np.array(
            [int(g.nodes.numel()) for g in self.groups], dtype=np.int64
        )
        self.members: list = [None] * total
        for j, group in enumerate(self.groups):
            nodes = group.nodes.tolist()
            for v in nodes:
                self.counts[j][v] = 1
                if self.members[v] is None:
                    self.members[v] = {j}
                else:
                    self.members[v].add(j)
            self.best_root[j] = int(nodes[0])
        self.splits = sorted({g.split for g in self.groups})
        self.index_of = {
            split: [j for j, g in enumerate(self.groups) if g.split == split]
            for split in self.splits
        }
        # "all" pools every evaluated group of this graph: the oracle cut is the
        # level that maximizes F1 over it, and the summary tables report it.
        self.index_of["all"] = list(range(len(self.groups)))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def group_metrics(self, j: int) -> tuple:
        best = int(self.best_count[j])
        recall = best / max(int(self.group_size[j]), 1)
        precision = best / max(int(self.size[self.best_root[j]]), 1)
        f1 = (
            2 * recall * precision / (recall + precision) if recall + precision else 0.0
        )
        detected = recall > self.threshold and precision > self.threshold
        return recall, precision, f1, float(detected)

    def snapshot(self) -> dict:
        out = {}
        for split, indices in self.index_of.items():
            if not indices:
                continue
            rows = np.array([self.group_metrics(j) for j in indices], dtype=np.float64)
            hit = int(self.best_count[indices].sum())
            total_nodes = int(self.group_size[indices].sum())
            blocks = int(sum(self.size[self.best_root[j]] for j in indices))
            micro_r = hit / max(total_nodes, 1)
            micro_p = hit / max(blocks, 1)
            micro_f1 = (
                2 * micro_r * micro_p / (micro_r + micro_p)
                if micro_r + micro_p
                else 0.0
            )
            out[split] = {
                "mean_recall": float(rows[:, 0].mean()),
                "mean_precision": float(rows[:, 1].mean()),
                "mean_f1": float(rows[:, 2].mean()),
                "detection_rate": float(rows[:, 3].mean()),
                "detected": int(rows[:, 3].sum()),
                "total": len(indices),
                "micro_recall": micro_r,
                "micro_precision": micro_p,
                "micro_f1": micro_f1,
            }
        return out

    def apply_merge(self, index: int, a: int, b: int) -> None:
        node = self.n + index
        ra, rb = self.find(a), self.find(b)
        self.parent[ra] = node
        self.parent[rb] = node
        self.size[node] = self.size[ra] + self.size[rb]
        left, right = self.members[ra], self.members[rb]
        touched = (left or set()) | (right or set()) if (left or right) else None
        self.members[node] = touched
        self.members[ra] = self.members[rb] = None
        if not touched:
            return
        for j in touched:
            ca = self.counts[j].pop(ra, 0)
            cb = self.counts[j].pop(rb, 0)
            merged = ca + cb
            if merged:
                self.counts[j][node] = merged
            if merged >= self.best_count[j]:
                self.best_count[j] = merged
                self.best_root[j] = node

    def per_group_rows(self) -> list:
        rows = []
        for j, group in enumerate(self.groups):
            recall, precision, f1, detected = self.group_metrics(j)
            rows.append(
                {
                    "group_id": group.group_id,
                    "split": group.split,
                    "group_type": group.group_type,
                    "size": int(self.group_size[j]),
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "detected": bool(detected),
                    "supernode_size": int(self.size[self.best_root[j]]),
                }
            )
        return rows


# --------------------------------------------------------------------------- #
# the component
# --------------------------------------------------------------------------- #
class Coarsening:
    """Builds the hierarchy, the trajectory and the selected operating point."""

    def __init__(self, config: CoarseningConfig):
        self.config = config

    # -- hierarchy ---------------------------------------------------------- #
    def build_hierarchy(self, graph, basis: torch.Tensor) -> "tuple[Hierarchy, dict]":
        c = self.config
        method = c.method
        n = int(graph.num_nodes)
        if method == "ward_tree":
            from src.ward_pr_sweep import ward_order

            children, distances, a0, metric = ward_order(
                graph.adjacency, basis, c.tau, laplacian="symmetric"
            )
            children = np.asarray(children, dtype=np.int64)
            scores = np.asarray(distances, dtype=np.float64)
            context = {"a0": a0, "metric": metric}
        elif method == "raw_ward":
            from src.raw_ward import raw_ward

            result = raw_ward(
                _to_scipy(graph.adjacency),
                basis.detach().cpu().numpy(),
                c.tau,
                geometry="symmetric",
                n_clusters=None,
                build_full_tree=True,
                max_cluster_size=c.max_cluster_size,
            )
            children = np.asarray(result.children_, dtype=np.int64).reshape(-1, 2)
            scores = np.array(
                [float(r["score"]) for r in result.merge_records_], dtype=np.float64
            )
            context = {}
        else:
            from src.deflated_coarsen import deflated_coarsen

            result = deflated_coarsen(
                _to_scipy(graph.adjacency),
                basis.detach().cpu().numpy(),
                c.tau,
                rule="dual-ward",
                n_clusters=None,
                build_full_tree=True,
                hops=c.deflated_hops,
                max_ball=c.deflated_max_ball,
                max_rescore=c.deflated_max_rescore,
                fanout=c.deflated_fanout,
                max_cluster_size=c.max_cluster_size,
                track_euclidean=False,
                record_curve=False,
            )
            children = np.asarray(result.children_, dtype=np.int64).reshape(-1, 2)
            scores = np.array(
                [float(r.get("score", float("nan"))) for r in result.merge_records_],
                dtype=np.float64,
            )
            context = {}

        constrained = int(children.shape[0])
        if constrained < n - 1:
            children, scores = self._complete(graph, basis, children, scores, n)
        return (
            Hierarchy(
                children=children,
                scores=scores,
                n_leaves=n,
                method=method,
                constrained_merges=constrained,
            ),
            context,
        )

    def _complete(self, graph, basis, children, scores, n):
        """Join the remaining (mutually non-adjacent) blocks down to one.

        A connectivity-constrained rule stops at the component count; the
        specification requires a hierarchy reaching two supernodes, so the
        leftovers are merged by the mass-weighted Ward increment on the screened
        level -- the same quantity the raw-Ward score is built from, without the
        adjacency constraint.
        """

        from src.pipeline.geometry import screened_geometry, screened_level

        geometry = screened_geometry(graph.a_hat, graph.adjacency)
        level = screened_level(basis, geometry, self.config.tau)
        values = level.values.detach().cpu().numpy()
        nodes = torch.arange(n)
        mass = (geometry.node_weights(nodes) ** 2).detach().cpu().numpy()

        labels = _labels_after(children, n, int(children.shape[0]))
        blocks = int(labels.max()) + 1
        block_mass = np.bincount(labels, weights=mass, minlength=blocks)
        block_mean = np.zeros((blocks, values.shape[1]))
        np.add.at(block_mean, labels, mass[:, None] * values)
        block_mean /= np.maximum(block_mass, 1e-300)[:, None]
        # tree ids of the surviving roots, in label order
        roots = np.full(blocks, -1, dtype=np.int64)
        parent = np.arange(n + int(children.shape[0]), dtype=np.int64)
        for t in range(int(children.shape[0])):
            parent[int(children[t, 0])] = n + t
            parent[int(children[t, 1])] = n + t
        for v in range(n):
            x = v
            while parent[x] != x:
                x = parent[x]
            roots[labels[v]] = x

        alive = list(range(blocks))
        extra_children, extra_scores = [], []
        next_id = n + int(children.shape[0])
        while len(alive) > 1:
            best, pair = math.inf, None
            for i in range(len(alive)):
                for j in range(i + 1, len(alive)):
                    a, b = alive[i], alive[j]
                    ma, mb = block_mass[a], block_mass[b]
                    delta = block_mean[a] - block_mean[b]
                    value = float(ma * mb / (ma + mb) * (delta @ delta))
                    if value < best:
                        best, pair = value, (i, j)
            i, j = pair
            a, b = alive[i], alive[j]
            extra_children.append((int(roots[a]), int(roots[b])))
            extra_scores.append(best)
            total = block_mass[a] + block_mass[b]
            block_mean[a] = (
                block_mass[a] * block_mean[a] + block_mass[b] * block_mean[b]
            ) / total
            block_mass[a] = total
            roots[a] = next_id
            next_id += 1
            alive.pop(j)
        if extra_children:
            children = np.vstack(
                [children.reshape(-1, 2), np.array(extra_children, dtype=np.int64)]
            )
            scores = np.concatenate([scores, np.array(extra_scores, dtype=np.float64)])
        return children, scores

    # -- epsilon ------------------------------------------------------------ #
    def _epsilon_axis(self, graph, basis, hierarchy: Hierarchy, context: dict):
        """``(epsilon_per_level, exact_flags, a0, metric)`` indexed by merge count."""

        from src.loukas_sgc_detection import (
            _l_orthonormalize,
            _normalized_laplacian,
            _screened_metric,
        )
        from src.ward_pr_sweep import adaptive_exact_epsilon

        a0 = context.get("a0")
        metric = context.get("metric")
        if a0 is None:
            metric = _screened_metric(
                _normalized_laplacian(graph.adjacency), self.config.tau
            )
            a0 = _l_orthonormalize(basis, metric)
        merges = int(hierarchy.children.shape[0])
        budget = self.config.exact_epsilon_budget
        if budget <= 0:
            sampled = np.array([0, merges], dtype=np.int64)
            values = np.array(
                [0.0, _exact_epsilon(a0, metric, hierarchy, merges)], dtype=float
            )
        else:
            sampled, values = adaptive_exact_epsilon(
                hierarchy.children, hierarchy.n_leaves, a0, metric, budget=budget
            )
        grid = np.arange(merges + 1)
        interpolated = np.interp(grid, sampled, values)
        exact = np.zeros(merges + 1, dtype=bool)
        exact[np.clip(sampled, 0, merges)] = True
        return interpolated, exact, a0, metric

    def _epsilon_q_evaluator(self, graph, basis, hierarchy: Hierarchy):
        """Exact ``eps_Q(t) = sqrt(lambda_max(H_P^tau))`` of the harmonic lift.

        The common ``epsilon`` axis scores the realized *uniform* block average;
        ``eps_Q`` scores the harmonic reconstruction, which is the constant the
        deflated sandwich is stated in.  It is defined for any partition, so it is
        available under every merge rule, not only ``deflated_ward``.
        """

        from src.deflated_coarsen import harmonic_rsa_epsilon
        from src.smooth_dual_ward import m_orthonormal_basis, screened_operators

        _a, d_tilde, _l, M = screened_operators(
            _to_scipy(graph.adjacency), self.config.tau
        )
        U, _rank = m_orthonormal_basis(basis.detach().cpu().numpy(), M)
        cache: dict = {}

        def evaluate(merges: int) -> float:
            merges = int(merges)
            if merges not in cache:
                labels = _labels_after(hierarchy.children, hierarchy.n_leaves, merges)
                cache[merges] = float(harmonic_rsa_epsilon(U, M, d_tilde, labels))
            return cache[merges]

        return evaluate

    def _score_sum_axis(self, hierarchy: Hierarchy):
        """Cumulative merge cost, normalized by the hierarchy's total.

        ``score_sum(t)`` is the fraction of the tree's total Ward increment spent
        by the first ``t`` merges, so it lives in ``[0, 1]`` and shares the
        ``epsilon_budget`` knob with the two epsilon rules.  The unnormalized sum
        is reported alongside as ``score_sum_raw``.
        """

        scores = np.asarray(hierarchy.scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"cut_rule='score_sum' needs a merge score for every merge, but "
                f"method={hierarchy.method!r} left "
                f"{int((~np.isfinite(scores)).sum())} of {scores.size} unscored"
            )
        raw = np.concatenate([[0.0], np.cumsum(np.maximum(scores, 0.0))])
        total = float(raw[-1])
        return raw, raw / total if total > 0.0 else np.zeros_like(raw)

    # -- trajectory --------------------------------------------------------- #
    def _trajectory(self, hierarchy, groups, columns: dict):
        """One row per level; ``columns`` maps a name to its per-merge array."""

        tracker = _Tracker(groups, hierarchy.n_leaves, self.config.detection_threshold)
        n = hierarchy.n_leaves
        rows = []

        def emit(merges):
            n_coarse = n - merges
            row = {
                "n_coarse": int(n_coarse),
                "merges": int(merges),
                "reduction": 1.0 - n_coarse / n,
                "retained": n_coarse / n,
                "merge_score": (
                    float(hierarchy.scores[merges - 1]) if merges else float("nan")
                ),
            }
            for name, values in columns.items():
                row[name] = (
                    bool(values[merges])
                    if values.dtype == bool
                    else float(values[merges])
                )
            for split, metrics in tracker.snapshot().items():
                for key, value in metrics.items():
                    row[f"{split}_{key}"] = value
            rows.append(row)

        emit(0)
        for t in range(int(hierarchy.children.shape[0])):
            tracker.apply_merge(
                t, int(hierarchy.children[t, 0]), int(hierarchy.children[t, 1])
            )
            if n - (t + 1) >= 2:
                emit(t + 1)
        return rows

    # -- cut selection ------------------------------------------------------ #
    def _warn_about_the_cut(self, graph, stopping, oracle, rule) -> None:
        """Say so when the operating point throws away most of what the tree offers.

        A cut rule can be configured far from the hierarchy's useful range -- a
        too-tight epsilon budget leaves the groups unmerged -- and the resulting
        metrics then look like a broken model rather than a misplaced cut.  The
        comparison is made on the pooled evaluation groups, so it works on
        transfer graphs too; it changes no decision.
        """

        achieved = stopping.metrics.get("all", {}).get("mean_f1")
        best = oracle.metrics.get("all", {}).get("mean_f1")
        if achieved is None or best is None:
            return
        if achieved < 0.8 * best - 1e-12:
            LOGGER.warning(
                f"[{graph.graph_id}] the {rule!r} cut sits well inside the hierarchy's "
                f"useful range: F1 {achieved:.3f} at n_coarse="
                f"{stopping.n_coarse:,} (epsilon {stopping.epsilon:.3f}), while "
                f"the same tree reaches {best:.3f} at n_coarse="
                f"{oracle.n_coarse:,} (epsilon {oracle.epsilon:.3f}).  The target "
                "subspace is not the limitation here -- the operating point is."
            )

    def _budgeted_cut(self, evaluate, hierarchy: Hierarchy, budget: float) -> int:
        """Coarsest cut whose **exact** axis value is within ``budget``.

        The trajectory's epsilon axes are interpolated between the sampled levels,
        so a cut picked off them can miss the budget by the interpolation error.
        All three budgeted axes are monotone along nested partitions, so the honest
        answer is a bisection on the exact value -- about ``log2(N)`` evaluations.
        """

        low, high = 0, max(0, hierarchy.n_leaves - 2)
        if evaluate(high) <= budget + 1e-12:
            return high
        while high - low > 1:
            middle = (low + high) // 2
            if evaluate(middle) <= budget + 1e-12:
                low = middle
            else:
                high = middle
        return low

    def _select(self, trajectory, rule, *, epsilon_budget, reduction, train_split):
        if rule == "f1":
            key = f"{train_split}_mean_f1"
            if key not in trajectory[0]:
                raise ValueError(
                    "cut_rule='f1' needs training groups on this graph; supply an "
                    "epsilon or reduction rule for transfer graphs"
                )
            return max(trajectory, key=lambda row: row[key])
        if rule in _BUDGETED_RULES:
            within = [row for row in trajectory if row[rule] <= epsilon_budget + 1e-12]
            return within[-1] if within else trajectory[0]
        target = 1.0 - float(reduction)
        return min(trajectory, key=lambda row: abs(row["retained"] - target))

    # -- run ---------------------------------------------------------------- #
    def run(
        self,
        graph,
        basis: torch.Tensor,
        *,
        train_groups: "list | None" = None,
        evaluation_groups: "list | None" = None,
        cut_rule: "CutRule | None" = None,
    ) -> CoarseningResult:
        """Coarsen ``graph`` under ``R = span(basis)`` and pick an operating point.

        ``train_groups`` (training groups of a *training* graph) are the only
        groups the ``f1`` rule may look at.  ``cut_rule``, when given, overrides
        the configured rule with an operating point learned elsewhere -- the
        transfer path.
        """

        c = self.config
        groups = (
            list(evaluation_groups)
            if evaluation_groups is not None
            else list(graph.groups)
        )
        if not groups:
            raise ValueError(f"graph {graph.graph_id!r} has no groups to evaluate")

        hierarchy, context = self.build_hierarchy(graph, basis)
        epsilon, exact_flags, a0, metric = self._epsilon_axis(
            graph, basis, hierarchy, context
        )

        if cut_rule is not None:
            rule = c.transfer_cut_rule
            budget = cut_rule.budget_for(rule)
            target = cut_rule.reduction
        else:
            rule = c.cut_rule
            budget, target = c.epsilon_budget, c.reduction
            if rule == "f1":
                allowed = {g.group_id for g in (train_groups or [])}
                if not allowed:
                    raise ValueError(
                        f"cut_rule='f1' requires train_groups; graph {graph.graph_id!r} "
                        f"supplied none (test and transfer graphs must use one of "
                        f"{_TRANSFERABLE_RULES})"
                    )
                leaked = [
                    g.group_id
                    for g in groups
                    if g.split == "train" and g.group_id not in allowed
                ]
                if leaked:
                    raise ValueError(
                        "the F1 cut rule would see groups outside train_groups: "
                        f"{leaked[:3]}"
                    )

        # eps_Q is a full eigensolve per level, so its axis is only materialized
        # when a rule -- here or on a later transfer -- actually reads it
        wants_q = "epsilon_q" in (rule, c.transfer_cut_rule)
        evaluate_q = (
            self._epsilon_q_evaluator(graph, basis, hierarchy) if wants_q else None
        )
        columns = {"epsilon": epsilon, "epsilon_is_exact": exact_flags}
        score_raw, score_norm = (
            self._score_sum_axis(hierarchy)
            if "score_sum" in (rule, c.transfer_cut_rule)
            else (None, None)
        )
        if score_norm is not None:
            columns["score_sum"] = score_norm
            columns["score_sum_raw"] = score_raw
        if evaluate_q is not None:
            q_values, q_exact = _adaptive_axis(
                evaluate_q, int(hierarchy.children.shape[0]), c.exact_epsilon_budget
            )
            columns["epsilon_q"] = q_values
            columns["epsilon_q_is_exact"] = q_exact
        trajectory = self._trajectory(hierarchy, groups, columns)

        if rule in _BUDGETED_RULES:
            if rule == "epsilon":
                evaluate = lambda t: _exact_epsilon(
                    a0, metric, hierarchy, t
                )  # noqa: E731
            elif rule == "epsilon_q":
                evaluate = evaluate_q
            else:
                evaluate = lambda t: float(score_norm[t])  # noqa: E731
            merges = self._budgeted_cut(evaluate, hierarchy, budget)
            if merges >= hierarchy.n_leaves - 2:
                source = (
                    f"the transferred {rule} {budget:.4g}"
                    if cut_rule is not None
                    else f"the {rule} budget {budget:g}"
                )
                tail = (
                    f"epsilon >= 1 as soon as n_coarse < rank(R) = "
                    f"{int(a0.shape[1]):,} (the capacity obstruction), so a wide "
                    "target saturates this axis -- set transfer_cut_rule="
                    "'reduction' to transfer the operating point by kept fraction "
                    "instead."
                    if rule in ("epsilon", "epsilon_q")
                    else "the whole tree costs less than the budget; score_sum is "
                    "normalized to [0, 1], so a budget at or above 1 never binds."
                )
                LOGGER.warning(
                    f"[{graph.graph_id}] {source} is not binding: the whole "
                    "hierarchy fits inside it, so the coarsest cut is selected and "
                    f"every group lands in one supernode.  {tail}"
                )
            selected = dict(next(row for row in trajectory if row["merges"] == merges))
        else:
            selected = dict(
                self._select(
                    trajectory,
                    rule,
                    epsilon_budget=budget,
                    reduction=target,
                    train_split="train",
                )
            )
        merges = int(selected["merges"])
        labels = _labels_after(hierarchy.children, hierarchy.n_leaves, merges)
        exact_epsilon = _exact_epsilon(a0, metric, hierarchy, merges)
        selected.update(
            rule=rule,
            epsilon=exact_epsilon,
            epsilon_is_exact=True,
            epsilon_interpolated=float(epsilon[merges]),
        )
        if evaluate_q is not None:
            selected.update(epsilon_q=evaluate_q(merges), epsilon_q_is_exact=True)
        for row in trajectory:
            if row["merges"] == merges:
                row["epsilon"] = exact_epsilon
                row["epsilon_is_exact"] = True
                if evaluate_q is not None:
                    row["epsilon_q"] = selected["epsilon_q"]
                    row["epsilon_q_is_exact"] = True
                row["selected"] = True
            else:
                row["selected"] = False

        stopping = self._evaluate_cut(
            hierarchy, groups, merges, selected, mode="stopping_rule", rule=rule
        )
        # the oracle maximizes F1 over EVERY evaluated group of this graph
        oracle_row = max(trajectory, key=lambda row: row["all_mean_f1"])
        oracle_merges = int(oracle_row["merges"])
        oracle_row = dict(oracle_row)
        oracle_row["epsilon"] = _exact_epsilon(a0, metric, hierarchy, oracle_merges)
        oracle_row["epsilon_is_exact"] = True
        oracle = self._evaluate_cut(
            hierarchy,
            groups,
            oracle_merges,
            oracle_row,
            mode="oracle",
            rule="f1_oracle",
        )
        self._warn_about_the_cut(graph, stopping, oracle, rule)

        learned_rule = None
        if cut_rule is None and rule == "f1":
            learned_rule = CutRule(
                epsilon=exact_epsilon,
                reduction=float(selected["reduction"]),
                n_coarse=int(selected["n_coarse"]),
                source_graph=graph.graph_id,
                rule="f1",
                epsilon_q=(
                    float("nan") if evaluate_q is None else float(selected["epsilon_q"])
                ),
                score_sum=(
                    float("nan") if score_norm is None else float(score_norm[merges])
                ),
            )
            if c.transfer_cut_rule in ("epsilon", "epsilon_q"):
                learned = learned_rule.budget_for(c.transfer_cut_rule)
                if learned >= 1.0:
                    LOGGER.warning(
                        f"[{graph.graph_id}] the learned {c.transfer_cut_rule}*="
                        f"{learned:.4f} is at the saturation point: it reaches 1 once "
                        f"n_coarse < rank(R) = {int(a0.shape[1]):,}, so transferring "
                        "it will not bind on another graph.  Use "
                        "transfer_cut_rule='reduction' or 'score_sum', or narrow the "
                        "target (fewer heads / lower degree) so the epsilon axis "
                        "still discriminates at the operating point."
                    )

        certificate = None
        if c.method == "deflated_ward":
            certificate = _deflated_certificate(
                graph, basis, labels, c.tau, exact_epsilon
            )

        return CoarseningResult(
            hierarchy=hierarchy,
            trajectory=trajectory,
            stopping_rule=stopping,
            oracle=oracle,
            pr_auc=_trajectory_pr_auc(trajectory),
            levels=_informative_levels(trajectory),
            cut_rule=learned_rule,
            target_rank=int(a0.shape[1]),
            deflated_certificate=certificate,
            config=asdict(c),
        )

    def _evaluate_cut(
        self, hierarchy, groups, merges, row, *, mode, rule
    ) -> CutEvaluation:
        """Materialize one level of the tree: labels, per-split metrics, per-group rows."""

        tracker = _Tracker(groups, hierarchy.n_leaves, self.config.detection_threshold)
        for t in range(int(merges)):
            tracker.apply_merge(
                t, int(hierarchy.children[t, 0]), int(hierarchy.children[t, 1])
            )
        labels = _labels_after(hierarchy.children, hierarchy.n_leaves, int(merges))
        return CutEvaluation(
            mode=mode,
            rule=rule,
            merges=int(merges),
            n_coarse=int(row["n_coarse"]),
            reduction=float(row["reduction"]),
            retained=float(row["retained"]),
            epsilon=float(row["epsilon"]),
            node_to_supernode=torch.from_numpy(labels).to(torch.long),
            metrics=tracker.snapshot(),
            per_group=tracker.per_group_rows(),
            row=dict(row),
        )


def _informative_levels(trajectory: list) -> int:
    """Levels at which the pooled metrics actually change (plus the first)."""

    keys = ("all_mean_recall", "all_mean_precision")
    changes, previous = 1, None
    for row in trajectory:
        current = tuple(row[k] for k in keys if k in row)
        if previous is not None and current != previous:
            changes += 1
        previous = current
    return changes


def _trajectory_pr_auc(trajectory: list) -> float:
    """PR-AUC over the whole hierarchy, via the repository's trapezoidal helper."""

    from src.ward_pr_sweep import pr_auc

    if "all_mean_recall" not in trajectory[0]:
        return float("nan")
    return float(
        pr_auc(
            [row["all_mean_recall"] for row in trajectory],
            [row["all_mean_precision"] for row in trajectory],
        )
    )


def _exact_epsilon(a0, metric, hierarchy: Hierarchy, merges: int) -> float:
    from src.loukas_sgc_detection import _exact_rsa_epsilon

    labels = _labels_after(hierarchy.children, hierarchy.n_leaves, merges)
    return float(
        _exact_rsa_epsilon(a0, metric, torch.from_numpy(labels).to(torch.long))
    )


def _adaptive_axis(evaluate, merges: int, budget: int, min_gap: float = 1e-3):
    """Sample a monotone curve at up to ``budget`` gap-driven levels, then interpolate.

    The same placement strategy :func:`src.ward_pr_sweep.adaptive_exact_epsilon`
    uses for the common epsilon, over an arbitrary callable, so ``eps_Q`` gets the
    same treatment without a second copy of the bisection bookkeeping.  Returns
    ``(values, exact_flags)`` indexed by merge count.
    """

    import heapq

    cache = {0: evaluate(0), merges: evaluate(merges)}
    heap: list = []

    def push(left, right):
        if right - left >= 2:
            heapq.heappush(heap, (-(cache[right] - cache[left]), left, right))

    push(0, merges)
    used = 2
    while used < budget and heap:
        gap, left, right = heapq.heappop(heap)
        if -gap <= min_gap:
            break
        middle = (left + right) // 2
        cache[middle] = evaluate(middle)
        used += 1
        push(left, middle)
        push(middle, right)

    sampled = np.array(sorted(cache), dtype=np.int64)
    values = np.array([cache[t] for t in sampled], dtype=float)
    exact = np.zeros(merges + 1, dtype=bool)
    exact[np.clip(sampled, 0, merges)] = True
    return np.interp(np.arange(merges + 1), sampled, values), exact


def _deflated_certificate(graph, basis, labels, tau, epsilon_common) -> dict:
    """``eps_Q``, ``mu`` and the RSA sandwich at the selected partition.

    The sandwich ``eps_Q <= eps_Pi <= mu eps_Q`` is stated for the *degree-weighted*
    block projector, so ``eps_Pi`` is recomputed in that convention here; the
    trajectory's ``epsilon`` (uniform block average, the common axis) is carried
    alongside as ``epsilon_common`` and is not part of the certificate.
    """

    from src.deflated_coarsen import block_distortion_mu, harmonic_rsa_epsilon
    from src.smooth_dual_ward import (
        exact_rsa_epsilon,
        m_orthonormal_basis,
        screened_operators,
    )

    W = _to_scipy(graph.adjacency)
    _a, d_tilde, _l, M = screened_operators(W, tau)
    U, _rank = m_orthonormal_basis(basis.detach().cpu().numpy(), M)
    eps_q = float(harmonic_rsa_epsilon(U, M, d_tilde, labels))
    eps_pi = float(exact_rsa_epsilon(U, M, d_tilde, labels))
    mu = float(block_distortion_mu(M, d_tilde, labels))
    return {
        "epsilon_q": eps_q,
        "epsilon_pi": eps_pi,
        "epsilon_common": float(epsilon_common),
        "mu": mu,
        "sandwich_upper": mu * eps_q,
        "sandwich_ok": bool(eps_q - 1e-9 <= eps_pi <= mu * eps_q + 1e-9),
    }
