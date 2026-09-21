r"""How planted-group density changes held-out detection.

Sweeps the synthetic ``group_density`` over ``0.00, 0.05, ..., 1.00`` (21 values,
both endpoints included) with **one** coarsening rule, ``deflated_ward_tight``,
and everything else held fixed.  Each cell is a complete
:func:`~src.pipeline.pipeline.run_pipeline` call, so the data, learning,
coarsening, evaluation, group-label and logging components are the pipeline's
own -- this module contributes the sweep, the aggregation and the figures, and
nothing else.

Why the comparison is paired
----------------------------
``build_synthetic_graph`` draws the node count, the group sizes, the group
membership and the whole Erdos-Renyi background *before* it plants any motif,
and the motif planter consumes the same number of random draws at every density
(one ``permutation(s)`` and one ``permutation(|pairs|)``, whatever the target
edge count).  So at a fixed seed the graphs at two densities differ **only** in
which intra-group edges are present: same nodes, same groups, same background,
same train/test split, same features.

What ``group_density = 0`` actually means
-----------------------------------------
Not "no internal edges".  ``group_types=("random",)`` plants a random spanning
path first so every planted group is connected, and only then adds pairs until
``round(density * s(s-1)/2)`` is reached.  At ``density = 0`` the target is
below the path, so the group keeps its ``s - 1`` path edges -- a realized
density of ``2/s``, which for ``group_size in [7, 20]`` is 0.10 to 0.29.  The
sweep therefore *saturates at the bottom*: the first few nominal densities
produce identical or near-identical graphs.  The realized density is measured
per graph and reported next to the nominal one rather than being assumed.

Outputs (``--out``)
-------------------
``density_sweep_config.json``    the resolved configuration, the density grid,
                                 the seed schedule and the fixed-value fingerprint
``density_sweep_per_graph.csv``  one row per (density, seed, graph, mode)
``density_sweep_per_seed.csv``   one row per (density, seed, mode), test graphs pooled
``density_sweep_summary.csv``    one row per (density, mode): mean, sd, n_runs
``density_sweep_results.json``   all of the above plus failures and undefined metrics
``figures/``                     the summary panel, the per-metric figures and the gap plot

Run::

    python -m src.run_synthetic_density_sweep --smoke        # validation only
    python -m src.run_synthetic_density_sweep --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

# the same preamble every runner in this tree uses: the pipeline's leaf modules
# import their neighbours by bare name (``from filters import ...``)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch

from src.pipeline.coarsening import _METHODS
from src.pipeline.learning import StaticSpectralLearning
from src.pipeline.logging_visualization import MODES, LoggingVisualizationConfig
from src.pipeline.pipeline import PipelineConfig, run_pipeline
from src.utils.utils import LOGGER, now

__all__ = [
    "DENSITIES",
    "METHOD",
    "build_config",
    "run_cell",
    "aggregate_by_density",
    "main",
]

#: exactly 21 values: 0.00, 0.05, ..., 1.00, both endpoints included
DENSITIES: "list[float]" = [round(0.05 * i, 2) for i in range(21)]

#: the one merge rule this experiment runs.  Nothing else is executed or compared.
METHOD = "deflated_ward_tight"

#: the two target subspaces compared at every density.  ``learned`` is the
#: polynomial closed-form bank; ``static_spectral`` is the paper's unsupervised
#: baseline R = span(U_q), the q lowest eigenvectors of L = I - A_hat, with q
#: matched per graph to the learned arm's effective target rank.
SUBSPACES = ("learned", "static_spectral")

#: metrics carried through per-seed aggregation and plotted against density
_PRIMARY = ("pr_auc", "f1", "detection_rate", "precision", "recall")
_SECONDARY = ("reduction", "retained", "n_coarse", "epsilon", "epsilon_q", "score_sum")
_AGGREGATED = _PRIMARY + _SECONDARY + (
    "runtime_s",
    "representation_seconds",
    "coarsening_seconds",
    "target_rank",
    "requested_q",

    "micro_f1",
    "micro_precision",
    "micro_recall",
    "detected",
    "n_groups",
    "realized_density",
    "label_coverage",
    "label_accuracy",
    "label_f1_macro",
)

#: one colour and one dash pattern per evaluation mode, used by every figure
_STYLE = {
    "stopping_rule": {"color": "#1f77b4", "linestyle": "-", "marker": "o",
                      "label": "stopping rule (deployable)"},
    "oracle": {"color": "#d62728", "linestyle": "--", "marker": "s",
               "label": "oracle (post-hoc upper bound)"},
}
_METRIC_COLOR = {
    "pr_auc": "#4c72b0",
    "f1": "#dd8452",
    "detection_rate": "#55a868",
    "precision": "#c44e52",
    "recall": "#8172b3",
}

#: one colour per subspace, one dash pattern per evaluation mode, so all four
#: curves of a panel are distinguishable by both channels
_SUBSPACE_STYLE = {
    ("learned", "stopping_rule"): dict(color="#1f77b4", linestyle="-", marker="o",
                                       label="learned — stopping rule"),
    ("learned", "oracle"): dict(color="#1f77b4", linestyle="--", marker="^",
                                label="learned — oracle (upper bound)"),
    ("static_spectral", "stopping_rule"): dict(color="#d95f02", linestyle="-",
                                               marker="s",
                                               label="static spectral — stopping rule"),
    ("static_spectral", "oracle"): dict(color="#d95f02", linestyle="--", marker="v",
                                        label="static spectral — oracle (upper bound)"),
}

#: budgets at which the two subspaces are compared on their own hierarchies, so a
#: difference cannot be a difference of selected cut
_MATCHED_N = (2000, 1500, 1200, 1000, 800, 600, 400, 200)
_MATCHED_REDUCTION = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_MATCHED_EPSILON = (0.6, 0.8, 0.9, 0.95, 0.99)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def build_config(
    density: float,
    seed: int,
    out_dir: Path,
    *,
    subspace: str = "learned",
    cut_rule: str = "f1",
    transfer_cut_rule: str = "score_sum",
    per_run_plots: bool = False,
) -> PipelineConfig:
    """The ``polynomial-closed-form`` example with one density and one seed set.

    The base configuration is imported from :mod:`src.run_modular_pipeline` and
    deep-copied, so the sweep cannot mutate the shared example and no setting is
    restated here.  Exactly four things are touched: ``group_density``, the three
    seeds, the coarsening method, and the output directory.

    ``cut_rule="f1"`` is what makes the deployable arm deployable: the operating
    point is chosen on the **training groups of the training graphs** (the
    component refuses to look at anything else) and the resulting
    :class:`~src.pipeline.coarsening.CutRule` is transferred to every held-out
    graph through ``transfer_cut_rule``.  No test label reaches cut selection.
    """

    from src.run_modular_pipeline import EXAMPLES

    if subspace not in SUBSPACES:
        raise ValueError(f"subspace must be one of {SUBSPACES}, got {subspace!r}")
    config = copy.deepcopy(EXAMPLES["polynomial-closed-form"])
    if subspace == "static_spectral":
        # the ONLY changed variable: the architecture that produces the target.
        # tau, the heads, the label head, the objective, the coarsening config,
        # the data and the seeds are all still the example's.
        config.learning.architecture = "static_spectral"
        config.learning.__post_init__()
    config.data.group_density = float(density)
    config.data.seed = int(seed)
    config.learning.seed = int(seed)
    config.learning.objective.seed = int(seed)
    config.coarsening.seed = int(seed)
    config.coarsening.method = METHOD
    config.coarsening.cut_rule = cut_rule
    config.coarsening.transfer_cut_rule = transfer_cut_rule
    config.logging = LoggingVisualizationConfig(
        output_dir=out_dir, make_plots=per_run_plots
    )
    return config


def _fingerprint_key(subspace: str) -> str:
    """Which fields the fixed-value check is allowed to see vary by subspace."""

    return "architecture" if subspace == "static_spectral" else ""


def _fixed_fingerprint(config: PipelineConfig) -> dict:
    """Everything the sweep promises to hold constant, as a comparable dict."""

    payload = config.to_dict()
    payload["data"].pop("group_density")
    payload["data"].pop("seed")
    payload["learning"].pop("seed", None)
    # the architecture IS the variable under test in the subspace comparison;
    # every other learning knob is checked for constancy
    payload["learning"].pop("architecture", None)
    payload["learning"].get("objective", {}).pop("seed", None)
    payload["coarsening"].pop("seed")
    payload.pop("logging")
    payload.pop("seeds")
    return payload


# --------------------------------------------------------------------------- #
# per-graph extraction
# --------------------------------------------------------------------------- #
def _realized_density(graph) -> float:
    """Mean over the graph's groups of ``|E(A)| / binom(|A|, 2)``.

    Measured, not assumed: the planter's spanning path puts a floor of ``2/s``
    under the nominal density, so the two disagree at the bottom of the sweep.
    """

    edges = graph.edge_index.cpu().numpy()
    upper = edges[0] < edges[1]
    u, v = edges[0][upper], edges[1][upper]
    member_of = np.full(int(graph.num_nodes), -1, dtype=np.int64)
    sizes = []
    for index, group in enumerate(graph.groups):
        member_of[group.nodes.cpu().numpy()] = index
        sizes.append(int(group.num_nodes))
    sizes = np.asarray(sizes, dtype=np.float64)
    if not sizes.size:
        return float("nan")
    inside = (member_of[u] >= 0) & (member_of[u] == member_of[v])
    counts = np.bincount(member_of[u][inside], minlength=sizes.size).astype(float)
    possible = sizes * (sizes - 1.0) / 2.0
    valid = possible > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(counts[valid] / possible[valid]))


def _maybe(value):
    """``None`` rather than a silent zero for a missing or non-finite metric."""

    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _graph_rows(
    evaluation, density: float, seed: int, subspace: str, extra: dict, undefined: list
) -> list:
    """One row per evaluation mode for one graph, with every requested field."""

    coarsening = evaluation.coarsening
    rows = []
    for mode in MODES:
        cut = coarsening.cut(mode)
        metrics = cut.metrics.get("all", {})
        row = {
            "density": float(density),
            "seed": int(seed),
            "subspace": subspace,
            "graph_id": evaluation.graph_id,
            "scope": evaluation.scope,
            "evaluation_mode": mode,
            "method": coarsening.hierarchy.method,
            "cut_rule": cut.rule,
            "realized_density": _maybe(evaluation.realized_density),
            # detection
            "pr_auc": _maybe(coarsening.pr_auc),
            "precision": _maybe(metrics.get("mean_precision")),
            "recall": _maybe(metrics.get("mean_recall")),
            "f1": _maybe(metrics.get("mean_f1")),
            "detection_rate": _maybe(metrics.get("detection_rate")),
            "detected": _maybe(metrics.get("detected")),
            "n_groups": _maybe(metrics.get("total")),
            "micro_precision": _maybe(metrics.get("micro_precision")),
            "micro_recall": _maybe(metrics.get("micro_recall")),
            "micro_f1": _maybe(metrics.get("micro_f1")),
            # the operating point
            "epsilon": _maybe(cut.epsilon),
            "epsilon_is_exact": bool(cut.row.get("epsilon_is_exact", False)),
            "epsilon_q": _maybe(cut.row.get("epsilon_q")),
            "epsilon_q_is_exact": bool(cut.row.get("epsilon_q_is_exact", False)),
            "score_sum": _maybe(cut.row.get("score_sum")),
            "score_sum_raw": _maybe(cut.row.get("score_sum_raw")),
            "reduction": _maybe(cut.reduction),
            "retained": _maybe(cut.retained),
            "n_coarse": _maybe(cut.n_coarse),
            "n_original": int(coarsening.hierarchy.n_leaves),
            "target_rank": int(coarsening.target_rank),
            "completion_merges": int(coarsening.hierarchy.completion_merges),
        }
        row.update(extra)
        report = evaluation.label_reports.get(mode)
        if report is not None:
            m = report.metrics
            row.update(
                label_coverage=_maybe(m.get("detected_group_label_coverage")),
                label_detected_groups=_maybe(m.get("n_detected_groups")),
                label_accuracy=_maybe(m.get("detected_group_label_accuracy")),
                label_f1_macro=_maybe(m.get("detected_group_label_f1_macro")),
                label_precision_macro=_maybe(
                    m.get("detected_group_label_precision_macro")
                ),
                label_recall_macro=_maybe(m.get("detected_group_label_recall_macro")),
            )
        for field in ("pr_auc", "f1", "precision", "recall", "epsilon", "epsilon_q"):
            if row.get(field) is None:
                undefined.append(
                    {
                        "density": float(density),
                        "seed": int(seed),
                        "graph_id": evaluation.graph_id,
                        "evaluation_mode": mode,
                        "field": field,
                        "reason": (
                            "eps_Q axis not materialized for this cut"
                            if field == "epsilon_q"
                            else "metric missing or not finite at the selected cut"
                        ),
                    }
                )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# one sweep cell
# --------------------------------------------------------------------------- #
def _matched_rows(
    evaluation,
    density,
    seed,
    subspace,
    *,
    n_grid=None,
    reduction_grid=None,
    epsilon_grid=None,
) -> list:
    """The same hierarchy read at fixed budgets, so no cut rule is involved.

    Three matched axes, each taking the coarsest level still within the budget:
    supernode count, reduction, and the common Loukas Def. 2 epsilon.  Both
    subspaces are read at the *same* budget on their *own* tree, which is what
    separates "a better hierarchy" from "a better selected cut".
    """

    trajectory = evaluation.coarsening.trajectory
    rows = []

    def emit(kind, budget, row):
        rows.append(
            {
                "density": float(density),
                "seed": int(seed),
                "subspace": subspace,
                "graph_id": evaluation.graph_id,
                "scope": evaluation.scope,
                "budget_kind": kind,
                "budget": float(budget),
                "n_coarse": int(row["n_coarse"]),
                "reduction": float(row["reduction"]),
                "retained": float(row["retained"]),
                "epsilon": _maybe(row.get("epsilon")),
                "epsilon_q": _maybe(row.get("epsilon_q")),
                "score_sum": _maybe(row.get("score_sum")),
                "pr_auc": _maybe(evaluation.coarsening.pr_auc),
                "f1": _maybe(row.get("all_mean_f1")),
                "precision": _maybe(row.get("all_mean_precision")),
                "recall": _maybe(row.get("all_mean_recall")),
                "detection_rate": _maybe(row.get("all_detection_rate")),
            }
        )

    for target in _MATCHED_N if n_grid is None else n_grid:
        below = [r for r in trajectory if r["n_coarse"] <= target]
        emit("n_coarse", target, below[0] if below else trajectory[-1])
    for target in _MATCHED_REDUCTION if reduction_grid is None else reduction_grid:
        within = [r for r in trajectory if r["reduction"] <= target + 1e-12]
        emit("reduction", target, within[-1] if within else trajectory[0])
    for target in _MATCHED_EPSILON if epsilon_grid is None else epsilon_grid:
        within = [r for r in trajectory if r.get("epsilon", np.inf) <= target + 1e-12]
        emit("epsilon", target, within[-1] if within else trajectory[0])
    return rows


def run_cell(
    density: float,
    seed: int,
    out_dir: Path,
    *,
    subspace: str = "learned",
    target_widths: "dict | None" = None,
    **options,
) -> dict:
    """One ``(density, seed, subspace)`` pipeline run; never raises, always reports."""

    config = build_config(density, seed, out_dir, subspace=subspace, **options)
    if config.coarsening.method != METHOD:
        raise AssertionError(
            f"this experiment runs {METHOD!r} only, got {config.coarsening.method!r}"
        )
    if subspace == "static_spectral":
        learner = StaticSpectralLearning(config.learning, target_widths=target_widths)
    else:
        from src.pipeline.learning import build_learner

        learner = build_learner(config.learning)

    # time the representation phase alone: the whole point of the cost comparison
    timings: dict = {}
    original_run = learner.run

    def timed_run(graphs):
        started_fit = time.time()
        try:
            return original_run(graphs)
        finally:
            timings["representation_seconds"] = time.time() - started_fit

    learner.run = timed_run

    started = time.time()
    try:
        result = run_pipeline(config, learner=learner)
    except Exception as error:  # noqa: BLE001 -- the sweep must survive one cell
        LOGGER.warning(
            f"[density {density:.2f} seed {seed} {subspace}] run failed and is "
            f"recorded as a failure, the sweep continues: "
            f"{type(error).__name__}: {error}"
        )
        return {
            "density": float(density),
            "seed": int(seed),
            "subspace": subspace,
            "ok": False,
            "rows": [],
            "matched": [],
            "undefined": [],
            "target_ranks": {},
            "failure": {
                "density": float(density),
                "seed": int(seed),
                "subspace": subspace,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=8),
            },
            "runtime_s": time.time() - started,
            "fingerprint": _fixed_fingerprint(config),
        }

    runtime = time.time() - started
    spectra = getattr(learner, "spectra_", {})
    undefined: list = []
    rows: list = []
    matched: list = []
    for evaluation in result.evaluations:
        evaluation.realized_density = _realized_density(evaluation.graph)
        record = spectra.get(evaluation.graph_id, {})
        extra = {
            "runtime_s": runtime,
            "representation_seconds": _maybe(timings.get("representation_seconds")),
            "coarsening_seconds": _maybe(
                runtime - (timings.get("representation_seconds") or 0.0)
            ),
            "requested_q": _maybe(record.get("requested_q")),
            "returned_q": _maybe(record.get("returned_q")),
            "eigenvalue_min": _maybe(record.get("eigenvalue_min")),
            "eigenvalue_max": _maybe(record.get("eigenvalue_max")),
            "eigensolver": record.get("solver"),
            "representation_width": int(
                result.learning.representations[evaluation.graph_id].shape[1]
            )
            if evaluation.graph_id in result.learning.representations
            else None,
        }
        rows.extend(
            _graph_rows(evaluation, density, seed, subspace, extra, undefined)
        )
        matched.extend(_matched_rows(evaluation, density, seed, subspace))
    return {
        "density": float(density),
        "seed": int(seed),
        "subspace": subspace,
        "ok": True,
        "rows": rows,
        "matched": matched,
        "undefined": undefined,
        "failure": None,
        "runtime_s": runtime,
        "representation_seconds": timings.get("representation_seconds"),
        "target_ranks": {
            e.graph_id: int(e.coarsening.target_rank) for e in result.evaluations
        },
        "cut_rule": None if result.cut_rule is None else result.cut_rule.to_dict(),
        "fingerprint": _fixed_fingerprint(config),
    }


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _mean_sd(values) -> tuple:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None, None, 0
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, sd, len(finite)


def per_seed_rows(graph_rows: list, *, scope: str = "test") -> list:
    """Pool one seed's ``scope`` graphs into one row per (density, seed, mode)."""

    buckets: dict = {}
    for row in graph_rows:
        if row["scope"] != scope:
            continue
        key = (row["density"], row["subspace"], row["seed"], row["evaluation_mode"])
        buckets.setdefault(key, []).append(row)
    out = []
    for (density, subspace, seed, mode), rows in sorted(buckets.items()):
        entry = {
            "density": density,
            "subspace": subspace,
            "seed": seed,
            "evaluation_mode": mode,
            "scope": scope,
            "n_graphs": len(rows),
            "method": rows[0]["method"],
        }
        for metric in _AGGREGATED:
            mean, _sd, count = _mean_sd([r.get(metric) for r in rows])
            entry[metric] = mean
            entry[f"{metric}_n_graphs"] = count
        entry["n_groups_total"] = sum(
            int(r["n_groups"]) for r in rows if r.get("n_groups") is not None
        )
        entry["detected_total"] = sum(
            int(r["detected"]) for r in rows if r.get("detected") is not None
        )
        out.append(entry)
    return out


def aggregate_by_density(seed_rows: list) -> list:
    """Mean, sd and successful-run count across seeds, per (density, mode)."""

    buckets: dict = {}
    for row in seed_rows:
        key = (row["density"], row["subspace"], row["evaluation_mode"])
        buckets.setdefault(key, []).append(row)
    out = []
    for (density, subspace, mode), rows in sorted(buckets.items()):
        entry = {
            "density": density,
            "subspace": subspace,
            "evaluation_mode": mode,
            "n_runs": len(rows),
            "seeds": sorted(int(r["seed"]) for r in rows),
        }
        for metric in _AGGREGATED:
            mean, sd, count = _mean_sd([r.get(metric) for r in rows])
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = sd
            entry[f"{metric}_n"] = count
        out.append(entry)
    return out


def paired_differences(seed_rows: list) -> list:
    """``learned - static_spectral`` on the same (density, seed, mode), seed-paired."""

    index = {
        (r["density"], r["subspace"], r["seed"], r["evaluation_mode"]): r
        for r in seed_rows
    }
    buckets: dict = {}
    for (density, subspace, seed, mode), row in index.items():
        if subspace != "learned":
            continue
        other = index.get((density, "static_spectral", seed, mode))
        if other is None:
            continue
        for metric in _PRIMARY + ("reduction", "retained", "n_coarse", "epsilon",
                                  "representation_seconds"):
            a, b = row.get(metric), other.get(metric)
            if a is None or b is None:
                continue
            buckets.setdefault((density, mode, metric), []).append(a - b)
    out = []
    for (density, mode, metric), values in sorted(buckets.items()):
        mean, sd, count = _mean_sd(values)
        out.append(
            {
                "density": density,
                "evaluation_mode": mode,
                "metric": metric,
                "mean_difference": mean,
                "std_difference": sd,
                "n_seeds": count,
                "learned_wins": int(sum(1 for v in values if v > 0)),
                "static_wins": int(sum(1 for v in values if v < 0)),
            }
        )
    return out


def aggregate_matched(matched_rows: list) -> list:
    """Mean/sd across seeds at each (density, subspace, budget), test graphs only."""

    per_seed: dict = {}
    for row in matched_rows:
        if row["scope"] != "test":
            continue
        key = (
            row["density"], row["subspace"], row["budget_kind"], row["budget"],
            row["seed"],
        )
        per_seed.setdefault(key, []).append(row)
    pooled: dict = {}
    for (density, subspace, kind, budget, _seed), rows in per_seed.items():
        entry = {}
        for metric in ("f1", "precision", "recall", "detection_rate", "pr_auc",
                       "n_coarse", "reduction", "retained", "epsilon", "epsilon_q"):
            mean, _sd, _n = _mean_sd([r.get(metric) for r in rows])
            entry[metric] = mean
        pooled.setdefault((density, subspace, kind, budget), []).append(entry)
    out = []
    for (density, subspace, kind, budget), entries in sorted(pooled.items()):
        row = {
            "density": density,
            "subspace": subspace,
            "budget_kind": kind,
            "budget": budget,
            "n_runs": len(entries),
        }
        for metric in entries[0]:
            mean, sd, count = _mean_sd([e[metric] for e in entries])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sd
            row[f"{metric}_n"] = count
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _log_density_table(summary: list, mode: str, subspace: str) -> None:
    title = (
        "STOPPING-RULE SUMMARY vs GROUP DENSITY "
        "(deployable: cut fitted on training groups, transferred to test graphs)"
        if mode == "stopping_rule"
        else "ORACLE SUMMARY vs GROUP DENSITY "
        "(post-hoc UPPER BOUND: best-F1 cut chosen with each test graph's labels)"
    ) + f"  [subspace = {subspace}]"
    columns = [
        ("density", 9, "{:.2f}"),
        ("realized", 10, "{:.3f}"),
        ("runs", 6, "{:.0f}"),
        ("pr_auc", 16, None),
        ("f1", 16, None),
        ("detection_rate", 16, None),
        ("precision", 16, None),
        ("recall", 16, None),
        ("epsilon", 14, None),
        ("retained", 14, None),
    ]
    labels = {"pr_auc": "PR-AUC", "detection_rate": "det rate", "epsilon": "eps"}
    header = "  " + "".join(
        labels.get(name, name).rjust(width) for name, width, _f in columns
    )
    LOGGER.info("\n" + "=" * max(len(header), 96))
    LOGGER.info(title)
    LOGGER.info("=" * max(len(header), 96))
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 2))
    chosen = [
        r for r in summary
        if r["evaluation_mode"] == mode and r["subspace"] == subspace
    ]
    for row in chosen:
        cells = []
        for name, width, fmt in columns:
            if name == "density":
                cells.append(fmt.format(row["density"]).rjust(width))
            elif name == "realized":
                value = row.get("realized_density_mean")
                cells.append(
                    ("n/a" if value is None else fmt.format(value)).rjust(width)
                )
            elif name == "runs":
                cells.append(f"{row['n_runs']:d}".rjust(width))
            else:
                mean, sd = row.get(f"{name}_mean"), row.get(f"{name}_std")
                cells.append(
                    ("undefined" if mean is None else f"{mean:.3f}+-{sd:.3f}").rjust(
                        width
                    )
                )
        LOGGER.info("  " + "".join(cells))
    LOGGER.info("  " + "-" * (len(header) - 2))
    LOGGER.info(
        "  mean +- sd across seeds; 'realized' is the measured intra-group edge "
        "fraction, which floors at 2/s because every planted group keeps a "
        "spanning path"
    )


def _log_difference_table(differences: list, mode: str) -> None:
    metrics = list(_PRIMARY)
    header = "  " + "density".rjust(9) + "".join(m.rjust(19) for m in metrics)
    LOGGER.info("\n" + "=" * max(len(header), 96))
    LOGGER.info(
        f"PAIRED DIFFERENCE  learned - static_spectral  [{mode}]  "
        "(positive = the learned subspace wins; W/L counts seeds)"
    )
    LOGGER.info("=" * max(len(header), 96))
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 2))
    densities = sorted({d["density"] for d in differences})
    index = {(d["density"], d["evaluation_mode"], d["metric"]): d for d in differences}
    for density in densities:
        cells = [f"{density:.2f}".rjust(9)]
        for metric in metrics:
            entry = index.get((density, mode, metric))
            cells.append(
                "n/a".rjust(19)
                if entry is None or entry["mean_difference"] is None
                else (
                    f"{entry['mean_difference']:+.3f} "
                    f"{entry['learned_wins']}/{entry['static_wins']}"
                ).rjust(19)
            )
        LOGGER.info("  " + "".join(cells))
    LOGGER.info("  " + "-" * (len(header) - 2))


def _write_csv(path: Path, rows: list) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    fields: list = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: ("" if v is None else ";".join(map(str, v)) if isinstance(v, list) else v)
                    for k, v in row.items()
                }
            )


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _series(summary: list, subspace: str, mode: str, metric: str):
    rows = [
        r for r in summary
        if r["evaluation_mode"] == mode and r["subspace"] == subspace
    ]
    rows.sort(key=lambda r: r["density"])
    x = np.array([r["density"] for r in rows], dtype=float)
    mean = np.array(
        [np.nan if r.get(f"{metric}_mean") is None else r[f"{metric}_mean"] for r in rows]
    )
    sd = np.array(
        [0.0 if r.get(f"{metric}_std") is None else r[f"{metric}_std"] for r in rows]
    )
    return x, mean, sd


def _draw(ax, summary, metric, subspaces, *, ylabel=None, legend=False):
    """One curve per (subspace, mode): colour is the subspace, dash is the mode."""

    for subspace in subspaces:
        for mode in MODES:
            x, mean, sd = _series(summary, subspace, mode, metric)
            if not x.size or not np.any(np.isfinite(mean)):
                continue
            style = _SUBSPACE_STYLE[(subspace, mode)]
            ax.plot(
                x, mean, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markersize=3.2, linewidth=1.6,
                label=style["label"],
            )
            ax.fill_between(
                x, mean - sd, mean + sd, color=style["color"], alpha=0.13, linewidth=0
            )
    ax.set_xlabel("planted group density")
    ax.set_ylabel(ylabel or metric)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(fontsize=7.5, framealpha=0.9)


def make_figures(summary: list, differences: list, out_dir: Path, subspaces) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    subspaces = [s for s in SUBSPACES if s in subspaces]

    panels = [
        ("pr_auc", "PR-AUC over the hierarchy"),
        ("f1", "macro F1"),
        ("detection_rate", "detection rate"),
        ("precision", "macro precision"),
        ("recall", "macro recall"),
        ("reduction", "reduction at the cut"),
        ("retained", "retained fraction"),
        ("epsilon", r"common $\varepsilon$ (Loukas Def. 2)"),
        ("epsilon_q", r"$\varepsilon_Q$"),
        ("n_coarse", "supernodes at the cut"),
        ("representation_seconds", "representation cost (s)"),
        ("runtime_s", "total runtime per run (s)"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 14))
    for ax, (metric, label) in zip(axes.ravel(), panels):
        _draw(ax, summary, metric, subspaces, ylabel=label, legend=(metric == "pr_auc"))
    fig.suptitle(
        f"Held-out detection vs planted group density — coarsener {METHOD}\n"
        "solid = deployable stopping rule, dashed = post-hoc oracle upper bound; "
        "band = ±1 sd across seeds",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / "density_sweep_summary_panel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    for metric, label in [
        ("pr_auc", "PR-AUC"),
        ("f1", "macro F1"),
        ("detection_rate", "detection rate"),
        ("precision", "macro precision"),
        ("recall", "macro recall"),
        ("reduction", "reduction at the selected cut"),
        ("representation_seconds", "representation cost (seconds)"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.5))
        _draw(ax, summary, metric, subspaces, ylabel=label, legend=True)
        ax.set_title(f"{label} vs group density ({METHOD})", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"density_sweep_{metric}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path)

    # oracle - stopping rule, per subspace
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for subspace in subspaces:
        for metric, label in (
            ("f1", "macro F1"), ("pr_auc", "PR-AUC"),
            ("detection_rate", "detection rate"),
        ):
            x, oracle, _s = _series(summary, subspace, "oracle", metric)
            _x, stopping, _s2 = _series(summary, subspace, "stopping_rule", metric)
            if not x.size:
                continue
            ax.plot(
                x, oracle - stopping, color=_METRIC_COLOR[metric],
                linestyle="-" if subspace == "learned" else "--",
                marker="o" if subspace == "learned" else "s", markersize=3.0,
                linewidth=1.5, label=f"{label} — {subspace}",
            )
    ax.axhline(0.0, color="#444444", linewidth=0.9, linestyle=":")
    ax.set_xlabel("planted group density")
    ax.set_ylabel("oracle − stopping rule")
    ax.set_title("What the deployable cut gives up to the post-hoc optimum", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    path = out_dir / "density_sweep_oracle_gap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    # paired learned - static, per metric, with the zero reference
    if differences:
        index = {
            (d["density"], d["evaluation_mode"], d["metric"]): d for d in differences
        }
        densities = sorted({d["density"] for d in differences})
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
        for ax, mode in zip(axes, MODES):
            for metric, label in (
                ("pr_auc", "PR-AUC"), ("f1", "macro F1"),
                ("detection_rate", "detection rate"),
                ("precision", "precision"), ("recall", "recall"),
            ):
                mean = np.array(
                    [
                        np.nan
                        if index.get((d, mode, metric)) is None
                        or index[(d, mode, metric)]["mean_difference"] is None
                        else index[(d, mode, metric)]["mean_difference"]
                        for d in densities
                    ]
                )
                sd = np.array(
                    [
                        0.0
                        if index.get((d, mode, metric)) is None
                        or index[(d, mode, metric)]["std_difference"] is None
                        else index[(d, mode, metric)]["std_difference"]
                        for d in densities
                    ]
                )
                x = np.array(densities, dtype=float)
                ax.plot(x, mean, color=_METRIC_COLOR[metric], marker="o",
                        markersize=3.2, linewidth=1.6, label=label)
                ax.fill_between(x, mean - sd, mean + sd,
                                color=_METRIC_COLOR[metric], alpha=0.12, linewidth=0)
            ax.axhline(0.0, color="#111111", linewidth=1.1)
            ax.set_xlabel("planted group density")
            ax.set_title(
                "stopping rule (deployable)" if mode == "stopping_rule"
                else "oracle (post-hoc upper bound)",
                fontsize=11,
            )
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("learned − static spectral")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "Paired difference: above the black line the learned subspace wins, "
            "below it the static spectral one does",
            fontsize=11.5,
        )
        fig.tight_layout()
        path = out_dir / "density_sweep_paired_difference.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(cells: list, densities: list, seeds: list) -> dict:
    """Structural checks that hold for the smoke test and the full sweep alike."""

    checks: dict = {}
    checks["density_grid_has_21_values"] = len(DENSITIES) == 21
    checks["density_grid_endpoints"] = (
        DENSITIES[0] == 0.0 and DENSITIES[-1] == 1.0
    )
    checks["density_grid_step"] = all(
        abs((DENSITIES[i + 1] - DENSITIES[i]) - 0.05) < 1e-9
        for i in range(len(DENSITIES) - 1)
    )
    rows = [row for cell in cells for row in cell["rows"]]
    checks["only_one_method_executed"] = {r["method"] for r in rows} == {METHOD}
    checks["subspaces_present"] = sorted({r["subspace"] for r in rows})
    checks["rows_carry_subspace"] = all("subspace" in r for r in rows)
    checks["q_is_dimension_matched"] = _q_matching_check(rows)
    fingerprints = {json.dumps(c["fingerprint"], sort_keys=True) for c in cells}
    checks["non_density_config_is_fixed"] = len(fingerprints) <= 1
    checks["seed_schedule_is_shared"] = all(
        sorted({c["seed"] for c in cells if c["density"] == d and c["subspace"] == sub})
        == sorted(seeds)
        for d in densities
        for sub in {c["subspace"] for c in cells}
    )
    checks["modes_kept_separate"] = {r["evaluation_mode"] for r in rows} == set(MODES)
    checks["stopping_rule_cut_uses_no_test_labels"] = all(
        r["cut_rule"] != "f1_oracle"
        for r in rows
        if r["evaluation_mode"] == "stopping_rule"
    )
    checks["oracle_rows_are_labelled_as_such"] = all(
        r["cut_rule"] == "f1_oracle"
        for r in rows
        if r["evaluation_mode"] == "oracle"
    )
    checks["rows_carry_density_seed_graph_mode"] = all(
        all(k in r for k in ("density", "seed", "graph_id", "evaluation_mode"))
        for r in rows
    )
    return checks


def _q_matching_check(rows: list) -> bool:
    """Every static graph's requested q equals the learned arm's rank on that graph."""

    learned = {
        (r["density"], r["seed"], r["graph_id"]): r["target_rank"]
        for r in rows
        if r["subspace"] == "learned"
    }
    static = [r for r in rows if r["subspace"] == "static_spectral"]
    if not static or not learned:
        return True  # only one arm ran; nothing to match
    return all(
        r.get("requested_q") is None
        or int(r["requested_q"])
        == learned.get((r["density"], r["seed"], r["graph_id"]), int(r["requested_q"]))
        for r in static
    )


def _determinism_check(density: float, seed: int, out_dir: Path, **options) -> dict:
    """Run one cell twice and compare every numeric field of every row."""

    first = run_cell(density, seed, out_dir / "det_a", **options)
    second = run_cell(density, seed, out_dir / "det_b", **options)
    if not (first["ok"] and second["ok"]):
        return {"deterministic": False, "reason": "a repeat run failed"}
    # wall-clock fields legitimately differ between two identical runs
    timing = {"runtime_s", "representation_seconds", "coarsening_seconds"}
    worst, field = 0.0, None
    for a, b in zip(first["rows"], second["rows"]):
        for key, value in a.items():
            if key in timing:
                continue
            other = b.get(key)
            if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                gap = abs(float(value) - float(other))
                if gap > worst:
                    worst, field = gap, key
            elif value != other:
                return {"deterministic": False, "reason": f"{key}: {value!r} != {other!r}"}
    return {"deterministic": worst == 0.0, "max_abs_difference": worst, "field": field}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3],
        help="seed schedule, applied identically at every density",
    )
    parser.add_argument(
        "--densities", type=float, nargs="+", default=None,
        help="override the 21-value grid (the smoke test uses this)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="three densities, one seed, plus the determinism check",
    )
    parser.add_argument("--cut-rule", default="f1")
    parser.add_argument("--transfer-cut-rule", default="score_sum")
    parser.add_argument(
        "--subspaces", nargs="+", default=list(SUBSPACES), choices=list(SUBSPACES),
        help="target subspaces to compare; the coarsener is the same for all",
    )
    parser.add_argument(
        "--per-run-plots", action="store_true",
        help="let each inner pipeline run draw its own figures (slow)",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--reuse-from", type=Path, default=None,
        help="a previous density_sweep_results.json: keep its rows for every "
        "subspace NOT being run now (and, for the static arm, its learned ranks "
        "for q-matching) instead of recomputing them",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    seeds = list(args.seeds)
    densities = list(args.densities) if args.densities else list(DENSITIES)
    subspaces = [s for s in SUBSPACES if s in args.subspaces]
    if args.smoke:
        densities = [DENSITIES[0], DENSITIES[10], DENSITIES[-1]]
        seeds = seeds[:1]
    out_dir = Path(args.out or f"results2/density_sweep/{now}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = dict(
        cut_rule=args.cut_rule,
        transfer_cut_rule=args.transfer_cut_rule,
        per_run_plots=args.per_run_plots,
    )

    if METHOD not in _METHODS:
        raise ValueError(f"{METHOD!r} is not a supported coarsening method")
    LOGGER.info(
        f"density sweep: {len(densities)} densities x {len(seeds)} seeds x "
        f"{len(subspaces)} subspaces = {len(densities) * len(seeds) * len(subspaces)} "
        f"runs, coarsener={METHOD!r} for all of them, "
        f"cut_rule={args.cut_rule!r} -> transfer={args.transfer_cut_rule!r}, "
        f"out={out_dir}"
    )
    LOGGER.info(
        "note: group_density=0 does NOT mean an edgeless group -- the planter keeps "
        "a spanning path, so the realized intra-group density floors at 2/s and is "
        "measured per graph as 'realized_density'"
    )
    if "static_spectral" in subspaces:
        LOGGER.info(
            "static spectral target: R = span(U_q), the q LOWEST eigenvectors of "
            "L_sym = I - D~^{-1/2}(W+I)D~^{-1/2} (eigenvalues ascending in lambda(L), "
            "0 = lambda_0 <= ... < 2, the constant eigenvector u_0 kept), computed "
            "per graph with no labels and never transferred between graphs; q is "
            "matched per graph to the learned arm's effective target rank"
        )

    reused_rows, reused_matched, reused_ranks = [], [], {}
    if args.reuse_from is not None:
        previous = json.loads(Path(args.reuse_from).read_text())
        reused_rows = [
            r for r in previous["per_graph"]
            if r["subspace"] not in subspaces and r["density"] in densities
            and r["seed"] in seeds
        ]
        reused_matched = [
            r for r in previous.get("matched_budgets", [])
            if r["subspace"] not in subspaces and r["density"] in densities
        ]
        for r in previous["per_graph"]:
            if r["subspace"] == "learned" and r["evaluation_mode"] == "stopping_rule":
                reused_ranks.setdefault((r["density"], r["seed"]), {})[
                    r["graph_id"]
                ] = int(r["target_rank"])
        LOGGER.info(
            f"reusing {len(reused_rows)} per-graph rows for "
            f"{sorted({r['subspace'] for r in reused_rows})} from {args.reuse_from}"
        )

    cells: list = []
    for density in densities:
        #: filled by the learned arm, then handed to the static arm so the two
        #: subspaces have the same dimension on the same graph
        ranks_by_seed: dict = {
            seed: reused_ranks[(density, seed)]
            for seed in seeds
            if (density, seed) in reused_ranks
        }
        for subspace in subspaces:
            for seed in seeds:
                cell = run_cell(
                    density,
                    seed,
                    out_dir / "runs" / f"d{density:.2f}_s{seed}_{subspace}",
                    subspace=subspace,
                    target_widths=ranks_by_seed.get(seed),
                    **options,
                )
                if subspace == "learned" and cell["ok"]:
                    ranks_by_seed[seed] = cell["target_ranks"]
                cells.append(cell)
                state = "ok" if cell["ok"] else "FAILED"
                LOGGER.info(
                    f"  density {density:.2f} seed {seed} {subspace:15s}: {state} "
                    f"({cell['runtime_s']:.1f}s total, "
                    f"{(cell.get('representation_seconds') or 0.0):.1f}s representation, "
                    f"{len(cell['rows'])} rows)"
                )

    if reused_rows:
        grouped: dict = {}
        for row in reused_rows:
            grouped.setdefault((row["density"], row["seed"], row["subspace"]), []).append(row)
        for (density, seed, subspace), rows in grouped.items():
            cells.append(
                {
                    "density": density, "seed": seed, "subspace": subspace,
                    "ok": True, "rows": rows, "matched": [], "undefined": [],
                    "failure": None, "runtime_s": rows[0].get("runtime_s") or 0.0,
                    "reused": True,
                    "fingerprint": _fixed_fingerprint(
                        build_config(density, seed, out_dir, subspace=subspace, **options)
                    ),
                }
            )
        subspaces = [s for s in SUBSPACES if s in subspaces or any(
            r["subspace"] == s for r in reused_rows)]

    graph_rows = [row for cell in cells for row in cell["rows"]]
    matched_rows = [row for cell in cells for row in cell.get("matched", [])]
    seed_rows = per_seed_rows(graph_rows, scope="test")
    summary = aggregate_by_density(seed_rows)
    differences = paired_differences(seed_rows)
    # aggregated per (density, subspace, budget), so a reused arm's rows can be
    # concatenated without ever being mixed with the arm that was rerun
    matched_summary = aggregate_matched(matched_rows) + reused_matched
    failures = [c["failure"] for c in cells if c["failure"] is not None]
    undefined = [u for cell in cells for u in cell["undefined"]]

    checks = validate(cells, densities, seeds)
    if args.smoke:
        # every subspace, not just the first: the static arm's eigensolver is the
        # one with a random start to get wrong
        for subspace in subspaces:
            result = _determinism_check(
                densities[0], seeds[0], out_dir / f"det_{subspace}",
                subspace=subspace, **options,
            )
            checks.update({f"{k}[{subspace}]": v for k, v in result.items()})
    for name, value in checks.items():
        LOGGER.info(f"  check {name}: {value}")

    for subspace in subspaces:
        for mode in MODES:
            _log_density_table(summary, mode, subspace)
    if len(subspaces) > 1:
        for mode in MODES:
            _log_difference_table(differences, mode)
    if undefined:
        LOGGER.warning(
            f"{len(undefined)} undefined metric(s) recorded (not replaced by zero); "
            f"first: {undefined[0]}"
        )
    if failures:
        LOGGER.warning(f"{len(failures)} run(s) failed; see density_sweep_results.json")

    resolved = {
        subspace: build_config(
            densities[0], seeds[0], out_dir, subspace=subspace, **options
        )
        for subspace in subspaces
    }
    config_payload = {
        "method": METHOD,
        "subspaces": subspaces,
        "densities": densities,
        "density_grid_full": DENSITIES,
        "seeds": seeds,
        "cut_rule": args.cut_rule,
        "transfer_cut_rule": args.transfer_cut_rule,
        "static_spectral_convention": {
            "operator": "L_sym = I - D~^{-1/2} (W + I) D~^{-1/2}",
            "ordering": "ascending in lambda(L); U_q = [u_0, ..., u_{q-1}]",
            "constant_eigenvector": "kept (u_0 is the first of the first q)",
            "orthonormalization": (
                "none here; the coarsener M_tau-orthonormalizes every target the "
                "same way, learned or static"
            ),
            "q": "matched per graph to the learned arm's effective target rank",
            "source": "paper, Sec. Preliminaries and Sec. Graph coarsening and "
            "Loukas' RSA (R = span(U_K))",
        },
        "matched_budgets": {
            "n_coarse": list(_MATCHED_N),
            "reduction": list(_MATCHED_REDUCTION),
            "epsilon": list(_MATCHED_EPSILON),
        },
        "resolved_pipeline_config": {
            k: v.to_dict() for k, v in resolved.items()
        },
        "fixed_fingerprint": _fixed_fingerprint(resolved[subspaces[0]]),
        "coarsening_config": asdict(resolved[subspaces[0]].coarsening),
    }
    (out_dir / "density_sweep_config.json").write_text(json.dumps(config_payload, indent=1))
    _write_csv(out_dir / "density_sweep_per_graph.csv", graph_rows)
    _write_csv(out_dir / "density_sweep_per_seed.csv", seed_rows)
    _write_csv(out_dir / "density_sweep_summary.csv", summary)
    _write_csv(out_dir / "density_sweep_matched.csv", matched_summary)
    _write_csv(out_dir / "density_sweep_differences.csv", differences)
    (out_dir / "density_sweep_results.json").write_text(
        json.dumps(
            {
                "config": config_payload,
                "checks": checks,
                "per_graph": graph_rows,
                "per_seed": seed_rows,
                "summary": summary,
                "matched_budgets": matched_summary,
                "matched_rows": matched_rows,
                "reused_from": None if args.reuse_from is None else str(args.reuse_from),
                "paired_differences": differences,
                "failures": failures,
                "undefined_metrics": undefined,
                "runtimes": [
                    {"density": c["density"], "seed": c["seed"],
                     "subspace": c["subspace"], "ok": c["ok"],
                     "runtime_s": c["runtime_s"],
                     "representation_seconds": c.get("representation_seconds")}
                    for c in cells
                ],
            },
            indent=1,
        )
    )
    if not args.no_figures:
        for path in make_figures(summary, differences, out_dir / "figures", subspaces):
            LOGGER.info(f"  wrote {path}")
    LOGGER.info(f"density sweep written to {out_dir}")


if __name__ == "__main__":
    main()
