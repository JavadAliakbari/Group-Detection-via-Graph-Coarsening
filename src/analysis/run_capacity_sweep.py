r"""Does learning beat the static spectral subspace at some model capacity?

A grid over the polynomial bank's **number of heads** and the **node-feature
dimension**, at a fixed synthetic setting (2,000 nodes, planted group density
0.4, closed-form training, ``deflated_ward_tight`` as the only coarsener).  At
every grid point the learned subspace is compared, on identical graphs, with the
static baseline ``R = span(U_q)`` -- the ``q`` lowest-frequency eigenvectors of
``L = I - D~^{-1/2}(W+I)D~^{-1/2}`` (:mod:`src.pipeline.learning.spectral`), with
``q`` matched per graph to the learned subspace's effective rank.

Three disjoint sets of graphs, per seed
---------------------------------------
``train``        fits the learned bank (once per grid point and seed) and each
                 subspace's own deployable stopping rule; training groups only.
``validation``   held-out graphs the grid is **reported** on and the configuration
                 is **selected** on.
``final_test``   fresh graphs, generated from a disjoint seed range and touched
                 exactly once: after the selection is locked, for the selected
                 configuration only, with the models and stopping rules already
                 fitted in the selection phase (nothing is refitted).

Graphs do not depend on the grid: the generator draws the whole structure
(nodes, groups, background) from a numpy stream and the features from a separate
torch stream, so at a fixed seed every grid point sees the same graphs and only
the node features' width changes.  That is verified, not assumed (``checks``).

Selection rule (pre-registered, validation only)
------------------------------------------------
Maximize the mean over seeds of the paired **stopping-rule F1** difference
``learned - static`` on the validation graphs; ties go to the higher learned F1.
``--select-metric`` swaps the criterion; what each alternative *would* have
chosen is reported next to the choice.  A configuration is called **reliable** if
the learned subspace wins on every seed *and* a two-sided Wilcoxon signed-rank
test over the (seed, graph) pairs gives p < 0.05 -- on validation for the grid,
and again on the fresh test graphs for the selected configuration.  No seed is
dropped or chosen on its results; the seed list is an input.

Cost
----
Grid cells are independent, so they run in ``--workers`` processes (BLAS threads
per worker set by ``--blas-threads``); results are identical to a serial run
with the same thread settings, and the smoke test checks repeat runs bit for bit.

Run::

    python -m src.run_capacity_sweep --smoke
    python -m src.run_capacity_sweep --heads 1 2 4 8 16 --feature-dims 8 16 32 64 \
        --seeds 1 2 3 4 5 --workers 5
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np  # noqa: E402

from src.utils.utils import LOGGER, now  # noqa: E402

__all__ = ["main", "cell_config", "run_cell", "select_configuration"]

NUM_NODES = 2000
DENSITY = 0.4
METHOD = "deflated_ward_tight"
SUBSPACES = ("learned", "static_spectral")
MODES = ("stopping_rule", "oracle")
PRIMARY = ("pr_auc", "f1", "precision", "recall", "detection_rate")

#: final-test graphs are generated from ``seed + FINAL_SEED_OFFSET``; with the
#: generator's ``+100 k`` / ``+9000`` offsets this range cannot meet the
#: selection phase's graphs for any seed below 10,000 (checked in ``validate``)
FINAL_SEED_OFFSET = 50_000

#: budgets for N = 2,000 (the density sweep's defaults were set for N = 3,000)
MATCHED_N = (1500, 1200, 1000, 800, 600, 400, 200)
MATCHED_REDUCTION = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MATCHED_EPSILON = (0.6, 0.8, 0.9, 0.95)
#: the useful operating band the matched-supernode heatmap averages over
MATCHED_BAND = (1200, 1000, 800, 600)

SELECTION_METRICS = {
    "diff_f1": ("stopping_rule", "f1", "difference"),
    "diff_pr_auc": ("stopping_rule", "pr_auc", "difference"),
    "diff_detection_rate": ("stopping_rule", "detection_rate", "difference"),
    "learned_f1": ("stopping_rule", "f1", "learned"),
}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
#: The frozen-level group-label head is a fixed-width linear layer on the
#: screened level, whose width follows each graph's effective rank.  Once a wide
#: learned bank is rank-deficient by different amounts on different graphs
#: (widths 994 vs 996 at heads x features = 1024) the head fitted on one graph
#: cannot be applied to another, and the run dies -- in exactly the high-capacity
#: cells.  Group labels are not part of this experiment, and the head cannot
#: influence the representation (the closed form never reads labels; the head is
#: fitted afterwards on the detached level), so it is switched off by default.
#: ``--label-head on`` restores it.
LABEL_HEAD = False


def cell_config(heads, feature_dim, seed, subspace, out_dir):
    """The density sweep's configuration at N=2000, density 0.4, one grid point.

    Built by :func:`src.run_synthetic_density_sweep.build_config` -- itself a deep
    copy of ``run_modular_pipeline``'s ``polynomial-closed-form`` example -- so no
    setting is restated here.  On top of it only the node count, the two grid
    knobs and the per-run output are set.  Per-run CSVs (dominated by the full
    trajectory, ~1.5 MB per graph) are switched off; this module extracts what
    it reports from the results in memory.
    """

    from src.pipeline.logging_visualization import LoggingVisualizationConfig
    from src.run_synthetic_density_sweep import build_config

    config = build_config(
        DENSITY, seed, out_dir, subspace=subspace,
        cut_rule="f1", transfer_cut_rule="score_sum",
    )
    config.data.num_nodes = NUM_NODES
    config.data.feature_dim = int(feature_dim)
    config.learning.num_heads = int(heads)
    config.learning.objective.label_enabled = bool(LABEL_HEAD)
    config.logging = LoggingVisualizationConfig(
        output_dir=out_dir, make_plots=False, write_csv=False
    )
    return config


def fingerprint(config) -> dict:
    """Everything that must not vary across the grid, as a comparable dict."""

    payload = config.to_dict()
    for key in ("group_density", "seed", "feature_dim"):
        payload["data"].pop(key, None)
    for key in ("seed", "architecture", "num_heads"):
        payload["learning"].pop(key, None)
    payload["learning"].get("objective", {}).pop("seed", None)
    payload["coarsening"].pop("seed", None)
    payload.pop("logging")
    payload.pop("seeds")
    payload["data"]["group_density_fixed"] = DENSITY
    return payload


def _graph_hash(graph) -> str:
    """Structure only (edges and groups), so feature width cannot enter it."""

    h = hashlib.sha1()
    h.update(np.ascontiguousarray(graph.edge_index.cpu().numpy()).tobytes())
    for group in graph.groups:
        h.update(np.ascontiguousarray(group.nodes.cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def _deflated_rank(graph, basis, tau) -> int:
    """The rank ``deflated_coarsen`` itself works with (``rank_tol = 1e-10``).

    ``target_rank`` -- what ``q`` is matched to -- comes from the common-epsilon
    orthonormalization (tolerance ``eps * max(shape)``); the deflated coarsener
    uses ``m_orthonormal_basis`` and can keep a few directions fewer.  Both are
    recorded so a mismatch is visible.
    """

    from smooth_dual_ward import m_orthonormal_basis, screened_operators

    from src.pipeline.coarsening import _to_scipy

    _a, _d, _l, M = screened_operators(_to_scipy(graph.adjacency), tau)
    _U, rank = m_orthonormal_basis(basis.detach().cpu().numpy(), M)
    return int(rank)


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
def _init_worker(log_dir: str, blas_threads: int) -> None:
    """Quiet, private logging per worker; BLAS threads fixed.

    Each process imports ``src.utils.utils``, which opens a timestamped log file
    as a side effect; with several workers starting in the same second they
    would share it.  The worker drops those handlers and writes its own file.
    It uses a plain stream handler on purpose: ``run_pipeline`` relocates every
    ``logging.FileHandler`` into each run's folder, and that must not carry the
    worker log away.
    """

    import logging
    import shutil

    import torch

    global LABEL_HEAD
    LABEL_HEAD = os.environ.get("CAPACITY_LABEL_HEAD", "0") == "1"
    torch.set_num_threads(int(blas_threads))
    torch.set_default_dtype(torch.float64)
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()
            try:
                Path(handler.baseFilename).unlink()
                parent = Path(handler.baseFilename).parent
                if parent.is_dir() and not any(parent.iterdir()):
                    shutil.rmtree(parent, ignore_errors=True)
            except OSError:
                pass
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stream = open(Path(log_dir) / f"worker_{os.getpid()}.log", "a")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def _timed(learner, timings: dict, key: str):
    """Time ``learner.run`` without changing it; undone before pickling."""

    original = learner.run

    def run(graphs):
        started = time.perf_counter()
        try:
            return original(graphs)
        finally:
            timings[key] = time.perf_counter() - started

    learner.run = run


def _strip(learner) -> None:
    """Drop per-graph caches (keyed by graph id!) and the timing wrapper."""

    learner.__dict__.pop("run", None)
    learner._geometry_cache.clear()
    # the polynomial bank's propagated-feature stack, [phi_k(A_hat) X]_k, is also
    # cached per graph id: (degree+1) x N x F per graph, ~34 MB at F=64, and it
    # would otherwise be pickled back to the parent with every cell
    if hasattr(learner, "_stack_cache"):
        learner._stack_cache.clear()
    if hasattr(learner, "spectra_"):
        learner.spectra_.clear()


#: a transferred score_sum* below this means the arm's own cumulative-cost axis is
#: flat over (almost) the whole hierarchy, so the budgeted cut lands at an
#: arbitrary coarse level -- the stopping-rule number then measures the rule, not
#: the subspace
DEGENERATE_SCORE_SUM = 1e-6


def _cut_rule_fields(rule) -> dict:
    if rule is None:
        return {"transferred_score_sum": None, "transferred_reduction": None,
                "stopping_axis_degenerate": None}
    return {
        "transferred_score_sum": float(rule.score_sum),
        "transferred_reduction": float(rule.reduction),
        "stopping_axis_degenerate": bool(rule.score_sum < DEGENERATE_SCORE_SUM),
    }


def _reduction_transfer_row(evaluation, rule, base, seed, subspace) -> "dict | None":
    """A second label-free deployable cut: the learned rule's *reduction*.

    Same training-group-fitted operating point, carried to the test graph by its
    kept fraction instead of by ``score_sum`` -- read off the tree already built,
    so it costs nothing and shares every other property of the primary rule.
    Reported as a robustness check, never as the primary result.
    """

    if rule is None:
        return None
    target = 1.0 - float(rule.reduction)
    row = min(evaluation.coarsening.trajectory, key=lambda r: abs(r["retained"] - target))
    return {
        "density": DENSITY, "seed": int(seed), "subspace": subspace,
        "graph_id": evaluation.graph_id, "scope": evaluation.scope,
        "budget_kind": "transfer_reduction", "budget": -1.0,
        "transfer_reduction_value": float(rule.reduction),
        "n_coarse": int(row["n_coarse"]), "reduction": float(row["reduction"]),
        "retained": float(row["retained"]), "epsilon": row.get("epsilon"),
        "epsilon_q": row.get("epsilon_q"), "score_sum": row.get("score_sum"),
        "pr_auc": float(evaluation.coarsening.pr_auc),
        "f1": row.get("all_mean_f1"), "precision": row.get("all_mean_precision"),
        "recall": row.get("all_mean_recall"),
        "detection_rate": row.get("all_detection_rate"), **base,
    }


def _realized_density(graph) -> float:
    from src.run_synthetic_density_sweep import _realized_density as measure

    return measure(graph)


def _rows(evaluation, *, heads, feature_dim, seed, subspace, role, extra, undefined,
          rule=None):
    from src.run_synthetic_density_sweep import _graph_rows, _matched_rows

    extra = {**extra, **_cut_rule_fields(rule)}
    base = {
        "heads": int(heads),
        "feature_dim": int(feature_dim),
        "nominal_width": int(heads) * int(feature_dim),
        "graph_role": role,
        **extra,
    }
    rows = _graph_rows(evaluation, DENSITY, seed, subspace, base, undefined)
    matched = _matched_rows(
        evaluation, DENSITY, seed, subspace,
        n_grid=MATCHED_N, reduction_grid=MATCHED_REDUCTION,
        epsilon_grid=MATCHED_EPSILON,
    )
    for row in matched:
        row.update(base)
    if role != "train":
        transfer = _reduction_transfer_row(evaluation, rule, base, seed, subspace)
        if transfer is not None:
            matched.append(transfer)
    return rows, matched


def run_cell(task: dict) -> dict:
    """One (heads, feature_dim, seed): fit once, run both arms on train+validation.

    Returns the rows plus the fitted learners and stopping rules, so the final
    test phase can reuse them without refitting.
    """

    import torch

    from src.pipeline.learning import StaticSpectralLearning, build_learner
    from src.pipeline.pipeline import run_pipeline

    torch.set_default_dtype(torch.float64)
    heads, feature_dim, seed = task["heads"], task["feature_dim"], task["seed"]
    out = Path(task["out"]) / "runs" / f"h{heads}_f{feature_dim}_s{seed}"
    started = time.perf_counter()
    try:
        # ---- learned arm: the ONLY fit of this configuration ----------------
        cfg_l = cell_config(heads, feature_dim, seed, "learned", out / "learned")
        learner = build_learner(cfg_l.learning)
        timings: dict = {}
        _timed(learner, timings, "learning_seconds")
        result_l = run_pipeline(cfg_l, learner=learner)
        ranks = {e.graph_id: int(e.coarsening.target_rank) for e in result_l.evaluations}

        # ---- static arm on the same graphs, q matched per graph --------------
        cfg_s = cell_config(heads, feature_dim, seed, "static_spectral", out / "static")
        static = StaticSpectralLearning(cfg_s.learning, target_widths=ranks)
        _timed(static, timings, "static_setup_seconds")
        result_s = run_pipeline(cfg_s, learner=static)
    except Exception as error:  # noqa: BLE001 -- one cell must not end the grid
        import traceback

        return {
            **task, "ok": False, "rows": [], "matched": [], "undefined": [],
            "failure": {**{k: task[k] for k in ("heads", "feature_dim", "seed")},
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(limit=8)},
            "seconds": time.perf_counter() - started,
        }

    rows, matched, undefined, hashes = [], [], [], {}
    tau = cfg_l.coarsening.tau
    for subspace, result, fitted in (
        ("learned", result_l, learner), ("static_spectral", result_s, static)
    ):
        for evaluation in result.evaluations:
            graph = evaluation.graph
            hashes[graph.graph_id] = _graph_hash(graph)
            basis = (
                result.learning.representations[graph.graph_id]
                if graph.graph_id in result.learning.representations
                else fitted.represent(graph)
            )
            spectrum = getattr(fitted, "spectra_", {}).get(graph.graph_id, {})
            extra = {
                "rank_common": int(evaluation.coarsening.target_rank),
                "rank_deflated": _deflated_rank(graph, basis, tau),
                "requested_q": spectrum.get("requested_q"),
                "learning_seconds": timings.get("learning_seconds")
                if subspace == "learned" else None,
                "eigensolver_seconds": spectrum.get("seconds"),
                "coarsening_seconds": evaluation.coarsening.timings.get("total"),
                "hierarchy_seconds": evaluation.coarsening.timings.get("hierarchy"),
                "graph_hash": hashes[graph.graph_id],
            }
            evaluation.realized_density = _realized_density(graph)
            role = "train" if evaluation.scope == "train" else "validation"
            r, m = _rows(
                evaluation, heads=heads, feature_dim=feature_dim, seed=seed,
                subspace=subspace, role=role, extra=extra, undefined=undefined,
                rule=result.cut_rule,
            )
            rows.extend(r)
            matched.extend(m)

    fp = fingerprint(cfg_l)
    labels = {
        "learned": result_l.learning.label_head is not None
        and cfg_l.learning.objective.label_enabled,
        "static_spectral": result_s.learning.label_head is not None
        and cfg_s.learning.objective.label_enabled,
    }
    _strip(learner)
    _strip(static)
    return {
        **task, "ok": True, "rows": rows, "matched": matched, "undefined": undefined,
        "failure": None, "seconds": time.perf_counter() - started,
        "graph_hashes": hashes, "fingerprint": fp,
        "learners": {"learned": learner, "static_spectral": static},
        "cut_rules": {"learned": result_l.cut_rule, "static_spectral": result_s.cut_rule},
        "labels_enabled": labels,
        "coarsening_config": copy.deepcopy(cfg_l.coarsening),
        "data_config": copy.deepcopy(cfg_l.data),
        "timings": timings,
    }


def run_final(task: dict) -> dict:
    """The selected configuration on fresh graphs, with the fitted models reused."""

    import torch

    from src.pipeline.coarsening import Coarsening
    from src.pipeline.data import Data
    from src.pipeline.pipeline import _evaluate

    torch.set_default_dtype(torch.float64)
    heads, feature_dim, seed = task["heads"], task["feature_dim"], task["seed"]
    data = copy.deepcopy(task["data_config"])
    data.seed = int(seed) + FINAL_SEED_OFFSET
    data.num_train_graphs = 1  # generated and discarded; only test graphs are used
    data.num_test_graphs = int(task["n_final"])
    graphs = Data(data).run().test_graphs
    for k, graph in enumerate(graphs):
        # graph ids key the learners' caches; a fresh graph must never reuse a
        # validation graph's cached geometry or eigenvectors
        graph.graph_id = f"F{k}"

    learners, cut_rules = task["learners"], task["cut_rules"]
    coarsener = Coarsening(task["coarsening_config"])
    rows, matched, undefined, hashes = [], [], [], {}
    for graph in graphs:
        hashes[graph.graph_id] = _graph_hash(graph)
        started = time.perf_counter()
        z = learners["learned"].represent(graph)
        represent_l = time.perf_counter() - started
        result_l = coarsener.run(graph, z, cut_rule=cut_rules["learned"])
        learners["static_spectral"].target_widths[graph.graph_id] = int(result_l.target_rank)
        u = learners["static_spectral"].represent(graph)
        result_s = coarsener.run(graph, u, cut_rule=cut_rules["static_spectral"])
        for subspace, result, basis in (
            ("learned", result_l, z), ("static_spectral", result_s, u)
        ):
            evaluation = _evaluate(
                learners[subspace], graph, result, "test", task["labels_enabled"][subspace]
            )
            evaluation.realized_density = _realized_density(graph)
            spectrum = getattr(learners[subspace], "spectra_", {}).get(graph.graph_id, {})
            extra = {
                "rank_common": int(result.target_rank),
                "rank_deflated": _deflated_rank(graph, basis, task["coarsening_config"].tau),
                "requested_q": spectrum.get("requested_q"),
                "learning_seconds": None,
                "represent_seconds": represent_l if subspace == "learned" else None,
                "eigensolver_seconds": spectrum.get("seconds"),
                "coarsening_seconds": result.timings.get("total"),
                "hierarchy_seconds": result.timings.get("hierarchy"),
                "graph_hash": hashes[graph.graph_id],
                "graph_generation_seed": int(data.seed),
            }
            r, m = _rows(
                evaluation, heads=heads, feature_dim=feature_dim, seed=seed,
                subspace=subspace, role="final_test", extra=extra, undefined=undefined,
                rule=cut_rules[subspace],
            )
            rows.extend(r)
            matched.extend(m)
    return {**task, "ok": True, "rows": rows, "matched": matched,
            "undefined": undefined, "graph_hashes": hashes,
            "learners": None, "cut_rules": None}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _mean_sd(values):
    values = _finite(values)
    if not values:
        return None, None, 0
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        len(values),
    )


def per_seed(rows: list, role: str) -> list:
    """One row per (config, seed, subspace, mode): the role's graphs pooled."""

    buckets: dict = {}
    for row in rows:
        if row["graph_role"] != role:
            continue
        key = (row["heads"], row["feature_dim"], row["seed"], row["subspace"],
               row["evaluation_mode"])
        buckets.setdefault(key, []).append(row)
    out = []
    for (heads, fdim, seed, subspace, mode), group in sorted(buckets.items()):
        entry = {"heads": heads, "feature_dim": fdim, "nominal_width": heads * fdim,
                 "seed": seed, "subspace": subspace, "evaluation_mode": mode,
                 "graph_role": role, "n_graphs": len(group)}
        for metric in PRIMARY + ("detected", "n_groups", "n_coarse", "reduction",
                                 "retained", "epsilon", "epsilon_q", "score_sum",
                                 "rank_common", "rank_deflated", "requested_q",
                                 "learning_seconds", "eigensolver_seconds",
                                 "coarsening_seconds"):
            entry[metric] = _mean_sd([r.get(metric) for r in group])[0]
        entry["transferred_score_sum"] = group[0].get("transferred_score_sum")
        entry["stopping_axis_degenerate"] = group[0].get("stopping_axis_degenerate")
        entry["rank_common_min"] = min(r["rank_common"] for r in group)
        entry["rank_common_max"] = max(r["rank_common"] for r in group)
        out.append(entry)
    return out


def grid_summary(seed_rows: list) -> list:
    """Mean, sd and run count across seeds per (config, subspace, mode)."""

    buckets: dict = {}
    for row in seed_rows:
        key = (row["heads"], row["feature_dim"], row["subspace"], row["evaluation_mode"],
               row["graph_role"])
        buckets.setdefault(key, []).append(row)
    out = []
    for (heads, fdim, subspace, mode, role), group in sorted(buckets.items()):
        entry = {"heads": heads, "feature_dim": fdim, "nominal_width": heads * fdim,
                 "subspace": subspace, "evaluation_mode": mode, "graph_role": role,
                 "n_runs": len(group), "seeds": sorted(r["seed"] for r in group),
                 "degenerate_stopping_axis_seeds": sum(
                     bool(r.get("stopping_axis_degenerate")) for r in group)}
        for metric in PRIMARY + ("n_coarse", "reduction", "epsilon", "epsilon_q",
                                 "rank_common", "rank_deflated", "requested_q",
                                 "learning_seconds", "eigensolver_seconds",
                                 "coarsening_seconds"):
            mean, sd, n = _mean_sd([r.get(metric) for r in group])
            entry[f"{metric}_mean"], entry[f"{metric}_std"], entry[f"{metric}_n"] = mean, sd, n
        out.append(entry)
    return out


def paired(rows: list, role: str) -> list:
    """``learned - static`` per (config, mode, metric): seed-level and graph-level.

    Seed level: the seed's graphs are averaged first, then the difference is
    taken -- the unit the reliability rule counts wins over.  Graph level: every
    (seed, graph) pair, which is what the Wilcoxon test runs on.
    """

    from scipy.stats import wilcoxon

    index = {
        (r["heads"], r["feature_dim"], r["seed"], r["graph_id"], r["subspace"],
         r["evaluation_mode"]): r
        for r in rows if r["graph_role"] == role
    }
    configs = sorted({(k[0], k[1]) for k in index})
    out = []
    for heads, fdim in configs:
        for mode in MODES:
            for metric in PRIMARY:
                graph_diffs, per_seed_diffs = [], {}
                for (h, f, seed, gid, sub, md), row in index.items():
                    if (h, f, sub, md) != (heads, fdim, "learned", mode):
                        continue
                    other = index.get((h, f, seed, gid, "static_spectral", md))
                    if other is None or row.get(metric) is None or other.get(metric) is None:
                        continue
                    d = float(row[metric]) - float(other[metric])
                    graph_diffs.append(d)
                    per_seed_diffs.setdefault(seed, []).append(d)
                seed_means = {s: float(np.mean(v)) for s, v in per_seed_diffs.items()}
                mean, sd, n = _mean_sd(list(seed_means.values()))
                nonzero = [d for d in graph_diffs if d != 0.0]
                p = (
                    float(wilcoxon(graph_diffs).pvalue)
                    if len(nonzero) >= 3 else float("nan")
                )
                out.append({
                    "heads": heads, "feature_dim": fdim, "nominal_width": heads * fdim,
                    "evaluation_mode": mode, "metric": metric, "graph_role": role,
                    "mean_difference": mean, "std_difference": sd, "n_seeds": n,
                    "per_seed_difference": {str(s): v for s, v in sorted(seed_means.items())},
                    "seed_wins": sum(v > 0 for v in seed_means.values()),
                    "seed_losses": sum(v < 0 for v in seed_means.values()),
                    "graph_pairs": len(graph_diffs),
                    "graph_wins": sum(d > 0 for d in graph_diffs),
                    "graph_losses": sum(d < 0 for d in graph_diffs),
                    "wilcoxon_p": p,
                    "reliable_learned_win": bool(
                        n > 0 and all(v > 0 for v in seed_means.values())
                        and math.isfinite(p) and p < 0.05
                    ),
                    "reliable_static_win": bool(
                        n > 0 and all(v < 0 for v in seed_means.values())
                        and math.isfinite(p) and p < 0.05
                    ),
                })
    return out


def matched_paired(matched: list, role: str) -> list:
    """Learned - static at identical budgets, seed-averaged, per config."""

    index = {}
    for r in matched:
        if r["graph_role"] != role:
            continue
        key = (r["heads"], r["feature_dim"], r["seed"], r["graph_id"],
               r["budget_kind"], r["budget"], r["subspace"])
        index[key] = r
    buckets: dict = {}
    for key, row in index.items():
        heads, fdim, seed, gid, kind, budget, sub = key
        if sub != "learned":
            continue
        other = index.get((heads, fdim, seed, gid, kind, budget, "static_spectral"))
        if other is None:
            continue
        slot = buckets.setdefault((heads, fdim, kind, budget), {})
        for metric in ("f1", "precision", "recall", "detection_rate"):
            a, b = row.get(metric), other.get(metric)
            if a is not None and b is not None:
                slot.setdefault(metric, {}).setdefault(seed, []).append(a - b)
        slot.setdefault("n_learned", {}).setdefault(seed, []).append(row["n_coarse"])
        slot.setdefault("n_static", {}).setdefault(seed, []).append(other["n_coarse"])
    out = []
    for (heads, fdim, kind, budget), slot in sorted(buckets.items()):
        entry = {"heads": heads, "feature_dim": fdim, "budget_kind": kind,
                 "budget": budget, "graph_role": role}
        for metric, per in slot.items():
            seed_means = [float(np.mean(v)) for v in per.values()]
            mean, sd, n = _mean_sd(seed_means)
            name = metric if metric.startswith("n_") else f"diff_{metric}"
            entry[f"{name}_mean"], entry[f"{name}_std"] = mean, sd
            if not metric.startswith("n_"):
                entry[f"{name}_seed_wins"] = sum(v > 0 for v in seed_means)
                entry[f"{name}_n_seeds"] = n
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def select_configuration(validation_paired: list, validation_summary: list,
                         criterion: str) -> dict:
    """Pick one configuration from validation results only.

    The function is handed validation aggregates and nothing else; the final
    test rows do not exist yet when it runs.
    """

    mode, metric, kind = SELECTION_METRICS[criterion]
    learned = {
        (r["heads"], r["feature_dim"]): r[f"{metric}_mean"]
        for r in validation_summary
        if r["subspace"] == "learned" and r["evaluation_mode"] == mode
    }
    diffs = {
        (r["heads"], r["feature_dim"]): r
        for r in validation_paired
        if r["evaluation_mode"] == mode and r["metric"] == metric
    }
    scores = {}
    for config in learned:
        if kind == "difference":
            value = diffs.get(config, {}).get("mean_difference")
        else:
            value = learned[config]
        if value is not None:
            scores[config] = (value, learned[config] or -1.0)
    best = max(scores, key=lambda c: scores[c])
    return {
        "criterion": criterion,
        "mode": mode,
        "metric": metric,
        "kind": kind,
        "selected": {"heads": best[0], "feature_dim": best[1]},
        "score": scores[best][0],
        "scores": {f"h{c[0]}_f{c[1]}": v[0] for c, v in sorted(scores.items())},
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _log_grid_table(summary, paired_rows, mode, role):
    title = (
        f"{role.upper()} {'STOPPING RULE (deployable)' if mode == 'stopping_rule' else 'ORACLE (post-hoc upper bound)'}"
        " -- learned | static | learned-static (seed W/L, Wilcoxon p)"
    )
    idx = {(r["heads"], r["feature_dim"], r["subspace"], r["evaluation_mode"]): r
           for r in summary if r["graph_role"] == role}
    pidx = {(r["heads"], r["feature_dim"], r["evaluation_mode"], r["metric"]): r
            for r in paired_rows}
    LOGGER.info("\n" + "=" * 150)
    LOGGER.info(title)
    LOGGER.info("=" * 150)
    LOGGER.info(
        "  heads fdim  q(L)   q(S)  | " + " | ".join(
            f"{m:^34s}" for m in ("F1", "PR-AUC", "detection rate")
        )
    )
    for heads, fdim in sorted({(k[0], k[1]) for k in idx}):
        L = idx.get((heads, fdim, "learned", mode))
        S = idx.get((heads, fdim, "static_spectral", mode))
        if not L or not S:
            continue
        cells = []
        for metric in ("f1", "pr_auc", "detection_rate"):
            p = pidx.get((heads, fdim, mode, metric), {})
            flag = "*" if p.get("reliable_learned_win") else (
                "-" if p.get("reliable_static_win") else " ")
            cells.append(
                f"{L[metric + '_mean']:.3f} {S[metric + '_mean']:.3f} "
                f"{p.get('mean_difference', float('nan')):+.3f}{flag}"
                f"({p.get('seed_wins', 0)}/{p.get('seed_losses', 0)},"
                f"{p.get('wilcoxon_p', float('nan')):.2g})"
            )
        degenerate = (
            f"  [static stopping axis degenerate on {S['degenerate_stopping_axis_seeds']}"
            f"/{S['n_runs']} seeds]"
            if mode == "stopping_rule" and S.get("degenerate_stopping_axis_seeds") else ""
        )
        LOGGER.info(
            f"  {heads:5d} {fdim:4d} {L['rank_common_mean']:5.0f} "
            f"{S['rank_common_mean']:6.0f} | " + " | ".join(c.ljust(34) for c in cells)
            + degenerate
        )
    LOGGER.info(
        "  * = reliable learned win (every seed + Wilcoxon p<0.05 over seed x graph "
        "pairs), - = reliable static win; q = mean effective rank"
    )


def _write_csv(path: Path, rows: list) -> None:
    from src.run_synthetic_density_sweep import _write_csv as write

    flat = []
    for row in rows:
        flat.append({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in row.items()})
    write(path, flat)


def make_figures(out_dir: Path, heads, fdims, val_paired, val_matched, selection,
                 final_paired, val_summary=()) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir.mkdir(parents=True, exist_ok=True)
    heads, fdims = sorted(heads), sorted(fdims)
    sel = (selection["selected"]["heads"], selection["selected"]["feature_dim"])
    final = {(r["evaluation_mode"], r["metric"]): r for r in final_paired}
    written = []

    def grid(lookup):
        values = np.full((len(heads), len(fdims)), np.nan)
        notes = [["" for _ in fdims] for _ in heads]
        for i, h in enumerate(heads):
            for j, f in enumerate(fdims):
                entry = lookup(h, f)
                if entry is None:
                    continue
                values[i, j], notes[i][j] = entry
        return values, notes

    def panel(ax, values, notes, title, vmax):
        image = ax.imshow(values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto",
                          origin="lower")
        ax.set_xticks(range(len(fdims)), [str(f) for f in fdims])
        ax.set_yticks(range(len(heads)), [str(h) for h in heads])
        ax.set_xlabel("node-feature dimension")
        ax.set_ylabel("polynomial heads")
        ax.set_title(title, fontsize=10)
        for i in range(len(heads)):
            for j in range(len(fdims)):
                if notes[i][j]:
                    ax.text(j, i, notes[i][j], ha="center", va="center", fontsize=7,
                            color="black")
        if sel[0] in heads and sel[1] in fdims:
            i, j = heads.index(sel[0]), fdims.index(sel[1])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=2.6))
        return image

    degenerate = {
        (r["heads"], r["feature_dim"]): r.get("degenerate_stopping_axis_seeds", 0)
        for r in val_summary
        if r["subspace"] == "static_spectral" and r["evaluation_mode"] == "stopping_rule"
        and r["graph_role"] == "validation"
    }

    def paired_lookup(mode, metric):
        index = {(r["heads"], r["feature_dim"]): r for r in val_paired
                 if r["evaluation_mode"] == mode and r["metric"] == metric}

        def lookup(h, f):
            r = index.get((h, f))
            if r is None or r["mean_difference"] is None:
                return None
            mark = "*" if r["reliable_learned_win"] else ("-" if r["reliable_static_win"] else "")
            dagger = "†" if mode == "stopping_rule" and degenerate.get((h, f)) else ""
            return (r["mean_difference"],
                    f"{r['mean_difference']:+.3f}{mark}{dagger}\n"
                    f"{r['seed_wins']}/{r['seed_losses']}")
        return lookup

    for mode, label in (("stopping_rule", "stopping rule (deployable)"),
                        ("oracle", "oracle (post-hoc upper bound)")):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
        for ax, (metric, name) in zip(axes, (("f1", "F1"), ("pr_auc", "PR-AUC"),
                                             ("detection_rate", "detection rate"))):
            values, notes = grid(paired_lookup(mode, metric))
            vmax = max(0.02, float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 0.02)
            image = panel(ax, values, notes, f"validation: learned − static {name}", vmax)
            fig.colorbar(image, ax=ax, shrink=0.85)
            f = final.get((mode, metric))
            if f is not None and f["mean_difference"] is not None:
                ax.text(
                    0.5, -0.24,
                    f"selected h={sel[0]}, f={sel[1]} on FRESH test graphs: "
                    f"{f['mean_difference']:+.3f} (seeds {f['seed_wins']}/{f['seed_losses']}, "
                    f"graphs {f['graph_wins']}/{f['graph_losses']}, p={f['wilcoxon_p']:.2g})",
                    transform=ax.transAxes, ha="center", fontsize=8,
                )
        fig.suptitle(
            f"Learned − static spectral subspace, {label} — N={NUM_NODES}, density "
            f"{DENSITY}, {METHOD}\nblue = learned better, red = static better; cell text "
            "= mean over seeds and seed W/L; * reliable learned win, − reliable static "
            "win; † static stopping axis degenerate (cut artifact); box = selected on "
            "validation",
            fontsize=10.5,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = out_dir / f"capacity_heatmap_{mode}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    # matched budgets: supernodes (useful band) and common epsilon
    band = {}
    for r in val_matched:
        if r["budget_kind"] == "n_coarse" and r["budget"] in MATCHED_BAND:
            band.setdefault((r["heads"], r["feature_dim"]), []).append(r["diff_f1_mean"])
    eps = {(r["heads"], r["feature_dim"]): r for r in val_matched
           if r["budget_kind"] == "epsilon" and abs(r["budget"] - 0.9) < 1e-9}
    transfer = {(r["heads"], r["feature_dim"]): r for r in val_matched
                if r["budget_kind"] == "transfer_reduction"}
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    values, notes = grid(lambda h, f: None if (h, f) not in transfer else (
        transfer[(h, f)]["diff_f1_mean"],
        f"{transfer[(h, f)]['diff_f1_mean']:+.3f}\n"
        f"{transfer[(h, f)]['diff_f1_seed_wins']}/{transfer[(h, f)]['diff_f1_n_seeds']}"))
    vmax = max(0.02, float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 0.02)
    image = panel(axes[2], values, notes,
                  "secondary deployable rule: transfer by REDUCTION\n"
                  "learned − static F1 (cell: mean, seed wins)", vmax)
    fig.colorbar(image, ax=axes[2], shrink=0.85)
    values, notes = grid(lambda h, f: None if (h, f) not in band else (
        float(np.mean(band[(h, f)])), f"{np.mean(band[(h, f)]):+.3f}"))
    vmax = max(0.02, float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 0.02)
    image = panel(axes[0], values, notes,
                  f"matched supernodes: learned − static F1\n(mean over n in {list(MATCHED_BAND)})",
                  vmax)
    fig.colorbar(image, ax=axes[0], shrink=0.85)
    values, notes = grid(lambda h, f: None if (h, f) not in eps else (
        eps[(h, f)]["diff_f1_mean"],
        f"{eps[(h, f)]['diff_f1_mean']:+.3f}\nn {eps[(h, f)]['n_learned_mean']:.0f}/"
        f"{eps[(h, f)]['n_static_mean']:.0f}"))
    vmax = max(0.02, float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 0.02)
    image = panel(axes[1], values, notes,
                  "matched common ε = 0.9: learned − static F1\n(cell: supernodes learned/static "
                  "— not like-for-like compression)", vmax)
    fig.colorbar(image, ax=axes[1], shrink=0.85)
    fig.suptitle("Validation, matched budgets: the same hierarchy level on each tree, "
                 "so cut selection cannot explain a difference", fontsize=10.5)
    fig.tight_layout()
    path = out_dir / "capacity_heatmap_matched.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)
    return written


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(cells, heads, fdims, seeds, final_cells) -> dict:
    rows = [r for c in cells for r in c["rows"]]
    checks = {
        "grid_complete": sorted({(c["heads"], c["feature_dim"]) for c in cells if c["ok"]})
        == sorted((h, f) for h in heads for f in fdims),
        "every_seed_at_every_grid_point": all(
            sorted(c["seed"] for c in cells if c["ok"] and (c["heads"], c["feature_dim"]) == (h, f))
            == sorted(seeds) for h in heads for f in fdims
        ),
        "only_one_method_executed": {r["method"] for r in rows} == {METHOD},
        "non_grid_config_is_fixed": len(
            {json.dumps(c["fingerprint"], sort_keys=True) for c in cells if c["ok"]}
        ) == 1,
        "q_matched_to_learned_rank": all(
            r["requested_q"] is None or int(r["requested_q"]) == int(
                next(x["rank_common"] for x in rows
                     if x["subspace"] == "learned" and x["heads"] == r["heads"]
                     and x["feature_dim"] == r["feature_dim"] and x["seed"] == r["seed"]
                     and x["graph_id"] == r["graph_id"])
            )
            for r in rows if r["subspace"] == "static_spectral"
            and r["evaluation_mode"] == "stopping_rule"
        ),
        "both_ranks_recorded": all(
            r.get("rank_common") is not None and r.get("rank_deflated") is not None
            for r in rows
        ),
    }
    # identical graphs across the grid for a seed
    by_seed: dict = {}
    for c in cells:
        if c["ok"]:
            by_seed.setdefault(c["seed"], set()).add(
                json.dumps(c["graph_hashes"], sort_keys=True))
    checks["identical_graphs_across_grid"] = all(len(v) == 1 for v in by_seed.values())
    if final_cells:
        selection_hashes = {h for c in cells if c["ok"] for h in c["graph_hashes"].values()}
        final_hashes = {h for c in final_cells for h in c["graph_hashes"].values()}
        checks["final_test_graphs_are_fresh"] = not (selection_hashes & final_hashes)
        checks["final_test_uses_only_selected_config"] = len(
            {(c["heads"], c["feature_dim"]) for c in final_cells}) == 1
    checks["modes_kept_separate"] = {r["evaluation_mode"] for r in rows} == set(MODES)
    checks["stopping_rule_cut_uses_no_test_labels"] = all(
        r["cut_rule"] != "f1_oracle" for r in rows if r["evaluation_mode"] == "stopping_rule")
    return checks


_TIMING_FIELDS = {"learning_seconds", "eigensolver_seconds", "coarsening_seconds",
                  "hierarchy_seconds", "runtime_s", "representation_seconds",
                  "represent_seconds"}


def _compare(rows_a: list, rows_b: list) -> dict:
    """Bit-for-bit comparison of two row lists, ignoring wall-clock fields."""

    if len(rows_a) != len(rows_b):
        return {"deterministic": False, "reason": f"{len(rows_a)} vs {len(rows_b)} rows"}
    worst, field = 0.0, None
    for x, y in zip(rows_a, rows_b):
        for key, value in x.items():
            if key in _TIMING_FIELDS:
                continue
            other = y.get(key)
            if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                gap = abs(float(value) - float(other))
                if gap > worst:
                    worst, field = gap, key
            elif value != other:
                return {"deterministic": False, "reason": f"{key}: {value!r} != {other!r}"}
    return {"deterministic": worst == 0.0, "max_abs_difference": worst, "field": field}


def fresh_process_check(task: dict, pooled: dict, out: Path, blas_threads: int) -> dict:
    """Recompute one pooled cell in a brand-new process and compare bit for bit.

    A repeat inside the same worker cannot see state that survives between tasks
    in a long-lived process (ARPACK's did); a fresh process can.  The narrowest
    configuration is the one to test: its static tree is the most sensitive to a
    last-bit difference in the basis.
    """

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=context, initializer=_init_worker,
        initargs=(str(out / "logs"), blas_threads),
    ) as fresh:
        repeat = fresh.submit(run_cell, dict(task)).result()
    if not (pooled["ok"] and repeat["ok"]):
        return {"deterministic": False, "reason": "a run failed"}
    return _compare(pooled["rows"], repeat["rows"])


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--feature-dims", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--final-test-graphs", type=int, default=8,
                        help="fresh test graphs per seed for the selected configuration")
    parser.add_argument("--select-metric", choices=sorted(SELECTION_METRICS),
                        default="diff_f1")
    parser.add_argument("--label-head", choices=("off", "on"), default="off",
                        help="fit the frozen-level group-label head (fails on wide, "
                        "rank-deficient banks; not needed for detection metrics)")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="a 2x2 grid, 2 seeds, 2 fresh graphs, plus a determinism check")
    args = parser.parse_args()

    global LABEL_HEAD
    LABEL_HEAD = args.label_head == "on"
    os.environ["CAPACITY_LABEL_HEAD"] = "1" if LABEL_HEAD else "0"
    heads, fdims, seeds = list(args.heads), list(args.feature_dims), list(args.seeds)
    n_final = int(args.final_test_graphs)
    if args.smoke:
        heads, fdims, seeds, n_final = heads[:2], fdims[:2], seeds[:2], 2
    if max(seeds) >= FINAL_SEED_OFFSET - 10_000:
        raise ValueError("seeds must stay below the final-test seed range")
    widest = max(heads) * max(fdims)
    if widest >= NUM_NODES // 2:
        LOGGER.warning(
            f"the widest configuration asks for q={widest} of N={NUM_NODES}; the static "
            "baseline then spans more than half the spectrum"
        )
    out = Path(args.out or f"results2/capacity_sweep/{now}/")
    out.mkdir(parents=True, exist_ok=True)
    # children read these at their own numpy import; the parent's BLAS is untouched
    for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS"):
        os.environ[var] = str(args.blas_threads)

    LOGGER.info(
        f"capacity sweep: heads={heads} x feature_dims={fdims} = {len(heads) * len(fdims)} "
        f"configurations x {len(seeds)} seeds, N={NUM_NODES}, density={DENSITY}, "
        f"coarsener={METHOD} only, subspaces={list(SUBSPACES)}, selection by "
        f"{args.select_metric!r} on validation graphs, {n_final} fresh test graphs per "
        f"seed for the selected configuration, {args.workers} workers, out={out}"
    )

    tasks = [
        {"heads": h, "feature_dim": f, "seed": s, "out": str(out)}
        for h in heads for f in fdims for s in seeds
    ]
    tasks.sort(key=lambda t: -t["heads"] * t["feature_dim"])  # widest first
    context = mp.get_context("spawn")
    cells: list = []
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context, initializer=_init_worker,
        initargs=(str(out / "logs"), args.blas_threads),
    ) as pool:
        for cell in pool.map(run_cell, tasks):
            cells.append(cell)
            LOGGER.info(
                f"  [{len(cells)}/{len(tasks)}] heads={cell['heads']:2d} "
                f"fdim={cell['feature_dim']:3d} seed={cell['seed']}: "
                f"{'ok' if cell['ok'] else 'FAILED ' + cell['failure']['error']} "
                f"({cell['seconds']:.0f}s, elapsed {time.time() - started:.0f}s)"
            )

        # ---- selection: validation rows only ---------------------------------
        rows = [r for c in cells for r in c["rows"]]
        matched = [r for c in cells for r in c["matched"]]
        val_seed = per_seed(rows, "validation")
        val_summary = grid_summary(val_seed)
        val_paired = paired(rows, "validation")
        val_matched = matched_paired(matched, "validation")
        selection = select_configuration(val_paired, val_summary, args.select_metric)
        selection["alternatives"] = {
            name: select_configuration(val_paired, val_summary, name)["selected"]
            for name in SELECTION_METRICS
        }
        sel = (selection["selected"]["heads"], selection["selected"]["feature_dim"])
        LOGGER.info(
            f"SELECTED on validation ({args.select_metric}): heads={sel[0]}, "
            f"feature_dim={sel[1]} (score {selection['score']:+.4f}); other criteria "
            f"would pick {selection['alternatives']}"
        )
        (out / "capacity_selection.json").write_text(json.dumps(selection, indent=1))

        # ---- final test: selected configuration, fresh graphs, models reused --
        by_key = {(c["heads"], c["feature_dim"], c["seed"]): c for c in cells if c["ok"]}
        final_tasks = [
            {k: by_key[(sel[0], sel[1], s)][k] for k in
             ("heads", "feature_dim", "seed", "learners", "cut_rules", "labels_enabled",
              "coarsening_config", "data_config")} | {"n_final": n_final}
            for s in seeds if (sel[0], sel[1], s) in by_key
        ]
        final_cells = list(pool.map(run_final, final_tasks))

        determinism = {}
        if args.smoke:
            narrowest = min(tasks, key=lambda t: (t["heads"] * t["feature_dim"], t["seed"]))
            pooled = next(c for c in cells if all(
                c[k] == narrowest[k] for k in ("heads", "feature_dim", "seed")))
            determinism = {
                f"{k}[fresh process vs pool, h{narrowest['heads']} f{narrowest['feature_dim']}]": v
                for k, v in fresh_process_check(
                    narrowest, pooled, out, args.blas_threads).items()
            }

    final_rows = [r for c in final_cells for r in c["rows"]]
    final_matched = [r for c in final_cells for r in c["matched"]]
    final_seed = per_seed(final_rows, "final_test")
    final_summary = grid_summary(final_seed)
    final_paired = paired(final_rows, "final_test")
    final_matched_paired = matched_paired(final_matched, "final_test")

    checks = validate(cells, heads, fdims, seeds, final_cells)
    checks.update(determinism)
    for name, value in checks.items():
        LOGGER.info(f"  check {name}: {value}")

    for mode in MODES:
        _log_grid_table(val_summary, val_paired, mode, "validation")
    for mode in MODES:
        _log_grid_table(final_summary, final_paired, mode, "final_test")
    reliable = [
        (r["heads"], r["feature_dim"]) for r in val_paired
        if r["evaluation_mode"] == "stopping_rule" and r["metric"] == "f1"
        and r["reliable_learned_win"]
    ]
    LOGGER.info(f"configurations with a reliable learned stopping-rule F1 win on "
                f"validation: {reliable or 'none'}")

    failures = [c["failure"] for c in cells if c["failure"]]
    undefined = [u for c in cells + final_cells for u in c["undefined"]]
    all_rows = rows + final_rows
    all_seed = val_seed + final_seed
    config_payload = {
        "method": METHOD, "num_nodes": NUM_NODES, "density": DENSITY,
        "heads": heads, "feature_dims": fdims, "seeds": seeds,
        "final_test_graphs_per_seed": n_final, "final_seed_offset": FINAL_SEED_OFFSET,
        "selection_metric": args.select_metric,
        "reliability_rule": "learned wins on every seed AND two-sided Wilcoxon p<0.05 "
                            "over (seed, graph) pairs",
        "matched_budgets": {"n_coarse": MATCHED_N, "reduction": MATCHED_REDUCTION,
                            "epsilon": MATCHED_EPSILON, "heatmap_band": MATCHED_BAND},
        "static_spectral_convention": "q lowest eigenvectors of L = I - D~^{-1/2}(W+I)"
                                      "D~^{-1/2}, ascending, constant eigenvector kept, "
                                      "shift-invert Lanczos; q = learned rank per graph",
        "fixed_fingerprint": cells[0]["fingerprint"] if cells and cells[0]["ok"] else None,
        "workers": args.workers, "blas_threads": args.blas_threads,
        "label_head": args.label_head,
    }
    (out / "capacity_config.json").write_text(json.dumps(config_payload, indent=1, default=str))
    _write_csv(out / "capacity_per_graph.csv", all_rows)
    _write_csv(out / "capacity_per_seed.csv", all_seed)
    _write_csv(out / "capacity_grid_summary.csv", val_summary + final_summary)
    _write_csv(out / "capacity_paired.csv", val_paired + final_paired)
    _write_csv(out / "capacity_matched.csv", val_matched + final_matched_paired)
    (out / "capacity_results.json").write_text(json.dumps({
        "config": config_payload, "checks": checks, "selection": selection,
        "validation": {"summary": val_summary, "paired": val_paired,
                       "matched": val_matched, "per_seed": val_seed},
        "final_test": {"summary": final_summary, "paired": final_paired,
                       "matched": final_matched_paired, "per_seed": final_seed},
        "per_graph": all_rows, "failures": failures, "undefined_metrics": undefined,
        "reliable_learned_configurations_validation": reliable,
    }, indent=1, default=str))
    for path in make_figures(out / "figures", heads, fdims, val_paired, val_matched,
                             selection, final_paired, val_summary):
        LOGGER.info(f"  wrote {path}")
    LOGGER.info(f"capacity sweep written to {out}")


if __name__ == "__main__":
    main()
