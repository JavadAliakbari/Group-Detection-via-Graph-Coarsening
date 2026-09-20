r"""Structured logging, machine-readable serialization and the retained figures.

One class owns all three, so no other component prints, writes files or imports
matplotlib.  ``LOGGER.info`` carries progress and results; warnings and errors
use their own levels.

Two evaluations are reported side by side for every graph:

``stopping_rule``
    the deployable cut -- chosen from **training groups only** on training
    graphs, then transferred to held-out and test graphs as the frozen
    ``epsilon*`` (or the learned reduction).  Test labels score it, never choose
    it.

``oracle``
    a post-hoc **upper bound**: the level of the same hierarchy that maximizes F1
    over that graph's evaluation groups.  It reads those labels and is therefore
    only a diagnostic -- it separates "the hierarchy cannot do better" from "the
    cut landed in the wrong place".  It never influences anything.

Output layout::

    config.json  results.json  comparison.csv/json
    training/        history CSVs, closed-form JSON, the training figure
    oracle/          tables/  per_group/  edge_diagnostics/  halo/
    stopping_rule/   tables/  per_group/  edge_diagnostics/  halo/
    trajectories/    one CSV + figure per graph

Every figure is written next to the CSV it was drawn from, and figure generation
never recomputes or changes a cut.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.utils import LOGGER

__all__ = ["LoggingVisualizationConfig", "LoggingVisualization", "GraphEvaluation"]

MODES = ("stopping_rule", "oracle")


@dataclass
class LoggingVisualizationConfig:
    output_dir: Path = Path("results/pipeline")
    write_csv: bool = True
    write_json: bool = True
    make_plots: bool = True
    trajectory_max_rows: int = 5000
    per_group_top_k: int = 12
    halo_groups_per_graph: int = 4
    halo_max_nodes: int = 250
    smoothing_window: int = 25
    oracle_figures: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.per_group_top_k < 1:
            raise ValueError("per_group_top_k must be at least 1")
        if self.halo_groups_per_graph < 0:
            raise ValueError("halo_groups_per_graph must be non-negative")


@dataclass
class GraphEvaluation:
    """Everything the reporter needs about one graph, already computed."""

    graph: object
    coarsening: object
    level: torch.Tensor
    scope: str  # "train" (a training graph) or "test" (a transfer graph)
    label_reports: dict = field(default_factory=dict)  # mode -> GroupLabelReport

    @property
    def graph_id(self) -> str:
        return self.graph.graph_id


# --------------------------------------------------------------------------- #
# edge diagnostics, on the centralized screened level
# --------------------------------------------------------------------------- #
def _undirected_edges(graph) -> torch.Tensor:
    index = graph.edge_index
    keep = index[0] < index[1]
    return index[:, keep]


def _edge_categories(graph, edges: torch.Tensor, groups: list) -> np.ndarray:
    """``0`` intra-group, ``1`` group boundary, ``2`` background, per edge.

    Computed group by group, so an edge internal to one overlapping group and
    boundary to another is counted as intra -- no single ``group_of`` array.
    """

    n = int(graph.num_nodes)
    category = np.full(edges.shape[1], 2, dtype=np.int64)
    for group in groups:
        member = torch.zeros(n, dtype=torch.bool)
        member[group.nodes] = True
        left, right = member[edges[0]].numpy(), member[edges[1]].numpy()
        category[(left | right) & (category == 2)] = 1
        category[left & right] = 0
    return category


def _per_group_edge_cost(graph, edges, cost: torch.Tensor, groups: list) -> list:
    """Median internal and boundary edge cost per group, overlap-safe."""

    n = int(graph.num_nodes)
    out = []
    for group in groups:
        member = torch.zeros(n, dtype=torch.bool)
        member[group.nodes] = True
        left, right = member[edges[0]], member[edges[1]]
        internal, boundary = left & right, (left ^ right)
        out.append(
            (
                float(cost[internal].median()) if int(internal.sum()) else float("nan"),
                float(cost[boundary].median()) if int(boundary.sum()) else float("nan"),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# summary rows
# --------------------------------------------------------------------------- #
_COLUMNS = [
    "graph",
    "groups",
    "levels",
    "pr_auc",
    "epsilon",
    "reduction",
    "n_coarse",
    "recall",
    "precision",
    "f1",
    "detection_rate",
]
_LABEL_COLUMNS = ["label_cov", "label_prec", "label_rec", "label_f1", "label_acc"]


def _summary_row(evaluation: GraphEvaluation, mode: str) -> dict:
    cut = evaluation.coarsening.cut(mode)
    metrics = cut.metrics["all"]
    row = {
        "graph": evaluation.graph_id,
        "scope": evaluation.scope,
        "groups": metrics["total"],
        "levels": evaluation.coarsening.levels,
        "pr_auc": evaluation.coarsening.pr_auc,
        "epsilon": cut.epsilon,
        "reduction": cut.reduction,
        "n_coarse": cut.n_coarse,
        "recall": metrics["mean_recall"],
        "precision": metrics["mean_precision"],
        "f1": metrics["mean_f1"],
        "detection_rate": metrics["detection_rate"],
        "detected": metrics["detected"],
    }
    report = evaluation.label_reports.get(mode)
    if report is not None and report.metrics.get("n_detected_groups"):
        m = report.metrics
        row.update(
            label_cov=m["detected_group_label_coverage"],
            label_prec=m["detected_group_label_precision_macro"],
            label_rec=m["detected_group_label_recall_macro"],
            label_f1=m["detected_group_label_f1_macro"],
            label_acc=m["detected_group_label_accuracy"],
        )
    return row


def _overall(evaluations: list, mode: str, scope: "str | None") -> dict:
    """Macro over pooled groups, detection over pooled counts, PR-AUC over graphs."""

    chosen = [e for e in evaluations if scope is None or e.scope == scope]
    if not chosen:
        return {}
    groups = [row for e in chosen for row in e.coarsening.cut(mode).per_group]
    if not groups:
        return {}
    detected = sum(int(row["detected"]) for row in groups)
    overall = {
        "graphs": len(chosen),
        "groups": len(groups),
        "pr_auc": float(np.mean([e.coarsening.pr_auc for e in chosen])),
        "recall": float(np.mean([row["recall"] for row in groups])),
        "precision": float(np.mean([row["precision"] for row in groups])),
        "f1": float(np.mean([row["f1"] for row in groups])),
        "detection_rate": detected / len(groups),
        "detected": detected,
    }
    label_rows = [
        row
        for e in chosen
        for row in (e.label_reports.get(mode).rows if e.label_reports.get(mode) else [])
    ]
    if label_rows:
        hits = [row for row in label_rows if row["detected"]]
        overall["label_cov"] = len(hits) / len(label_rows)
        if hits:
            overall["label_acc"] = float(np.mean([row["correct"] for row in hits]))
            from sklearn.metrics import precision_recall_fscore_support

            macro = precision_recall_fscore_support(
                [row["true_label"] for row in hits],
                [row["predicted_label"] for row in hits],
                average="macro",
                zero_division=0,
            )
            overall["label_prec"], overall["label_rec"], overall["label_f1"] = (
                float(macro[0]),
                float(macro[1]),
                float(macro[2]),
            )
    return overall


# --------------------------------------------------------------------------- #
# the component
# --------------------------------------------------------------------------- #
class LoggingVisualization:
    """Reports progress, serializes results and draws the retained figures."""

    def __init__(self, config: LoggingVisualizationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------- #
    def _path(self, *parts: str) -> Path:
        path = self.config.output_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_csv(self, rows, name: "tuple[str, ...] | str", *, subsample=False):
        if not self.config.write_csv or rows is None:
            return None
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if frame.empty:
            return None
        if subsample and len(frame) > self.config.trajectory_max_rows:
            step = int(np.ceil(len(frame) / self.config.trajectory_max_rows))
            keep = (np.arange(len(frame)) % step) == 0
            for marker in ("selected", "oracle_cut"):
                if marker in frame:
                    keep |= frame[marker].to_numpy(dtype=bool)
            frame = frame[keep]
        parts = (name,) if isinstance(name, str) else name
        path = self._path(*parts)
        frame.to_csv(path, index=False)
        return path

    def _write_json(self, payload, name: "tuple[str, ...] | str"):
        if not self.config.write_json:
            return None
        parts = (name,) if isinstance(name, str) else name
        path = self._path(*parts)
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return path

    def _figure(self):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            return plt
        except ImportError:  # pragma: no cover
            LOGGER.warning("matplotlib unavailable -- skipping figures")
            return None

    # -- configuration and dataset ----------------------------------------- #
    def log_configuration(self, payload: dict) -> None:
        LOGGER.info("=" * 96)
        LOGGER.info("RESOLVED CONFIGURATION")
        LOGGER.info("=" * 96)
        for section, values in payload.items():
            LOGGER.info(f"  [{section}]")
            for key, value in (values.items() if isinstance(values, dict) else []):
                LOGGER.info(f"    {key}: {value}")
        self._write_json(payload, "config.json")

    def log_dataset(self, bundle) -> None:
        LOGGER.info("-" * 96)
        LOGGER.info(
            f"DATASET {bundle.name}: {len(bundle.train_graphs)} training graph(s), "
            f"{len(bundle.test_graphs)} transfer graph(s), "
            f"feature dim {bundle.feature_dim}, {bundle.num_classes} classes"
        )
        for graph in bundle.graphs:
            LOGGER.info(
                f"  {graph.graph_id:<12} N={graph.num_nodes:,}  "
                f"E={graph.edge_index.shape[1] // 2:,}  "
                f"groups: {len(graph.train_groups)} train / "
                f"{len(graph.test_groups)} held out"
            )

    # -- learning ----------------------------------------------------------- #
    def log_learning(self, result) -> None:
        LOGGER.info("-" * 96)
        LOGGER.info(
            f"LEARNING ({result.config['architecture']} / {result.config['training_mode']})"
        )
        if result.history:
            first, last = result.history[0], result.history[-1]
            ratio_form = bool(result.dinkelbach and result.dinkelbach.get("rho_trace"))
            if ratio_form:
                # the surrogate is measured against a moving rho; report the ratio
                LOGGER.info(
                    f"  ratio {first['ratio']:.5g} -> {last['ratio']:.5g}  "
                    f"| Dinkelbach surrogate (moving rho, not comparable across "
                    f"outer iterations) {first['edge_objective']:.3g} -> "
                    f"{last['edge_objective']:.3g}"
                )
            else:
                LOGGER.info(
                    f"  total loss {first['total_loss']:.5g} -> "
                    f"{last['total_loss']:.5g}  | edge objective "
                    f"{first['edge_objective']:.5g} -> {last['edge_objective']:.5g}"
                )
            LOGGER.info(
                f"  boundary {last['boundary']:.5g}  internal {last['internal']:.5g}  "
                f"ratio {last['ratio']:.4g}  host {last['host_loss']:.5g}  "
                f"label {last['label_loss']:.5g}"
            )
            best_epoch = (result.dinkelbach or {}).get("best_epoch")
            if best_epoch is not None and best_epoch != last["epoch"]:
                restored = result.history[best_epoch]
                LOGGER.info(
                    f"  restored the best iterate: epoch {best_epoch}/"
                    f"{last['epoch']} (ratio {restored['ratio']:.4g})"
                )
            rows = [
                r for r in result.history if np.isfinite(r.get("train_capture", np.nan))
            ]
            if rows:
                row = rows[-1]
                LOGGER.info(
                    f"  capture train {row['train_capture']:.4f} "
                    f"(min {row['train_capture_min']:.4f})  held out "
                    f"{row['heldout_capture']:.4f}  confusability "
                    f"{row['confusability']:.4f}"
                )
            sampled = {r["graph_id"] for r in result.history}
            if len(sampled) > 1:
                LOGGER.info(f"  epochs sampled graphs {sorted(sampled)}")
        if result.dinkelbach and result.dinkelbach.get("rho_trace"):
            info = result.dinkelbach
            LOGGER.info(
                f"  Dinkelbach: {info['outer_iterations']} outer iteration(s) x "
                f"{info['inner_epochs_per_iteration']} inner epochs; rho "
                f"{' -> '.join(f'{v:.4g}' for v in info['rho_trace'])}"
                f" ({'converged' if info['rho_converged'] else 'NOT converged'})"
            )
        if result.closed_form:
            self._log_closed_form(result.closed_form)
        if result.label_history:
            LOGGER.info(
                f"  label head: loss {result.label_history[0]['label_loss']:.5g} -> "
                f"{result.label_history[-1]['label_loss']:.5g}"
            )
        self._write_csv(result.history, ("training", "learning_history.csv"))
        self._write_csv(result.label_history, ("training", "label_head_history.csv"))
        if self.config.make_plots:
            self._plot_training(result)

    def _log_closed_form(self, report: dict) -> None:
        LOGGER.info(
            f"  closed form [{report['form']}]: {report['n_pencils']} pencil(s), "
            f"dictionary Gram rank {report['gram_rank_median']:.0f}/"
            f"{report['gram_dimension']} (condition "
            f"{report['gram_condition_median']:.3g}), cliff lambda_max(N,P) "
            f"{report['penalty_star_median']:.4g}, positive directions "
            f"{report['positive_directions_per_channel']}"
        )
        for graph_id in report["final"]:
            before, after = report["initial"][graph_id], report["final"][graph_id]
            LOGGER.info(
                f"    {graph_id}: J {before['edge_objective']:.5g} -> "
                f"{after['edge_objective']:.5g}"
                f" | boundary {before['boundary']:.4g} -> {after['boundary']:.4g}"
                f" | internal {before['internal']:.4g} -> {after['internal']:.4g}"
                f" | host {before['host_loss']:.4g} -> {after['host_loss']:.4g}"
                f" | capture {before['train_capture']:.4f} -> {after['train_capture']:.4f}"
                f" | chi {before['confusability']:.4f} -> {after['confusability']:.4f}"
            )
        if report.get("ratio_solver"):
            ratio = report["ratio_solver"]
            LOGGER.info(
                f"    ratio solver: {ratio['iterations_max']} Dinkelbach iterations, "
                f"converged {ratio['converged']}, residual "
                f"{ratio['fixed_point_residual_max']:.2e}"
            )
        self._write_json(
            {k: v for k, v in report.items() if k != "coefficients"},
            ("training", "closed_form.json"),
        )

    def _smooth(self, values: np.ndarray) -> "np.ndarray | None":
        window = int(self.config.smoothing_window)
        if window < 3 or values.size < 3 * window:
            return None
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _plot_training(self, result) -> None:
        """Multi-panel gradient-training figure; raw curves, smoothing on top."""

        if not result.history:
            return
        plt = self._figure()
        if plt is None:
            return
        frame = pd.DataFrame(result.history)
        panels = [
            (("edge_objective", "total_loss"), "optimized objective and total loss"),
            (("train_capture", "train_capture_min", "heldout_capture"), "capture"),
            (("confusability",), "confusability (chibar)"),
            (("internal",), "mean internal edge cost (screened level)"),
            (("boundary",), "mean boundary edge cost (screened level)"),
            (("ratio",), "boundary-to-internal ratio"),
        ]
        if (
            frame.get("label_loss") is not None
            and float(frame["label_loss"].abs().max()) > 0
        ):
            panels.append((("label_loss",), "label loss"))
        if (
            frame.get("host_loss") is not None
            and float(frame["host_loss"].abs().max()) > 0
        ):
            panels.append((("host_loss",), "host-response loss"))

        columns = 3
        rows = int(np.ceil(len(panels) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(6 * columns, 3.6 * rows), squeeze=False
        )
        flat = list(axes.flat)
        for axis, (names, title) in zip(flat, panels):
            for name in names:
                if name not in frame:
                    continue
                values = frame[name].to_numpy(dtype=float)
                finite = np.isfinite(values)
                if not finite.any():
                    continue
                axis.plot(
                    frame["epoch"][finite],
                    values[finite],
                    lw=0.9,
                    alpha=0.55,
                    label=f"{name} (raw)",
                )
                smoothed = self._smooth(values[finite])
                if smoothed is not None:
                    axis.plot(
                        frame["epoch"][finite],
                        smoothed,
                        lw=1.8,
                        label=f"{name} (smoothed, w={self.config.smoothing_window})",
                    )
            axis.set_title(title)
            axis.set_xlabel("epoch")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=6)
        for axis in flat[len(panels) :]:
            axis.axis("off")
        sampled = sorted({r["graph_id"] for r in result.history})
        figure.suptitle(
            f"gradient training -- {result.config['architecture']}"
            + (f"   sampled graphs: {', '.join(sampled)}" if len(sampled) > 1 else "")
        )
        figure.tight_layout()
        figure.savefig(self._path("training", "training_curves.png"), dpi=130)
        plt.close(figure)

    # -- per-graph coarsening ----------------------------------------------- #
    def log_coarsening(self, evaluation: GraphEvaluation) -> None:
        result = evaluation.coarsening
        stopping, oracle = result.stopping_rule, result.oracle
        LOGGER.info("-" * 96)
        LOGGER.info(
            f"COARSENING [{evaluation.graph_id}] {result.hierarchy.method}: "
            f"N={result.hierarchy.n_leaves:,}  levels={result.levels:,}  "
            f"PR-AUC={result.pr_auc:.3f}  target rank={result.target_rank:,}"
        )
        if result.hierarchy.completion_merges:
            LOGGER.warning(
                f"  {result.hierarchy.completion_merges} completion merge(s) were "
                "needed to reach two supernodes: the constrained rule exhausted "
                "adjacent pairs (the graph is disconnected)"
            )
        for cut in (stopping, oracle):
            tag = (
                "stopping rule"
                if cut.mode == "stopping_rule"
                else "ORACLE (upper bound)"
            )
            LOGGER.info(
                f"  {tag} [{cut.rule}]: n_coarse={cut.n_coarse:,} "
                f"(reduction {cut.reduction:.3f} removed / {cut.retained:.3f} "
                f"retained), epsilon={cut.epsilon:.4g}"
            )
            header = (
                f"    {'split':<7}{'groups':>7}{'recall':>9}{'precision':>11}"
                f"{'F1':>8}{'detection':>11}{'det/tot':>10}"
            )
            LOGGER.info(header)
            LOGGER.info("    " + "-" * (len(header) - 4))
            for split, metrics in sorted(cut.metrics.items()):
                LOGGER.info(
                    f"    {split:<7}{metrics['total']:>7}{metrics['mean_recall']:>9.3f}"
                    f"{metrics['mean_precision']:>11.3f}{metrics['mean_f1']:>8.3f}"
                    f"{metrics['detection_rate']:>11.1%}"
                    f"{metrics['detected']:>5}/{metrics['total']:<4}"
                )
        LOGGER.info(
            "  (mean_* are per-group macro averages; micro_* pool node counts and "
            "are in the CSV.  The oracle reads this graph's evaluation labels and "
            "is a post-hoc bound, never a selection.)"
        )
        if result.deflated_certificate:
            cert = result.deflated_certificate
            LOGGER.info(
                f"  deflated certificate at the stopping cut: eps_Q="
                f"{cert['epsilon_q']:.4f} <= eps_Pi={cert['epsilon_pi']:.4f} <= "
                f"mu*eps_Q={cert['sandwich_upper']:.4f} (mu={cert['mu']:.3f}) -> "
                f"sandwich {'OK' if cert['sandwich_ok'] else 'VIOLATED'}; common "
                f"epsilon {cert['epsilon_common']:.4f}"
            )
            if not cert["sandwich_ok"]:
                LOGGER.warning(
                    "  the RSA sandwich does not hold at this cut; mu is estimated "
                    "by power iteration and converges from below"
                )
        self._log_score_certificate(result)

        self._write_csv(
            result.trajectory,
            ("trajectories", f"{evaluation.graph_id}.csv"),
            subsample=True,
        )
        for mode in MODES:
            cut = result.cut(mode)
            self._write_csv(
                cut.per_group, (mode, "per_group", f"{evaluation.graph_id}.csv")
            )
        if self.config.make_plots:
            self._plot_trajectory(evaluation)
            for mode in MODES:
                if mode == "oracle" and not self.config.oracle_figures:
                    continue
                self._plot_per_group(evaluation, mode)
                self._plot_edge_diagnostics(evaluation, mode)
                self._plot_halo(evaluation, mode)

    def _log_score_certificate(self, result) -> None:
        """The ``eps_Q^2 <= S_n <= q eps_Q^2`` audit, level by level."""

        certificate = getattr(result, "score_certificate", None)
        if not certificate:
            return
        verdict = "HOLDS" if certificate["holds_on_deflated_prefix"] else "VIOLATED"
        LOGGER.info(
            f"  cumulative-score bound eps_Q^2 <= S_n <= q*eps_Q^2 (q="
            f"{certificate['q']:,}, S_n = score_sum_raw, exact eps_Q): {verdict} "
            f"over the {certificate['constrained_merges']:,} deflated merges "
            f"[solve={certificate.get('solve')!r}, "
            f"commit_solve={certificate.get('commit_solve')!r}]"
        )
        header = (
            f"    {'merge':>9} {'n_coarse':>9} {'eps_Q':>8} {'eps_Q^2':>10} "
            f"{'S_n(raw)':>11} {'S_n(norm)':>10} {'lower_gap':>11} "
            f"{'upper_gap':>11} {'ratio':>9}  type"
        )
        LOGGER.info(header)
        LOGGER.info("    " + "-" * (len(header) - 4))
        for row in certificate["rows"]:
            LOGGER.info(
                f"    {row['merge_index']:>9,} {row['n_coarse']:>9,} "
                f"{row['epsilon_q']:>8.4f} {row['epsilon_q_squared']:>10.6f} "
                f"{row['score_sum_raw']:>11.6f} {row['score_sum_normalized']:>10.4f} "
                f"{row['lower_gap']:>+11.3e} {row['upper_gap']:>+11.3e} "
                f"{row['ratio']:>9.3f}  {row['merge_type']}"
            )
        LOGGER.info(
            f"    worst lower gap {certificate['worst_lower_gap']:+.3e}, worst upper "
            f"gap {certificate['worst_upper_gap']:+.3e}, largest |S_n - trace(H_P)| "
            f"{certificate['max_abs_score_error']:.3e} "
            f"(tolerance {certificate['tolerance']:.1e}; every eps_Q here is exact, "
            "never the interpolated axis)"
        )
        self._write_csv(
            certificate["rows"], ("trajectories", "cumulative_score_bound.csv")
        )

    def log_cut_rule(self, rule) -> None:
        if rule is None:
            return
        LOGGER.info(
            f"  transferable stopping rule from {rule.source_graph}: "
            f"epsilon*={rule.epsilon:.4f}, reduction*={rule.reduction:.4f} "
            f"(n_coarse={rule.n_coarse:,}); transferred to held-out graphs"
        )

    # -- summary tables ----------------------------------------------------- #
    def log_summaries(self, evaluations: list, *, labels_enabled: bool) -> dict:
        columns = list(_COLUMNS) + (list(_LABEL_COLUMNS) if labels_enabled else [])
        payload: dict = {}
        for mode in MODES:
            rows = [_summary_row(e, mode) for e in evaluations]
            title = (
                "STOPPING-RULE SUMMARY (deployable: cut chosen without test labels)"
                if mode == "stopping_rule"
                else "ORACLE SUMMARY (post-hoc UPPER BOUND: cut chosen with this "
                "graph's evaluation labels)"
            )
            overall = {
                "train": _overall(evaluations, mode, "train"),
                "test": _overall(evaluations, mode, "test"),
                "all": _overall(evaluations, mode, None),
            }
            self._log_table(title, rows, columns, overall)
            self._write_csv(rows, (mode, "tables", "summary.csv"))
            self._write_json(
                {"rows": rows, "overall": overall}, (mode, "tables", "summary.json")
            )
            payload[mode] = {"rows": rows, "overall": overall}
        self._log_comparison(payload, labels_enabled)
        return payload

    def _log_table(self, title, rows, columns, overall) -> None:
        widths = {
            "graph": 20,
            "groups": 8,
            "levels": 8,
            "pr_auc": 9,
            "epsilon": 9,
            "reduction": 11,
            "n_coarse": 10,
            "recall": 9,
            "precision": 11,
            "f1": 8,
            "detection_rate": 9,
            "label_cov": 11,
            "label_prec": 12,
            "label_rec": 11,
            "label_f1": 10,
            "label_acc": 11,
        }
        labels = {"pr_auc": "PR-AUC", "epsilon": "eps", "detection_rate": "det"}

        def render(values: dict) -> str:
            cells = []
            for column in columns:
                width = widths[column]
                value = values.get(column)
                if value is None:
                    cells.append("".rjust(width))
                elif column == "graph":
                    cells.append(str(value).ljust(width))
                elif column in ("groups", "levels", "n_coarse"):
                    cells.append(f"{int(value):,}".rjust(width))
                elif column == "detection_rate":
                    cells.append(f"{value:.1%}".rjust(width))
                else:
                    cells.append(f"{value:.3f}".rjust(width))
            return "  " + "".join(cells)

        header = "  " + "".join(
            (
                labels.get(c, c).ljust(widths[c])
                if c == "graph"
                else labels.get(c, c).rjust(widths[c])
            )
            for c in columns
        )
        LOGGER.info("\n" + "=" * max(len(header), 96))
        LOGGER.info(title)
        LOGGER.info("=" * max(len(header), 96))
        LOGGER.info(header)
        LOGGER.info("  " + "-" * (len(header) - 2))
        for row in rows:
            LOGGER.info(render(row))
        LOGGER.info("  " + "-" * (len(header) - 2))
        if rows:
            mean = {
                c: float(np.mean([r[c] for r in rows if r.get(c) is not None]))
                for c in columns
                if c not in ("graph", "groups", "levels", "n_coarse")
                and any(r.get(c) is not None for r in rows)
            }
            LOGGER.info(render({"graph": "mean", **mean}))
        for scope, label in (
            ("train", "OVERALL train graphs"),
            ("test", "OVERALL test graphs"),
        ):
            values = overall.get(scope)
            if values:
                LOGGER.info(render({"graph": label, **values}))
        LOGGER.info(
            "  overall rows: precision/recall/F1 macro-averaged over all groups of "
            "that scope, detection = detected/total, PR-AUC = mean over graphs"
        )

    def _log_comparison(self, payload: dict, labels_enabled: bool) -> None:
        rows = []
        for scope in ("train", "test"):
            for mode in MODES:
                values = payload[mode]["overall"].get(scope)
                if not values:
                    continue
                row = {
                    "split": scope,
                    "evaluation": mode,
                    "pr_auc": values["pr_auc"],
                    "recall": values["recall"],
                    "precision": values["precision"],
                    "f1": values["f1"],
                    "detection_rate": values["detection_rate"],
                }
                if labels_enabled:
                    row.update({k: values.get(k) for k in _LABEL_COLUMNS})
                rows.append(row)
        if not rows:
            return
        LOGGER.info("\n" + "=" * 96)
        LOGGER.info("OVERALL COMPARISON  (oracle rows are POST-HOC UPPER BOUNDS)")
        LOGGER.info("=" * 96)
        columns = [
            "split",
            "evaluation",
            "pr_auc",
            "recall",
            "precision",
            "f1",
            "detection_rate",
        ] + (list(_LABEL_COLUMNS) if labels_enabled else [])
        header = (
            "  "
            + "split".ljust(8)
            + "evaluation".ljust(16)
            + "".join(c.rjust(12) for c in columns[2:])
        )
        LOGGER.info(header)
        LOGGER.info("  " + "-" * (len(header) - 2))
        for row in rows:
            cells = "".join(
                (
                    "".rjust(12)
                    if row.get(c) is None
                    else (
                        f"{row[c]:.1%}".rjust(12)
                        if c == "detection_rate"
                        else f"{row[c]:.3f}".rjust(12)
                    )
                )
                for c in columns[2:]
            )
            LOGGER.info(
                "  " + row["split"].ljust(8) + row["evaluation"].ljust(16) + cells
            )
        self._write_csv(rows, "comparison.csv")
        self._write_json(rows, "comparison.json")

    # -- group labels ------------------------------------------------------- #
    def log_group_labels(self, evaluations: list) -> dict:
        rows, summary = [], {}
        for mode in MODES:
            reports = [
                e.label_reports[mode] for e in evaluations if e.label_reports.get(mode)
            ]
            if not reports:
                continue
            for report in reports:
                rows.extend(report.rows)
            summary[mode] = {
                scope: _overall(evaluations, mode, scope) for scope in ("train", "test")
            }
        if not rows:
            return {}
        LOGGER.info("\n" + "=" * 96)
        LOGGER.info(
            "GROUP-LABEL PREDICTION BY MAJORITY VOTE  (conditional on detection -- "
            "always read with the coverage)"
        )
        LOGGER.info("=" * 96)
        for mode in summary:
            for scope, values in summary[mode].items():
                if not values or "label_acc" not in values:
                    continue
                LOGGER.info(
                    f"  {mode:<14}{scope:<6} coverage="
                    f"{values['label_cov']:.1%}  accuracy={values['label_acc']:.3f}  "
                    f"macro precision={values['label_prec']:.3f} "
                    f"recall={values['label_rec']:.3f} F1={values['label_f1']:.3f}"
                )
        for evaluation in evaluations:
            for mode, report in evaluation.label_reports.items():
                per_class = report.metrics.get("per_class")
                if not per_class:
                    continue
                LOGGER.info(
                    f"  [{evaluation.graph_id}/{mode}] confusion "
                    f"{report.metrics['confusion_matrix']} over classes "
                    f"{report.metrics['confusion_matrix_labels']}; per-class "
                    + ", ".join(
                        f"{c}: P={v['precision']:.2f} R={v['recall']:.2f} "
                        f"F1={v['f1']:.2f} n={v['support']}"
                        for c, v in per_class.items()
                    )
                )
        self._write_csv(rows, "group_label_votes.csv")
        self._write_json(
            {
                mode: {
                    e.graph_id: e.label_reports[mode].metrics
                    for e in evaluations
                    if e.label_reports.get(mode)
                }
                for mode in MODES
            },
            "group_label_metrics.json",
        )
        return summary

    # -- figures ------------------------------------------------------------ #
    def _plot_per_group(self, evaluation: GraphEvaluation, mode: str) -> None:
        from src.analyze_elliptic_coarsening import plot_gang_pr

        plt = self._figure()
        if plt is None:
            return
        cut = evaluation.coarsening.cut(mode)
        groups = list(evaluation.graph.groups)
        by_id = {row["group_id"]: row for row in cut.per_group}
        results = [
            type(
                "_R",
                (),
                {
                    "recall": by_id[g.group_id]["recall"],
                    "precision": by_id[g.group_id]["precision"],
                    "detected": by_id[g.group_id]["detected"],
                },
            )()
            for g in groups
        ]
        view = [
            type("_G", (), {"num_nodes": g.num_nodes, "id": g.group_id})()
            for g in groups
        ]
        train_ids = {g.group_id for g in groups if g.split == "train"}
        plot_gang_pr(
            results,
            view,
            train_ids,
            self._path(mode, "per_group", f"{evaluation.graph_id}.png"),
            top=self.config.per_group_top_k,
            title=(
                f"Per-group recall / precision -- {evaluation.graph_id} "
                f"[{mode} cut, n_coarse={cut.n_coarse:,}, eps={cut.epsilon:.3f}]"
                f"   ★ detected  ✗ missed   (tr = training group)"
            ),
        )

    def _edge_frame(self, evaluation: GraphEvaluation, mode: str):
        """Edge costs on the centralized whitened level, plus their categories."""

        from src.analyze_elliptic_coarsening import edge_costs

        edges = _undirected_edges(evaluation.graph)
        cost = edge_costs(evaluation.level, edges)
        groups = list(evaluation.graph.groups)
        categories = _edge_categories(evaluation.graph, edges, groups)
        per_group = _per_group_edge_cost(evaluation.graph, edges, cost, groups)
        return edges, cost, categories, per_group, groups

    def _plot_edge_diagnostics(self, evaluation: GraphEvaluation, mode: str) -> None:
        from src.analyze_elliptic_coarsening import plot_edge_cost

        plt = self._figure()
        if plt is None:
            return
        edges, cost, categories, per_group, groups = self._edge_frame(evaluation, mode)
        cut = evaluation.coarsening.cut(mode)
        by_id = {row["group_id"]: row for row in cut.per_group}
        detected = [int(by_id[g.group_id]["detected"]) for g in groups]
        sizes = [g.num_nodes for g in groups]
        medians, below, total = plot_edge_cost(
            cost,
            categories,
            per_group,
            sizes,
            detected,
            self._path(mode, "edge_diagnostics", f"{evaluation.graph_id}.png"),
            title=(
                f"Internal vs boundary screened edge cost -- "
                f"{evaluation.graph_id} [{mode} cut]"
            ),
        )
        self._write_csv(
            [
                {
                    "group_id": g.group_id,
                    "size": g.num_nodes,
                    "split": g.split,
                    "median_internal_cost": mi,
                    "median_boundary_cost": mb,
                    "internal_below_boundary": bool(mi < mb),
                    "detected": bool(by_id[g.group_id]["detected"]),
                    "evaluation_mode": mode,
                }
                for g, (mi, mb) in zip(groups, per_group)
            ],
            (mode, "edge_diagnostics", f"{evaluation.graph_id}.csv"),
        )
        LOGGER.info(
            f"  [{evaluation.graph_id}/{mode}] internal < boundary edge cost for "
            f"{below}/{total} groups; medians {medians}"
        )

    def _plot_halo(self, evaluation: GraphEvaluation, mode: str) -> None:
        from src.analyze_elliptic_coarsening import plot_gang_graph

        budget = self.config.halo_groups_per_graph
        if budget <= 0:
            return
        plt = self._figure()
        if plt is None:
            return
        edges, cost, _categories, _per_group, groups = self._edge_frame(
            evaluation, mode
        )
        cut = evaluation.coarsening.cut(mode)
        by_id = {row["group_id"]: row for row in cut.per_group}
        ordered = sorted(groups, key=lambda g: g.num_nodes, reverse=True)
        hits = [g for g in ordered if by_id[g.group_id]["detected"]]
        misses = [g for g in ordered if not by_id[g.group_id]["detected"]]
        chosen, half = [], max(1, budget // 2)
        chosen.extend(hits[:half])
        chosen.extend(misses[: budget - len(chosen)])
        chosen.extend(g for g in ordered if g not in chosen)
        for group in chosen[:budget]:
            detected = by_id[group.group_id]["detected"]
            plot_gang_graph(
                edges,
                cost,
                group.group_id,
                group.nodes.tolist(),
                self._path(
                    mode,
                    "halo",
                    f"{evaluation.graph_id}_{group.group_id}.png".replace("/", "_"),
                ),
                max_halo=self.config.halo_max_nodes,
                title=(
                    f"Group {group.group_id} ({group.num_nodes} nodes, red) + halo "
                    f"(blue) -- {'DETECTED' if detected else 'MISSED'} at the "
                    f"{mode} cut\nedge colour = learned screened edge cost"
                ),
            )

    def _plot_trajectory(self, evaluation: GraphEvaluation) -> None:
        plt = self._figure()
        if plt is None:
            return
        result = evaluation.coarsening
        frame = pd.DataFrame(result.trajectory)
        splits = sorted(
            {c.split("_")[0] for c in frame.columns if c.endswith("_mean_f1")}
        )
        stopping, oracle = result.stopping_rule, result.oracle

        figure, axes = plt.subplots(2, 3, figsize=(18, 8.5))
        for axis, (column, label, logx) in zip(
            axes[0],
            (
                ("n_coarse", "number of supernodes", True),
                ("epsilon", "epsilon (common Loukas Def. 2)", False),
                ("reduction", "reduction (fraction removed)", False),
            ),
        ):
            for split in splits:
                for metric, style in (
                    ("mean_recall", "-"),
                    ("mean_precision", "--"),
                    ("mean_f1", "-."),
                    ("detection_rate", ":"),
                ):
                    key = f"{split}_{metric}"
                    if key in frame:
                        axis.plot(
                            frame[column],
                            frame[key],
                            style,
                            lw=1.0,
                            label=f"{split} {metric.replace('mean_', '')}",
                        )
            for cut, colour in ((stopping, "k"), (oracle, "tab:purple")):
                axis.axvline(
                    {
                        "n_coarse": cut.n_coarse,
                        "epsilon": cut.epsilon,
                        "reduction": cut.reduction,
                    }[column],
                    color=colour,
                    ls="--",
                    alpha=0.8,
                    label=f"{cut.mode} cut",
                )
            axis.set_xlabel(label)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.3)
            if logx:
                axis.set_xscale("log")
                axis.invert_xaxis()
        axes[0][0].set_ylabel("metric")
        axes[0][-1].legend(fontsize=6, ncol=2)

        # bottom-left: everything against the transferable stopping coordinate
        versus = axes[1][0]
        if "score_sum" in frame:
            x, x_label = frame["score_sum"], "normalized cumulative merge score"
        else:
            x, x_label = frame["reduction"], "reduction (fraction removed)"
        for split in splits:
            for metric, style in (
                ("mean_recall", "-"),
                ("mean_precision", "--"),
                ("mean_f1", "-."),
                ("detection_rate", ":"),
            ):
                key = f"{split}_{metric}"
                if key in frame:
                    versus.plot(
                        x,
                        frame[key],
                        style,
                        lw=1.0,
                        label=f"{split} {metric.replace('mean_', '')}",
                    )
        versus.plot(x, frame["epsilon"], lw=1.8, color="k", label="common epsilon")
        if "epsilon_q" in frame:
            versus.plot(x, frame["epsilon_q"], lw=1.8, color="tab:green", label="eps_Q")
        for cut, colour in ((stopping, "k"), (oracle, "tab:purple")):
            key = "score_sum" if "score_sum" in frame else "reduction"
            if key in cut.row:
                versus.axvline(
                    float(cut.row[key]),
                    color=colour,
                    ls="--",
                    alpha=0.8,
                    label=f"{cut.mode} cut",
                )
        versus.set_xlabel(x_label)
        versus.set_ylabel("metric / epsilon")
        versus.set_ylim(0, 1.02)
        versus.grid(alpha=0.3)
        versus.legend(fontsize=6, ncol=2)

        eps = axes[1][1]
        eps.plot(frame["n_coarse"], frame["epsilon"], lw=1.2, label="common epsilon")
        exact = frame[frame["epsilon_is_exact"]]
        eps.plot(
            exact["n_coarse"],
            exact["epsilon"],
            "k|",
            ms=6,
            alpha=0.7,
            label=f"exact samples ({len(exact)})",
        )
        if "epsilon_q" in frame:
            eps.plot(
                frame["n_coarse"],
                frame["epsilon_q"],
                lw=1.2,
                color="tab:green",
                label="eps_Q (harmonic)",
            )
        # sqrt(S_n / q) <= eps_Q, read off the UNNORMALIZED score.  The companion
        # upper bound sqrt(S_n) is far too loose to share an axis with it.
        certificate = result.score_certificate
        if "score_sum_raw" in frame and "epsilon_q" in frame and certificate:
            q = max(int(certificate["q"]), 1)
            eps.plot(
                frame["n_coarse"],
                np.sqrt(frame["score_sum_raw"].to_numpy() / q),
                lw=1.0,
                ls="--",
                color="tab:red",
                label="sqrt(S_n / q)  lower bound on eps_Q",
            )
            if result.hierarchy.completion_merges:
                eps.axvline(
                    result.hierarchy.n_leaves - result.hierarchy.constrained_merges,
                    color="grey",
                    ls="-.",
                    lw=1.0,
                    label="first completion merge",
                )
        eps.set_xscale("log")
        eps.invert_xaxis()
        eps.set_xlabel("number of supernodes")
        eps.set_ylabel("epsilon  (S_n = raw cumulative score)")
        eps.set_ylim(0, None)
        eps.grid(alpha=0.3)
        eps.legend(fontsize=6)

        pr = axes[1][2]
        order = np.argsort(frame["all_mean_recall"].to_numpy())
        pr.plot(
            frame["all_mean_recall"].to_numpy()[order],
            frame["all_mean_precision"].to_numpy()[order],
            lw=1.2,
        )
        for cut, colour in ((stopping, "k"), (oracle, "tab:purple")):
            metrics = cut.metrics["all"]
            pr.scatter(
                [metrics["mean_recall"]],
                [metrics["mean_precision"]],
                s=90,
                marker="*",
                color=colour,
                zorder=4,
                label=f"{cut.mode} cut",
            )
        pr.set_xlabel("recall")
        pr.set_ylabel("precision")
        pr.set_xlim(0, 1.02)
        pr.set_ylim(0, 1.02)
        pr.set_title(f"PR curve over the hierarchy -- PR-AUC={result.pr_auc:.3f}")
        pr.grid(alpha=0.3)
        pr.legend(fontsize=7, loc="upper right")

        deflated = result.deflated_certificate
        notes = []
        if deflated:
            notes.append(
                f"deflated certificate @ stopping cut\n"
                f"eps_Q={deflated['epsilon_q']:.4f}  "
                f"eps_Pi={deflated['epsilon_pi']:.4f}\n"
                f"mu={deflated['mu']:.3f}  "
                f"mu*eps_Q={deflated['sandwich_upper']:.4f}\n"
                f"sandwich {'OK' if deflated['sandwich_ok'] else 'VIOLATED'}"
            )
        if certificate:
            notes.append(
                f"eps_Q^2 <= S_n <= q eps_Q^2 (q={certificate['q']}): "
                f"{'OK' if certificate['holds_on_deflated_prefix'] else 'VIOLATED'}\n"
                f"worst lower gap {certificate['worst_lower_gap']:+.2e}  "
                f"commit_solve={certificate.get('commit_solve')!r}"
            )
        if notes:
            pr.text(
                0.02,
                0.02,
                "\n".join(notes),
                transform=pr.transAxes,
                fontsize=6.5,
                va="bottom",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            )
        figure.suptitle(
            f"{evaluation.graph_id} -- {result.hierarchy.method} "
            f"(stopping rule: n_coarse={stopping.n_coarse:,}, eps={stopping.epsilon:.3f}; "
            f"oracle: n_coarse={oracle.n_coarse:,}, eps={oracle.epsilon:.3f})"
        )
        figure.tight_layout()
        figure.savefig(
            self._path("trajectories", f"{evaluation.graph_id}.png"), dpi=130
        )
        plt.close(figure)

    # -- final -------------------------------------------------------------- #
    def write_results(self, payload: dict) -> Path:
        path = self._write_json(payload, "results.json")
        LOGGER.info(f"\nresults written to {self.config.output_dir}")
        return path


def relocate_log_file(output_dir: Path) -> None:
    """Move the module-level LOGGER file handler's file into ``output_dir``.

    ``src/utils/utils.py`` opens a log file in a fresh ``results/<timestamp>/``
    folder as an import side effect; this run owns ``output_dir`` instead.
    """

    import shutil

    for handler in list(LOGGER.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        handler.close()
        LOGGER.removeHandler(handler)
        source = Path(handler.baseFilename)
        if not source.is_file() or source.is_symlink():
            continue  # e.g. a /dev/null handler in a headless environment
        try:
            shutil.move(str(source), str(Path(output_dir) / source.name))
            parent = source.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            LOGGER.warning(f"could not relocate the log file {source}")
