"""Apply the modular :class:`CollectiveBankDetector` to the Elliptic++ Actors graph.

This is a *thin data adapter*: it loads the Elliptic++ wallet-address transaction
graph, forms the illicit gangs (connected components of the illicit subgraph) as
:class:`Pattern` objects, wraps everything in a dataset-agnostic
:class:`~src.collective_detector.GraphData`, and hands it to the same
:class:`~src.collective_detector.CollectiveBankDetector` that runs on the
synthetic benchmark -- no algorithm code is duplicated.

Run::

    conda activate FedStruct
    python -m src.run_elliptic_modular --day-start 24 --day-end 26 \
        --feature-mode wallet --coarsening-method ward-tree --ward-stop f1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.analyze_elliptic_coarsening import analyze_coarsening
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.run_collective_bank_detection import build_node_split
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    split_train_test,
    random_structural_features,
)

# pure diagnostic helpers (no coarsening) reused so the missed-gang plots are
# produced here from the coarsenings this script already computes -- no separate
# analyze_missed_gangs run.
from src.analyze_missed_gangs import gang_moments, gang_structure, plot_diagnostics
from src.plot_training import write_training_report
from src.utils.utils import LOGGER, now


def _relocate_logger_file(out_dir: Path) -> None:
    """Move the LOGGER's log file into ``out_dir`` and remove the now-empty
    ``results/<timestamp>/`` folder created for it in ``src/utils/utils.py``.

    ``src/utils/utils.py`` creates a standalone results folder as a module-level
    side effect purely to host the LOGGER's file handler. This run has its own
    ``out_dir`` (``args.out``), so the log file belongs there instead -- this
    relocates it and cleans up the now-empty scratch folder.
    """

    for handler in list(LOGGER.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        handler.close()
        LOGGER.removeHandler(handler)
        src_path = Path(handler.baseFilename)
        if not src_path.exists():
            continue
        dest_path = Path(out_dir) / src_path.name
        shutil.move(str(src_path), str(dest_path))
        parent = src_path.parent
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def _parse_day_groups(spec: str) -> list:
    """Day spec -> list of ``(lo, hi)`` windows, **one graph per window**.

    The comma/dash distinction is the whole point:

    * ``"26"``        -> ``[(26, 26)]``                     one graph
    * ``"24,25,26"``  -> ``[(24,24), (25,25), (26,26)]``    three separate graphs
    * ``"24-26"``     -> ``[(24, 26)]``                     ONE merged graph
    * ``"24-25,28"``  -> ``[(24,25), (28,28)]``             two graphs

    A dash merges the window into a single graph (gangs spanning the days become
    one connected component); a comma keeps the days as independent graphs that
    the trainer samples between.
    """

    groups: list = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            groups.append((int(lo), int(hi)))
        else:
            groups.append((int(part), int(part)))
    return groups


def _load_training_day(window, args, feature_columns):
    """Load one training WINDOW ``(lo, hi)`` as a single graph + its gang split.

    ``lo == hi`` is one day; ``lo < hi`` merges the window into one graph.  The
    split uses the same ``--train-ratio`` and a window-specific seed, so every
    group contributes its own held-out gangs that the filter never sees.
    Returns ``(GraphData, train_patterns)``; ``(None, None)`` when empty.
    """

    lo, hi = window
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, lo, hi)
    if args.feature_mode in ("wallet", "wallet+random"):
        Xfeat = load_node_features(
            args.data_dir, nodes_df, lo, hi, keep_columns=feature_columns
        )
        if args.feature_mode == "wallet+random":
            Xfeat = torch.cat(
                [
                    Xfeat,
                    random_structural_features(
                        int(A_unw.shape[0]), args.random_width, args.seed + lo
                    ),
                ],
                dim=1,
            )
    else:
        Xfeat = random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed + lo
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=False)
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    if not gang_sets:
        LOGGER.warning(f"    days {lo}-{hi}: no gangs -> skipped")
        return None, None, None
    day_gangs = make_patterns(gang_sets, "alert", "gang", "g")
    d_train, d_test = split_train_test(
        day_gangs, args.train_ratio, np.random.default_rng(args.seed + lo)
    )
    return GraphData.from_graph(graph), d_train, d_test


def _gang_diagnostic_rows(det, data, gangs, gang_sets, edge_index, day, node_to_super):
    """Per-gang structural stats + detection outcome for the missed-gang analysis.

    Uses an **already-computed** coarsening (``node_to_super``) -- no re-coarsening.
    Records the theory's predictors (conductance ``Phi``, boundary-edge mean
    ``mbar1``, retained capture, density, star-ness, degree ratio) beside each gang's
    detection outcome, matching the schema :func:`plot_diagnostics` expects.
    """

    res, _ = evaluate_loukas_patterns(
        gangs, node_to_super, data.y, threshold=det.config.threshold
    )
    phi, mbar1 = gang_moments(
        data.a_hat, data.adjacency, gangs, geometry=det.geometry(data)
    )
    cap = det.capture(data, gangs)["per_gang_capture"]
    gang_of = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for gi, S in enumerate(gang_sets):
        gang_of[torch.as_tensor(list(S), dtype=torch.long)] = gi
    struct = gang_structure(edge_index, gang_of, len(gangs), data.num_nodes)
    return [
        {
            "day": day,
            "gang": gangs[gi].id,
            "size": struct[gi]["size"],
            "Phi": float(phi[gi]),
            "mbar1": float(mbar1[gi]),
            "capture": float(cap[gi]),
            "density": struct[gi]["density"],
            "starness": struct[gi]["starness"],
            "deg_ratio": struct[gi]["deg_ratio"],
            "recall": res[gi].recall,
            "precision": res[gi].precision,
            "f1": res[gi].f1,
            "detected": int(res[gi].detected),
        }
        for gi in range(len(gangs))
    ]


def _write_missed_gang_diagnostics(rows: list, out: Path) -> None:
    """Pool per-gang rows (training + transfer days) -> CSV, plot, AUC report."""

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_gang_diagnostics.csv", index=False)
    plot_diagnostics(df, out / "missed_gang_diagnostics.png")

    feats = ["size", "Phi", "mbar1", "capture", "density", "starness", "deg_ratio"]
    LOGGER.info("\n" + "=" * 78)
    LOGGER.info(
        f"POOLED MISSED-GANG DIAGNOSTICS: {len(df)} gangs over {df.day.nunique()} days"
        f"   detected {int(df.detected.sum())}/{len(df)} ({df.detected.mean():.1%})"
    )
    LOGGER.info("=" * 78)
    LOGGER.info(
        f"{'statistic':<12}{'detected median':>17}{'missed median':>15}{'AUC(detect)':>13}"
    )
    LOGGER.info("-" * 78)
    summary = {}
    have_det, have_mis = (df.detected == 1).any(), (df.detected == 0).any()
    for f in feats:
        d_med = float(df[df.detected == 1][f].median()) if have_det else float("nan")
        m_med = float(df[df.detected == 0][f].median()) if have_mis else float("nan")
        try:
            auc = float(roc_auc_score(df.detected, df[f]))
        except Exception:
            auc = float("nan")
        summary[f] = {"detected_median": d_med, "missed_median": m_med, "auc": auc}
        LOGGER.info(f"{f:<12}{d_med:>17.4g}{m_med:>15.4g}{auc:>13.2f}")
    LOGGER.info(
        "\nAUC > 0.5: higher value -> more likely detected;  AUC < 0.5: -> missed."
    )
    (out / "missed_gang_summary.json").write_text(
        json.dumps(
            {"n_gangs": len(df), "detected": int(df.detected.sum()), "stats": summary},
            indent=2,
        )
        + "\n"
    )
    LOGGER.info(f"missed-gang CSV + plot + summary -> {out}")


def _run_pr_sweep(
    det, data, basis, gang_sets, tag, args, dataset="elliptic++"
) -> "dict | None":
    """Full incremental Ward PR-sweep for one graph (training day-range or a transfer day).

    Walks the whole Ward merge order (finest -> 2 clusters) recording every metric
    at every level, so the epsilon budget is swept 0 -> 1 with all intermediates.
    With ``--pr-sweep-exact-budget B > 0`` the epsilon axis is the *exact* RSA
    constant, sampled at B adaptively-placed levels (largest-gap bisection) and
    interpolated to every level; otherwise it is the free cumulative Ward
    distortion.  Reuses the already-computed ``basis`` -- no extra coarsening.

    ``dataset`` only labels the figure (so the synthetic driver in
    :mod:`src.run_synthetic_modular` can share this sweep verbatim rather than
    keeping a second copy of it).
    """

    from src.ward_pr_sweep import (
        adaptive_exact_epsilon,
        calibrate_epsilon,
        plot_sweep,
        summarize_sweep,
        sweep_metrics,
        ward_order,
    )

    if not gang_sets:
        return None
    lap = det.config.coarsening_laplacian
    children, distances, a0, metric = ward_order(
        data.adjacency, basis, det.config.tau, laplacian=lap
    )
    gsets_idx = [list(map(int, np.asarray(list(s)))) for s in gang_sets]
    traj = sweep_metrics(children, distances, data.num_nodes, gsets_idx, args.threshold)
    eps_key, checkpoints = "epsilon", None
    if args.pr_sweep_exact_budget > 0:
        lv, ex = adaptive_exact_epsilon(
            children, data.num_nodes, a0, metric, budget=args.pr_sweep_exact_budget
        )
        calibrate_epsilon(traj, lv, ex, data.num_nodes)
        eps_key = "epsilon_exact"
        checkpoints = (data.num_nodes - lv, ex)
    summ = summarize_sweep(traj, eps_budget=args.epsilon, eps_key=eps_key)
    pd.DataFrame(traj).to_csv(args.out / f"pr_sweep_{tag}.csv", index=False)
    auc, _ = plot_sweep(
        traj,
        f"{dataset} {tag}",
        args.out / f"pr_sweep_{tag}.png",
        eps_budget=args.epsilon,
        eps_key=eps_key,
        exact_checkpoints=checkpoints,
    )
    b = summ["best_f1"]
    LOGGER.info(
        f"  PR-sweep [{tag}]: levels={len(traj)}  PR-AUC={auc:.3f}  "
        f"best-F1 @ eps={b[eps_key]:.3f} (n_coarse={b['n_coarse']}): "
        f"R={b['mean_recall']:.3f} P={b['mean_precision']:.3f} F1={b['mean_f1']:.3f} "
        f"det={b['det_rate']:.1%}"
    )
    out = {
        "tag": tag,
        "levels": len(traj),
        "pr_auc": auc,
        "eps_key": eps_key,
        "n_gangs": len(gsets_idx),
        **{f"best_{k}": v for k, v in b.items()},
    }
    if "at_epsilon" in summ:
        a = summ["at_epsilon"]
        LOGGER.info(
            f"    at budget eps<={a['epsilon_budget']:g} (eps={a[eps_key]:.3f}, "
            f"n_coarse={a['n_coarse']}): R={a['mean_recall']:.3f} "
            f"P={a['mean_precision']:.3f} F1={a['mean_f1']:.3f} det={a['det_rate']:.1%}"
        )
        out.update({f"budget_{k}": v for k, v in a.items()})
    return out


def _log_deflated_certificate(coarsening, prefix: str = "  ") -> dict | None:
    """Log (and return) the RSA sandwich certificate of a deflated coarsening.

    ``None`` for every other coarsener -- only :mod:`src.deflated_coarsen`
    produces the harmonic constant ``eps_Q``, the block-distortion factor
    ``mu_P^tau`` and hence the certified interval for the realized Euclidean RSA
    error ``eps_Pi in [eps_Q, mu eps_Q]``.
    """

    cert = getattr(coarsening, "deflated_certificate", None)
    if cert is None:
        return None
    ok = cert.get("sandwich_ok")
    LOGGER.info(
        f"{prefix}deflated certificate ({cert['rule']}): "
        f"eps_Q={cert['epsilon_q_exact']:.4f} <= eps_Pi={cert['epsilon_pi']:.4f} "
        f"<= mu*eps_Q={cert['sandwich_upper']:.4f}  "
        f"(mu={cert['mu']:.3f}, tr H={cert['trace_h']:.4f}, "
        f"mean leakage eta0={cert['leakage_mean']:.4f}, "
        f"local-solve drift={cert['eps_q_drift']:.2e})"
        + ("" if ok is None else f"  sandwich {'OK' if ok else 'VIOLATED'}")
    )
    LOGGER.info(
        f"{prefix}  merge scores ||a||^2: median {cert['a_sq_median']:.3g}  "
        f"p99 {cert['a_sq_p99']:.3g}  "
        f"{cert['a_sq_zero_frac']:.1%} of merges are invisible to the target "
        f"(score < 1e-12, so their order is decided by tie-breaking)"
    )
    return cert


def _evaluate_transfer_day(
    det: CollectiveBankDetector,
    args: argparse.Namespace,
    day: int,
    feature_columns: "list[str] | None",
) -> dict:
    """Apply the *already-fit* detector to a single held-out day ``day``.

    The learned filter bank ``det.theta_`` is frozen; only the target subspace
    ``R = span(g_Theta(A_hat) X)`` and the RSA/Ward coarsening are recomputed on
    this day's graph.  The Ward cut is chosen by the label-free epsilon budget
    (``det.config.ward_stop == "epsilon"``), so the day's own gang labels never
    influence the coarsening -- they are used only to *score* the resulting
    supernodes.  Returns a per-day record (report + graph sizes).
    """

    LOGGER.info(f"\n--- transfer day {day} ---")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, day, day)
    if args.feature_mode in ("wallet", "wallet+random"):
        Xfeat = load_node_features(
            args.data_dir, nodes_df, day, day, keep_columns=feature_columns
        )
        if args.feature_mode == "wallet+random":
            Xfeat = torch.cat(
                [
                    Xfeat,
                    random_structural_features(
                        int(A_unw.shape[0]), args.random_width, args.seed + day
                    ),
                ],
                dim=1,
            )
    else:
        Xfeat = random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed + day
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=args.weighted)

    illicit_idx = np.where(cls == 1)[0]
    gang_sets = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    day_gangs = make_patterns(gang_sets, "alert", "gang", "g")
    LOGGER.info(f"  gangs (illicit CC>={args.min_gang_size}): {len(day_gangs)}")

    data = GraphData.from_graph(graph)
    # bank theta is (H, K+1, d) -> feature dim is the LAST axis; the closed-form
    # solver has no bank, its coefficients are (P, m) with P = (K+1)*d instead
    if det.theta_ is not None:
        trained_d = int(det.theta_.shape[-1])
    elif det.pencil_theta_ is not None:
        trained_d = int(det.pencil_theta_.shape[0]) // (det.config.degree + 1)
    else:
        trained_d = data.feature_dim
    if data.feature_dim != trained_d:
        raise ValueError(
            f"day {day} feature-dim {data.feature_dim} != trained filter "
            f"feature-dim {trained_d}; cannot apply frozen filter."
        )

    record: dict = {
        "day": day,
        "n_nodes": data.num_nodes,
        "n_gangs": len(day_gangs),
        "report": None,
    }
    if not day_gangs:
        LOGGER.info("  no gangs on this day -> skipping evaluation")
        return record

    basis = det.target_subspace(data, day_gangs)  # R = span(Z); labels unused
    coarsening, _ = det.coarsen(data, basis, day_gangs)
    report = det.evaluate(data, coarsening, {"all": day_gangs})
    record["report"] = report.get("all")
    record["coarsening"] = {
        "n_original": int(coarsening.n_original),
        "n_coarse": int(coarsening.n_coarse),
        "epsilon": float(getattr(coarsening, "epsilon", float("nan"))),
    }
    cert = _log_deflated_certificate(coarsening)
    if cert is not None:
        record["coarsening"]["deflated_certificate"] = cert
    # per-gang missed-gang diagnostics from THIS day's coarsening (no re-coarsening)
    record["gang_rows"] = _gang_diagnostic_rows(
        det,
        data,
        day_gangs,
        gang_sets,
        graph.edge_index,
        day,
        coarsening.node_to_supernode,
    )
    if args.pr_sweep:  # full epsilon sweep on this day, reusing the same basis
        record["pr_sweep"] = _run_pr_sweep(
            det, data, basis, gang_sets, f"day{day}", args
        )
    r = record["report"]
    if r is not None:
        LOGGER.info(
            f"  recall={r['mean_recall']:.3f} precision={r['mean_precision']:.3f} "
            f"f1={r['mean_f1']:.3f} detection={r['detection_rate']:.1%} "
            f"({r['detected']}/{r['total']})"
        )
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # --- dataset ---
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)

    ap.add_argument(
        "--collective-solver",
        choices=["gradient", "closed-form", "trace-ratio", "aeq"],
        default="gradient",
        help="'gradient' = ascend --capture-objective (soft-min lambda_min) on "
        "the filter bank; 'closed-form' = Theta_beta = (G + beta*W_all)^{-1} Bhat, "
        "one shared factorization + m linear solves.  At --pencil-beta 0 the "
        "closed form is the EXACT maximizer of lambda_min(Gamma) over every "
        "target in the dictionary (Theorem A), so the minimax training is "
        "provably unnecessary; beta>0 buys confusability suppression with the "
        "certified sandwich N_beta <= Gamma <= N_0.  'trace-ratio' keeps the "
        "BANK class (so it generalizes) but replaces Adam with a Dinkelbach "
        "iteration: each step is d small (K+1)x(K+1) eigenproblems and the heads "
        "come out orthogonal per channel, so it needs tens of eigensolves rather "
        "than hundreds of epochs.  'aeq' solves the signal-to-confusion "
        "generalized eigenproblem A_eq w = lambda (H + rho I) w instead of the "
        "max-min certificate: A_eq = Bbar (Bbar^T Bbar + alpha I)^{-1} Bbar^T is "
        "the (nearly) group-indicator projector, H the |S_j|-1 normalized mean "
        "confusability, and each retained direction has an interpretable capture-"
        "per-confusability eigenvalue.  One eigendecomposition, global optimum, "
        "and unlike lambda_min it stays informative when m > d.",
    )
    # --- A_eq pencil (--collective-solver aeq) ---
    ap.add_argument(
        "--aeq-alpha",
        type=float,
        default=1e-3,
        help="ridge inside A_eq; alpha -> 0 makes A_eq the projector onto the "
        "realizable group-indicator span (treats independent group combinations "
        "equally instead of weighting by how strongly a group appears). Keep it "
        "small but nonzero when the indicators are nearly dependent.",
    )
    ap.add_argument(
        "--aeq-rho",
        type=float,
        default=1e-3,
        help="ridge on the aggregate confusability H; prevents unstable ratios "
        "where H has a null space",
    )
    ap.add_argument(
        "--aeq-width",
        type=int,
        default=0,
        help="target width d (0 = one direction per training gang, matching the "
        "closed-form solver's width)",
    )
    ap.add_argument(
        "--aeq-lambda-floor",
        type=float,
        default=0.0,
        help="drop directions whose distinguishability lambda_k falls below this "
        "(0 = keep all --aeq-width); the generalized eigenvalues quantify how "
        "many filters are actually useful",
    )
    ap.add_argument(
        "--aeq-reweight-iters",
        type=int,
        default=0,
        help=">0 walks the mean confusability penalty towards the worst case: "
        "solve, evaluate every chi_j, set omega_j = softmax(kappa*chi_j), rebuild "
        "H and solve again.  Each iteration is still one eigenproblem.",
    )
    ap.add_argument(
        "--aeq-signal-communities",
        type=int,
        default=0,
        help="widen A_eq's signal operator with this many LABEL-FREE community "
        "indicators (k-means on the smoothed features).  A_eq's rank is the "
        "number of signal groups, so with gangs alone the target can be neither "
        "wider than m nor able to hold a gang it never saw -- col(A_eq) is "
        "exactly the training gangs' indicator span.  The extra groups enter "
        "A_eq only; H stays built from the training gangs.",
    )
    ap.add_argument(
        "--aeq-signal-normals",
        action="store_true",
        default=False,
        help="also add the licit components (--max-normal-patterns) to A_eq's "
        "signal operator",
    )
    ap.add_argument(
        "--aeq-rho-scale",
        choices=["relative", "absolute"],
        default="relative",
        help="'relative' reads --aeq-rho as a multiple of the mean nonzero "
        "eigenvalue tr(H)/rank(H), so the ridge means the same thing on every "
        "day; 'absolute' "
        "uses it as written",
    )
    ap.add_argument(
        "--aeq-kappa",
        type=float,
        default=5.0,
        help="sharpness of the --aeq-reweight-iters softmax (larger = closer to "
        "the maximum over groups)",
    )
    ap.add_argument(
        "--train-days",
        default="22, 23, 24,25,26",
        help="the ONLY day option.  A comma separates GRAPHS, a dash merges a "
        "window into one graph: '26' = one day; '24,25,26' = three separate "
        "graphs, one drawn per epoch so the filter must work across graphs; "
        "'24-26' = the three days merged into a single graph (gangs spanning "
        "days become one component); '24-25,28' = two graphs.  The LAST group is "
        "the graph that is coarsened, reported on, and from which transfer days "
        "continue.  All groups must share the feature dimension.",
    )
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--weighted", action="store_true", default=False)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument(
        "--transfer-days",
        type=int,
        default=10,
        help="apply the trained (frozen) filter to this many single days *after* "
        "--day-end, evaluating each day's graph individually (0 = disable). Each "
        "day builds its own graph g_k (day-end+k) and is scored with the label-free "
        "epsilon-budget Ward cut so the transfer is honest.",
    )
    ap.add_argument(
        "--feature-mode",
        choices=["wallet", "random", "wallet+random"],
        default="wallet+random",
        help="'wallet' uses the real z-scored wallet features as the bank input X; "
        "'random' uses an isotropic structural range-finder of --random-width columns "
        "(more capacity when there are many training gangs); 'wallet+random' "
        "concatenates both -- the random channels raise the capture reachability "
        "ceiling (Thm 6.5) where the wallet features cannot reach a gang's bands.",
    )
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument(
        "--max-train-gangs",
        type=int,
        default=0,
        help="cap the number of training gangs (0 = all). The collective objective "
        "saturates when #train-gangs > feature-dim (capacity threshold); capping or "
        "raising the feature width avoids lambda_min collapsing to 0.",
    )
    # --- detector hyperparameters (mirror DetectorConfig) ---
    ap.add_argument("--degree", type=int, default=32, help="polynomial degree K")
    ap.add_argument(
        "--basis",
        choices=["chebyshev", "monomial", "lanczos"],
        default="chebyshev",
        help="filter dictionary: chebyshev (minimax-robust, graph-independent), "
        "monomial (legacy, ill-conditioned), or lanczos (instance-optimal: "
        "orthogonal polys of this graph's M_tau-weighted spectral density, "
        "identity channel Gram, but graph-dependent so theta is not comparable "
        "across days).",
    )
    ap.add_argument("--tau", type=float, default=0.5, help="screening (0 = Cor 4.7)")
    ap.add_argument("--epochs", type=int, default=500, help="training epochs")
    ap.add_argument(
        "--learning-rate", type=float, default=0.05, help="Adam learning rate"
    )
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument(
        "--optimizer",
        choices=["projected", "riemannian", "lbfgs"],
        default="projected",
        help="'projected'/'riemannian' = Adam (first-order); 'lbfgs' = full-batch "
        "quasi-Newton -- the objective is deterministic and low-dimensional, so "
        "L-BFGS reaches in tens of epochs what Adam needs thousands for "
        "(unavailable with negative sampling, which makes the objective stochastic)",
    )
    ap.add_argument(
        "--day-aggregate",
        choices=["sample", "mean", "min"],
        default="min",
        help="how multiple training graphs are combined each epoch: 'sample' "
        "draws one (stochastic, the iterate never settles); 'mean' steps on the "
        "average over all graphs (deterministic, settles); 'min' steps on the "
        "worst graph (maximin, scale-robust when days disagree).",
    )
    ap.add_argument(
        "--trace-ratio-iters",
        type=int,
        default=40,
        help="iterations of the trace-ratio solver (each = d small eigensolves)",
    )
    ap.add_argument(
        "--pencil-beta",
        type=float,
        default=0.0,
        help="beta of the closed-form collective solver (0 = pure capture "
        "optimum; larger suppresses confusability, cost bounded by the "
        "reported optimality gap lambda_min(N_0) - lambda_min(N_beta))",
    )
    ap.add_argument("--softmin-temperature", type=float, default=0.2)
    ap.add_argument(
        "--margin-alpha",
        type=float,
        default=0.25,
        help="softmin temperature over the per-gang certified margins "
        "(--capture-objective certified_margin).  0 = the hard worst-gang minimum, "
        "which is what the certificate is stated on; >0 spreads the ascent over "
        "every group, which matters because the binding group changes per step.",
    )
    ap.add_argument(
        "--margin-softplus",
        type=float,
        default=0.5,
        help="smooth the (.)_+ hinge of the boundary functional to "
        "softplus(beta x)/beta.  The hard hinge is exactly zero whenever a group's "
        "capture is below 1 - kappa^2 -- the normal state on a large graph, where "
        "kappa ~ sqrt(tau w_min / vol S) is small -- and then the objective has NO "
        "capture gradient at all.  A finite beta keeps that gradient while agreeing "
        "with the hinge wherever the margin is comfortably positive.  The reported "
        "certificate always uses the exact hinge.",
    )
    ap.add_argument(
        "--warm-start",
        choices=["ones", "closed_form", "resolvent"],
        default="closed_form",
        help="'closed_form' initializes theta at the best single filter consistent "
        "with the per-gang Theorem 6.2 optima instead of the flat low-pass; "
        "'resolvent' is the other end of the seed family M_tau^{-s} v_S -- the "
        "least-squares fit theta = G_K^{-1} T_K^T v_S of the resolvent direction, "
        "which by the master identity needs only the RAW indicators (no PPR solve, "
        "no eigenproblem) and has chi = 0 exactly at s = 1",
    )
    ap.add_argument(
        "--softmin-anneal",
        type=float,
        default=1.0,
        help=">1 starts the soft-min temperature this many times higher and "
        "anneals geometrically down to --softmin-temperature",
    )
    ap.add_argument(
        "--capture-objective",
        choices=["lambda_min", "trace", "softmin_diag", "certified_margin"],
        default="certified_margin",
        help="what the bank ascends: 'lambda_min' (capture + cross-gang separation, "
        "carries the m>d capacity wall) | 'trace' (mean per-gang capture, no "
        "separation, no capacity wall) | 'softmin_diag' (worst gang's capture, no "
        "separation). trace/softmin_diag drop the cross-gang separation lambda_min "
        "buys, which the connectivity-constrained coarsener provides for free "
        "(Prop 8.5); the needed neighbour separation is the confusability chi. "
        "'certified_margin' is the certified-margin objective (src.certified_margin): "
        "softmin_j [ s_boundary(C_j) - chi_j ] with "
        "s_boundary = (1/C)[(kappa - sqrt(1-C))_+]^2 and kappa the graph-only "
        "adjacency floor.  It has NO --conf-weight beta and no collective Gram, and "
        "its value is itself a recovery certificate: positive at every training "
        "group implies raw-score Ward recovers all of them.",
    )
    ap.add_argument(
        "--label-weight",
        type=float,
        default=0.0,
        help="beta for a supervised illicit/licit node head trained JOINTLY with "
        "the filter: loss = -capture + label_weight*CE(head(Z), y). 0 = off. Uses "
        "--neg-per-pos host nodes per gang node as the negative class.",
    )
    ap.add_argument(
        "--neg-per-pos",
        type=float,
        default=1.0,
        help="host (non-gang) nodes sampled per gang node as the negative class for "
        "the --label-weight head (only used when --label-weight > 0).",
    )
    ap.add_argument(
        "--conf-weight",
        type=float,
        default=10.0,
        help="weight of the confusability penalty (soft-min chi) in the objective",
    )
    ap.add_argument("--conf-reduce", choices=["max", "mean"], default="mean")
    ap.add_argument("--conf-delta", type=float, default=0.00)
    ap.add_argument("--structural-width", type=int, default=0)
    ap.add_argument(
        "--coarsen-target",
        choices=["bank", "indicators", "dictionary", "bank+dictionary"],
        default="bank",
        help="'bank' = span(Z), the learned bank's own span (H*d columns, "
        "inductive); 'indicators' = v_hat projected onto span(Z) (capped at the "
        "bank's capture); 'dictionary' = Theorem 6.2 closed-form projection onto "
        "the FULL Chebyshev dictionary (per-gang capture ceiling, needs candidate "
        "node sets); 'bank+dictionary' = both concatenated.",
    )
    ap.add_argument(
        "--heads",
        type=int,
        default=8,
        help="number of filter heads H: the bank is Theta (H, K+1, d) and the "
        "target is the concatenated span of its heads (H*d columns).  H=1 is the "
        "single shared filter and reproduces the classic behaviour exactly; H>1 "
        "removes the one-hop-profile-per-channel bottleneck so gangs stop "
        "competing.  Same objective, epochs and optimizer either way.",
    )
    ap.add_argument(
        "--head-diversity",
        type=float,
        default=0.0,
        help="weight of the head-decorrelation penalty (mean squared cosine "
        "between distinct heads' per-channel filters).  0 relies on random-init "
        "symmetry breaking alone; >0 guarantees the heads stay distinct.",
    )
    # --- coarsening ---
    ap.add_argument(
        "--coarsening-method",
        choices=[
            "edges",
            "neighborhood",
            "capped",
            "star",
            "kmeans",
            "linkage",
            "ward",
            "ward-tree",
            "dual-ward",
            "deflated-dual-ward",
            "deflated-minimax",
        ],
        default="deflated-minimax",
        help="'edges' scales best on the ~50k-node graph; 'ward-tree' builds the "
        "full Ward tree (heavier) and stops per --ward-stop; 'dual-ward' is Smooth "
        "Dual Ward (src.smooth_dual_ward), a Ward variant scored in the screened "
        "dual metric -- see --dual-ward-* below.  'deflated-dual-ward' / "
        "'deflated-minimax' are the screened-consistent agglomeration "
        "(src.deflated_coarsen): merges are scored against the M_tau-orthogonal "
        "(harmonic) block projector rather than the Euclidean block average, which "
        "makes the merge calculus exactly rank-one PSD, the score monotone, and "
        "eps_Q = sqrt(lambda_max(H)) the EXACT screened RSA of the harmonic "
        "reconstruction; both build the full tree and stop per --ward-stop.  See "
        "--deflated-* below.",
    )
    ap.add_argument(
        "--coarsening-laplacian",
        choices=["symmetric", "combinatorial"],
        default="symmetric",
        help="THE geometry switch -- it is not confined to the coarsener (the name "
        "is historical).  It selects the screened metric M_tau the filter bank is "
        "trained in, the group indicator v_S that capture and confusability are "
        "fractions of, the operator the Chebyshev bank propagates on, and the "
        "metric the RSA distortion is measured in.  'symmetric': L = I - A_hat, "
        "v_S = D_tilde^{1/2} 1_S / sqrt(vol(S)), Phi(S) = cut/vol (conductance).  "
        "'combinatorial': L = D - W, v_S = 1_S / sqrt(|S|), Phi(S) = cut/|S| -- "
        "the paper's Section 3.  NOTE tau screens relative to the Laplacian's own "
        "scale, and lambda_max(D - W) >> lambda_max(L_sym) <= 2, so the same "
        "numeric --tau is a much weaker screening under 'combinatorial'; the log "
        "prints the like-for-like equivalent.",
    )
    ap.add_argument(
        "--propagation",
        choices=["auto", "a_hat"],
        default="auto",
        help="operator the Chebyshev bank filters on.  'auto' follows "
        "--coarsening-laplacian (the paper's pairing L_tilde = 2L/lambda_max - I); "
        "'a_hat' pins it to A_hat so a combinatorial run differs from a symmetric "
        "one ONLY in the metric and the indicator -- which separates the geometry "
        "effect from the fact that I - 2(D-W)/lambda_max is near-identity on "
        "low-degree nodes and so diffuses far less per hop.",
    )
    ap.add_argument(
        "--indicator",
        choices=["geometry", "degree_weighted", "plain"],
        default="geometry",
        help="gang signal the REPORTED capture is a fraction of.  'geometry' "
        "(default) takes it from --coarsening-laplacian so the reported number is "
        "the one the fit actually ascended; the other two pin it explicitly, which "
        "is what you want to score the SAME signal under both geometries.",
    )
    ap.add_argument(
        "--pr-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="walk the whole Ward merge order (finest -> 2 clusters) recording "
        "recall/precision/f1/jaccard/detection + epsilon at every level: a full "
        "epsilon sweep 0->1 with all intermediates, plus a PR curve (+AUC) and a "
        "metrics-vs-epsilon plot per day.",
    )
    ap.add_argument(
        "--pr-sweep-exact-budget",
        type=int,
        default=200,
        help="if >0, use the EXACT RSA constant as the sweep's epsilon axis, "
        "computed at this many adaptively-placed levels (endpoints anchored, then "
        "the largest epsilon gap split each step) and interpolated to every level.",
    )
    ap.add_argument("--reduction", type=float, default=0.3)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--max-levels", type=int, default=10)

    ap.add_argument("--ward-stop", choices=["epsilon", "f1"], default="f1")
    ap.add_argument("--ward-num-cuts", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.51)
    # --- deflated / screened-consistent agglomeration (src.deflated_coarsen) ---
    ap.add_argument(
        "--deflated-hops",
        type=int,
        default=2,
        help="radius of the coarse ball the deflation solve (L_c + tau I) c = b is "
        "truncated to.  Screening localizes the harmonic projector -- the "
        "truncation error is O(q^r), q = (sqrt(kappa)-1)/(sqrt(kappa)+1) with "
        "kappa <= (lambda_max + tau)/tau -- so r = 2 is already within ~1e-4 of the "
        "global solve.",
    )
    ap.add_argument(
        "--deflated-max-ball",
        type=int,
        default=32,
        help="cap on the number of blocks in that ball (hub blocks would otherwise "
        "make it global).  Measured: 32 matches 64 to ~1e-4 in eps_Q at ~2x the "
        "speed.",
    )
    ap.add_argument("--deflated-max-rescore", type=int, default=8)
    ap.add_argument(
        "--deflated-fanout",
        type=int,
        default=32,
        help="queue entries pushed per merge for the new block (its best-keyed "
        "neighbours; 0 = all of them).  Elliptic++ day 25 has a degree-4,960 node, "
        "and a block that absorbs it would otherwise push thousands of entries per "
        "merge -- that queue growth, not the linear algebra, is what makes the "
        "hierarchy superlinear (measured: >10 min vs 0.6 min for the full 17k tree).",
    )
    ap.add_argument(
        "--deflated-max-cluster-size",
        type=int,
        default=0,
        help="cap on supernode cardinality (0 = uncapped).",
    )
    ap.add_argument(
        "--deflated-epsilon-key",
        choices=["epsilon_pi", "epsilon_q"],
        default="epsilon_pi",
        help="which RSA constant --epsilon is spent in.  'epsilon_pi' is the "
        "realized Euclidean constant every other coarsener reports (so budgets "
        "stay comparable); 'epsilon_q' is the intrinsic harmonic constant the "
        "algorithm optimizes, which is monotone along the hierarchy.",
    )
    ap.add_argument(
        "--deflated-certify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute eps_Q by brute force and the block-distortion factor "
        "mu_P^tau at the chosen cut, giving the certified RSA sandwich interval "
        "eps_Pi in [eps_Q, mu eps_Q] and the O(q^r) drift of the local solves.",
    )
    # --- Smooth Dual Ward (coarsening_method="dual-ward"; see src.smooth_dual_ward) ---
    ap.add_argument(
        "--dual-ward-alpha",
        type=float,
        default=0.0,
        help="smoothing exponent: 0 = normalized sigma_DW in [0,1] (mass-free, "
        "conductance-like, empirically the best of the family), 1 = raw dual-Ward "
        "increment Delta_DW",
    )
    ap.add_argument(
        "--dual-ward-tau",
        type=float,
        default=0.5,
        help="screening level of M_tau = L_sym + tau I (must be > 0)",
    )
    ap.add_argument(
        "--dual-ward-max-size",
        type=int,
        default=0,
        help="super-node cardinality cap (0 = uncapped; classical Ward chains to "
        "comparable sizes on this graph, so this is a cost bound, not a fix)",
    )
    ap.add_argument(
        "--dual-ward-embedding",
        choices=["dual", "primal"],
        default="dual",
        help="'dual' (the spec) scores merges against M_tau U_tau, which acts as "
        "a high-pass filter on the target and empirically loses to Ward; "
        "'primal' scores against U_tau instead, matching the evaluated RSA "
        "constant and closing most of that gap -- kept as the default here",
    )
    ap.add_argument(
        "--compare-coarseners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in addition to --coarsening-method, also coarsen with plain 'ward' "
        "and 'dual-ward' (at --dual-ward-alpha) on the SAME learned target basis "
        "and report all three side by side",
    )
    ap.add_argument("--max-normal-patterns", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    out_dir = f"results/elliptic_modular/{now}/"
    ap.add_argument("--out", default=out_dir, type=Path)
    args = ap.parse_args()
    # --train-days is the single day option.  The last group is the graph that is
    # coarsened / reported on; day_start..day_end are derived from it so every
    # downstream consumer (output naming, transfer, JSON) is unchanged.
    day_groups = _parse_day_groups(args.train_days)
    if not day_groups:
        ap.error("--train-days must name at least one day, e.g. '26' or '24,25,26'")
    args.day_start, args.day_end = day_groups[-1]
    args.day_groups = day_groups

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- 1. load the Elliptic++ graph + illicit gangs -----------------------
    LOGGER.info(f"=== Elliptic++ (modular) | days {args.day_start}-{args.day_end} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    feature_columns: "list[str] | None" = None
    if args.feature_mode in ("wallet", "wallet+random"):
        Xfeat, feature_columns = load_node_features(
            args.data_dir, nodes_df, args.day_start, args.day_end, return_columns=True
        )
        if args.feature_mode == "wallet+random":
            Xfeat = torch.cat(
                [
                    Xfeat,
                    random_structural_features(
                        int(A_unw.shape[0]), args.random_width, args.seed
                    ),
                ],
                dim=1,
            )
            LOGGER.info(
                f"  + {args.random_width} random structural channels -> "
                f"X: {Xfeat.shape[0]:,} x {Xfeat.shape[1]}"
            )
    else:
        Xfeat = random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed
        )
        LOGGER.info(
            f"  Feature matrix X: {Xfeat.shape[0]:,} x {Xfeat.shape[1]} (random structural)"
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=args.weighted)

    illicit_idx = np.where(cls == 1)[0]
    gang_sets = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    LOGGER.info(
        f"  Gangs (illicit CC>={args.min_gang_size}): {len(gangs)}  "
        f"| sizes: {sorted((p.num_nodes for p in gangs), reverse=True)[:12]}..."
    )
    licit_sets = connected_components_sets(
        A_unw, np.where(cls == 2)[0], args.min_gang_size
    )
    licit_sets = sorted(licit_sets, key=len, reverse=True)[: args.max_normal_patterns]
    normals = make_patterns(licit_sets, "normal", "normal", "n")
    LOGGER.info(
        f"  gangs={len(gangs)} (sizes {sorted((p.num_nodes for p in gangs), reverse=True)[:8]}...)  normals={len(normals)}"
    )

    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    if args.max_train_gangs and len(gang_train) > args.max_train_gangs:
        gang_train = gang_train[: args.max_train_gangs]
    LOGGER.info(
        f"  train gangs: {len(gang_train)}  test gangs: {len(gang_test)}  "
        f"feature-dim: {graph.x.shape[1]}"
    )
    if len(gang_train) > graph.x.shape[1]:
        LOGGER.info(
            f"  WARNING: #train-gangs ({len(gang_train)}) > feature-dim "
            f"({graph.x.shape[1]}): capacity threshold -> lambda_min may be ~0. "
            "Use --feature-mode random --random-width, or --max-train-gangs."
        )

    # --- 2. dataset-agnostic wrapper + detector -----------------------------
    data = GraphData.from_graph(graph)  # features already set on graph.x
    cfg = DetectorConfig(
        degree=args.degree,
        basis=args.basis,
        tau=args.tau,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge=args.ridge,
        optimizer=args.optimizer,
        softmin_temperature=args.softmin_temperature,
        warm_start=args.warm_start,
        softmin_anneal=args.softmin_anneal,
        capture_objective=args.capture_objective,
        margin_alpha=args.margin_alpha,
        margin_softplus=args.margin_softplus,
        collective_solver=args.collective_solver,
        pencil_beta=args.pencil_beta,
        aeq_alpha=args.aeq_alpha,
        aeq_rho=args.aeq_rho,
        aeq_width=args.aeq_width,
        aeq_lambda_floor=args.aeq_lambda_floor,
        aeq_reweight_iters=args.aeq_reweight_iters,
        aeq_kappa=args.aeq_kappa,
        aeq_rho_scale=args.aeq_rho_scale,
        trace_ratio_iters=args.trace_ratio_iters,
        day_aggregate=args.day_aggregate,
        label_weight=args.label_weight,
        conf_weight=args.conf_weight,
        conf_reduce=args.conf_reduce,
        conf_delta=args.conf_delta,
        structural_width=args.structural_width,
        coarsen_target=args.coarsen_target,
        heads=args.heads,
        head_diversity=args.head_diversity,
        coarsening_method=args.coarsening_method,
        coarsening_laplacian=args.coarsening_laplacian,
        propagation=args.propagation,
        indicator=args.indicator,
        reduction=args.reduction,
        epsilon=args.epsilon,
        max_levels=args.max_levels,
        ward_stop=args.ward_stop,
        ward_num_cuts=args.ward_num_cuts,
        threshold=args.threshold,
        dual_ward_alpha=args.dual_ward_alpha,
        dual_ward_tau=args.dual_ward_tau,
        dual_ward_max_size=args.dual_ward_max_size,
        dual_ward_embedding=args.dual_ward_embedding,
        deflated_hops=args.deflated_hops,
        deflated_max_ball=args.deflated_max_ball,
        deflated_max_rescore=args.deflated_max_rescore,
        deflated_fanout=args.deflated_fanout,
        deflated_max_cluster_size=args.deflated_max_cluster_size,
        deflated_epsilon_key=args.deflated_epsilon_key,
        deflated_certify=args.deflated_certify,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)

    # optional joint supervised head: build a node-level illicit/host split on the
    # TRAIN gangs (labels are used only for the head; the coarsening cut stays
    # label-free via --ward-stop epsilon).
    label_y = label_idx = None
    if args.label_weight > 0.0:
        label_y, label_idx, _ = build_node_split(
            gang_train,
            gang_test,
            data.num_nodes,
            (data.y == 1),
            neg_per_pos=args.neg_per_pos,
            seed=args.seed,
        )
        LOGGER.info(
            f"  joint label head: beta={args.label_weight:g}  "
            f"{len(label_idx):,} labelled nodes "
            f"({int((label_y[label_idx] == 1).sum()):,} illicit)"
        )

    geo = det.geometry(data)
    LOGGER.info(
        f"\n  Geometry: {geo.describe()}"
        f"   propagation={'A_hat' if geo.prop is data.a_hat else 'I - 2L/lambda_max'}"
    )
    if geo.kind == "combinatorial":
        LOGGER.info(
            f"    screening: tau={cfg.tau:g} here is like tau="
            f"{cfg.tau / geo.tau_equivalent:.4g} in the symmetric geometry "
            f"(lambda_max ratio {geo.tau_equivalent:.4g}); the block-distortion "
            f"bound mu_P^tau <= sqrt((lambda_max+tau)/tau) is "
            f"{((geo.lambda_max + cfg.tau) / (cfg.tau + 1e-12)) ** 0.5:.3g}"
        )
    LOGGER.info(
        f"\n  Fitting collective bank (basis={cfg.basis}, tau={cfg.tau}, "
        f"K={cfg.degree}, opt={cfg.optimizer}, objective={cfg.capture_objective}"
        f"{f', label_w={cfg.label_weight:g}' if cfg.label_weight > 0 else ''}) …"
    )

    # Every solver takes the same list of training groups; only the LAST group
    # is coarsened / reported on.  A single group routes to each solver's plain
    # single-graph call, so `--train-days 26 --collective-solver gradient`
    # reproduces the classic fit exactly.
    day_specs = []
    for grp in args.day_groups:
        if grp == (args.day_start, args.day_end):
            d_data, d_train, d_test = data, gang_train, gang_test  # already loaded
        else:
            d_data, d_train, d_test = _load_training_day(grp, args, feature_columns)
            if d_train is None:
                continue
        day_specs.append(
            (
                f"{grp[0]}-{grp[1]}" if grp[0] != grp[1] else str(grp[0]),
                d_data,
                d_train,
                d_test,
            )
        )
    if not day_specs:
        raise ValueError(f"--train-days {args.train_days} yielded no usable groups")
    LOGGER.info(
        f"  training groups: {[spec[0] for spec in day_specs]}"
        f"{' (one drawn per epoch)' if len(day_specs) > 1 else ''}"
        f"   |   coarsened + reported on: {day_specs[-1][0]}"
    )

    # A_eq signal enrichment: extra indicator columns per training graph.  The
    # rank of A_eq is the number of signal groups, so this is what lets the
    # target be wider than m and stop being a basis for these gangs specifically.
    signal_patterns = None
    if cfg.collective_solver == "aeq" and (
        args.aeq_signal_communities > 0 or args.aeq_signal_normals
    ):
        from src.margin_pencil import community_patterns

        signal_patterns = {}
        for lbl, d_data, _d_train, _d_test in day_specs:
            extra = []
            if args.aeq_signal_communities > 0:
                extra += community_patterns(
                    d_data.a_hat,
                    d_data.X,
                    n_clusters=args.aeq_signal_communities,
                    seed=args.seed,
                )
            if args.aeq_signal_normals and lbl == day_specs[-1][0]:
                extra += normals
            signal_patterns[lbl] = extra
        LOGGER.info(
            "  A_eq signal groups: "
            + ", ".join(f"{k}: +{len(v)}" for k, v in signal_patterns.items())
        )

    det.fit(day_specs, signal_patterns=signal_patterns)

    # Every training group is coarsened and scored, not just the reported one:
    # with several graphs the single-group table hides how the shared filter
    # actually does on the days it was trained on.
    per_group_reports = {}
    if len(day_specs) > 1:
        LOGGER.info("\n" + "=" * 74)
        LOGGER.info(
            "PER-GROUP TRAINING-DAY PERFORMANCE (shared filter, own coarsening)"
        )
        LOGGER.info("=" * 74)
        LOGGER.info(
            f"  {'group':<8}{'gangs':>7}{'recall':>9}{'precision':>11}"
            f"{'f1':>8}{'detection':>11}{'det/tot':>10}"
        )
        LOGGER.info("  " + "-" * 62)
        for lbl, g_data, g_train, _g_test in day_specs:
            g_basis = det.target_subspace(g_data, g_train)
            g_co, _ = det.coarsen(g_data, g_basis, g_train)
            rep = det.evaluate(g_data, g_co, {"train": g_train})["train"]
            rep["n_coarse"] = int(g_co.n_coarse)
            rep["epsilon"] = float(g_co.epsilon)
            per_group_reports[lbl] = rep
            LOGGER.info(
                f"  {lbl:<8}{rep['total']:>7}{rep['mean_recall']:>9.3f}"
                f"{rep['mean_precision']:>11.3f}{rep['mean_f1']:>8.3f}"
                f"{rep['detection_rate']:>11.1%}"
                f"{rep['detected']:>5}/{rep['total']:<4}"
            )

    basis_ = det.target_subspace(data, gang_train)
    coarsening, trajectory = det.coarsen(data, basis_, gang_train)
    splits = {"train": gang_train, "test": gang_test, "all": gangs}
    result = {
        "config": cfg.to_dict(),
        "theta": det.theta_,
        "fit": det.fit_info_,
        "basis": basis_,
        "coarsening": coarsening,
        "trajectory": trajectory,
        "report": det.evaluate(data, coarsening, splits),
        "captures": {n: det.capture(data, p) for n, p in splits.items() if p},
        "per_group_report": per_group_reports,
        "deflated_certificate": getattr(coarsening, "deflated_certificate", None),
    }
    _log_deflated_certificate(coarsening)

    fit = result["fit"]
    if cfg.collective_solver == "aeq":
        LOGGER.info(
            f"    A_eq signal-to-confusion pencil (alpha={cfg.aeq_alpha:g}, "
            f"rho={cfg.aeq_rho:g}[{cfg.aeq_rho_scale}]={fit['rho_used']:.3g}, "
            f"d={fit['width']}, P={fit['dictionary_dim']:,}, "
            f"rank={fit['dictionary_rank']:,}):"
        )
        ev = fit["eigenvalues"]
        LOGGER.info(
            f"      distinguishability lambda_k: max {max(ev):.4g}  "
            f"median {float(np.median(ev)):.4g}  min {min(ev):.4g}"
        )
        LOGGER.info(
            f"      signal groups: {fit['n_signal']} "
            f"({fit['n_extra_signal']} beyond the training gangs)"
        )
        LOGGER.info(
            f"      certificate: lambda_min(Gamma) {fit['lambda_min_Gamma']:.6g} "
            f"(Theorem A ceiling {fit['lambda_min_N0']:.6g})  "
            f"max chi {fit['chi_subspace_max']:.4g}  "
            f"margin {fit['margin']:.6g}  tr(Gamma) {fit['trace_Gamma']:.4g}"
        )
        if cfg.aeq_reweight_iters > 0:
            h = fit["reweight_history"]
            LOGGER.info(
                "      reweighting: "
                + "  ".join(
                    f"[{r['iter']}] chi_max={r['chi_max']:.3g} "
                    f"margin={r['margin']:.4g}"
                    for r in h
                )
            )
    elif cfg.collective_solver == "closed-form":
        LOGGER.info(
            f"    closed-form collective solve (beta={cfg.pencil_beta:g}, "
            f"P={fit['dictionary_dim']:,}, rank={fit['dictionary_rank']:,}):"
        )
        LOGGER.info(
            f"      lambda_min: N_beta {fit['lambda_min_N_beta']:.6g} <= "
            f"Gamma {fit['lambda_min_Gamma']:.6g} <= N_0 {fit['lambda_min_N0']:.6g}"
            f"   [sandwich holds: {fit['sandwich_ok']}]"
        )
        LOGGER.info(
            f"      optimality gap (headroom for ANY minimax): "
            f"{fit['optimality_gap']:.6g}   max chi {max(fit['chi_cross_max']):.3e} "
            f"<= bound {max(fit['chi_bound']):.3e}"
        )
        LOGGER.info(
            f"      certificate: lambda_min(Gamma) {fit['lambda_min_Gamma']:.6g}  "
            f"max chi {fit['chi_subspace_max']:.4g}  margin {fit['margin']:.6g}  "
            f"tr(Gamma) {fit['trace_Gamma']:.4g}"
        )
    elif cfg.capture_objective == "certified_margin":
        LOGGER.info(
            f"    certified margin (softmin over {len(gang_train)} train gangs): "
            f"{fit['init_objective']:.4g} -> {fit['margin']:.4g}"
        )
        rch = np.asarray(fit.get("reachability") or [])
        if rch.size:
            LOGGER.info(
                f"      reachability R_K (what the dictionary can express at all, "
                f"pre-training): median {np.median(rch):.4g} "
                f"[{rch.min():.3g}, {rch.max():.3g}]  =>  delta_min^2 = 1 - R_K"
            )
        kap = np.asarray(fit.get("margin_kappa") or [])
        thr = np.asarray(fit.get("margin_capture_threshold") or [])
        forced = np.asarray(fit.get("margin_chi_forced") or [])
        if kap.size:
            LOGGER.info(
                f"      boundary floor kappa: median {np.median(kap):.4g} "
                f"[{kap.min():.3g}, {kap.max():.3g}]  =>  a group needs capture "
                f"C > 1 - kappa^2 = {np.median(thr):.6f} (median) before its "
                f"boundary functional is even positive"
            )
            LOGGER.info(
                f"      conflict bound (chi forced by full capture): median "
                f"{np.median(forced):.4g}  |  nu^2 median "
                f"{np.median(np.asarray(fit.get('margin_nu2') or [0])):.4g}"
            )
        LOGGER.info(
            f"      CERTIFICATE (exact hinge, worst gang): {fit['certificate']:.4g}  "
            f"-> {fit['n_certified']}/{len(gang_train)} training gangs certified"
            f"; {fit['margin_feasible_capture']}/{len(gang_train)} reach the "
            f"capture threshold at all"
        )
        if fit["n_certified"] == 0:
            LOGGER.info(
                "      note: a non-positive certificate does NOT mean the "
                "coarsening fails -- the theorem is one-directional (positive => "
                "recovery).  It means this instance's capture is far below the "
                "level at which the boundary floor can dominate chi."
            )
    else:
        LOGGER.info(
            f"    capture ({cfg.capture_objective}) / lambda_min(Gamma): "
            f"{fit['init_objective']:.4g} -> {fit['objective']:.4g}"
        )
        if cfg.conf_weight > 0:
            LOGGER.info(
                f"    confusability chi: {fit['confusability_init']:.4g} -> "
                f"{fit['confusability']:.4g}"
            )
    # loss / capture / confusability / per-gang capture over the fit (the
    # closed-form solver has no training trace, so this is a no-op there)
    fig = write_training_report(
        fit,
        args.out,
        title=(
            f"collective bank fit -- elliptic++ train="
            f"{'+'.join(spec[0] for spec in day_specs)} eval=d{args.day_end} "
            f"({cfg.capture_objective}, beta={cfg.conf_weight:g}, K={cfg.degree}, "
            f"tau={cfg.tau:g}, {len(gang_train)} train gangs)"
        ),
    )
    if fig is not None:
        LOGGER.info(f"    training curves + history CSV -> {fig}")
    co = result["coarsening"]
    LOGGER.info(
        f"  coarsening ({cfg.coarsening_method}): N={co.n_original:,} -> "
        f"n_coarse={co.n_coarse:,}  epsilon={co.epsilon:.4g}"
    )

    # --- 3. report ----------------------------------------------------------
    LOGGER.info("\n" + "=" * 74)
    LOGGER.info(
        f"ELLIPTIC++ GANG DETECTION (modular)  days {args.day_start}-{args.day_end}  "
        f"N={data.num_nodes:,}  {len(gangs)} gangs"
    )
    LOGGER.info("=" * 74)
    hdr = f"  {'split':<6} {'recall':>8} {'precision':>10} {'f1':>7} {'detection':>10} {'det/tot':>10}"
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for name in ("train", "test", "all"):
        r = result["report"].get(name)
        if r is None:
            continue
        LOGGER.info(
            f"  {name:<6} {r['mean_recall']:>8.3f} {r['mean_precision']:>10.3f} "
            f"{r['mean_f1']:>7.3f} {r['detection_rate']:>10.1%} "
            f"{r['detected']:>4}/{r['total']:<5}"
        )

    # --- 3'. coarsener comparison: same basis + gangs, different coarsening ---
    # ward-tree can't be recut by loukas_coarsen_pytorch (it owns its own tree),
    # so the comparison always includes plain 'ward' and 'dual-ward' plus
    # whichever method was actually configured (if different from those two).
    coarsener_reports: dict = {}
    if args.compare_coarseners:
        LOGGER.info("\n" + "=" * 74)
        LOGGER.info("COARSENER COMPARISON (same learned target basis + train gangs)")
        LOGGER.info("=" * 74)
        variants = [("ward", {})]
        methods = {"ward": "ward"}
        if cfg.coarsening_laplacian not in ("combinatorial", "comb"):
            # Smooth Dual Ward scores merges in L_sym + tau I only, so under the
            # combinatorial geometry it would coarsen in a different metric than
            # the target and the RSA budget are stated in -- drop it rather than
            # print a number that is not comparable to the others in this table.
            variants.append((f"dual-ward a={args.dual_ward_alpha:g}", {}))
            methods[variants[-1][0]] = "dual-ward"
        if cfg.coarsening_method not in ("ward", "dual-ward"):
            variants.append((cfg.coarsening_method, {}))
            methods[cfg.coarsening_method] = cfg.coarsening_method
        chdr = (
            f"  {'coarsener':<20}{'n_coarse':>10}{'eps':>8}{'recall':>9}"
            f"{'precision':>11}{'f1':>8}{'detection':>11}{'det/tot':>10}"
        )
        LOGGER.info(chdr)
        LOGGER.info("  " + "-" * (len(chdr) - 2))
        for tag, _ in variants:
            method = methods[tag]
            variant_cfg = replace(cfg, coarsening_method=method)
            variant_det = CollectiveBankDetector(variant_cfg)
            v_co, _ = variant_det.coarsen(data, basis_, gang_train)
            v_rep = variant_det.evaluate(data, v_co, {"all": gangs})["all"]
            v_rep["n_coarse"] = int(v_co.n_coarse)
            v_rep["epsilon"] = float(v_co.epsilon)
            coarsener_reports[tag] = v_rep
            LOGGER.info(
                f"  {tag:<20}{v_co.n_coarse:>10,}{v_co.epsilon:>8.3f}"
                f"{v_rep['mean_recall']:>9.3f}{v_rep['mean_precision']:>11.3f}"
                f"{v_rep['mean_f1']:>8.3f}{v_rep['detection_rate']:>11.1%}"
                f"{v_rep['detected']:>5}/{v_rep['total']:<4}"
            )
        LOGGER.info(
            "\n  Read: all three coarsen the SAME target subspace learned by the "
            f"configured filter (coarsening_method={cfg.coarsening_method!r} is the "
            "one actually reported above/below); this isolates the coarsening "
            "algorithm from the learned target. 'dual-ward' uses "
            f"--dual-ward-embedding={args.dual_ward_embedding!r} "
            f"(tau={args.dual_ward_tau:g}); 'primal' scores merges against U_tau "
            "instead of the spec's M_tau U_tau, which removes a high-pass "
            "distortion of the target that otherwise costs dual-ward most of its "
            "detections relative to Ward on Elliptic++."
        )
        result["coarsener_comparison"] = coarsener_reports

    # --- 3a. full epsilon sweep on the training graph (reuses the fitted filter) --
    pr_sweep_records: list[dict] = []
    if args.pr_sweep:
        LOGGER.info("\n" + "=" * 74)
        LOGGER.info(
            "FULL WARD PR-SWEEP (finest -> 2 clusters; every metric at every level)"
            + (
                f"\n  epsilon axis: EXACT RSA, {args.pr_sweep_exact_budget} adaptive samples"
                if args.pr_sweep_exact_budget > 0
                else "\n  epsilon axis: cumulative Ward distortion (free)"
            )
        )
        LOGGER.info("=" * 74)
        train_basis = det.target_subspace(data, gangs)
        rec = _run_pr_sweep(
            det,
            data,
            train_basis,
            gang_sets,
            f"train_d{args.day_start}-{args.day_end}",
            args,
        )
        if rec is not None:
            pr_sweep_records.append(rec)

    # --- 3b. transfer: apply the frozen filter to the next N single days ----
    transfer_records: list[dict] = []
    transfer_summary: dict | None = None
    if args.transfer_days > 0:
        # Freeze the learned filter; recompute only the subspace + coarsening per
        # day. Force the label-free epsilon Ward stop so no day's own gangs steer
        # its coarsening (honest transfer).
        det_transfer = CollectiveBankDetector(replace(cfg, ward_stop="epsilon"))
        det_transfer.theta_ = det.theta_
        det_transfer.fit_info_ = det.fit_info_
        # closed-form solver: the frozen dictionary coefficients transfer too
        det_transfer.pencil_theta_ = det.pencil_theta_

        LOGGER.info("\n" + "=" * 74)
        LOGGER.info(
            f"TRANSFER: frozen filter (trained on days {args.day_start}-{args.day_end}) "
            f"applied to days {args.day_end + 1}-{args.day_end + args.transfer_days}"
        )
        LOGGER.info("=" * 74)
        for k in range(1, args.transfer_days + 1):
            transfer_records.append(
                _evaluate_transfer_day(
                    det_transfer, args, args.day_end + k, feature_columns
                )
            )

        # per-day table + average over the days that had gangs
        LOGGER.info("\n" + "=" * 74)
        LOGGER.info("PER-DAY TRANSFER PERFORMANCE (frozen filter, honest epsilon cut)")
        LOGGER.info("=" * 74)
        thdr = (
            f"  {'day':<5} {'nodes':>8} {'gangs':>6} {'recall':>8} {'precision':>10} "
            f"{'f1':>7} {'detection':>10} {'det/tot':>10}"
        )
        LOGGER.info(thdr)
        LOGGER.info("  " + "-" * (len(thdr) - 2))
        scored = [rec for rec in transfer_records if rec["report"] is not None]
        for rec in transfer_records:
            r = rec["report"]
            if r is None:
                LOGGER.info(
                    f"  {rec['day']:<5} {rec['n_nodes']:>8,} {rec['n_gangs']:>6} "
                    f"{'--':>8} {'--':>10} {'--':>7} {'--':>10} {'--':>10}"
                )
                continue
            LOGGER.info(
                f"  {rec['day']:<5} {rec['n_nodes']:>8,} {rec['n_gangs']:>6} "
                f"{r['mean_recall']:>8.3f} {r['mean_precision']:>10.3f} "
                f"{r['mean_f1']:>7.3f} {r['detection_rate']:>10.1%} "
                f"{r['detected']:>4}/{r['total']:<5}"
            )
        if scored:
            keys = ("mean_recall", "mean_precision", "mean_f1", "detection_rate")
            avg = {
                k: float(np.mean([rec["report"][k] for rec in scored])) for k in keys
            }
            tot_det = int(sum(rec["report"]["detected"] for rec in scored))
            tot_all = int(sum(rec["report"]["total"] for rec in scored))
            transfer_summary = {
                "n_days_scored": len(scored),
                **avg,
                "detected": tot_det,
                "total": tot_all,
            }
            LOGGER.info("  " + "-" * (len(thdr) - 2))
            LOGGER.info(
                f"  {'avg':<5} {'':>8} {'':>6} "
                f"{avg['mean_recall']:>8.3f} {avg['mean_precision']:>10.3f} "
                f"{avg['mean_f1']:>7.3f} {avg['detection_rate']:>10.1%} "
                f"{tot_det:>4}/{tot_all:<5}"
            )

        for rec in transfer_records:  # collect each day's sweep for the summary table
            if rec.get("pr_sweep"):
                pr_sweep_records.append(rec["pr_sweep"])

        # echo the training-day test-pattern performance for side-by-side reading
        test_r = result["report"].get("test")
        if test_r is not None:
            LOGGER.info(
                f"\n  training-day (d{args.day_start}-{args.day_end}) TEST patterns: "
                f"recall={test_r['mean_recall']:.3f} "
                f"precision={test_r['mean_precision']:.3f} "
                f"f1={test_r['mean_f1']:.3f} "
                f"detection={test_r['detection_rate']:.1%} "
                f"({test_r['detected']}/{test_r['total']})"
            )

    # --- PR-sweep summary across the training graph + every transfer day -----
    if pr_sweep_records:
        S = pd.DataFrame(pr_sweep_records)
        S.to_csv(args.out / "pr_sweep_summary.csv", index=False)
        ek = pr_sweep_records[0]["eps_key"]
        LOGGER.info("\n" + "=" * 96)
        LOGGER.info(
            "PR-SWEEP SUMMARY  (best-F1 stop over the full epsilon sweep, per graph)"
        )
        LOGGER.info("=" * 96)
        h = (
            f"  {'graph':<18}{'gangs':>6}{'levels':>8}{'PR-AUC':>8}"
            f"{'eps*':>8}{'n_coarse':>10}{'recall':>8}{'prec':>8}{'F1':>8}{'det':>8}"
        )
        LOGGER.info(h)
        LOGGER.info("  " + "-" * (len(h) - 2))
        for r in pr_sweep_records:
            LOGGER.info(
                f"  {r['tag']:<18}{r['n_gangs']:>6}{r['levels']:>8}{r['pr_auc']:>8.3f}"
                f"{r[f'best_{ek}']:>8.3f}{r['best_n_coarse']:>10}"
                f"{r['best_mean_recall']:>8.3f}{r['best_mean_precision']:>8.3f}"
                f"{r['best_mean_f1']:>8.3f}{r['best_det_rate']:>8.1%}"
            )
        if len(pr_sweep_records) > 1:
            LOGGER.info("  " + "-" * (len(h) - 2))
            LOGGER.info(
                f"  {'mean':<18}{'':>6}{'':>8}{S.pr_auc.mean():>8.3f}"
                f"{S[f'best_{ek}'].mean():>8.3f}{'':>10}"
                f"{S.best_mean_recall.mean():>8.3f}{S.best_mean_precision.mean():>8.3f}"
                f"{S.best_mean_f1.mean():>8.3f}{S.best_det_rate.mean():>8.1%}"
            )
        LOGGER.info(f"\n  PR-sweep CSVs + plots -> {args.out}")

    out_json = args.out / f"elliptic_modular_d{args.day_start}-{args.day_end}.json"
    payload = {
        "dataset": "elliptic++",
        "day_start": args.day_start,
        "day_end": args.day_end,
        "n_nodes": data.num_nodes,
        "feature_mode": args.feature_mode,
        "feature_dim": data.feature_dim,
        "n_gangs": len(gangs),
        "n_train_gangs": len(gang_train),
        "config": cfg.to_dict(),
        # the closed-form solver has no training trace; it reports the Theorem
        # A/B triple instead (N_beta <= Gamma <= N_0) and its optimality gap
        "lambda_min_init": fit.get("init_objective", fit.get("lambda_min_N_beta")),
        "lambda_min_final": fit.get("objective", fit.get("lambda_min_Gamma")),
        # the closed-form / A_eq solvers report the shared max-min certificate
        # (lambda_min(Gamma), max_j chi_j, margin) instead of a training trace
        "closed_form": {
            k: fit[k]
            for k in (
                "lambda_min_N0",
                "lambda_min_N_beta",
                "lambda_min_Gamma",
                "optimality_gap",
                "sandwich_ok",
                "beta",
                "chi_subspace_max",
                "margin",
                "trace_Gamma",
            )
            if k in fit
        }
        or None,
        "aeq": (
            {
                k: fit[k]
                for k in (
                    "eigenvalues",
                    "width",
                    "alpha",
                    "rho",
                    "rho_used",
                    "rho_scale",
                    "kappa",
                    "reweight_iters",
                    "reweight_history",
                    "n_signal",
                    "n_extra_signal",
                    "capture",
                    "chi_subspace",
                )
                if k in fit
            }
            if cfg.collective_solver == "aeq"
            else None
        ),
        "coarsening": {
            "n_original": co.n_original,
            "n_coarse": co.n_coarse,
            "epsilon": co.epsilon,
            # only src.deflated_coarsen produces this (eps_Q, mu, RSA sandwich)
            "deflated_certificate": result.get("deflated_certificate"),
        },
        "report": result["report"],
        # per-split capture C_S = Gamma_jj, measured in the run's own geometry
        # (--coarsening-laplacian) against its own indicator (--indicator), so it
        # is comparable across runs only when those two match
        "captures": {
            name: {k: v for k, v in cap.items() if k != "per_gang_capture"}
            for name, cap in (result.get("captures") or {}).items()
        },
        # certified-margin objective: the certificate and the pre-training
        # diagnostics that explain it (graph-only, so identical across arms)
        "certified_margin": (
            {
                "objective_init": fit["init_objective"],
                "objective_final": fit["margin"],
                "certificate": fit.get("certificate"),
                "n_certified": fit.get("n_certified"),
                "n_feasible_capture": fit.get("margin_feasible_capture"),
                "kappa": fit.get("margin_kappa"),
                "capture_threshold": fit.get("margin_capture_threshold"),
                "chi_forced": fit.get("margin_chi_forced"),
                "nu2": fit.get("margin_nu2"),
                "reachability": fit.get("reachability"),
            }
            if cfg.capture_objective == "certified_margin"
            else None
        ),
        "reachability": fit.get("reachability") or None,
        "coarsener_comparison": result.get("coarsener_comparison") or None,
        # how the SHARED filter did on each training group's own graph (empty for a
        # single group); printed above, kept here so the export path can read it
        "per_group_report": result.get("per_group_report") or None,
        "transfer": {
            # per-gang diagnostic rows live in the CSV, not here (keeps JSON small)
            "days": [
                {k: v for k, v in rec.items() if k != "gang_rows"}
                for rec in transfer_records
            ],
            "average": transfer_summary,
        },
        "pr_sweep": pr_sweep_records or None,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    LOGGER.info(f"\nJSON report: {out_json}")

    basis = det.target_subspace(data, gang_train)
    coarsening, _ = det.coarsen(data, basis, gang_train)
    n2s = coarsening.node_to_supernode

    analyze_coarsening(
        data,
        basis,
        gangs,
        det,
        gang_train,
        args,
        cfg,
        coarsening,
        n2s,
        normals,
    )

    # --- 4. pooled missed-gang diagnostics (reuses coarsenings already computed) --
    # Training day: the coarsening just built above (n2s).  Transfer days: the
    # per-gang rows collected inside _evaluate_transfer_day.  No re-coarsening.
    diag_rows = _gang_diagnostic_rows(
        det, data, gangs, gang_sets, graph.edge_index, args.day_start, n2s
    )
    for rec in transfer_records:
        diag_rows += rec.get("gang_rows", [])
    _write_missed_gang_diagnostics(diag_rows, args.out)

    # move the LOGGER's log file into out_dir and drop the scratch results/
    # folder created for it at import time (src/utils/utils.py).
    _relocate_logger_file(args.out)


if __name__ == "__main__":
    main()
