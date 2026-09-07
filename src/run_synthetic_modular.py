"""Trainable collective filter-bank detector on SYNTHETIC random graphs with
custom, variable-size planted motifs.

This is the synthetic twin of :mod:`src.run_elliptic_modular`: it drives the very
same :class:`~src.collective_detector.CollectiveBankDetector` pipeline (fit ->
target subspace -> coarsen -> evaluate), exposes the same knobs, and prints the
same reports -- only the *data* is swapped.  Instead of Elliptic++ days it plants
motifs on Erdos-Renyi backgrounds, using the graph construction of
:func:`src.run_collective_bank_detection.build_synthetic_graph` but with two
generalizations that make the benchmark harder and more realistic:

* **variable motif sizes** -- each motif's size is drawn uniformly from
  ``[--motif-size-min, --motif-size-max]`` rather than a single fixed size, so
  the bank must capture gangs across a whole size range at once (the small ones
  are the hard, low-``M_tau``-energy tail);
* **mixed motif types** -- ``--motif-types clique,cycle,star,random`` plants a
  random mix, so a single shared filter must reach several distinct spectral
  signatures (a clique is a flat low-pass gang, a long cycle lives in a narrow
  band, a star is a single high-degree hub).

Multi-graph training (the analogue of ``--train-days 24,25,26``) is driven by
``--num-graphs``: each is an independent ER+motif instance with its own
train/test motif split, one drawn/aggregated per epoch exactly as the day-list
fit does.  ``--transfer-graphs`` then applies the *frozen* filter to fresh unseen
graphs (the analogue of ``--transfer-days``) so generalization is measured
honestly.

Example (mixed motifs, sizes 4-40, three training graphs + five transfer graphs)::

    python -m src.run_synthetic_modular \
        --num-graphs 3 --transfer-graphs 5 \
        --num-nodes 1500 --num-motifs 12 \
        --motif-size-min 4 --motif-size-max 40 \
        --motif-types clique,cycle,star,random \
        --heads 8 --conf-weight 10 --epochs 400
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
from torch_geometric.data import Data

import pandas as pd

from src.analyze_elliptic_coarsening import analyze_coarsening
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.pattern_models import create_pattern, make_patterns
from src.plot_training import write_training_report
from src.run_collective_bank_detection import (
    _motif_edges,
    build_node_split,
    inject_gang_features,
    make_negative_sampler,
)

# The sweep / diagnostics live in the Elliptic driver and are dataset-agnostic, so
# they are REUSED here rather than reimplemented -- one implementation, two datasets.
from src.run_elliptic_modular import (
    _gang_diagnostic_rows,
    _run_pr_sweep,
    _write_missed_gang_diagnostics,
)
from src.utils.utils import LOGGER, now


# --------------------------------------------------------------------------- #
# synthetic graph with VARIABLE-SIZE, MIXED-TYPE planted motifs
# --------------------------------------------------------------------------- #
def build_varsize_motif_graph(
    *,
    num_nodes: int,
    num_motifs: int,
    motif_types: list[str],
    size_min: int,
    size_max: int,
    avg_degree: float,
    feature_dim: int,
    rng_seed: int,
    motif_density: float = 1.0,
    motif_conductance: float = -1.0,
) -> tuple[Data, list]:
    """Erdos-Renyi background + ``num_motifs`` disjoint motifs of *varying* size/type.

    Generalizes :func:`src.run_collective_bank_detection.build_synthetic_graph`:
    each motif independently draws a size ``s ~ U[size_min, size_max]`` and a type
    uniformly from ``motif_types``.  Returns the graph (``x``, ``edge_index``,
    ``y``) and the list of :class:`Pattern` objects (label ``"alert"``), each
    tagged with its actual type and size so the report can break detection down by
    both.  ``y`` marks every motif node class 1.
    """

    if size_min < 2 or size_max < size_min:
        raise ValueError("require 2 <= --motif-size-min <= --motif-size-max")
    rng = np.random.default_rng(rng_seed)

    # --- draw per-motif sizes and types, then carve disjoint node blocks ------
    sizes = rng.integers(size_min, size_max + 1, size=num_motifs)
    kinds = [motif_types[int(i)] for i in rng.integers(0, len(motif_types), num_motifs)]
    total = int(sizes.sum())
    if total > num_nodes:
        raise ValueError(
            f"sum of motif sizes ({total}) exceeds num_nodes ({num_nodes}); "
            "lower --num-motifs / --motif-size-max or raise --num-nodes."
        )
    perm = rng.permutation(num_nodes)
    offs = np.concatenate([[0], np.cumsum(sizes)])
    motif_node_lists = [perm[offs[m] : offs[m + 1]].tolist() for m in range(num_motifs)]

    edges: set[tuple[int, int]] = set()

    # --- Erdos-Renyi background ----------------------------------------------
    n_background = int(num_nodes * avg_degree / 2)
    src = rng.integers(0, num_nodes, size=n_background)
    dst = rng.integers(0, num_nodes, size=n_background)
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            edges.add((min(u, v), max(u, v)))

    # --- planted motifs -------------------------------------------------------
    patterns = []
    y = np.zeros(num_nodes, dtype=np.int64)
    for m in range(num_motifs):
        nodes = motif_node_lists[m]
        kind = kinds[m]
        for u, v in _motif_edges(nodes, kind, density=motif_density, rng=rng):
            edges.add((min(u, v), max(u, v)))
        y[nodes] = 1
        p = create_pattern(f"{kind}_{m}", nodes, kind, label="alert")
        # stash the type/size so the report can bin by them (Pattern is permissive)
        p.motif_kind = kind
        p.motif_size = len(nodes)
        patterns.append(p)

    # --- optional conductance tuning (motif -> host wiring) -------------------
    if motif_conductance is not None and motif_conductance >= 0.0:
        if motif_conductance >= 1.0:
            raise ValueError("--motif-conductance must be in [0, 1)")
        motif_of = -np.ones(num_nodes, dtype=np.int64)
        for m in range(num_motifs):
            motif_of[np.asarray(motif_node_lists[m])] = m
        host_nodes = np.nonzero(motif_of < 0)[0]
        cut = np.zeros(num_motifs, dtype=np.int64)
        vol = np.zeros(num_motifs, dtype=np.int64)
        for u, v in edges:
            mu, mv = motif_of[u], motif_of[v]
            if mu >= 0:
                vol[mu] += 1
            if mv >= 0:
                vol[mv] += 1
            if mu != mv:
                if mu >= 0:
                    cut[mu] += 1
                if mv >= 0:
                    cut[mv] += 1
        for m in range(num_motifs):
            k = int(
                round((motif_conductance * vol[m] - cut[m]) / (1.0 - motif_conductance))
            )
            if k <= 0:
                continue
            nodes = np.asarray(motif_node_lists[m])
            added, attempts, cap = 0, 0, 20 * k + 100
            while added < k and attempts < cap:
                attempts += 1
                u = int(rng.choice(nodes))
                w = int(rng.choice(host_nodes))
                e = (min(u, w), max(u, w))
                if e in edges:
                    continue
                edges.add(e)
                added += 1

    edge_array = np.array(sorted(edges), dtype=np.int64).T
    edge_index = torch.from_numpy(
        np.concatenate([edge_array, edge_array[::-1]], axis=1)
    ).long()

    gen = torch.Generator().manual_seed(rng_seed)
    X = torch.randn(num_nodes, feature_dim, dtype=torch.float64, generator=gen)
    X = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)

    graph = Data(x=X, edge_index=edge_index, y=torch.from_numpy(y), num_nodes=num_nodes)
    return graph, patterns


def split_motifs(patterns, train_ratio, rng):
    """Shuffle and split motifs into (train, test) by ``train_ratio``."""
    idx = rng.permutation(len(patterns))
    k = max(1, int(round(train_ratio * len(patterns))))
    order = [patterns[int(i)] for i in idx]
    return order[:k], order[k:]


def _load_graph(seed, args):
    """Build one synthetic graph instance + its train/test motif split."""
    graph, patterns = build_varsize_motif_graph(
        num_nodes=args.num_nodes,
        num_motifs=args.num_motifs,
        motif_types=args.motif_types,
        size_min=args.motif_size_min,
        size_max=args.motif_size_max,
        avg_degree=args.avg_degree,
        feature_dim=args.feature_dim,
        rng_seed=seed,
        motif_density=args.motif_density,
        motif_conductance=args.motif_conductance,
    )
    if args.feat_shared > 0.0 or args.feat_signature > 0.0:
        graph.x = inject_gang_features(
            graph.x,
            patterns,
            shared=args.feat_shared,
            signature=args.feat_signature,
            seed=seed,
        )
    tr, te = split_motifs(patterns, args.train_ratio, np.random.default_rng(seed + 1))
    if args.max_train_gangs and len(tr) > args.max_train_gangs:
        tr = tr[: args.max_train_gangs]
    return GraphData.from_graph(graph), tr, te, patterns


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def _fmt_split(name, r):
    return (
        f"  {name:<7}{r['total']:>6}{r['mean_recall']:>9.3f}{r['mean_precision']:>11.3f}"
        f"{r['mean_f1']:>8.3f}{r['detection_rate']:>11.1%}"
        f"{r['detected']:>5}/{r['total']:<4}"
    )


def _detection_by(patterns, node_to_super, y, threshold, key):
    """Detection rate bucketed by a per-motif key (type or size band)."""
    from src.loukas_sgc_detection import evaluate_loukas_patterns

    buckets: dict = {}
    for p in patterns:
        results, by_label = evaluate_loukas_patterns(
            [p], node_to_super, y, threshold=threshold
        )
        det = by_label.get("alert", {}).get("detected", 0)
        b = buckets.setdefault(key(p), [0, 0])
        b[0] += int(det)
        b[1] += 1
    return buckets


def _size_band(p):
    s = getattr(p, "motif_size", len(p.node_indices))
    for lo, hi in ((2, 5), (6, 10), (11, 20), (21, 40), (41, 10_000)):
        if lo <= s <= hi:
            return f"{lo}-{hi if hi < 10_000 else 'inf'}"
    return "?"


def _background_patterns(data, patterns, args, seed):
    """Non-motif background blobs -- the synthetic stand-in for Elliptic's licit CCs.

    The ``[4] non-gang PR`` panel asks how often the coarsener collapses sets that
    are *not* planted gangs; that needs negative sets drawn from the same graph.
    Reuses the repo's negative sampler (random connected neighbourhoods of random
    size that avoid every motif node), so they are genuine background structure
    rather than gangs.
    """

    avoid = [int(v) for p in patterns for v in p.node_indices]
    sampler = make_negative_sampler(
        data.adjacency.coalesce().indices(),
        data.num_nodes,
        num_sets=args.max_normal_patterns,
        size_min=max(2, args.motif_size_min),
        size_max=max(3, args.motif_size_max),
        avoid=sorted(set(avoid)),
        rng=np.random.default_rng(seed + 4242),
    )
    return make_patterns(sampler(), "normal", "normal", "n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)

    # --- synthetic dataset ---
    ap.add_argument(
        "--num-graphs",
        type=int,
        default=3,
        help="number of independent training graphs (the day-list analogue)",
    )
    ap.add_argument(
        "--transfer-graphs",
        type=int,
        default=5,
        help="fresh unseen graphs the frozen filter is applied to (0=off)",
    )
    ap.add_argument("--num-nodes", type=int, default=1500)
    ap.add_argument("--num-motifs", type=int, default=12)
    ap.add_argument("--motif-size-min", type=int, default=7)
    ap.add_argument("--motif-size-max", type=int, default=20)
    ap.add_argument(
        "--motif-types",
        type=str,
        default="random",
        help="comma list drawn from {clique,cycle,star,random}",
    )
    ap.add_argument(
        "--motif-density",
        type=float,
        default=0.4,
        help="edge density of 'random' motifs (1.0 = clique)",
    )
    ap.add_argument(
        "--motif-conductance",
        type=float,
        default=-1.0,
        help=">=0 wires each motif to hosts to hit this Phi (-1 = planted)",
    )
    ap.add_argument("--avg-degree", type=float, default=6.0)
    ap.add_argument(
        "--feature-dim",
        type=int,
        default=32,
        help="isotropic random feature width d (the reachability channel)",
    )
    ap.add_argument(
        "--feat-shared",
        type=float,
        default=0.0,
        help="inject a shared gang feature signature of this strength",
    )
    ap.add_argument(
        "--feat-signature",
        type=float,
        default=0.0,
        help="inject a per-gang feature signature of this strength",
    )
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--max-train-gangs", type=int, default=0)

    # --- detector hyperparameters (mirror DetectorConfig / run_elliptic_modular) ---
    ap.add_argument(
        "--collective-solver",
        choices=["gradient", "closed-form", "trace-ratio"],
        default="gradient",
    )
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument(
        "--basis", choices=["chebyshev", "monomial", "lanczos"], default="chebyshev"
    )
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--ridge", type=float, default=1e-7)
    ap.add_argument(
        "--optimizer", choices=["projected", "riemannian", "lbfgs"], default="projected"
    )
    ap.add_argument("--day-aggregate", choices=["sample", "mean", "min"], default="min")
    ap.add_argument("--trace-ratio-iters", type=int, default=40)
    ap.add_argument("--pencil-beta", type=float, default=0.0)
    ap.add_argument("--softmin-temperature", type=float, default=0.02)
    ap.add_argument(
        "--warm-start",
        choices=["ones", "closed_form", "resolvent"],
        default="closed_form",
    )
    ap.add_argument("--softmin-anneal", type=float, default=5.0)
    ap.add_argument(
        "--capture-objective",
        choices=["lambda_min", "trace", "softmin_diag", "certified_margin"],
        default="lambda_min",
    )
    ap.add_argument("--margin-alpha", type=float, default=0.0)
    ap.add_argument("--margin-softplus", type=float, default=0.0)
    ap.add_argument("--label-weight", type=float, default=0.0)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--conf-weight", type=float, default=50.0)
    ap.add_argument("--conf-reduce", choices=["max", "mean"], default="mean")
    ap.add_argument("--conf-delta", type=float, default=0.0)
    ap.add_argument("--conf-halo-hops", type=int, default=1)
    ap.add_argument("--structural-width", type=int, default=0)
    ap.add_argument(
        "--coarsen-target",
        choices=["bank", "indicators", "dictionary", "bank+dictionary"],
        default="bank",
    )
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-diversity", type=float, default=0.0)

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
        ],
        default="ward-tree",
    )
    ap.add_argument(
        "--coarsening-laplacian",
        choices=["symmetric", "combinatorial"],
        default="symmetric",
    )
    ap.add_argument("--reduction", type=float, default=0.3)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--max-levels", type=int, default=10)
    ap.add_argument("--ward-stop", choices=["epsilon", "f1"], default="f1")
    ap.add_argument("--ward-num-cuts", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.51)

    # --- sweeps + diagnostics (the epsilon* / PR-AUC reporting) ---
    ap.add_argument(
        "--pr-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="walk the whole Ward merge order (finest -> 2 clusters) recording "
        "recall/precision/f1/jaccard/detection + epsilon at every level: a full "
        "epsilon sweep 0->1, a PR curve (+AUC) and a metrics-vs-epsilon plot per graph.",
    )
    ap.add_argument(
        "--pr-sweep-exact-budget",
        type=int,
        default=200,
        help="if >0, use the EXACT RSA constant as the sweep's epsilon axis, "
        "computed at this many adaptively-placed levels and interpolated.",
    )
    ap.add_argument(
        "--analyze-coarsening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the per-gang PR / edge-cost / conductance / non-gang PR figures.",
    )
    ap.add_argument(
        "--num-gang-graphs",
        type=int,
        default=2,
        help="how many individual motif subgraphs to draw (figure 2c)",
    )
    ap.add_argument(
        "--max-normal-patterns",
        type=int,
        default=120,
        help="background (non-motif) sets sampled for the non-gang PR panel",
    )
    ap.add_argument(
        "--missed-gang-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pooled per-motif structural stats vs detection outcome (+AUC table).",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=f"results/synthetic_modular/{now}/", type=Path)
    args = ap.parse_args()

    args.motif_types = [t.strip() for t in args.motif_types.split(",") if t.strip()]
    valid = {"clique", "cycle", "star", "random"}
    if not set(args.motif_types) <= valid:
        ap.error(f"--motif-types must be a subset of {sorted(valid)}")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    LOGGER.info("=" * 74)
    LOGGER.info(
        f"=== SYNTHETIC (modular) | {args.num_graphs} train graph(s), "
        f"motifs {args.motif_types} sizes {args.motif_size_min}-"
        f"{args.motif_size_max} ==="
    )
    LOGGER.info("=" * 74)

    # --- 1. build the training graphs ---------------------------------------
    day_specs = []
    for g in range(args.num_graphs):
        data_g, tr_g, te_g, all_g = _load_graph(args.seed + 100 * g, args)
        sizes = sorted(
            (getattr(p, "motif_size", len(p.node_indices)) for p in all_g), reverse=True
        )
        kinds = {}
        for p in all_g:
            kinds[getattr(p, "motif_kind", "?")] = (
                kinds.get(getattr(p, "motif_kind", "?"), 0) + 1
            )
        LOGGER.info(
            f"  graph G{g}: N={data_g.num_nodes:,}  motifs={len(all_g)} "
            f"(train {len(tr_g)} / test {len(te_g)})  sizes={sizes}  mix={kinds}"
        )
        day_specs.append((f"G{g}", data_g, tr_g, te_g))
    data, gang_train, gang_test, gangs = (
        day_specs[-1][1],
        day_specs[-1][2],
        day_specs[-1][3],
        day_specs[-1][2] + day_specs[-1][3],
    )
    if len(gang_train) > data.feature_dim * args.heads:
        LOGGER.info(
            f"  WARNING #train-motifs ({len(gang_train)}) > H*d "
            f"({data.feature_dim * args.heads}): capacity threshold, lambda_min ~ 0. "
            "Raise --feature-dim/--heads or lower --num-motifs."
        )

    # --- 2. detector config (every knob) ------------------------------------
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
        trace_ratio_iters=args.trace_ratio_iters,
        day_aggregate=args.day_aggregate,
        label_weight=args.label_weight,
        conf_weight=args.conf_weight,
        conf_reduce=args.conf_reduce,
        conf_delta=args.conf_delta,
        conf_halo_hops=args.conf_halo_hops,
        structural_width=args.structural_width,
        coarsen_target=args.coarsen_target,
        heads=args.heads,
        head_diversity=args.head_diversity,
        coarsening_method=args.coarsening_method,
        coarsening_laplacian=args.coarsening_laplacian,
        reduction=args.reduction,
        epsilon=args.epsilon,
        max_levels=args.max_levels,
        ward_stop=args.ward_stop,
        ward_num_cuts=args.ward_num_cuts,
        threshold=args.threshold,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)

    # optional joint supervised node head (only meaningful for a single graph)
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
            f"{len(label_idx):,} labelled nodes"
        )

    LOGGER.info(
        f"\n  Fitting collective bank (solver={cfg.collective_solver}, basis={cfg.basis}, "
        f"tau={cfg.tau}, K={cfg.degree}, H={cfg.heads}, opt={cfg.optimizer}, "
        f"objective={cfg.capture_objective}, beta={cfg.conf_weight:g}, "
        f"delta={cfg.conf_delta:g}, aggregate={cfg.day_aggregate}) ..."
    )

    det.fit(day_specs, label_y=label_y, label_idx=label_idx)
    fit = det.fit_info_

    if cfg.capture_objective == "certified_margin":
        import numpy as _np

        kap = _np.asarray(fit.get("margin_kappa") or [])
        rch = _np.asarray(fit.get("reachability") or [])
        LOGGER.info(
            f"    certified margin: {fit['init_objective']:.4g} -> "
            f"{fit['margin']:.4g}   CERTIFICATE (exact hinge) "
            f"{fit.get('certificate', float('nan')):.4g}  -> "
            f"{fit.get('n_certified', 0)}/{fit['n_train_patterns']} gangs certified, "
            f"{fit.get('margin_feasible_capture', 0)} above the capture threshold"
        )
        if kap.size:
            LOGGER.info(
                f"      kappa median {_np.median(kap):.4g} => capture needed "
                f"{_np.median(1 - kap**2):.4g}   |   R_K median "
                f"{(_np.median(rch) if rch.size else float('nan')):.4g}"
            )
    elif cfg.collective_solver != "closed-form":
        LOGGER.info(
            f"    capture ({cfg.capture_objective}) / lambda_min(Gamma): "
            f"{fit['init_objective']:.4g} -> {fit['objective']:.4g}"
        )
        if cfg.conf_weight > 0:
            LOGGER.info(
                f"    confusability chi: {fit['confusability_init']:.4g} -> "
                f"{fit['confusability']:.4g}"
            )
    if "per_day" in fit and len(fit["per_day"]) > 1:
        LOGGER.info("    per-graph lambda_min at the retained filter:")
        for lbl, v in fit["per_day"].items():
            LOGGER.info(
                f"      {lbl}: lambda_min={v['lambda_min']:.4g}  "
                f"mean C={v['mean_capture']:.4g}  ({v['n_train_gangs']} motifs)"
            )

    fig = write_training_report(
        fit,
        args.out,
        title=(
            f"synthetic bank fit -- {args.num_graphs} graphs, "
            f"motifs {'+'.join(args.motif_types)} sizes "
            f"{args.motif_size_min}-{args.motif_size_max} "
            f"({cfg.capture_objective}, beta={cfg.conf_weight:g}, K={cfg.degree}, "
            f"H={cfg.heads})"
        ),
    )
    if fig is not None:
        LOGGER.info(f"    training curves + history CSV -> {fig}")

    # --- 3. per-training-graph performance (shared filter, own coarsening) ---
    per_group_reports = {}
    if len(day_specs) > 1:
        LOGGER.info("\n" + "=" * 74)
        LOGGER.info("PER-GRAPH TRAINING PERFORMANCE (shared filter, own coarsening)")
        LOGGER.info("=" * 74)
        LOGGER.info(
            f"  {'graph':<7}{'motifs':>6}{'recall':>9}{'precision':>11}"
            f"{'f1':>8}{'detection':>11}{'det/tot':>10}"
        )
        LOGGER.info("  " + "-" * 62)
        for lbl, g_data, g_train, _g_test in day_specs:
            g_basis = det.target_subspace(g_data, g_train)
            g_co, _ = det.coarsen(g_data, g_basis, g_train)
            rep = det.evaluate(g_data, g_co, {"train": g_train})["train"]
            per_group_reports[lbl] = rep
            LOGGER.info(_fmt_split(lbl, rep))

    # --- 4. full report on the LAST graph (train/test/all) ------------------
    basis_ = det.target_subspace(data, gang_train)
    coarsening, _ = det.coarsen(data, basis_, gang_train)
    splits = {"train": gang_train, "test": gang_test, "all": gangs}
    report = det.evaluate(data, coarsening, splits)
    captures = {n: det.capture(data, p) for n, p in splits.items() if p}

    LOGGER.info(
        f"  coarsening ({cfg.coarsening_method}): N={coarsening.n_original:,} "
        f"-> n_coarse={coarsening.n_coarse:,}  epsilon={coarsening.epsilon:.4g}"
    )
    LOGGER.info("\n" + "=" * 74)
    LOGGER.info(
        f"SYNTHETIC GANG DETECTION (reported graph {day_specs[-1][0]})  "
        f"N={data.num_nodes:,}  {len(gangs)} motifs"
    )
    LOGGER.info("=" * 74)
    LOGGER.info(
        f"  {'split':<7}{'motifs':>6}{'recall':>9}{'precision':>11}"
        f"{'f1':>8}{'detection':>11}{'det/tot':>10}"
    )
    LOGGER.info("  " + "-" * 62)
    for name in ("train", "test", "all"):
        if report.get(name):
            LOGGER.info(_fmt_split(name, report[name]))

    # --- 4a. detection broken down by motif TYPE and by SIZE band -----------
    n2s = coarsening.node_to_supernode
    LOGGER.info("\n  detection by motif TYPE (all motifs on reported graph):")
    by_type = _detection_by(
        gangs, n2s, data.y, cfg.threshold, lambda p: getattr(p, "motif_kind", "?")
    )
    for k, (d_, t_) in sorted(by_type.items()):
        LOGGER.info(f"    {k:<10} {d_:>3}/{t_:<3}  ({d_ / max(t_, 1):.0%})")
    LOGGER.info("  detection by motif SIZE band:")
    by_size = _detection_by(gangs, n2s, data.y, cfg.threshold, _size_band)
    for k in sorted(by_size, key=lambda s: int(s.split("-")[0])):
        d_, t_ = by_size[k]
        LOGGER.info(f"    size {k:<8} {d_:>3}/{t_:<3}  ({d_ / max(t_, 1):.0%})")

    # --- 4b. capture diagnostics --------------------------------------------
    LOGGER.info("\n  capture (retained M_tau energy) on the reported graph:")
    for name, cap in captures.items():
        LOGGER.info(
            f"    {name:<5} mean_C={cap['mean_capture']:.4f}  "
            f"min_C={cap['min_capture']:.4f}  "
            f"lambda_min(Gamma)={cap['lambda_min_gamma']:.4f}"
        )

    # --- 4c. FULL WARD PR-SWEEP on the reported graph (epsilon* + PR-AUC) ----
    # `_run_pr_sweep` wants day_start/day_end for nothing but naming; the shared
    # `analyze_coarsening` below reads them too, so set them once here.
    args.day_start, args.day_end = 0, len(day_specs) - 1
    gang_sets = [list(map(int, p.node_indices)) for p in gangs]
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
        sweep_basis = det.target_subspace(data, gangs)
        rec = _run_pr_sweep(
            det,
            data,
            sweep_basis,
            gang_sets,
            f"train_{day_specs[-1][0]}",
            args,
            dataset="synthetic",
        )
        if rec is not None:
            pr_sweep_records.append(rec)

    # --- 5. TRANSFER: frozen filter on fresh unseen graphs ------------------
    diag_rows = []
    if args.missed_gang_diagnostics:
        diag_rows += _gang_diagnostic_rows(
            det, data, gangs, gang_sets, data.edge_index, day_specs[-1][0], n2s
        )
    transfer_rows = []
    if args.transfer_graphs > 0:
        LOGGER.info("\n" + "=" * 74)
        LOGGER.info(
            f"TRANSFER: frozen filter on {args.transfer_graphs} fresh unseen graphs"
        )
        LOGGER.info("=" * 74)
        LOGGER.info(
            f"  {'graph':<7}{'motifs':>6}{'recall':>9}{'precision':>11}"
            f"{'f1':>8}{'detection':>11}{'det/tot':>10}"
        )
        LOGGER.info("  " + "-" * 62)
        for k in range(args.transfer_graphs):
            t_data, _t_tr, _t_te, t_all = _load_graph(args.seed + 9000 + 100 * k, args)
            t_basis = det.target_subspace(t_data, t_all)
            t_co, _ = det.coarsen(t_data, t_basis, t_all)
            rep = det.evaluate(t_data, t_co, {"all": t_all})["all"]
            transfer_rows.append(rep)
            LOGGER.info(_fmt_split(f"T{k}", rep))
            t_sets = [list(map(int, p.node_indices)) for p in t_all]
            if args.pr_sweep:  # full epsilon sweep on this graph, same basis
                t_rec = _run_pr_sweep(
                    det,
                    t_data,
                    t_basis,
                    t_sets,
                    f"T{k}",
                    args,
                    dataset="synthetic",
                )
                if t_rec is not None:
                    pr_sweep_records.append(t_rec)
            if args.missed_gang_diagnostics:
                diag_rows += _gang_diagnostic_rows(
                    det,
                    t_data,
                    t_all,
                    t_sets,
                    t_data.edge_index,
                    f"T{k}",
                    t_co.node_to_supernode,
                )
        if transfer_rows:
            m = {
                kk: float(np.mean([r[kk] for r in transfer_rows]))
                for kk in ("mean_recall", "mean_precision", "mean_f1", "detection_rate")
            }
            LOGGER.info("  " + "-" * 62)
            LOGGER.info(
                f"  {'MEAN':<7}{'':>6}{m['mean_recall']:>9.3f}"
                f"{m['mean_precision']:>11.3f}{m['mean_f1']:>8.3f}"
                f"{m['detection_rate']:>11.1%}"
            )

    # --- 5b. PR-sweep summary across the reported graph + every transfer graph -
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
            f"  {'graph':<18}{'motifs':>7}{'levels':>8}{'PR-AUC':>8}"
            f"{'eps*':>8}{'n_coarse':>10}{'recall':>8}{'prec':>8}{'F1':>8}{'det':>8}"
        )
        LOGGER.info(h)
        LOGGER.info("  " + "-" * (len(h) - 2))
        for r in pr_sweep_records:
            LOGGER.info(
                f"  {r['tag']:<18}{r['n_gangs']:>7}{r['levels']:>8}{r['pr_auc']:>8.3f}"
                f"{r[f'best_{ek}']:>8.3f}{r['best_n_coarse']:>10}"
                f"{r['best_mean_recall']:>8.3f}{r['best_mean_precision']:>8.3f}"
                f"{r['best_mean_f1']:>8.3f}{r['best_det_rate']:>8.1%}"
            )
        if len(pr_sweep_records) > 1:
            LOGGER.info("  " + "-" * (len(h) - 2))
            LOGGER.info(
                f"  {'mean':<18}{'':>7}{'':>8}{S.pr_auc.mean():>8.3f}"
                f"{S[f'best_{ek}'].mean():>8.3f}{'':>10}"
                f"{S.best_mean_recall.mean():>8.3f}{S.best_mean_precision.mean():>8.3f}"
                f"{S.best_mean_f1.mean():>8.3f}{S.best_det_rate.mean():>8.1%}"
            )
        LOGGER.info(f"\n  PR-sweep CSVs + plots -> {args.out}")

    # --- 5c. coarsening figures (per-gang PR / edge cost / conductance / non-gang)
    if args.analyze_coarsening:
        normals = _background_patterns(data, gangs, args, args.seed)
        LOGGER.info(
            f"\n  background (non-motif) sets for the non-gang PR: {len(normals)}"
        )
        analyze_coarsening(
            data,
            basis_,
            gangs,
            det,
            gang_train,
            args,
            cfg,
            coarsening,
            n2s,
            normals,
        )

    # --- 5d. pooled missed-motif diagnostics (reported + transfer graphs) -----
    if args.missed_gang_diagnostics:
        _write_missed_gang_diagnostics(diag_rows, args.out)

    # --- 6. dump the JSON report --------------------------------------------
    out_json = args.out / "synthetic_modular_report.json"
    with open(out_json, "w") as fh:
        json.dump(
            {
                "config": cfg.to_dict(),
                "args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "fit": {
                    k: fit[k]
                    for k in (
                        "init_objective",
                        "objective",
                        "margin",
                        "confusability_init",
                        "confusability",
                        "train_days",
                        "per_day",
                    )
                    if k in fit
                },
                "report": report,
                "captures": captures,
                "per_group_report": per_group_reports,
                "pr_sweep": pr_sweep_records or None,
                "detection_by_type": {k: v for k, v in by_type.items()},
                "detection_by_size": {k: v for k, v in by_size.items()},
                "transfer": transfer_rows,
            },
            fh,
            indent=2,
            default=float,
        )
    LOGGER.info(f"\nJSON report -> {out_json}")


if __name__ == "__main__":
    main()
