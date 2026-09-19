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

Comparison mode
---------------
``--compare`` (implied by ``--density-sweep``) replaces the single-configuration
report with a **(density x laplacian x capture-objective x coarsener) grid**,
which answers four questions in one pass:

1. does the bank learn better under ``lambda_min`` -- capture *and* cross-gang
   separation, carrying the ``m > H*d`` capacity wall -- or under ``trace``, the
   mean per-gang capture with no separation term and no wall?
2. how does the rank-``q`` **raw-Ward** coarsener (:mod:`src.raw_ward`) compare
   with the contiguity-constrained Ward tree and with deflated dual Ward?
3. how do both answers move with graph density?
4. how do they move between the symmetric geometry (``M_tau = L_sym + tau I``,
   volumes and conductance) and the combinatorial one (``M_tau = D - W + tau I``,
   cardinalities -- **no volumes anywhere**: raw Ward's level loses its
   ``D_t^{-1/2}`` factor, its block mean becomes the plain average and
   ``Phi = cut/|A|``)?

Everything not under test is held fixed: for one density every cell sees the SAME
graphs, and for one ``(density, laplacian, objective)`` every coarsener sees the
SAME fitted filter and the SAME target subspace ``R = span(Z)``, so a difference
in the table is a difference in the coarsener alone.

Two caveats the tables carry explicitly.  ``--coarsening-laplacian`` is a whole
*pipeline* switch, not a coarsener switch -- it selects the metric the bank is
trained in, the indicator capture is a fraction of, and the RSA axis -- so the
bank is refit per laplacian and the two sides are different experiments, not one
experiment coarsened twice.  And the deflated family (plus ``dual-ward``) is
symmetric-only by construction, so the combinatorial columns compare ``ward-tree``
against ``raw-ward`` only.

**Every performance number is reported at ``epsilon*``** -- the sweep level where
mean F1 peaks -- with ``epsilon`` recomputed from the partition on one common
axis (Loukas Def. 2, uniform block average in ``M_tau``), because the coarseners
otherwise report the constant in their own conventions.  A second, stop-free
reading at a matched supernode budget (``--compare-kept``) checks the merge order
at a reduction every method is held to equally.  Results land in
``comparison_rows.csv`` / ``comparison_curves.csv`` /
``comparison_matched_budget.csv`` / ``comparison_fit.csv`` plus two figures::

    python -m src.run_synthetic_modular \
        --density-sweep 3,4,6,8,12,16 --density-knob avg-degree \
        --compare-laplacians symmetric,combinatorial \
        --compare-objectives lambda_min,trace \
        --compare-coarseners ward-tree,raw-ward,deflated-dual-ward \
        --num-graphs 2 --transfer-graphs 4 --epochs 1000 --ward-stop f1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
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
    _relocate_logger_file,
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
# detector config (shared by the single run and the comparison grid)
# --------------------------------------------------------------------------- #
def _make_cfg(args, **overrides) -> DetectorConfig:
    """Every knob of ``args`` as a :class:`DetectorConfig`, with ``overrides``.

    The comparison grid varies only ``capture_objective`` / ``coarsening_method``
    / ``ward_num_cuts``, so it goes through the *same* constructor as the single
    run -- nothing can drift between the two paths.
    """

    kw = dict(
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
        level_beta=getattr(args, "level_beta", 1.0),
        level_chi_temperature=getattr(args, "level_chi_temperature", 0.05),
        ridge_relative=getattr(args, "ridge_relative", 0.0),
        edge_gamma=getattr(args, "edge_gamma", 1.0),
        cf_share_filters=getattr(args, "cf_share_filters", False),
        cf_feature_draws=getattr(args, "cf_feature_draws", 0),
        cf_host_mode=getattr(args, "cf_host_mode", "none"),
        cf_host_weight=getattr(args, "cf_host_weight", 0.0),
        cf_host_count=getattr(args, "cf_host_count", 0),
        cf_param_scale=getattr(args, "cf_param_scale", "absolute"),
        cf_solver_form=getattr(args, "cf_solver_form", "difference"),
    )
    kw.update(overrides)
    return DetectorConfig(**kw)


# --------------------------------------------------------------------------- #
# objective x coarsener x density comparison grid
# --------------------------------------------------------------------------- #
# Three questions, one grid, one set of graphs:
#
#   1. does the bank learn better under ``lambda_min`` (capture + cross-gang
#      separation, with the m > H*d capacity wall) or under ``trace`` (mean
#      per-gang capture, no separation term, no wall)?
#   2. how does the rank-q **raw-Ward** coarsener (:mod:`src.raw_ward`) compare
#      with the contiguity-constrained Ward tree and with deflated dual Ward?
#   3. how do both answers move with graph density?
#
# Everything that is not the variable under test is held fixed: for one density
# the SAME graphs are used by every objective, and for one (density, objective)
# the SAME fitted filter and the SAME target subspace are handed to every
# coarsener.  Only the coarsening differs, so a difference in the table is a
# difference in the coarsener.
_DENSITY_KNOBS = {
    "avg-degree": "avg_degree",  # ER background degree: the graph's own density
    "motif-density": "motif_density",  # internal edge density of 'random' motifs
    "motif-conductance": "motif_conductance",  # motif -> host wiring (Phi target)
}


def _common_epsilon(det, data, basis, node_to_supernode) -> float:
    """Exact RSA distortion of a partition, computed the SAME way for every method.

    Each coarsener reports an ``epsilon`` in its own convention (the Ward tree and
    raw Ward in the uniform block-average constant, the deflated family alongside
    a harmonic one), so the numbers on the ``coarsening.epsilon`` field are not
    all on one axis.  This recomputes Loukas Def. 2 from the partition alone --
    same ``M_tau``-orthonormalized basis, same uniform projector -- so the
    comparison table has one epsilon column that means one thing.
    """

    from src.loukas_sgc_detection import (
        _exact_rsa_epsilon,
        _l_orthonormalize,
        _laplacian,
        _normalized_laplacian,
        _screened_metric,
    )

    c = det.config
    base_fn = (
        _laplacian
        if c.coarsening_laplacian in ("combinatorial", "comb")
        else _normalized_laplacian
    )
    metric = _screened_metric(base_fn(data.adjacency), c.tau)
    a0 = _l_orthonormalize(basis, metric)
    return float(_exact_rsa_epsilon(a0, metric, node_to_supernode))


def _convergence(fit) -> dict:
    """Did the ascent settle?  Checked on the FIXED components of the objective.

    A tau sweep is only interpretable if every cell is read at its own optimum
    rather than wherever its ascent happened to be when the epoch budget ran out
    -- tau changes the conditioning of ``M_tau``, so a fixed epoch count is not a
    fixed amount of progress.

    The trace of the *ascended* value is the wrong thing to test: with
    ``softmin_anneal > 1`` the soft-min temperature moves every epoch, so that
    value is not a single function of ``Theta`` and can fall while the fit is
    improving.  These traces are: ``history`` = ``lambda_min(Gamma)``,
    ``energy_history`` = mean capture, ``conf_history`` = confusability ``chi``.
    Each is declared settled when its drift over the last tenth of training is
    under 2% of the range it covered.
    """

    out: dict = {}
    drifts = []
    for key, tag in (
        ("history", "lambda_min"),
        ("energy_history", "capture"),
        ("conf_history", "chi"),
        ("levelobj_chi_history", "chibar"),
    ):
        h = np.asarray([v for v in (fit.get(key) or []) if np.isfinite(v)], dtype=float)
        if h.size < 20:
            continue
        # smooth first: per-epoch jitter of a stochastic ascent is not drift
        w = max(1, h.size // 50)
        if w > 1:
            h = np.convolve(h, np.ones(w) / w, mode="valid")
        scale = max(float(h.max() - h.min()), 1e-12)
        k = max(1, h.size // 10)
        drift = abs(float(h[-k:].mean() - h[-2 * k : -k].mean())) / scale
        out[f"drift_{tag}"] = drift
        drifts.append(drift)
    lam = np.asarray(
        [v for v in (fit.get("history") or []) if np.isfinite(v)], dtype=float
    )
    out["best_epoch_frac"] = (
        float(int(np.argmax(lam)) / max(lam.size - 1, 1))
        if lam.size > 1
        else float("nan")
    )
    out["tail_drift"] = max(drifts) if drifts else float("nan")
    out["converged"] = bool(drifts and out["tail_drift"] < 0.02)
    out["n_epochs"] = int(lam.size)
    return out


def _level_diagnostics(det, data, patterns) -> dict:
    """Exact ``C_S``, exact ``chi_S`` and the level bound ``chibar_S`` -- any objective.

    Computed from the frozen filter's own bank ``Z`` on ``patterns`` (the groups
    the fit saw), the same way for every arm, so the lambda_min and level
    objectives are compared on what each one claims to optimize rather than on
    its own training value.  ``chi_S`` is the exact confusability (a small
    generalized eigenproblem on the internal-fluctuation space), ``chibar_S =
    lambda_max(G^-1/2 Sigma_S G^-1/2)/tau`` the Sec. 5 bound the level objective
    minimizes; their ratio says how loose that surrogate is.
    """

    from scipy.linalg import eigh

    from src.run_collective_bank_detection import (
        _basis_stack,
        _filtered_bank,
        theta_degree,
    )

    c = det.config
    geo = det.geometry(data)
    if not patterns or getattr(geo, "kind", "symmetric") == "combinatorial":
        return {}
    tau = float(c.tau)
    with torch.no_grad():
        prop = _basis_stack(
            geo.prop, data.X, theta_degree(det.theta_), c.basis, tau, geometry=geo
        )
        Z = _filtered_bank(prop, det.theta_)
        MZ = geo.m_apply(Z, tau)
    Z, MZ = Z.double().cpu().numpy(), MZ.double().cpu().numpy()
    G = 0.5 * (Z.T @ MZ + (Z.T @ MZ).T)
    ev, V = np.linalg.eigh(G)
    keep = ev > 1e-10 * ev.max()
    W = V[:, keep] / np.sqrt(ev[keep])  # G^+1/2 on the numerical range
    MU = MZ @ W  # M_tau U, U = Z W is M_tau-orthonormal
    caps, chis, bounds = [], [], []
    for p in patterns:
        S = np.asarray(sorted(map(int, p.node_indices)))
        if S.size < 2:
            continue
        idx = torch.as_tensor(S, dtype=torch.long, device=data.X.device)
        sd = geo.node_weights(idx).double().cpu().numpy()  # sqrt(d~)
        onehot = torch.zeros(
            data.num_nodes, S.size, dtype=data.X.dtype, device=data.X.device
        )
        onehot[idx, torch.arange(S.size, device=data.X.device)] = 1.0
        with torch.no_grad():
            Mss = geo.m_apply(onehot, tau)[idx].double().cpu().numpy()
        Mss = 0.5 * (Mss + Mss.T)
        # exact capture
        v = sd / np.sqrt(sd @ Mss @ sd)  # v_hat on S (unit M_tau norm)
        caps.append(float(np.sum((MU[S].T @ v) ** 2)))
        # level bound: lambda_max(E G^+ E^T)/tau, E rows sqrt(d~)(ell - lbar)
        ell = MZ[S] / sd[:, None]
        lbar = (sd[:, None] ** 2 * ell).sum(0) / (sd**2).sum()
        E = sd[:, None] * (ell - lbar)
        EW = E @ W
        bounds.append(float(np.linalg.eigvalsh(EW @ EW.T)[-1] / tau))
        # exact chi on {w : supp S, <w, sqrt(d~)> = 0}
        Nb = np.linalg.qr(np.eye(S.size) - np.outer(sd, sd) / (sd @ sd))[0][
            :, : S.size - 1
        ]
        P = MU[S]
        lhs = Nb.T @ P @ P.T @ Nb
        rhs = Nb.T @ Mss @ Nb
        chis.append(
            float(eigh(0.5 * (lhs + lhs.T), 0.5 * (rhs + rhs.T), eigvals_only=True)[-1])
        )
    if not caps:
        return {}
    caps, chis, bounds = map(np.asarray, (caps, chis, bounds))
    return {
        "diag_cap_mean": float(caps.mean()),
        "diag_cap_min": float(caps.min()),
        "diag_chi_mean": float(chis.mean()),
        "diag_chi_max": float(chis.max()),
        "diag_chibar_mean": float(bounds.mean()),
        "diag_chibar_max": float(bounds.max()),
        "diag_bound_looseness": float(np.median(bounds / np.maximum(chis, 1e-12))),
    }


def _fit_one(args, day_specs, objective, laplacian, tau=None):
    """Fit the bank once under ``objective`` in ``laplacian``; ``(det, summary)``.

    ``coarsening_laplacian`` is the geometry switch of the whole pipeline, not a
    coarsener-only knob: it selects the screened metric ``M_tau`` the bank is
    trained in, the group indicator the capture is a fraction of, and the metric
    the RSA distortion is measured in.  So the two laplacians are two *different
    experiments end to end*, not one experiment coarsened two ways -- which is
    why the fit is redone per laplacian rather than reused.
    """

    # "level@0.3" = capture_objective="level" with level_beta=0.3, so one grid can
    # carry several betas; the label is kept verbatim as the table column
    obj_kind, _, obj_arg = str(objective).partition("@")
    over = dict(capture_objective=obj_kind, coarsening_laplacian=laplacian)
    if obj_kind == "level" and obj_arg:
        over["level_beta"] = float(obj_arg)
    elif obj_kind == "edge" and obj_arg:  # "edge@3" = edge variant, edge_gamma=3
        over["edge_gamma"] = float(obj_arg)
    if tau is not None:
        over["tau"] = float(tau)
    cfg = _make_cfg(args, **over)
    det = CollectiveBankDetector(cfg)
    t0 = time.time()
    det.fit(day_specs)
    fit = det.fit_info_
    summary = {
        "laplacian": laplacian,
        "objective": objective,
        "tau": float(cfg.tau),
        **_convergence(fit),
        "level_obj_init": float(
            (fit.get("level_objective_init") or {}).get("level_obj", float("nan"))
        ),
        "level_obj_best": float(
            (fit.get("level_objective_best") or {}).get("level_obj", float("nan"))
        ),
        "fit_seconds": time.time() - t0,
        "init_objective": float(fit.get("init_objective", float("nan"))),
        "final_objective": float(fit.get("objective", float("nan"))),
        "confusability_init": float(fit.get("confusability_init", float("nan"))),
        "confusability": float(fit.get("confusability", float("nan"))),
    }
    return det, summary


def _coarsen_and_score(det, data, basis, patterns, method, args):
    """Coarsen ``data`` with ``method`` under the already-fitted ``det``; score it.

    ``det.config`` is mutated for the duration of the call only -- the fit is
    already frozen in ``det.theta_``, so switching the coarsener costs nothing and
    guarantees every method sees the identical target subspace.

    Two cuts are reported.  The **chosen** cut is whatever ``--ward-stop`` picked.
    The **epsilon\\*** cut is the level of the sweep where the mean F1 peaks, with
    its epsilon recomputed on the common axis -- the fair anchor for comparing
    coarseners, since each method's own stop may land somewhere else.  (Under
    ``--ward-stop f1`` the two coincide by construction, which is a useful
    internal check: the ``f1`` and ``f1_star`` columns must agree.)
    """

    from src.loukas_sgc_detection import evaluate_loukas_patterns

    prev_method = det.config.coarsening_method
    prev_cuts = det.config.ward_num_cuts
    det.config.coarsening_method = method
    det.config.ward_num_cuts = args.compare_num_cuts
    try:
        t0 = time.time()
        coarsening, trajectory = det.coarsen(data, basis, patterns)
        seconds = time.time() - t0
    finally:
        det.config.coarsening_method = prev_method
        det.config.ward_num_cuts = prev_cuts
    rep = det.evaluate(data, coarsening, {"all": patterns})["all"]
    n2s = coarsening.node_to_supernode
    out = {
        "n_coarse": int(coarsening.n_coarse),
        "kept": coarsening.n_coarse / max(coarsening.n_original, 1),
        "epsilon_native": float(coarsening.epsilon),
        "epsilon_common": _common_epsilon(det, data, basis, n2s),
        "coarsen_seconds": seconds,
        "mean_recall": float(rep["mean_recall"]),
        "mean_precision": float(rep["mean_precision"]),
        "mean_f1": float(rep["mean_f1"]),
        "detection_rate": float(rep["detection_rate"]),
        "detected": int(rep["detected"]),
        "total": int(rep["total"]),
    }

    # The whole fine->coarse sweep, scalars only (the trajectory also carries a
    # full labels array per level, which must not be accumulated).  This is what
    # lets the comparison be read at a MATCHED supernode budget instead of at
    # each method's own stop.
    curve = [
        {
            "n_coarse": int(e["n_coarse"]),
            "f1": float(e["train_f1"]),
            "recall": float(e["recall"]),
            "precision": float(e["precision"]),
            "epsilon_native": float(e["epsilon"]),
        }
        for e in (trajectory or [])
    ]

    # ---- epsilon*: the level where F1 peaks ---------------------------------
    if trajectory:
        star = max(trajectory, key=lambda e: e["train_f1"])
        n2s_star = torch.from_numpy(star["labels"]).to(data.y.device)
        results, by_label = evaluate_loukas_patterns(
            patterns, n2s_star, data.y, threshold=det.config.threshold
        )
        alert = by_label.get("alert", {})
        out.update(
            {
                "epsilon_star": _common_epsilon(det, data, basis, n2s_star),
                "epsilon_star_native": float(star["epsilon"]),
                "n_coarse_star": int(star["n_coarse"]),
                "kept_star": star["n_coarse"] / max(coarsening.n_original, 1),
                "f1_star": float(np.mean([r.f1 for r in results])),
                "recall_star": float(alert.get("mean_recall", 0.0) or 0.0),
                "precision_star": float(alert.get("mean_precision", 0.0) or 0.0),
                "detection_star": float(alert.get("detection_rate", 0.0) or 0.0),
            }
        )
    else:  # the greedy Loukas methods return no sweep
        out.update(
            {
                "epsilon_star": out["epsilon_common"],
                "epsilon_star_native": out["epsilon_native"],
                "n_coarse_star": out["n_coarse"],
                "kept_star": out["kept"],
                "f1_star": out["mean_f1"],
                "recall_star": out["mean_recall"],
                "precision_star": out["mean_precision"],
                "detection_star": out["detection_rate"],
            }
        )
    return curve, out


def _fmt_table(df, index, columns, value, fmt="{:.3f}", title=""):
    """Pivot ``df`` and render it as a fixed-width LOGGER table."""

    if df.empty:
        return None
    piv = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    cols = list(piv.columns)
    w = max(12, max((len(str(c)) for c in cols), default=12) + 2)
    idx_w = max(10, max((len(str(i)) for i in piv.index), default=10) + 2)
    if title:
        LOGGER.info(f"\n  {title}")
    LOGGER.info("  " + str(index).rjust(idx_w) + "".join(str(c).rjust(w) for c in cols))
    LOGGER.info("  " + "-" * (idx_w + w * len(cols)))
    for i, row in piv.iterrows():
        cells = "".join(
            (fmt.format(row[c]) if np.isfinite(row[c]) else "n/a").rjust(w)
            for c in cols
        )
        LOGGER.info("  " + str(i).rjust(idx_w) + cells)
    return piv


def _at_matched_budget(C, knob, kept_targets):
    """Each sweep read at the level whose kept fraction is closest to each target.

    The three coarseners produce different level *schedules* (the deflated tree
    and raw Ward both stall at the component count on a disconnected graph), so
    there is no exact shared n_coarse; picking the nearest level per sweep is the
    honest matching, and ``kept_actual`` records how close it landed.
    """

    out = []
    keys = [knob, "laplacian", "objective", "coarsener", "graph"]
    for kept in kept_targets:
        for key, g in C.groupby(keys):
            j = (g.kept - kept).abs().idxmin()
            row = C.loc[j]
            out.append(
                {
                    **dict(zip(keys, key)),
                    "kept_target": kept,
                    "kept_actual": float(row.kept),
                    "n_coarse": int(row.n_coarse),
                    "f1": float(row.f1),
                    "recall": float(row.recall),
                    "precision": float(row.precision),
                }
            )
    return pd.DataFrame(out)


# Coarseners whose merge calculus is stated in M_tau = L_sym + tau I only.  The
# deflated family's harmonic projector, its coarse system L_c + tau I and its
# block-Gram closed form are all symmetric constructions (src.deflated_coarsen
# raises rather than silently coarsening in a metric it was not derived in), and
# smooth dual Ward likewise.  Raw Ward and the Ward tree run in both geometries.
_SYMMETRIC_ONLY = ("deflated-dual-ward", "deflated-minimax", "dual-ward")


def run_comparison(args) -> None:
    """The (density x laplacian x objective x coarsener) grid."""

    densities = [float(x) for x in args.density_sweep.split(",") if x.strip()] or [
        getattr(args, _DENSITY_KNOBS[args.density_knob])
    ]
    objectives = [x.strip() for x in args.compare_objectives.split(",") if x.strip()]
    coarseners = [x.strip() for x in args.compare_coarseners.split(",") if x.strip()]
    laplacians = [x.strip() for x in args.compare_laplacians.split(",") if x.strip()]
    taus = [float(x) for x in args.tau_sweep.split(",") if x.strip()] or [args.tau]
    knob = _DENSITY_KNOBS[args.density_knob]

    bad = set(laplacians) - {"symmetric", "combinatorial"}
    if bad:
        raise SystemExit(
            f"--compare-laplacians must be symmetric/combinatorial, got {sorted(bad)}"
        )
    if args.label_weight > 0.0:
        LOGGER.info(
            "  NOTE --label-weight is ignored in comparison mode: the joint "
            "supervised node head is only meaningful for a single graph, and the "
            "grid fits one filter per (density, laplacian, objective)."
        )
    skipped = [m for m in coarseners if m in _SYMMETRIC_ONLY]
    if skipped and "combinatorial" in laplacians:
        LOGGER.info(
            f"  NOTE {skipped} are symmetric-only by construction and are SKIPPED "
            "under the combinatorial laplacian (their merge calculus is derived in "
            "M_tau = L_sym + tau I).  The combinatorial columns therefore compare "
            f"{[m for m in coarseners if m not in _SYMMETRIC_ONLY]} only."
        )

    LOGGER.info("=" * 96)
    LOGGER.info("COMPARISON GRID")
    LOGGER.info(
        f"  density knob   : --{args.density_knob} in {densities}\n"
        f"  laplacians     : {laplacians}  (the FULL geometry switch: metric, "
        f"indicator, propagation and RSA axis)\n"
        f"  taus           : {taus}\n"
        f"  objectives     : {objectives}\n"
        f"  coarseners     : {coarseners}  (stop={args.ward_stop}, "
        f"cuts={args.compare_num_cuts}, epsilon={args.epsilon})\n"
        f"  graphs         : {args.num_graphs} train + {args.transfer_graphs} transfer, "
        f"N={args.num_nodes:,}, {args.num_motifs} motifs "
        f"sizes {args.motif_size_min}-{args.motif_size_max} {args.motif_types}"
    )
    LOGGER.info("=" * 96)

    rows: list[dict] = []
    fit_rows: list[dict] = []
    curve_rows: list[dict] = []
    for density in densities:
        d_args = copy.copy(args)
        setattr(d_args, knob, density)

        # ---- the graphs: built ONCE per density, shared by every cell --------
        day_specs = []
        for g in range(d_args.num_graphs):
            data_g, tr_g, te_g, _all_g = _load_graph(d_args.seed + 100 * g, d_args)
            day_specs.append((f"G{g}", data_g, tr_g, te_g))
        transfer = []
        for k in range(d_args.transfer_graphs):
            t_data, _t_tr, _t_te, t_all = _load_graph(
                d_args.seed + 9000 + 100 * k, d_args
            )
            transfer.append((f"T{k}", t_data, t_all))
        last = day_specs[-1]
        # held-out motifs on the reported training graph, then the frozen filter
        # on every fresh unseen graph
        eval_specs = [("train:test", last[1], last[3])] + transfer
        eval_specs = [(lbl, d, p) for lbl, d, p in eval_specs if p]

        deg = float(
            np.mean(
                [
                    d.adjacency.coalesce().indices().shape[1] / d.num_nodes
                    for _l, d, _t, _e in day_specs
                ]
            )
        )
        LOGGER.info(
            f"\n{'=' * 96}\n[{args.density_knob}={density:g}]  realized mean degree "
            f"{deg:.2f}  ({len(eval_specs)} evaluation graphs)\n{'=' * 96}"
        )

        for laplacian in laplacians:
            methods = [
                m
                for m in coarseners
                if laplacian == "symmetric" or m not in _SYMMETRIC_ONLY
            ]
            if not methods:
                continue
            LOGGER.info(f"  --- laplacian = {laplacian} ---")
            for tau_v, objective in [(t, o) for t in taus for o in objectives]:
                det, fs = _fit_one(d_args, day_specs, objective, laplacian, tau_v)
                fs[knob] = density
                fs["mean_degree"] = deg
                cap = det.capture(last[1], last[2] + last[3])
                fs.update(
                    {
                        "mean_capture": cap["mean_capture"],
                        "min_capture": cap["min_capture"],
                        "lambda_min_gamma": cap["lambda_min_gamma"],
                    }
                )
                fs.update(_level_diagnostics(det, last[1], last[2]))
                fit_rows.append(fs)
                LOGGER.info(
                    f"  fit[tau={tau_v:<5g} {laplacian[:4]}/{objective:<10}] "
                    f"{fs['init_objective']:.4g} -> {fs['final_objective']:.4g}   "
                    f"chi {fs['confusability_init']:.4g} -> {fs['confusability']:.4g}"
                    f"   mean_C={cap['mean_capture']:.4f} "
                    f"min_C={cap['min_capture']:.4f} "
                    f"lambda_min={cap['lambda_min_gamma']:.4f}  "
                    f"[{'converged' if fs['converged'] else 'NOT CONVERGED'}: "
                    f"tail drift {fs['tail_drift']:.1%}, best at epoch "
                    f"{fs['best_epoch_frac']:.0%}]  ({fs['fit_seconds']:.1f}s)"
                )
                if "diag_cap_mean" in fs:
                    LOGGER.info(
                        f"      level view of the training groups: C mean "
                        f"{fs['diag_cap_mean']:.4f} min {fs['diag_cap_min']:.4f} | "
                        f"exact chi mean {fs['diag_chi_mean']:.4f} max "
                        f"{fs['diag_chi_max']:.4f} | chibar max "
                        f"{fs['diag_chibar_max']:.4f} (bound {fs['diag_bound_looseness']:.1f}x)"
                        + (
                            f" | J {fs['level_obj_init']:.4f} -> {fs['level_obj_best']:.4f}"
                            if np.isfinite(fs.get("level_obj_best", float("nan")))
                            else ""
                        )
                    )

                # ---- one target subspace per graph, reused by every coarsener -
                bases = [
                    (lbl, d, p, det.target_subspace(d, p)) for lbl, d, p in eval_specs
                ]
                for method in methods:
                    for lbl, d, p, b in bases:
                        curve, r = _coarsen_and_score(det, d, b, p, method, d_args)
                        tag = {
                            knob: density,
                            "tau": tau_v,
                            "laplacian": laplacian,
                            "objective": objective,
                            "coarsener": method,
                            "graph": lbl,
                        }
                        for pt in curve:
                            curve_rows.append(
                                {
                                    **tag,
                                    "n_original": int(d.num_nodes),
                                    "kept": pt["n_coarse"] / max(int(d.num_nodes), 1),
                                    **pt,
                                }
                            )
                        r.update({**tag, "mean_degree": deg})
                        rows.append(r)
                    sub = [
                        r
                        for r in rows
                        if r["coarsener"] == method
                        and r["objective"] == objective
                        and r["laplacian"] == laplacian
                        and r["tau"] == tau_v
                        and r[knob] == density
                    ]
                    LOGGER.info(
                        f"    {method:<20} "
                        f"F1*={np.mean([r['f1_star'] for r in sub]):.3f}  "
                        f"R*={np.mean([r['recall_star'] for r in sub]):.3f}  "
                        f"P*={np.mean([r['precision_star'] for r in sub]):.3f}  "
                        f"det*={np.mean([r['detection_star'] for r in sub]):.1%}  "
                        f"eps*={np.mean([r['epsilon_star'] for r in sub]):.3f}  "
                        f"n*={np.mean([r['n_coarse_star'] for r in sub]):.0f}  "
                        f"({np.mean([r['coarsen_seconds'] for r in sub]):.1f}s/graph)"
                    )

    # ---- persist ------------------------------------------------------------
    D = pd.DataFrame(rows)
    F = pd.DataFrame(fit_rows)
    C = pd.DataFrame(curve_rows)
    D.to_csv(args.out / "comparison_rows.csv", index=False)
    F.to_csv(args.out / "comparison_fit.csv", index=False)
    if not C.empty:
        C.to_csv(args.out / "comparison_curves.csv", index=False)

    laps = list(dict.fromkeys(D.laplacian))
    LOGGER.info("\n" + "=" * 96)
    LOGGER.info(
        "ALL PERFORMANCE NUMBERS BELOW ARE AT epsilon* -- the sweep level where "
        "mean F1 peaks,\n  with epsilon recomputed on ONE common axis (Loukas "
        "Def. 2, uniform block average in M_tau)\n  so the three coarseners' "
        "distortions are directly comparable."
    )
    LOGGER.info("=" * 96)

    # ---- [0] tau ------------------------------------------------------------
    if len(taus) > 1:
        LOGGER.info("\n" + "=" * 96)
        LOGGER.info("[0] SCREENING LEVEL tau, at eps*  (mean over graphs)")
        LOGGER.info(
            "  tau moves the WHOLE geometry: the metric M_tau = L + tau I the bank "
            "trains in, the\n  indicator capture is a fraction of, the RSA axis, and "
            "the block-distortion bound\n  mu_P^tau <= sqrt((lambda_max+tau)/tau).  So "
            "capture and eps* are in tau-dependent UNITS\n  and must not be compared "
            "across rows; the detection columns are unitless and can be."
        )
        LOGGER.info("=" * 96)
        bad = F[~F.converged.astype(bool)]
        if len(bad):
            LOGGER.info(
                f"\n  WARNING {len(bad)} of {len(F)} fits did NOT converge -- their "
                "rows are not comparable:\n    "
                + ", ".join(
                    f"tau={r.tau:g} (drift {r.tail_drift:.1%}, best at "
                    f"{r.best_epoch_frac:.0%})"
                    for _i, r in bad.iterrows()
                )
            )
        else:
            LOGGER.info(
                f"\n  all {len(F)} fits converged "
                f"(max tail drift {F.tail_drift.max():.2%}, "
                f"latest best iterate at epoch {F.best_epoch_frac.max():.0%})"
            )
        _fmt_table(D, "tau", "coarsener", "f1_star", title="mean F1* (comparable)")
        _fmt_table(
            D,
            "tau",
            "coarsener",
            "detection_star",
            "{:.1%}",
            "detection at eps* (comparable)",
        )
        _fmt_table(D, "tau", "coarsener", "recall_star", title="recall at eps*")
        _fmt_table(D, "tau", "coarsener", "precision_star", title="precision at eps*")
        _fmt_table(
            D, "tau", "coarsener", "n_coarse_star", "{:.0f}", "supernodes at eps*"
        )
        _fmt_table(
            D,
            "tau",
            "coarsener",
            "epsilon_star",
            title="eps* -- tau-dependent units, do NOT compare down this column",
        )
        _fmt_table(
            F,
            "tau",
            "objective",
            "mean_capture",
            title="mean capture C -- also tau-dependent units",
        )
        LOGGER.info(
            "\n  mu bound sqrt((lambda_max+tau)/tau) (lambda_max <= 2), and the "
            "fit's convergence:"
        )
        LOGGER.info(
            f"  {'tau':>8}{'mu_bound':>12}{'lambda_min':>12}{'chi':>10}"
            f"{'tail drift':>12}{'best@':>8}{'fit s':>8}"
        )
        for t in sorted(set(F.tau)):
            r = F[F.tau == t].mean(numeric_only=True)
            LOGGER.info(
                f"  {t:>8g}{math.sqrt((2.0 + t) / t):>12.2f}"
                f"{r['final_objective']:>12.4f}{r['confusability']:>10.4f}"
                f"{r['tail_drift']:>12.2%}{r['best_epoch_frac']:>8.0%}"
                f"{r['fit_seconds']:>8.0f}"
            )
        best = D.groupby("tau").f1_star.mean()
        LOGGER.info(
            f"\n  => best tau overall: {best.idxmax():g} (mean F1* {best.max():.4f}); "
            f"worst {best.idxmin():g} ({best.min():.4f})  spread "
            f"{best.max() - best.min():.4f}"
        )
        for m in sorted(set(D.coarsener)):
            b = D[D.coarsener == m].groupby("tau").f1_star.mean()
            LOGGER.info(
                f"       {m:<22}best tau {b.idxmax():g} (F1* {b.max():.4f}), "
                f"spread {b.max() - b.min():.4f}"
            )

    # ---- [1] objective ------------------------------------------------------
    LOGGER.info("\n" + "=" * 96)
    LOGGER.info("[1] OBJECTIVE COMPARISON  (mean over coarseners and graphs, at eps*)")
    LOGGER.info("=" * 96)
    for lap in laps:
        LOGGER.info(f"\n  ---------- laplacian = {lap} ----------")
        _fmt_table(
            D[D.laplacian == lap], knob, "objective", "f1_star", title="mean F1*"
        )
        _fmt_table(
            D[D.laplacian == lap],
            knob,
            "objective",
            "detection_star",
            "{:.1%}",
            "detection rate at eps*",
        )
        _fmt_table(
            F[F.laplacian == lap],
            knob,
            "objective",
            "mean_capture",
            title="mean capture C (fit)",
        )
        for col, ttl in (
            ("diag_cap_min", "exact capture, worst training group"),
            ("diag_chi_max", "exact confusability chi, worst training group"),
            ("diag_chibar_max", "level bound chibar, worst training group"),
        ):
            if col in F.columns:
                _fmt_table(F[F.laplacian == lap], knob, "objective", col, title=ttl)
    m = D.groupby(["laplacian", "objective"]).f1_star.mean()
    LOGGER.info("\n  => pooled mean F1* per (laplacian, objective):")
    for (lap, obj), v in m.items():
        LOGGER.info(f"       {lap:<15}{obj:<15}{v:.4f}")

    # ---- [2-3] coarsener ----------------------------------------------------
    LOGGER.info("\n" + "=" * 96)
    LOGGER.info("[2-3] COARSENER vs DENSITY, at eps*  (mean over objectives, graphs)")
    LOGGER.info("=" * 96)
    for lap in laps:
        sub = D[D.laplacian == lap]
        LOGGER.info(f"\n  ---------- laplacian = {lap} ----------")
        _fmt_table(sub, knob, "coarsener", "f1_star", title="mean F1*")
        _fmt_table(
            sub, knob, "coarsener", "detection_star", "{:.1%}", "detection at eps*"
        )
        _fmt_table(sub, knob, "coarsener", "recall_star", title="recall at eps*")
        _fmt_table(sub, knob, "coarsener", "precision_star", title="precision at eps*")
        _fmt_table(sub, knob, "coarsener", "epsilon_star", title="eps* (common axis)")
        _fmt_table(
            sub, knob, "coarsener", "n_coarse_star", "{:.0f}", "supernodes at eps*"
        )
        _fmt_table(sub, knob, "coarsener", "coarsen_seconds", "{:.1f}", "seconds/graph")
        rank = sub.groupby("coarsener").f1_star.mean().sort_values(ascending=False)
        LOGGER.info(f"\n  => [{lap}] coarsener ranking by pooled mean F1*:")
        for name, v in rank.items():
            LOGGER.info(f"       {name:<22}{v:.4f}")
        piv = sub.pivot_table(index=knob, columns="coarsener", values="f1_star")
        LOGGER.info(f"  => [{lap}] winner per density:")
        for d_, row in piv.iterrows():
            LOGGER.info(
                f"       {args.density_knob}={d_:<8g} {row.idxmax():<22}"
                f"F1*={row.max():.4f}  (spread {row.max() - row.min():.4f})"
            )

    # ---- [4] laplacian ------------------------------------------------------
    if len(laps) > 1:
        shared = sorted(
            set(D[D.laplacian == laps[0]].coarsener).intersection(
                *[set(D[D.laplacian == l].coarsener) for l in laps[1:]]
            )
        )
        LOGGER.info("\n" + "=" * 96)
        LOGGER.info(
            f"[4] LAPLACIAN COMPARISON at eps*  (coarseners available in both: {shared})"
        )
        LOGGER.info(
            "  NB this is a whole-pipeline switch, not a coarsener switch: the bank "
            "is refit\n  in each metric, and capture / RSA are fractions of a "
            "different indicator on each side."
        )
        LOGGER.info("=" * 96)
        S = D[D.coarsener.isin(shared)]
        _fmt_table(S, knob, "laplacian", "f1_star", title="mean F1*")
        _fmt_table(
            S, knob, "laplacian", "detection_star", "{:.1%}", "detection at eps*"
        )
        _fmt_table(S, knob, "laplacian", "epsilon_star", title="eps* (common axis)")
        _fmt_table(
            S, "coarsener", "laplacian", "f1_star", title="mean F1* per coarsener"
        )
        _fmt_table(
            S, "coarsener", "laplacian", "n_coarse_star", "{:.0f}", "supernodes at eps*"
        )
        pooled = S.groupby("laplacian").f1_star.mean()
        LOGGER.info("\n  => pooled mean F1* per laplacian:")
        for name, v in pooled.items():
            LOGGER.info(f"       {name:<22}{v:.4f}")

    # ---- [5] matched budget -------------------------------------------------
    if not C.empty:
        LOGGER.info("\n" + "=" * 96)
        LOGGER.info(
            "[5] COARSENER AT A MATCHED SUPERNODE BUDGET  (mean F1 at the level "
            "closest to each kept-fraction)"
        )
        LOGGER.info(
            "  A second, stop-free reading: eps* is each method's own best level, so "
            "this checks\n  the merge ORDER at a budget every method is held to "
            "equally."
        )
        LOGGER.info("=" * 96)
        M = _at_matched_budget(C, knob, args.compare_kept)
        M.to_csv(args.out / "comparison_matched_budget.csv", index=False)
        for lap in laps:
            LOGGER.info(f"\n  ---------- laplacian = {lap} ----------")
            for kept in args.compare_kept:
                sub = M[(M.kept_target == kept) & (M.laplacian == lap)]
                _fmt_table(
                    sub,
                    knob,
                    "coarsener",
                    "f1",
                    title=f"mean F1 at n_coarse/N = {kept:g}",
                )
            pooled = (
                M[M.laplacian == lap]
                .groupby("coarsener")
                .f1.mean()
                .sort_values(ascending=False)
            )
            LOGGER.info(f"\n  => [{lap}] ranking at matched budget (pooled):")
            for name, v in pooled.items():
                LOGGER.info(f"       {name:<22}{v:.4f}")

    _plot_comparison(D, F, knob, args)
    LOGGER.info(f"\n  rows -> {args.out / 'comparison_rows.csv'}")
    LOGGER.info(f"  fits -> {args.out / 'comparison_fit.csv'}")


def _plot_comparison(D, F, knob, args) -> None:
    """Two figures: objective effect and coarsener-vs-density effect."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        LOGGER.info("  (matplotlib unavailable -- skipping comparison figures)")
        return

    laps = list(dict.fromkeys(D.laplacian))
    style = {"symmetric": "-", "combinatorial": "--"}

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for name, panel, ylab in (
        ("f1_star", axes[0], "mean F1 at eps*"),
        ("detection_star", axes[1], "detection at eps*"),
        ("mean_capture", axes[2], "mean capture C"),
        ("lambda_min_gamma", axes[3], "lambda_min(Gamma)"),
    ):
        src = D if name in D.columns else F
        for (lap, obj), g in src.groupby(["laplacian", "objective"]):
            agg = g.groupby(knob)[name].mean().sort_index()
            panel.plot(
                agg.index,
                agg.values,
                marker="o",
                linestyle=style.get(lap, "-"),
                label=f"{obj} ({lap})",
            )
        panel.set_xlabel(args.density_knob)
        panel.set_ylabel(ylab)
        panel.grid(alpha=0.3)
        panel.legend(fontsize=7)
    fig.suptitle(
        "[1] learning objective: lambda_min vs trace (solid=symmetric, dashed=combinatorial)"
    )
    fig.tight_layout()
    f1 = args.out / "compare_objectives.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(laps), 4, figsize=(20, 4.2 * len(laps)), squeeze=False)
    for row, lap in enumerate(laps):
        sub = D[D.laplacian == lap]
        for name, panel, ylab in (
            ("f1_star", axes[row][0], "mean F1 at eps*"),
            ("detection_star", axes[row][1], "detection at eps*"),
            ("epsilon_star", axes[row][2], "eps* (common axis)"),
            ("n_coarse_star", axes[row][3], "supernodes at eps*"),
        ):
            for method, g in sub.groupby("coarsener"):
                agg = g.groupby(knob)[name].mean().sort_index()
                panel.plot(agg.index, agg.values, marker="o", label=method)
            panel.set_xlabel(args.density_knob)
            panel.set_ylabel(f"{ylab}\n[{lap}]")
            panel.grid(alpha=0.3)
            panel.legend(title="coarsener", fontsize=8)
    fig.suptitle("[2-4] coarsener vs density at eps*, per laplacian")
    fig.tight_layout()
    f2 = args.out / "compare_coarseners.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    LOGGER.info(f"  figures -> {f1}\n             {f2}")


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
    ap.add_argument("--num-nodes", type=int, default=3000)
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
    ap.add_argument("--avg-degree", type=float, default=4.0)
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
        choices=["gradient", "closed-form", "trace-ratio", "channel-closed-form"],
        default="channel-closed-form",
        help="channel-closed-form = per-channel generalized eigenproblems of the "
        "capture_objective's pencil (level -> trace surrogate, edge -> exact), "
        "pooled over the training graphs; no epochs (src.closed_form_level)",
    )
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument(
        "--basis", choices=["chebyshev", "monomial", "lanczos"], default="chebyshev"
    )
    ap.add_argument("--tau", type=float, default=1)
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
        choices=[
            "lambda_min",
            "trace",
            "softmin_diag",
            "certified_margin",
            "level",
            "edge",
        ],
        default="edge",
    )
    ap.add_argument(
        "--level-beta",
        type=float,
        default=5,
        help="capture_objective=level: weight of the smooth worst level-covariance "
        "bound chibar against the smooth worst level capture",
    )
    ap.add_argument(
        "--level-chi-temperature",
        type=float,
        default=0.05,
        help="log-sum-exp temperature of the chibar soft-max (chibar units)",
    )
    ap.add_argument(
        "--edge-gamma",
        type=float,
        default=2.0,
        help="capture_objective=edge: weight of the internal-edge level contrast "
        "against the boundary-edge one",
    )
    ap.add_argument(
        "--ridge-relative",
        type=float,
        default=0.0,
        help="scale-free ridge eps*mean(diag G) on the channel Gram (both objectives); "
        "0 = the historical absolute --ridge only",
    )
    ap.add_argument(
        "--cf-share-filters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="channel-closed-form: one set of H filters shared by all input channels "
        "(one pooled (K+1)x(K+1) pencil) instead of one filter set per channel",
    )
    ap.add_argument(
        "--cf-feature-draws",
        type=int,
        default=0,
        help="channel-closed-form: extra fresh random-feature realizations per "
        "training graph pooled into the pencil (0 = each graph's own X only)",
    )
    ap.add_argument(
        "--cf-host-mode",
        choices=["none", "neighbours", "random"],
        default="none",
        help="channel-closed-form: which nodes the squared screened-response "
        "penalty -lambda_H C_H acts on.  'neighbours' = the non-group endpoints of "
        "edges leaving a training group; 'random' = a uniform sample of non-group "
        "nodes from the whole graph.  Both exclude the TRAINING groups only",
    )
    ap.add_argument(
        "--cf-host-weight",
        type=float,
        default=0.25,
        help="lambda_H >= 0, the weight of the host term (0 = off, the exact "
        "no-host baseline)",
    )
    ap.add_argument(
        "--cf-host-count",
        type=int,
        default=100,
        help="--cf-host-mode random: how many host nodes to draw per graph "
        "(0 = as many as the 'neighbours' mask would give)",
    )
    ap.add_argument(
        "--cf-param-scale",
        choices=["absolute", "relative"],
        default="absolute",
        help="channel-closed-form: 'relative' reads --edge-gamma / --level-beta and "
        "--cf-host-weight as fractions of each block's own scale -- the penalty of "
        "lambda_max(N, P), the cliff above which no direction has a positive objective "
        "(keep < 1), the host weight of lambda_max(A, S) / lambda_max(C_H, S)",
    )
    ap.add_argument(
        "--cf-solver-form",
        choices=["difference", "ratio"],
        default="ratio",
        help="channel-closed-form: 'difference' = top eigenvectors of N - penalty P - "
        "host C_H; 'ratio' = maximize tr N / tr(P + host C_H) by Dinkelbach iteration, "
        "so the penalty is solved for (--edge-gamma / --level-beta are ignored)",
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
            "raw-ward",
            "dual-ward",
            "deflated-dual-ward",
            "deflated-minimax",
        ],
        default="deflated-dual-ward",
        help="ward-tree = contiguity-constrained Ward tree; raw-ward = the rank-q "
        "raw-Ward score s^0_R on the screened level (src.raw_ward); "
        "deflated-* = the M_tau-orthogonal (harmonic) agglomeration",
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
    # --- comparison grid: objective x coarsener x density --------------------
    ap.add_argument(
        "--compare",
        action="store_true",
        help="run the (density x capture-objective x coarsener) comparison grid "
        "instead of the single-configuration report.  Implied by --density-sweep.",
    )
    ap.add_argument(
        "--compare-objectives",
        type=str,
        default="lambda_min,trace",
        help="comma list of --capture-objective values to fit and compare",
    )
    ap.add_argument(
        "--compare-coarseners",
        type=str,
        default="ward-tree,raw-ward,deflated-dual-ward",
        help="comma list of --coarsening-method values applied to the SAME fitted "
        "filter and the SAME target subspace",
    )
    ap.add_argument(
        "--density-sweep",
        type=str,
        default="",
        help="comma list of density values to sweep (empty = the single value "
        "already set by the corresponding flag)",
    )
    ap.add_argument(
        "--density-knob",
        choices=["avg-degree", "motif-density", "motif-conductance"],
        default="avg-degree",
        help="which density --density-sweep varies",
    )
    ap.add_argument(
        "--tau-sweep",
        type=str,
        default="",
        help="comma list of screening levels tau to compare (empty = just --tau).  "
        "tau is the whole geometry's scale: M_tau = L + tau I, so it moves the "
        "metric the bank trains in, the indicator capture is a fraction of, the "
        "block-distortion bound mu <= sqrt((lambda_max+tau)/tau), and the RSA "
        "axis.  Capture / epsilon are therefore NOT comparable across tau; the "
        "detection numbers at eps* are.",
    )
    ap.add_argument(
        "--compare-laplacians",
        type=str,
        default="symmetric",
        help="comma list from {symmetric, combinatorial}.  This is the FULL "
        "geometry switch (metric, indicator, propagation, RSA axis), so the bank "
        "is refit per laplacian; the deflated family and dual-ward are "
        "symmetric-only and are skipped on the combinatorial side.",
    )
    ap.add_argument(
        "--compare-kept",
        type=str,
        default="0.5,0.3,0.2,0.1",
        help="comma list of kept fractions n_coarse/N at which the coarseners are "
        "additionally compared head-to-head, independent of their own stop rule",
    )
    ap.add_argument(
        "--compare-num-cuts",
        type=int,
        default=120,
        help="tree-cut levels evaluated per coarsening inside the grid (the "
        "single-run default --ward-num-cuts is usually far larger)",
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=f"results/synthetic_modular/{now}/", type=Path)
    args = ap.parse_args()

    args.compare_kept = [
        float(x) for x in str(args.compare_kept).split(",") if x.strip()
    ]
    args.motif_types = [t.strip() for t in args.motif_types.split(",") if t.strip()]
    valid = {"clique", "cycle", "star", "random"}
    if not set(args.motif_types) <= valid:
        ap.error(f"--motif-types must be a subset of {sorted(valid)}")
    if args.cf_feature_draws > 0 and (
        args.feat_shared > 0.0 or args.feat_signature > 0.0
    ):
        ap.error(
            "--cf-feature-draws redraws pure-noise features; it would drop the "
            "injected --feat-shared/--feat-signature gang signal"
        )
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.compare or args.density_sweep.strip():
        run_comparison(args)
        return

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
    cfg = _make_cfg(args)
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
    elif cfg.collective_solver == "channel-closed-form":
        npos = fit["n_positive_per_channel"]
        LOGGER.info(
            f"    closed form ({'SHARED' if fit['share_filters'] else 'per-channel'} filters, "
            f"{fit['pencil']} pencil, pooled over {fit['n_instances']} instances = "
            f"{len(fit['train_days'])} graphs x {1 + fit['feature_draws']} feature draws): "
            f"pencils {fit['pencil_seconds']:.2f}s + "
            f"solve {fit['solve_seconds']:.3f}s; positive directions per channel "
            f"min {min(npos)} / mean {np.mean(npos):.1f} / max {max(npos)} (H={cfg.heads})"
        )
        if fit["param_scale"] != "absolute" or fit["solver_form"] != "difference":
            LOGGER.info(
                f"    {fit['solver_form']} form, {fit['param_scale']} parameters: "
                f"{'rho* (solved penalty)' if fit['solver_form'] == 'ratio' else 'penalty'} "
                f"median {fit['penalty_abs_median']:.4g} (range {fit['penalty_abs_range'][0]:.4g}"
                f"-{fit['penalty_abs_range'][1]:.4g}); cliff lambda_max(N,P) median "
                f"{fit['penalty_star_median']:.4g}; host weight median {fit['host_abs_median']:.4g}"
                + (
                    f"; Dinkelbach <= {fit['ratio_iters_max']} iters, converged "
                    f"{fit['ratio_converged']}"
                    if fit["solver_form"] == "ratio"
                    else ""
                )
            )
        for lbl, v in fit["per_day"].items():
            LOGGER.info(
                f"      {lbl}: J_level={v['J_level']:.4g}  J_edge={v['J_edge']:.4g}  "
                f"cap_min={v['cap_min']:.4g}  chibar_max={v['chibar_max']:.4g}"
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
                        # channel-closed-form: what the solver resolved
                        "n_positive_per_channel",
                        "param_scale",
                        "solver_form",
                        "penalty_star_median",
                        "penalty_abs_median",
                        "penalty_abs_range",
                        "host_abs_median",
                        "ratio_iters_max",
                        "ratio_converged",
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

    # move the LOGGER's log file into out_dir and drop the scratch results/
    # folder created for it at import time (src/utils/utils.py).
    _relocate_logger_file(args.out)


if __name__ == "__main__":
    main()
