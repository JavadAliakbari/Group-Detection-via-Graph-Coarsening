"""Why are some gangs missed?  Pooled per-gang diagnostics across transfer days.

Trains the collective bank on ``--day-start`` (exactly as ``run_elliptic_modular``
does), freezes it, applies it to each of the next ``--transfer-days`` days, and for
**every gang on every day** records the theory's structural statistics alongside the
detection outcome:

* ``size``      -- |S|
* ``Phi``       -- conductance cut(S)/vol(S)  (Definition 3.1; ``= m_1``)
* ``mbar1``     -- boundary-edge mean ``m_2/m_1`` (Definition 3.3).  Theorem 9.2 says
                   capture depends on S *only* through this scalar, and the paper's
                   structural failure mode is exactly ``mbar1`` bounded away from 0
                   for sparse fan/star motifs -- regardless of how low Phi is.
* ``capture``   -- retained ``M_tau``-energy ``C_S`` of the learned target (the
                   sufficiency scalar that upper-bounds detectability)
* ``density``   -- internal edge density 2E/(s(s-1))
* ``starness``  -- max/mean internal degree (fan-in/fan-out hub structure)
* ``deg_ratio`` -- mean external degree / mean internal degree

Then it reports, for detected vs missed gangs, the medians and each statistic's
single-feature AUC for predicting detection -- i.e. *which* property explains the
misses -- and writes diagnostic plots.

Run::

    python -m src.analyze_missed_gangs --day-start 24 --transfer-days 10 \
        --feature-mode wallet --out results/missed_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.run_collective_bank_detection import _l_apply, degree_weighted_indicators
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    _NON_FEATURE_COLS,
    build_torch_graph,
    make_patterns,
    split_train_test,
    random_structural_features,
)

_FEAT_DF = None  # the 600MB wallet-feature CSV, read once and reused per day


def cached_wallet_features(
    data_dir, nodes_df, day, keep_columns=None, return_columns=False
):
    """``load_node_features`` for one day, but reading the CSV only once per process."""

    global _FEAT_DF
    if _FEAT_DF is None:
        _FEAT_DF = pd.read_csv(
            data_dir / "wallets_features.csv", dtype={"address": str}
        )
    feat = _FEAT_DF[_FEAT_DF["Time step"].between(day, day)]
    feat = feat.drop_duplicates(subset=["address"], keep="last").set_index("address")
    feat = feat.reindex(nodes_df["address"].values)
    num = feat.drop(columns=[c for c in _NON_FEATURE_COLS if c in feat.columns])
    num = num.select_dtypes(include=[np.number])
    if keep_columns is not None:
        num = num.reindex(columns=list(keep_columns))
    X = np.nan_to_num(num.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    if keep_columns is None:
        keep = sd > 1e-12
        cols = list(num.columns[keep])
        X = (X[:, keep] - mu[keep]) / sd[keep]
    else:
        cols = list(keep_columns)
        X = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    T = torch.from_numpy(X).to(torch.float64)
    return (T, cols) if return_columns else T


# --------------------------------------------------------------------------- #
def gang_moments(a_hat, adjacency, patterns, geometry=None):
    """``(Phi, mbar1)`` per gang: the boundary ratio and the boundary-edge mean m_2/m_1.

    ``geometry`` fixes both the indicator and the Laplacian, so the reported
    ``Phi`` is the same quantity the detector trained on: the conductance
    ``cut(S)/vol(S)`` under the symmetric convention, and the cardinality-
    normalized ``cut(S)/|S|`` under the combinatorial one.  ``None`` keeps the
    historical symmetric reading.
    """

    from src.run_collective_bank_detection import _geometry_or_symmetric

    geo = _geometry_or_symmetric(geometry, a_hat, adjacency)
    V = geo.indicators(patterns)  # (N, m), ||v||_2 = 1
    LV = geo.l_apply(V)
    m1 = (V * LV).sum(0)  # v^T L v = Phi
    L2V = geo.l_apply(LV)
    m2 = (V * L2V).sum(0)  # v^T L^2 v
    return m1, m2 / m1.clamp_min(1e-12)


def gang_structure(edge_index, gang_of, n_gangs, num_nodes):
    """Per-gang internal density, star-ness (max/mean internal degree), degree ratio."""

    u, v = edge_index[0], edge_index[1]
    gu, gv = gang_of[u], gang_of[v]
    internal = (gu >= 0) & (gu == gv)
    deg_int = torch.zeros(num_nodes)
    deg_int.index_add_(0, u[internal], torch.ones(int(internal.sum())))
    deg_int.index_add_(0, v[internal], torch.ones(int(internal.sum())))
    deg_all = torch.zeros(num_nodes)
    deg_all.index_add_(0, u, torch.ones(u.numel()))
    deg_all.index_add_(0, v, torch.ones(v.numel()))
    out = []
    for gi in range(n_gangs):
        nodes = torch.nonzero(gang_of == gi, as_tuple=False).flatten()
        s = int(nodes.numel())
        di = deg_int[nodes]
        e_int = float(di.sum()) / 2.0
        density = 2 * e_int / max(s * (s - 1), 1)
        starness = float(di.max() / di.mean().clamp_min(1e-9)) if s else float("nan")
        ext = deg_all[nodes] - di
        deg_ratio = float(ext.mean() / di.mean().clamp_min(1e-9))
        out.append(
            {
                "size": s,
                "density": density,
                "starness": starness,
                "deg_ratio": deg_ratio,
                "n_internal_edges": e_int,
            }
        )
    return out


def day_records(det, data, gangs, gang_sets, edge_index, day):
    """Per-gang diagnostics + detection outcome for one day (frozen filter)."""

    cfg = det.config
    basis = det.target_subspace(data, gangs)  # labels only pick the span, not the cut
    coarsening, _ = det.coarsen(data, basis, gangs)
    res, _ = evaluate_loukas_patterns(
        gangs, coarsening.node_to_supernode, data.y, threshold=cfg.threshold
    )
    phi, mbar1 = gang_moments(data.a_hat, data.adjacency, gangs)
    cap = det.capture(data, gangs)["per_gang_capture"]
    gang_of = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for gi, S in enumerate(gang_sets):
        gang_of[torch.as_tensor(list(S), dtype=torch.long)] = gi
    struct = gang_structure(edge_index, gang_of, len(gangs), data.num_nodes)
    rows = []
    for gi in range(len(gangs)):
        rows.append(
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
        )
    return rows


# --------------------------------------------------------------------------- #
def plot_diagnostics(df: pd.DataFrame, out: Path):
    feats = ["size", "Phi", "mbar1", "capture", "density", "starness", "deg_ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    for ax, f in zip(axes.ravel(), feats):
        d = df[df.detected == 1][f].dropna()
        m = df[df.detected == 0][f].dropna()
        ax.boxplot([d, m], tick_labels=["detected", "missed"], showfliers=False)
        ax.scatter(
            np.random.normal(1, 0.05, d.size), d, s=7, alpha=0.35, color="tab:green"
        )
        ax.scatter(
            np.random.normal(2, 0.05, m.size), m, s=7, alpha=0.35, color="tab:red"
        )
        if f in ("size", "mbar1", "starness", "deg_ratio"):
            ax.set_yscale("log")
        try:
            auc = roc_auc_score(df.detected, df[f])
        except Exception:
            auc = float("nan")
        ax.set_title(f"{f}   AUC={auc:.2f}")
        ax.grid(axis="y", alpha=0.3)
    # the theory's two axes: Phi (external quietness) vs mbar1 (internal wiring)
    ax = axes.ravel()[7]
    for lab, sub, c in [
        ("detected", df[df.detected == 1], "tab:green"),
        ("missed", df[df.detected == 0], "tab:red"),
    ]:
        ax.scatter(
            sub.Phi,
            sub.mbar1,
            s=8 + 2 * np.sqrt(sub["size"]),
            alpha=0.55,
            color=c,
            label=lab,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.set_xlabel(r"conductance $\Phi$")
    ax.set_ylabel(r"boundary-edge mean $\bar m_1$")
    ax.set_yscale("log")
    ax.set_title("Theory axes: external quietness vs internal wiring")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("Detected vs missed gangs, pooled over transfer days", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def analyze_missed_gangs(det, cols, args):
    # --- apply frozen filter to each transfer day ---------------------------
    rows = []
    for k in range(args.transfer_days + 1):
        day = args.day_start + k
        A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, day, day)
        if args.feature_mode == "wallet":
            Xd = cached_wallet_features(args.data_dir, nodes_df, day, keep_columns=cols)
        else:
            Xd = random_structural_features(
                int(A_unw.shape[0]), args.random_width, args.seed + day
            )
        g = build_torch_graph(A_w, A_unw, cls, Xd, weighted=False)
        d_sets = connected_components_sets(
            A_unw, np.where(cls == 1)[0], args.min_gang_size
        )
        d_gangs = make_patterns(d_sets, "alert", "gang", "g")
        if not d_gangs:
            continue
        dd = GraphData.from_graph(g)
        r = day_records(det, dd, d_gangs, d_sets, g.edge_index, day)
        rows.extend(r)
        det_n = sum(x["detected"] for x in r)
        print(
            f"  day {day}: N={dd.num_nodes:,} gangs={len(d_gangs):3d} detected={det_n:3d} "
            f"({det_n/len(d_gangs):.0%})"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_gang_diagnostics.csv", index=False)
    plot_diagnostics(df, args.out / "missed_gang_diagnostics.png")

    # --- report -------------------------------------------------------------
    feats = ["size", "Phi", "mbar1", "capture", "density", "starness", "deg_ratio"]
    print("\n" + "=" * 78)
    print(
        f"POOLED: {len(df)} gangs over {df.day.nunique()} days   "
        f"detected {int(df.detected.sum())}/{len(df)} ({df.detected.mean():.1%})"
    )
    print("=" * 78)
    print(
        f"{'statistic':<12}{'detected median':>17}{'missed median':>15}{'AUC(detect)':>13}"
    )
    print("-" * 78)
    summary = {}
    for f in feats:
        d_med = float(df[df.detected == 1][f].median())
        m_med = float(df[df.detected == 0][f].median())
        try:
            auc = float(roc_auc_score(df.detected, df[f]))
        except Exception:
            auc = float("nan")
        summary[f] = {"detected_median": d_med, "missed_median": m_med, "auc": auc}
        print(f"{f:<12}{d_med:>17.4g}{m_med:>15.4g}{auc:>13.2f}")
    print(
        "\nAUC > 0.5: higher value → more likely detected;  AUC < 0.5: higher → missed."
    )
    (args.out / "summary.json").write_text(
        json.dumps(
            {"n_gangs": len(df), "detected": int(df.detected.sum()), "stats": summary},
            indent=2,
        )
        + "\n"
    )
    print(f"\nCSV + plot + summary -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--feature-mode", choices=["wallet", "random"], default="wallet")
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--ward-num-cuts", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/missed_analysis", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- train on day-start (frozen thereafter) -----------------------------
    print(f"=== training day {args.day_start} ===")
    A_unw, A_w, cls, nodes_df = build_graph(
        args.data_dir, args.day_start, args.day_start
    )
    cols = None
    if args.feature_mode == "wallet":
        Xf, cols = cached_wallet_features(
            args.data_dir, nodes_df, args.day_start, return_columns=True
        )
    else:
        Xf = random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xf, weighted=False)
    gsets = connected_components_sets(A_unw, np.where(cls == 1)[0], args.min_gang_size)
    gangs = make_patterns(gsets, "alert", "gang", "g")
    tr, _ = split_train_test(gangs, args.train_ratio, np.random.default_rng(args.seed))
    data = GraphData.from_graph(graph)
    cfg = DetectorConfig(
        degree=args.degree,
        tau=args.tau,
        epochs=args.epochs,
        conf_weight=args.conf_weight,
        conf_reduce="mean",
        optimizer="riemannian",
        coarsen_target="bank",
        coarsening_method="ward-tree",
        ward_stop="epsilon",
        epsilon=args.epsilon,
        ward_num_cuts=args.ward_num_cuts,
        threshold=args.threshold,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)
    det.fit(data, tr)
    print(
        f"  lambda_min {det.fit_info_['init_objective']:.4g} -> {det.fit_info_['objective']:.4g}"
        f"  (feature-dim {data.feature_dim}, {len(tr)} train gangs)"
    )


if __name__ == "__main__":
    main()
