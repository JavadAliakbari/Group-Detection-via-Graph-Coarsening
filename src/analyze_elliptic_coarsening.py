"""Analysis + plots of the *learned* coarsening on the Elliptic++ Actors graph.

Runs the modular :class:`CollectiveBankDetector`, then measures and visualizes
what the learned target subspace buys the coarsening.  The unifying finding is
that **a gang's conductance controls everything**: low-conductance (tight) gangs
get near-constant target rows, hence cheap internal edges, hence they collapse
into one supernode and are detected; the few sprawling high-conductance illicit
components do not.  Four deliverables:

1. **Per-gang precision / recall** (large gangs first).
2. **Edge-cost geometry.** The coarsener charges each edge the RSA local variation
   ``c(u,v) = ||A[u]-A[v]||^2`` on the ``M_tau``-orthonormal target rows ``A``.  We
   show *per gang* that internal edges are cheaper than that gang's boundary edges
   (a scatter below the diagonal), the aggregate split, and a node-link drawing of
   a tight gang with edges colored by cost (cheap interior, costly boundary).
3. **Supernode conductance** ``Phi(S)=cut(S)/vol(S)`` -- gang supernodes vs
   background -- and the conductance -> detection link that explains (1)-(2).
4. **Non-gang (licit) precision / recall** and the false-collapse rate.

Run::

    python -m src.analyze_elliptic_coarsening --day-start 24 --day-end 24 \
        --feature-mode random --random-width 128 --epochs 400 \
        --out results/elliptic_analysis
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
import networkx as nx
import numpy as np
import torch

from src.pattern_models import make_patterns
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import (
    _degrees,
    _l_orthonormalize,
    _laplacian,
    _normalized_laplacian,
    _screened_metric,
    evaluate_loukas_patterns,
)
from src.run_elliptic_gang_conductance import connected_components_sets


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def edge_costs(A: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-edge RSA local variation ``c(u,v) = ||A[u]-A[v]||^2`` on the target rows."""

    return (A[edge_index[0]] - A[edge_index[1]]).pow(2).sum(dim=1)


def _cut_vol_conductance(adjacency, labels, n_groups):
    """``(cut, vol, Phi)`` per group id in ``labels`` (Phi = cut/min(vol, volbar))."""

    deg = _degrees(adjacency)
    vol = torch.zeros(n_groups, dtype=deg.dtype).index_add_(0, labels.clamp_min(0), deg)
    idx = adjacency.coalesce().indices()
    w = adjacency.coalesce().values()
    lu, lv = labels[idx[0]], labels[idx[1]]
    boundary = lu != lv
    cut = torch.zeros(n_groups, dtype=deg.dtype)
    valid = boundary & (lu >= 0)
    cut.index_add_(0, lu[valid], w[valid])
    vol_total = deg.sum()
    phi = (cut / torch.minimum(vol, vol_total - vol).clamp_min(1e-12)).clamp(0, 1)
    return cut, vol, phi


def gang_conductance(adjacency, gang_of, n_gangs):
    """Conductance of each planted gang on the *original* graph."""

    _, _, phi = _cut_vol_conductance(adjacency, gang_of, n_gangs)
    return phi


def supernode_conductance(adjacency, node_to_super):
    """Conductance + size of every coarsened supernode."""

    n_super = int(node_to_super.max().item()) + 1
    cut, vol, phi = _cut_vol_conductance(adjacency, node_to_super, n_super)
    sizes = torch.zeros(n_super, dtype=torch.long).index_add_(
        0, node_to_super, torch.ones_like(node_to_super)
    )
    return phi, vol, sizes


def dominant_supernode(node_to_super, nodes):
    sup = node_to_super[torch.as_tensor(nodes, dtype=torch.long)]
    vals, counts = torch.unique(sup, return_counts=True)
    return int(vals[int(counts.argmax())])


def per_gang_edge_cost(cost, edge_index, gang_of, n_gangs):
    """Median internal- and boundary-edge cost for each gang."""

    gu, gv = gang_of[edge_index[0]], gang_of[edge_index[1]]
    out = []
    for gi in range(n_gangs):
        internal = (gu == gi) & (gv == gi)
        boundary = ((gu == gi) | (gv == gi)) & ~internal
        mi = float(cost[internal].median()) if int(internal.sum()) else float("nan")
        mb = float(cost[boundary].median()) if int(boundary.sum()) else float("nan")
        out.append((mi, mb))
    return out


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_gang_pr(results, gangs, train_ids, out: Path, top: int = 25):
    order = sorted(range(len(gangs)), key=lambda i: gangs[i].num_nodes, reverse=True)[
        :top
    ]
    sizes = [gangs[i].num_nodes for i in order]
    rec = [results[i].recall for i in order]
    prec = [results[i].precision for i in order]
    det = [results[i].detected for i in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.5), 5))
    ax.bar(x - 0.2, rec, 0.4, label="recall", color="tab:blue")
    ax.bar(x + 0.2, prec, 0.4, label="precision", color="tab:orange")
    for xi, i, d in zip(x, order, det):
        ax.plot(xi, 1.05, marker="*" if d else "x", color="green" if d else "red", ms=9)
        if gangs[i].id in train_ids:
            ax.text(xi, -0.09, "tr", ha="center", fontsize=6, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("gang size (nodes)   [tr = training gang]")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.14)
    ax.set_title(
        f"(1) Per-gang recall / precision (top {len(order)} by size)   ★ detected  ✗ missed"
    )
    ax.legend(loc="center right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_edge_cost(cost, categories, pg_cost, gang_sizes, detected, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) per-gang medians (each gang weighted equally -- not dominated by the few
    #     giant sprawling gangs that hold most of the intra edges)
    mi_all = np.array([c[0] for c in pg_cost])
    mb_all = np.array([c[1] for c in pg_cost])
    okm = np.isfinite(mi_all) & np.isfinite(mb_all)
    axes[0].boxplot(
        [np.log10(mi_all[okm]), np.log10(mb_all[okm])],
        tick_labels=["gang\ninterior", "gang\nboundary"],
        showfliers=False,
    )
    for yi, yb in zip(mi_all[okm], mb_all[okm]):
        axes[0].plot(
            [1, 2], [np.log10(yi), np.log10(yb)], color="gray", alpha=0.25, lw=0.6
        )
    axes[0].set_ylabel(r"$\log_{10}$ median edge cost (per gang)")
    axes[0].set_title("(2a) Per-gang interior vs boundary\n(each line = one gang)")
    axes[0].grid(axis="y", alpha=0.3)

    # (b) per-gang internal vs boundary (below diagonal = internal cheaper)
    mi = np.array([c[0] for c in pg_cost])
    mb = np.array([c[1] for c in pg_cost])
    ok = np.isfinite(mi) & np.isfinite(mb)
    below = int((mi[ok] < mb[ok]).sum())
    sc = axes[1].scatter(
        mb[ok],
        mi[ok],
        s=20 + 3 * np.sqrt(np.array(gang_sizes)[ok]),
        c=np.array(detected)[ok],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        edgecolors="k",
        linewidths=0.4,
        alpha=0.85,
    )
    lo = np.nanmin([mi[ok].min(), mb[ok].min()]) * 0.5
    hi = np.nanmax([mi[ok].max(), mb[ok].max()]) * 2
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    axes[1].fill_between([lo, hi], [lo, hi], [lo, lo], color="tab:green", alpha=0.07)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("median BOUNDARY edge cost")
    axes[1].set_ylabel("median INTERNAL edge cost")
    axes[1].set_title(
        f"(2b) Per gang: internal < boundary for {below}/{int(ok.sum())} gangs\n"
        "(below diagonal = interior cheaper; green = detected)"
    )
    axes[1].grid(alpha=0.3, which="both")

    # (c) conductance vs cost ratio
    axes[2].axis("off")
    fig.colorbar(sc, ax=axes[2], fraction=0.5, label="detected (0/1)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    lab = ["intra-gang", "gang-boundary", "background"]
    med = {
        lab[k]: (
            float(cost[categories == k][cost[categories == k] > 0].median())
            if int((categories == k).sum())
            else None
        )
        for k in range(3)
    }
    med["per_gang_interior_median"] = (
        float(np.median(mi_all[okm])) if okm.any() else None
    )
    med["per_gang_boundary_median"] = (
        float(np.median(mb_all[okm])) if okm.any() else None
    )
    return med, below, int(ok.sum())


def plot_gang_graph(edge_index, cost, gang_id, gang_nodes, out: Path, max_halo=250):
    gang_set = set(int(v) for v in gang_nodes)
    u, v = edge_index[0].numpy(), edge_index[1].numpy()
    inc = np.where(np.isin(u, list(gang_set)) | np.isin(v, list(gang_set)))[0]
    halo = set()
    for e in inc:
        a, b = int(u[e]), int(v[e])
        halo.add(a if a not in gang_set else b)
    halo -= gang_set
    if len(halo) > max_halo:
        halo = set(list(halo)[:max_halo])
    keep = gang_set | halo
    mask = np.isin(u, list(keep)) & np.isin(v, list(keep))
    sub_e = np.where(mask)[0]
    if sub_e.size == 0:
        return
    G = nx.Graph()
    G.add_nodes_from(keep)
    ec_all = cost.numpy()
    for e in sub_e:
        G.add_edge(int(u[e]), int(v[e]), c=float(ec_all[e]))
    pos = nx.spring_layout(G, seed=0, iterations=60, k=1.5 / np.sqrt(max(len(G), 2)))
    e_c = np.log10(np.array([G[a][b]["c"] for a, b in G.edges()]) + 1e-12)
    node_color = ["#d62728" if n in gang_set else "#9ecae1" for n in G.nodes()]
    node_size = [26 if n in gang_set else 8 for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(8, 7))
    ec = nx.draw_networkx_edges(
        G, pos, edge_color=e_c, edge_cmap=plt.cm.RdYlBu_r, width=1.3, alpha=0.75, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, node_color=node_color, node_size=node_size, linewidths=0, ax=ax
    )
    fig.colorbar(ec, ax=ax, label=r"$\log_{10}$ edge cost")
    ax.set_title(
        f"(2c) Gang {gang_id} ({len(gang_set)} nodes, red) + halo (blue)\n"
        "edges by learned cost: cheap interior, costly boundary"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_conductance(phi_super, sizes, gang_super_ids, gang_phi, gang_det, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    multi = sizes >= 2
    gmask = torch.zeros_like(phi_super, dtype=torch.bool)
    gmask[torch.as_tensor(sorted(set(gang_super_ids)), dtype=torch.long)] = True
    phi_gang = phi_super[gmask & multi].numpy()
    phi_bg = phi_super[(~gmask) & multi].numpy()
    bins = np.linspace(0, 1, 41)
    ax1.hist(
        phi_bg,
        bins=bins,
        alpha=0.6,
        density=True,
        color="tab:gray",
        label=f"background ({phi_bg.size})",
    )
    ax1.hist(
        phi_gang,
        bins=bins,
        alpha=0.7,
        density=True,
        color="tab:red",
        label=f"gang supernodes ({phi_gang.size})",
    )
    if phi_gang.size:
        ax1.axvline(
            np.median(phi_gang),
            color="tab:red",
            ls="--",
            label=f"gang median Φ={np.median(phi_gang):.3f}",
        )
    if phi_bg.size:
        ax1.axvline(
            np.median(phi_bg),
            color="k",
            ls=":",
            label=f"bg median Φ={np.median(phi_bg):.3f}",
        )
    ax1.set_xlabel(r"supernode conductance $\Phi$")
    ax1.set_ylabel("density")
    ax1.set_title("(3a) Coarsened-supernode conductance (size≥2)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    gp = np.array(gang_phi)
    gd = np.array(gang_det, dtype=bool)
    ax2.scatter(
        gp[~gd],
        np.zeros((~gd).sum()) + 0.02 * np.random.randn((~gd).sum()),
        s=30,
        color="tab:red",
        alpha=0.6,
        label="missed",
    )
    ax2.scatter(
        gp[gd],
        np.ones(gd.sum()) + 0.02 * np.random.randn(gd.sum()),
        s=30,
        color="tab:green",
        alpha=0.6,
        label="detected",
    )
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["missed", "detected"])
    ax2.set_xlabel(r"gang conductance $\Phi$(gang)")
    ax2.set_title("(3b) Detection vs gang conductance\n(low Φ → detected)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {
        "gang_median_conductance": (
            float(np.median(phi_gang)) if phi_gang.size else None
        ),
        "background_median_conductance": (
            float(np.median(phi_bg)) if phi_bg.size else None
        ),
        "n_gang_supernodes": int(phi_gang.size),
        "n_background_supernodes": int(phi_bg.size),
    }


def plot_nongang_pr(normal_results, out: Path):
    rec = np.array([r.recall for r in normal_results])
    prec = np.array([r.precision for r in normal_results])
    collapsed = np.array([r.detected for r in normal_results])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.boxplot([rec, prec], tick_labels=["recall", "precision"], showfliers=False)
    ax1.scatter(
        np.random.normal(1, 0.05, rec.size), rec, s=8, alpha=0.4, color="tab:blue"
    )
    ax1.scatter(
        np.random.normal(2, 0.05, prec.size), prec, s=8, alpha=0.4, color="tab:orange"
    )
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("score")
    ax1.set_title(f"(4a) Non-gang (licit) components  (n={len(normal_results)})")
    ax1.grid(axis="y", alpha=0.3)
    rate = float(collapsed.mean()) if collapsed.size else 0.0
    ax2.bar(
        ["collapse\n(false pos.)", "not collapsed"],
        [rate, 1 - rate],
        color=["tab:red", "tab:green"],
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fraction")
    ax2.set_title(f"(4b) Licit false-collapse rate = {rate:.1%}")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {
        "normal_mean_recall": float(rec.mean()) if rec.size else None,
        "normal_mean_precision": float(prec.mean()) if prec.size else None,
        "normal_false_collapse_rate": rate,
    }


def analyze_coarsening(
    data: GraphData,
    basis,
    gangs,
    det: CollectiveBankDetector,
    gang_train,
    args,
    cfg: DetectorConfig,
    coarsening,
    n2s,
    normals,
):
    # the metric must be the one the target basis and the RSA budget are stated
    # in, or every epsilon reported below belongs to a different geometry than
    # the coarsening that produced it
    base_laplacian = (
        _laplacian
        if cfg.coarsening_laplacian in ("combinatorial", "comb")
        else _normalized_laplacian
    )
    metric = _screened_metric(base_laplacian(data.adjacency), cfg.tau)
    A = _l_orthonormalize(basis, metric)
    gang_of = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for gi, g in enumerate(gangs):
        gang_of[torch.as_tensor(list(g.nodes), dtype=torch.long)] = gi

    summary = {
        "config": cfg.to_dict(),
        "day_start": args.day_start,
        "day_end": args.day_end,
        "n_nodes": data.num_nodes,
        "n_gangs": len(gangs),
        "n_coarse": coarsening.n_coarse,
        "epsilon": coarsening.epsilon,
        # gradient fit reports "objective"; the closed-form collective solver
        # reports the Theorem A/B triple instead (lambda_min of Gamma)
        "lambda_min": det.fit_info_.get(
            "objective", det.fit_info_.get("lambda_min_Gamma")
        ),
    }

    # (1) per-gang PR
    train_ids = {p.id for p in gang_train}
    gang_res, _ = evaluate_loukas_patterns(gangs, n2s, data.y, threshold=cfg.threshold)
    detected = [bool(r.detected) for r in gang_res]
    plot_gang_pr(gang_res, gangs, train_ids, args.out / "1_gang_precision_recall.png")
    big = sorted(range(len(gangs)), key=lambda i: gangs[i].num_nodes, reverse=True)[:15]
    summary["gangs_large"] = [
        {
            "size": gangs[i].num_nodes,
            "recall": gang_res[i].recall,
            "precision": gang_res[i].precision,
            "f1": gang_res[i].f1,
            "detected": detected[i],
            "train": gangs[i].id in train_ids,
        }
        for i in big
    ]
    print(f"  [1] per-gang PR  (detected {sum(detected)}/{len(gangs)})")

    # (2) edge cost
    cost = edge_costs(A, data.edge_index)
    gu, gv = gang_of[data.edge_index[0]], gang_of[data.edge_index[1]]
    categories = torch.full_like(gu, 1)
    categories[(gu >= 0) & (gu == gv)] = 0
    categories[(gu < 0) & (gv < 0)] = 2
    pg_cost = per_gang_edge_cost(cost, data.edge_index, gang_of, len(gangs))
    gang_sizes = [g.num_nodes for g in gangs]
    med, below, nvalid = plot_edge_cost(
        cost, categories, pg_cost, gang_sizes, detected, args.out / "2_edge_cost.png"
    )
    summary["edge_cost_median"] = med
    summary["gangs_internal_cheaper_than_boundary"] = f"{below}/{nvalid}"
    print(
        f"  [2] edge cost: internal<boundary for {below}/{nvalid} gangs; "
        f"per-gang median interior={med['per_gang_interior_median']:.2e} "
        f"boundary={med['per_gang_boundary_median']:.2e}"
    )
    # draw moderate-size gangs with the STRONGEST interior-cheap / boundary-costly
    # contrast (clearest visual), not the giant sprawling ones
    cand = [
        (i, pg_cost[i][1] / pg_cost[i][0])
        for i in range(len(gangs))
        if np.isfinite(pg_cost[i][0])
        and np.isfinite(pg_cost[i][1])
        and pg_cost[i][0] < pg_cost[i][1]
        and 12 <= gangs[i].num_nodes <= 90
    ]
    clean = [
        i
        for i, _ in sorted(cand, key=lambda t: t[1], reverse=True)[
            : args.num_gang_graphs if hasattr(args, "num_gang_graphs") else 2
        ]
    ]
    for i in clean:
        plot_gang_graph(
            data.edge_index,
            cost,
            gangs[i].id,
            gangs[i].nodes,
            args.out / f"2c_gang_graph_{gangs[i].id}_n{gangs[i].num_nodes}.png",
        )
    print(f"  [2c] drew {len(clean)} clean gang graphs")

    # (3) conductance
    phi_super, vol, sizes = supernode_conductance(data.adjacency, n2s)
    gsuper = [dominant_supernode(n2s, list(g.nodes)) for g in gangs]
    gphi = gang_conductance(data.adjacency, gang_of, len(gangs)).tolist()
    cond = plot_conductance(
        phi_super, sizes, gsuper, gphi, detected, args.out / "3_conductance.png"
    )
    summary["conductance"] = cond
    summary["gang_conductance_detected_vs_missed"] = {
        "detected_median": (
            float(np.median([gphi[i] for i in range(len(gangs)) if detected[i]]))
            if any(detected)
            else None
        ),
        "missed_median": (
            float(np.median([gphi[i] for i in range(len(gangs)) if not detected[i]]))
            if not all(detected)
            else None
        ),
    }
    # every median here is None when its group is empty -- a coarsening that
    # detects nothing (or everything) is a legitimate outcome to report, not a
    # crash, so format defensively.
    def _fmt(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(
        f"  [3] conductance: gang supernodes Φ={_fmt(cond['gang_median_conductance'])} "
        f"vs bg Φ={_fmt(cond['background_median_conductance'])}; "
        f"detected gangs Φ="
        f"{_fmt(summary['gang_conductance_detected_vs_missed']['detected_median'])}"
    )

    # (4) non-gang PR
    normal_res, _ = evaluate_loukas_patterns(
        normals, n2s, data.y, threshold=cfg.threshold
    )
    ng = plot_nongang_pr(normal_res, args.out / "4_nongang_precision_recall.png")
    ng["n_normals"] = len(normals)
    summary["nongang"] = ng
    print(
        f"  [4] non-gang PR: recall={ng['normal_mean_recall']:.3f} prec={ng['normal_mean_precision']:.3f} "
        f"false-collapse={ng['normal_false_collapse_rate']:.1%}"
    )

    (args.out / "analysis.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(f"\nFigures + analysis.json -> {args.out}")
