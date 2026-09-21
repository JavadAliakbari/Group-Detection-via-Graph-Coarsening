r"""Figures for the coarsener comparison.

All panels read the stored trajectories, so every curve is the *same* hierarchy
the tables score, and every matched-budget panel compares methods at identical
levels rather than at each method's preferred cut.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {
    "ward_tree": "#1f77b4",
    "raw_ward": "#7f7f7f",
    "deflated_ward": "#d62728",
    "deflated_ward_tight": "#2ca02c",
    "deflated_ward_abs": "#9467bd",
    "deflated_nocap": "#ff7f0e",
    "deflated_cert": "#17becf",
    "deflated_ball": "#bcbd22",
    "deflated_nofanout": "#8c564b",
}


def _grid(trajectories, method, key, n_grid):
    """``key`` at each ``n_coarse`` in ``n_grid``, one row per (seed, graph)."""

    runs = [t for (s, m, g), t in trajectories.items() if m == method]
    out = np.full((len(runs), len(n_grid)), np.nan)
    for i, traj in enumerate(runs):
        n = np.array([r["n_coarse"] for r in traj])
        v = np.array([r[key] for r in traj])
        order = np.argsort(n)
        out[i] = np.interp(n_grid, n[order], v[order])
    return out


def hierarchy_panels(trajectories, methods, out_dir, n_max):
    n_grid = np.unique(np.round(np.geomspace(20, n_max, 220)).astype(int))
    keys = [
        ("all_mean_f1", "F1"),
        ("all_mean_precision", "precision"),
        ("all_mean_recall", "recall"),
        ("all_detection_rate", "detection rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), keys):
        for method in methods:
            data = _grid(trajectories, method, key, n_grid)
            mean, sd = np.nanmean(data, 0), np.nanstd(data, 0, ddof=1)
            ax.plot(n_grid, mean, label=method, color=COLORS.get(method), lw=1.6)
            ax.fill_between(
                n_grid, mean - sd, mean + sd, color=COLORS.get(method), alpha=0.12
            )
        ax.set_xscale("log")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("supernodes (coarser to the right)")
    axes[0, 0].invert_xaxis()
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Detection over the hierarchy, at matched supernode budgets")
    fig.tight_layout()
    fig.savefig(out_dir / "hierarchy_metrics.png", dpi=150)
    plt.close(fig)


def epsilon_panel(trajectories, methods, out_dir, n_max):
    n_grid = np.unique(np.round(np.geomspace(20, n_max, 220)).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in methods:
        eps = _grid(trajectories, method, "epsilon", n_grid)
        f1 = _grid(trajectories, method, "all_mean_f1", n_grid)
        axes[0].plot(n_grid, np.nanmean(eps, 0), color=COLORS.get(method), label=method)
        order = np.argsort(np.nanmean(eps, 0))
        axes[1].plot(
            np.nanmean(eps, 0)[order],
            np.nanmean(f1, 0)[order],
            color=COLORS.get(method),
            label=method,
        )
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("supernodes")
    axes[0].set_ylabel(r"common $\varepsilon$ (Loukas Def. 2)")
    axes[1].set_xlabel(r"common $\varepsilon$")
    axes[1].set_ylabel("F1")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Matched common epsilon")
    fig.tight_layout()
    fig.savefig(out_dir / "matched_epsilon.png", dpi=150)
    plt.close(fig)


def score_panel(trajectories, methods, out_dir, n_max):
    """Cumulative raw merge score along the hierarchy (the eps_Q^2 sandwich axis)."""

    n_grid = np.unique(np.round(np.geomspace(20, n_max, 220)).astype(int))
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        if not any(
            "score_sum_raw" in t[0] for (s, m, g), t in trajectories.items() if m == method
        ):
            continue
        data = _grid(trajectories, method, "score_sum_raw", n_grid)
        ax.plot(n_grid, np.nanmean(data, 0), color=COLORS.get(method), label=method)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("supernodes")
    ax.set_ylabel(r"cumulative raw merge score  $S_n=\sum_{k>n}s_k$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_score.png", dpi=150)
    plt.close(fig)


def forensic_panels(forensic_dir: Path, out_dir: Path):
    """Internal vs boundary merges, and group survival, from the merge replay."""

    rows = np.load(forensic_dir / "merge_rows.npy", allow_pickle=True)
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, sub in by.items():
        n0 = max(r["n_coarse"] for r in sub) + 1
        grid = np.unique(np.round(np.geomspace(20, n0, 120)).astype(int))
        for role, ax, style in (("internal", axes[0], "-"), ("boundary", axes[0], "--")):
            picks = np.array(
                sorted(r["n_coarse"] for r in sub if r["role"] == role)
            )
            if not picks.size:
                continue
            counts = np.array([(picks >= g).sum() for g in grid], float)
            counts /= max(len(by), 1) and max(
                1, len({(r["seed"], r["graph"]) for r in sub})
            )
            ax.plot(
                grid, counts, style, color=COLORS.get(method),
                label=f"{method} {role}", lw=1.4,
            )
        sizes = np.array([(r["n_coarse"], r["new_card"]) for r in sub], float)
        order = np.argsort(-sizes[:, 0])
        running = np.maximum.accumulate(sizes[order, 1])
        axes[1].plot(
            sizes[order, 0], running, color=COLORS.get(method), label=method, lw=1.4
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("supernodes")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("merges of this kind committed so far (per graph)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("largest block so far (nodes)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=7)
    fig.suptitle("Internal / boundary merge order and block growth")
    fig.tight_layout()
    fig.savefig(out_dir / "merge_forensics.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("results2/coarsener_final"))
    parser.add_argument("--forensics", type=Path, default=Path("results2/merge_forensics"))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (args.dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.dir / "trajectories.pkl", "rb") as fh:
        trajectories = pickle.load(fh)
    methods = args.methods or sorted({m for (_s, m, _g) in trajectories})
    n_max = max(r["n_coarse"] for t in trajectories.values() for r in t)

    hierarchy_panels(trajectories, methods, out_dir, n_max)
    epsilon_panel(trajectories, methods, out_dir, n_max)
    score_panel(trajectories, methods, out_dir, n_max)
    if (args.forensics / "merge_rows.npy").exists():
        forensic_panels(args.forensics, out_dir)
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
