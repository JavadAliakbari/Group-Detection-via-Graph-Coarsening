"""Head-to-head: Ward vs the screened-consistent (deflated) hierarchies.

The pipeline comparison in ``run_elliptic_modular`` conflates two things: the
*hierarchy* a coarsener builds and the *stop rule* used to cut it.  On transfer
days the stop is the label-free ``epsilon`` budget, and the deflated coarsener
holds the RSA constant down for far longer than Ward does, so at the same budget
it cuts much coarser -- which changes detection for reasons that have nothing to
do with merge quality.

This script separates them.  For each day it fits nothing new: it reuses the
frozen filter bank, builds *both* hierarchies over the *same* target subspace,
and reports, for each,

* the cut chosen by the shared ``epsilon`` budget (what the pipeline reports), and
* the best cut anywhere on that hierarchy (the oracle stop),

so the second column is a property of the tree alone.  For the deflated arms it
also prints the RSA sandwich certificate ``eps_Q <= eps_Pi <= mu eps_Q``.

Run::

    conda activate FedStruct
    python -m src.run_deflated_tree_comparison --train-days 25 --days 26,27,28,29,30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch

from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.deflated_coarsen import deflated_tree_coarsen
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.run_collective_bank_detection import ward_tree_coarsen
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
)
from src.utils.utils import LOGGER


def _load_day(data_dir: Path, day: int, min_gang_size: int, feature_columns):
    """One day's graph, wallet features and illicit gangs -- the same loading path
    ``run_elliptic_modular`` uses, so the numbers are directly comparable."""

    A_unw, A_w, cls, nodes_df = build_graph(data_dir, day, day)
    X, columns = load_node_features(
        data_dir, nodes_df, day, day, keep_columns=feature_columns, return_columns=True
    )
    graph = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    return GraphData.from_graph(graph), gangs, gang_sets, columns


def _score(coarsening_labels, gangs, y, threshold: float) -> dict:
    results, by_label = evaluate_loukas_patterns(
        gangs, coarsening_labels, y, threshold=threshold
    )
    alert = by_label.get("alert", {})
    return {
        "f1": float(np.mean([r.f1 for r in results])) if results else 0.0,
        "recall": float(alert.get("mean_recall", 0.0) or 0.0),
        "precision": float(alert.get("mean_precision", 0.0) or 0.0),
        "detected": int(sum(1 for r in results if r.recall > threshold and r.precision > threshold)),
        "total": len(results),
    }


def _row(name, stop, entry, gangs, y, threshold):
    labels = torch.from_numpy(entry["labels"]).to(y.device)
    return {
        "arm": name,
        "stop": stop,
        "n_coarse": int(entry["n_coarse"]),
        "epsilon_pi": float(entry["epsilon"]),
        "epsilon_q": float(entry.get("epsilon_q", float("nan"))),
        **_score(labels, gangs, y, threshold),
    }


def _arms(name, trajectory, gangs, y, threshold, eps_pi_budget, eps_q_budgets):
    """One row per stop rule, all read off the *same* hierarchy.

    ``trajectory`` is ordered fine -> coarse and carries both RSA constants and
    the F1 at every cut, so each stop rule is a scan rather than a rebuild.  That
    is the whole point: the arms below differ only in where the tree is cut.
    """

    rows = []
    feasible = [e for e in trajectory if e["epsilon"] <= eps_pi_budget]
    if feasible:
        rows.append(
            _row(name, f"eps_Pi<={eps_pi_budget:g}", feasible[-1], gangs, y, threshold)
        )
    for b in eps_q_budgets:
        f = [
            e
            for e in trajectory
            if np.isfinite(e.get("epsilon_q", np.nan)) and e["epsilon_q"] <= b
        ]
        if f:
            rows.append(_row(name, f"eps_Q<={b:g}", f[-1], gangs, y, threshold))
    if trajectory:
        best = max(trajectory, key=lambda e: e["train_f1"])
        rows.append(_row(name, "oracle-f1", best, gangs, y, threshold))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--train-days", default="25")
    ap.add_argument("--days", default="26,27,28,29,30")
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--num-cuts", type=int, default=300)
    ap.add_argument(
        "--eps-q-budgets",
        default="0.9,0.95,0.99",
        help="budgets to spend in the HARMONIC constant eps_Q for the deflated "
        "arms.  eps_Q is what those rules actually descend and it is monotone "
        "along the hierarchy, whereas the Euclidean eps_Pi is neither -- spending "
        "the budget in eps_Pi stops them before they coarsen at all.",
    )
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="results/deflated_tree_comparison", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_day = int(args.train_days)
    data, gangs, _sets, columns = _load_day(
        args.data_dir, train_day, args.min_gang_size, None
    )
    cfg = DetectorConfig(
        degree=args.degree,
        heads=args.heads,
        tau=args.tau,
        epochs=args.epochs,
        threshold=args.threshold,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)
    LOGGER.info(f"fitting the shared filter bank on day {train_day} ({len(gangs)} gangs)")
    det.fit([(str(train_day), data, gangs, [])])

    rows: list = []
    for day in [train_day] + [int(d) for d in args.days.split(",") if d.strip()]:
        if day == train_day:
            d_data, d_gangs = data, gangs
        else:
            d_data, d_gangs, _s, _c = _load_day(
                args.data_dir, day, args.min_gang_size, columns
            )
        if not d_gangs:
            LOGGER.info(f"day {day}: no gangs, skipped")
            continue
        basis = det.target_subspace(d_data, d_gangs)  # same target for every arm
        LOGGER.info(f"\n=== day {day}: N={d_data.num_nodes:,}  {len(d_gangs)} gangs ===")

        eps_q_budgets = [float(b) for b in args.eps_q_budgets.split(",") if b.strip()]
        day_rows = []
        co, traj = ward_tree_coarsen(
            d_data.adjacency, basis, d_gangs, d_data.y,
            tau=args.tau, laplacian="symmetric", threshold=args.threshold,
            stop="f1", epsilon_budget=args.epsilon, num_cuts=args.num_cuts,
        )
        day_rows += _arms(
            "ward-tree", traj, d_gangs, d_data.y, args.threshold, args.epsilon, []
        )
        for rule, tag in (("dual-ward", "deflated-dual-ward"), ("minimax", "deflated-minimax")):
            co, traj = deflated_tree_coarsen(
                d_data.adjacency, basis, d_gangs, d_data.y,
                tau=args.tau, rule=rule, laplacian="symmetric",
                threshold=args.threshold, stop="f1",
                epsilon_budget=args.epsilon, num_cuts=args.num_cuts,
            )
            cert = co.deflated_certificate
            LOGGER.info(
                f"  {tag}: eps_Q={cert['epsilon_q_exact']:.4f} <= "
                f"eps_Pi={cert['epsilon_pi']:.4f} <= mu*eps_Q={cert['sandwich_upper']:.4f} "
                f"(mu={cert['mu']:.3f})  sandwich "
                f"{'OK' if cert['sandwich_ok'] else 'VIOLATED'}  "
                f"tr H={cert['trace_h']:.3g}  mean eta0={cert['leakage_mean']:.4f}"
            )
            day_rows += _arms(
                tag, traj, d_gangs, d_data.y, args.threshold, args.epsilon, eps_q_budgets
            )

        hdr = (
            f"  {'arm':<20}{'stop':<13}{'n_coarse':>9}{'eps_Pi':>8}{'eps_Q':>7}"
            f"{'recall':>8}{'prec':>7}{'f1':>7}{'det':>9}"
        )
        LOGGER.info(hdr)
        LOGGER.info("  " + "-" * (len(hdr) - 2))
        for row in day_rows:
            row["day"] = day
            row["n_gangs"] = len(d_gangs)
            rows.append(row)
            LOGGER.info(
                f"  {row['arm']:<20}{row['stop']:<13}{row['n_coarse']:>9,}"
                f"{row['epsilon_pi']:>8.3f}{row['epsilon_q']:>7.3f}"
                f"{row['recall']:>8.3f}{row['precision']:>7.3f}"
                f"{row['f1']:>7.3f}{row['detected']:>5}/{row['total']:<3}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "tree_comparison.csv", index=False)
    LOGGER.info("\n" + "=" * 78)
    LOGGER.info("AVERAGE OVER ALL DAYS")
    LOGGER.info("=" * 78)
    hdr = f"  {'arm':<20}{'stop':<13}{'n_coarse':>10}{'f1':>8}{'recall':>9}{'prec':>8}{'det/tot':>10}"
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for (arm, stop), g in df.groupby(["arm", "stop"], sort=False):
        LOGGER.info(
            f"  {arm:<20}{stop:<13}{g['n_coarse'].mean():>10,.0f}{g['f1'].mean():>8.3f}"
            f"{g['recall'].mean():>9.3f}{g['precision'].mean():>8.3f}"
            f"{int(g['detected'].sum()):>6}/{int(g['total'].sum()):<3}"
        )
    (args.out / "tree_comparison.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n"
    )
    LOGGER.info(f"\nwrote {args.out / 'tree_comparison.csv'}")


if __name__ == "__main__":
    main()
