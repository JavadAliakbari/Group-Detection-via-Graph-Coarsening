r"""Step 1: reproduce ``ward_tree`` vs ``raw_ward`` vs ``deflated_ward``.

Everything but the merge rule is held fixed: the same generated graphs, the same
train/test group split, the same node features, the same seeds, the same frozen
polynomial closed-form representation (fitted **once** per seed and reused by
every arm), the same screened geometry and ``tau``, the same detection
threshold, the same evaluation levels and the same stopping-rule protocol.

Writes one row per (seed, graph, method) with the stopping-rule and oracle
metrics, the PR-AUC over the whole hierarchy, and -- separately -- the full
trajectory, so matched-budget comparisons are done afterwards on identical
levels rather than at each method's own preferred cut.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from analysis.coarsener_ablation import BASE_COARSENING, frozen_representation
from src.pipeline.coarsening import Coarsening

METHODS = ("ward_tree", "raw_ward", "deflated_ward", "deflated_ward_tight")
_TRAJ_KEYS = (
    "n_coarse",
    "merges",
    "reduction",
    "retained",
    "epsilon",
    "epsilon_is_exact",
    "merge_score",
    "all_mean_precision",
    "all_mean_recall",
    "all_mean_f1",
    "all_detection_rate",
    "all_micro_f1",
    "test_mean_f1",
    "test_detection_rate",
    "train_mean_f1",
    "score_sum",
    "score_sum_raw",
)


#: extra diagnostic arms: each changes exactly one thing relative to the
#: production deflated coarsener, and the comment says which of the four
#: categories it belongs to
ARMS = {
    "deflated_nocap": dict(method="deflated_ward", deflated_max_rescore=4096),
    "deflated_cert": dict(
        method="deflated_ward", deflated_queue_key="certified",
        deflated_max_rescore=4096,
    ),  # search
    "deflated_ball": dict(
        method="deflated_ward", deflated_hops=3, deflated_max_ball=64
    ),  # score accuracy
    "deflated_nofanout": dict(method="deflated_ward", deflated_fanout=0),  # search
}


def _cut_row(cut) -> dict:
    m = cut.metrics.get("all", {})
    return {
        "n_coarse": int(cut.n_coarse),
        "reduction": float(cut.reduction),
        "epsilon": float(cut.epsilon),
        "precision": float(m.get("mean_precision", float("nan"))),
        "recall": float(m.get("mean_recall", float("nan"))),
        "f1": float(m.get("mean_f1", float("nan"))),
        "detection_rate": float(m.get("detection_rate", float("nan"))),
        "micro_f1": float(m.get("micro_f1", float("nan"))),
        "detected": int(m.get("detected", 0)),
        "total": int(m.get("total", 0)),
    }


def run_seed(seed: int, *, num_nodes, num_train, num_test, methods=METHODS):
    bundle, learner, learning, bases = frozen_representation(
        seed, num_nodes=num_nodes, num_train=num_train, num_test=num_test
    )
    graphs = [(g, "train") for g in bundle.train_graphs]
    graphs += [(g, "test") for g in bundle.test_graphs]

    rows, trajectories = [], {}
    for method in methods:
        coarsener = Coarsening(replace(BASE_COARSENING, **ARMS.get(method, {"method": method})))
        for graph, scope in graphs:
            t0 = time.time()
            result = coarsener.run(
                graph,
                bases[graph.graph_id],
                train_groups=graph.train_groups if scope == "train" else None,
            )
            elapsed = time.time() - t0
            row = {
                "seed": seed,
                "method": method,
                "graph": graph.graph_id,
                "scope": scope,
                "n": int(graph.num_nodes),
                "target_rank": int(result.target_rank),
                "pr_auc": float(result.pr_auc),
                "constrained_merges": int(result.hierarchy.constrained_merges),
                "completion_merges": int(result.hierarchy.completion_merges),
                "runtime_s": elapsed,
            }
            row.update({f"stop_{k}": v for k, v in _cut_row(result.stopping_rule).items()})
            row.update({f"oracle_{k}": v for k, v in _cut_row(result.oracle).items()})
            rows.append(row)
            trajectories[(seed, method, graph.graph_id)] = [
                {k: r[k] for k in _TRAJ_KEYS if k in r} for r in result.trajectory
            ]
            print(
                f"  seed {seed} {method:14s} {graph.graph_id:6s} ({scope:5s}) "
                f"stopF1={row['stop_f1']:.3f} oracleF1={row['oracle_f1']:.3f} "
                f"PRAUC={row['pr_auc']:.3f} n*={row['stop_n_coarse']:5d} "
                f"[{elapsed:.1f}s]",
                flush=True,
            )
    return rows, trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-nodes", type=int, default=3000)
    parser.add_argument("--num-train", type=int, default=3)
    parser.add_argument("--num-test", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", type=Path, default=Path("results2/coarsener_repro"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows, all_traj = [], {}
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        rows, traj = run_seed(
            seed,
            num_nodes=args.num_nodes,
            num_train=args.num_train,
            num_test=args.num_test,
            methods=tuple(args.methods),
        )
        all_rows.extend(rows)
        all_traj.update(traj)
        (args.out / "rows.json").write_text(json.dumps(all_rows, indent=1))
        with open(args.out / "trajectories.pkl", "wb") as fh:
            pickle.dump(all_traj, fh)
    print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
