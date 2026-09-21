r"""Step 4: compare the hierarchies at the **merge** level, not at their cuts.

For each method and graph this replays the committed merge order once and
records, per merge:

``role``        ``internal`` (both blocks meet the same ground-truth group),
                ``boundary`` (at least one block meets a group and they do not
                meet a common one) or ``background`` (neither meets a group)
``score``       the merge score the method itself stored
``mass``        the classical Ward mass factor ``|A||B|/(|A|+|B|)`` of the pair,
                and its volume-weighted twin -- neither is used by any decision,
                they are measured so the size profile of the merge order can be
                compared across scores
``sizes``       the two block cardinalities

and per group:

``first_impure``   the merge index at which the group's dominant block first
                   contains a node outside the group
``completed``      the merge index at which every node of the group lies in one
                   block (``None`` if that never happens before the group has
                   already been diluted)
``survival``       ``exact`` levels: the group is *exactly* one block, no more

Nothing here feeds a decision; it is measured off the committed hierarchies.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from analysis.coarsener_ablation import BASE_COARSENING, frozen_representation
from src.pipeline.coarsening import Coarsening

METHODS = ("ward_tree", "raw_ward", "deflated_ward")


def replay(children, n, group_sets, scores):
    """One pass over a merge order -> per-merge rows and per-group milestones."""

    members = [ {v} for v in range(n) ]
    members += [None] * int(children.shape[0])
    parent = np.arange(n + int(children.shape[0]), dtype=np.int64)
    card = np.ones(n + int(children.shape[0]), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    touches = [set() for _ in range(n + int(children.shape[0]))]
    for gi, s in enumerate(group_sets):
        for v in s:
            touches[v].add(gi)

    n_groups = len(group_sets)
    first_impure = [None] * n_groups
    completed = [None] * n_groups
    #: count of each group's nodes in each alive block, and the block holding most
    counts = [defaultdict(int) for _ in range(n_groups)]
    for gi, s in enumerate(group_sets):
        for v in s:
            counts[gi][v] = 1

    rows = []
    first_boundary = None
    n_internal_before_boundary = 0
    for t in range(int(children.shape[0])):
        a, b = int(children[t, 0]), int(children[t, 1])
        ra, rb = find(a), find(b)
        ga, gb = touches[ra], touches[rb]
        if not ga and not gb:
            role = "background"
        elif ga & gb:
            role = "internal"
        else:
            role = "boundary"
        if role == "boundary" and first_boundary is None:
            first_boundary = t
        if role == "internal" and first_boundary is None:
            n_internal_before_boundary += 1

        ca, cb = int(card[ra]), int(card[rb])
        new = n + t
        parent[ra] = parent[rb] = new
        card[new] = ca + cb
        touches[new] = ga | gb
        mems = members[ra] | members[rb]
        members[ra] = members[rb] = None
        members[new] = mems

        for gi in touches[new]:
            hit = counts[gi].pop(ra, 0) + counts[gi].pop(rb, 0)
            if hit:
                counts[gi][new] = hit
            dominant = max(counts[gi].values())
            dom_block = max(counts[gi], key=counts[gi].get)
            if first_impure[gi] is None and card[dom_block] > dominant:
                first_impure[gi] = t
            if completed[gi] is None and dominant == len(group_sets[gi]):
                completed[gi] = t

        rows.append(
            {
                "merge": t,
                "n_coarse": n - t - 1,
                "role": role,
                "score": float(scores[t]) if t < len(scores) else float("nan"),
                "card_a": ca,
                "card_b": cb,
                "ward_mass": ca * cb / (ca + cb),
                "max_card": max(ca, cb),
                "new_card": ca + cb,
            }
        )

    milestones = [
        {
            "group": gi,
            "size": len(group_sets[gi]),
            "first_impure": first_impure[gi],
            "completed": completed[gi],
            "exact": (
                completed[gi] is not None
                and (first_impure[gi] is None or first_impure[gi] >= completed[gi])
            ),
        }
        for gi in range(n_groups)
    ]
    return rows, milestones, first_boundary, n_internal_before_boundary


def summarize(rows, milestones, first_boundary, internal_before, n):
    by_role = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)

    def q(values, name):
        values = np.asarray(values, float)
        if not values.size:
            return {}
        return {
            f"{name}_mean": float(values.mean()),
            f"{name}_median": float(np.median(values)),
            f"{name}_p90": float(np.percentile(values, 90)),
        }

    out = {
        "first_boundary_merge": first_boundary,
        "internal_merges_before_first_boundary": internal_before,
        "n_merges": len(rows),
    }
    for role in ("internal", "boundary", "background"):
        sub = by_role.get(role, [])
        out[f"{role}_count"] = len(sub)
        out.update(
            {f"{role}_{k}": v for k, v in q([r["score"] for r in sub], "score").items()}
        )
        out.update(
            {
                f"{role}_{k}": v
                for k, v in q([r["ward_mass"] for r in sub], "mass").items()
            }
        )
        if sub:
            out[f"{role}_median_merge_index"] = float(
                np.median([r["merge"] for r in sub])
            )
    # score AUC: P(internal score < boundary score) over the whole hierarchy
    si = np.array([r["score"] for r in by_role.get("internal", [])], float)
    sb = np.array([r["score"] for r in by_role.get("boundary", [])], float)
    if si.size and sb.size:
        from scipy.stats import mannwhitneyu

        out["score_auc_internal_below_boundary"] = float(
            mannwhitneyu(si, sb, alternative="less").statistic / (si.size * sb.size)
        )
    exact = [m for m in milestones if m["exact"]]
    out["groups"] = len(milestones)
    out["groups_ever_exact"] = len(exact)
    out["median_completed_merge"] = (
        float(np.median([m["completed"] for m in milestones if m["completed"] is not None]))
        if any(m["completed"] is not None for m in milestones) else None
    )
    out["median_first_impure_merge"] = (
        float(np.median([m["first_impure"] for m in milestones if m["first_impure"] is not None]))
        if any(m["first_impure"] is not None for m in milestones) else None
    )
    out["groups_completed"] = sum(m["completed"] is not None for m in milestones)
    # growth profile: how fast the largest block grows
    cards = np.array([r["new_card"] for r in rows])
    running_max = np.maximum.accumulate(cards)
    for frac in (0.25, 0.5, 0.75):
        idx = int(frac * len(rows))
        out[f"largest_block_at_{int(frac * 100)}pct_merges"] = int(running_max[idx])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--num-nodes", type=int, default=3000)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", type=Path, default=Path("results2/merge_forensics"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)

    per_run, all_rows = [], []
    for seed in args.seeds:
        bundle, _learner, _res, bases = frozen_representation(
            seed, num_nodes=args.num_nodes, num_train=1, num_test=1
        )
        for graph in list(bundle.train_graphs) + list(bundle.test_graphs):
            group_sets = [set(g.nodes.tolist()) for g in graph.groups]
            for method in args.methods:
                coarsener = Coarsening(replace(BASE_COARSENING, method=method))
                result = coarsener.run(
                    graph, bases[graph.graph_id],
                    train_groups=graph.train_groups if graph in bundle.train_graphs else None,
                )
                rows, milestones, fb, ib = replay(
                    result.hierarchy.children,
                    int(graph.num_nodes),
                    group_sets,
                    result.hierarchy.scores,
                )
                summary = summarize(rows, milestones, fb, ib, int(graph.num_nodes))
                summary.update(seed=seed, graph=graph.graph_id, method=method,
                               constrained=int(result.hierarchy.constrained_merges))
                per_run.append(summary)
                for r in rows:
                    r.update(seed=seed, graph=graph.graph_id, method=method)
                all_rows.extend(rows)
                print(
                    f"seed {seed} {graph.graph_id:4s} {method:14s} "
                    f"internal-before-first-boundary={ib:4d} "
                    f"exact groups={summary['groups_ever_exact']:2d}/{summary['groups']} "
                    f"completed={summary['groups_completed']:2d} "
                    f"median complete@merge={summary['median_completed_merge']} "
                    f"median impure@merge={summary['median_first_impure_merge']} "
                    f"AUC(int<bnd)={summary.get('score_auc_internal_below_boundary', float('nan')):.3f} "
                    f"largest@50%={summary['largest_block_at_50pct_merges']}",
                    flush=True,
                )

    (args.out / "per_run.json").write_text(json.dumps(per_run, indent=1))
    np.save(args.out / "merge_rows.npy", np.array(all_rows, dtype=object), allow_pickle=True)

    print("\n=== pooled means ===")
    keys = [
        "internal_merges_before_first_boundary", "groups_ever_exact",
        "groups_completed", "median_completed_merge", "median_first_impure_merge",
        "score_auc_internal_below_boundary", "internal_score_median",
        "boundary_score_median", "background_score_median",
        "internal_median_merge_index", "boundary_median_merge_index",
        "background_median_merge_index",
        "largest_block_at_25pct_merges", "largest_block_at_50pct_merges",
    ]
    print(f"{'metric':44s} " + " ".join(f"{m:>16s}" for m in args.methods))
    for key in keys:
        cells = []
        for m in args.methods:
            vals = [r[key] for r in per_run if r["method"] == m and r.get(key) is not None]
            cells.append(f"{np.mean(vals):16.4g}" if vals else " " * 16)
        print(f"{key:44s} " + " ".join(cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
