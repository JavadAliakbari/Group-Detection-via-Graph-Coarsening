r"""Step 3: verify the deflated-Ward implementation against its own definitions.

Two independent audits, both on graphs small enough that everything can be
recomputed from scratch (:mod:`src.deflated_reference`):

``selection``
    replay the production hierarchy merge by merge.  At every committed merge,
    recompute the exact deflated score of the committed pair **and of every
    eligible adjacent pair** at that partition, and report the committed pair's
    rank.  A rank of 1 means the implementation picked the true minimum-score
    pair; anything else is a search approximation, and the audit says which knob
    it came from by re-running with that knob disabled.

``certificate``
    check ``(eps_Pn^Q)^2 <= sum_{k>n} s_k <= q (eps_Pn^Q)^2`` using the
    *unnormalized* cumulative theoretical score, the exact ``eps_Q`` from a
    from-scratch harmonic projector, and the effective target rank ``q``.  The
    normalized ``score_sum`` axis of the pipeline is never used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from analysis.coarsener_ablation import frozen_representation
from src.deflated_coarsen import deflated_coarsen
from analysis.deflated_reference import (
    CoarseExactDeflation,
    ExactDeflation,
    exact_greedy_hierarchy,
)
from src.pipeline.coarsening import _labels_after, _to_scipy


def _block_of(children, n, t):
    """Partition labels after ``t`` merges, plus tree-id -> label map."""

    return _labels_after(children, n, t)


def _tree_labels(children, n, t):
    """Map every tree node id alive after ``t`` merges to its partition label."""

    parent = np.arange(n + int(children.shape[0]), dtype=np.int64)
    for k in range(t):
        parent[int(children[k, 0])] = n + k
        parent[int(children[k, 1])] = n + k

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x

    labels = _labels_after(children, n, t)
    out = {}
    for v in range(n):
        out[root(v)] = int(labels[v])
    return labels, out


def group_role(members_a, members_b, group_sets):
    """``internal`` / ``boundary`` / ``background`` for one committed merge."""

    ga = {i for i, s in enumerate(group_sets) if s & members_a}
    gb = {i for i, s in enumerate(group_sets) if s & members_b}
    if not ga and not gb:
        return "background"
    if ga & gb:
        return "internal"
    return "boundary"


def audit_selection(
    W, Z, tau, *, max_merges, coarsen_kwargs, group_sets, label="", route="fine",
    merge_indices=None,
):
    """Replay a production hierarchy and rank every committed merge exactly."""

    n = W.shape[0]
    result = deflated_coarsen(
        W, Z, tau, rule="dual-ward", build_full_tree=True,
        track_euclidean=False, record_curve=False, **coarsen_kwargs,
    )
    children = np.asarray(result.children_, np.int64).reshape(-1, 2)
    ex = (ExactDeflation if route == "fine" else CoarseExactDeflation)(W, Z, tau)
    constrained = int(children.shape[0])

    if merge_indices is None:
        merge_indices = range(min(max_merges, constrained))
    rows = []
    for t in merge_indices:
        if t >= constrained:
            continue
        labels, tree_to_label = _tree_labels(children, n, t)
        a_id, b_id = int(children[t, 0]), int(children[t, 1])
        la, lb = tree_to_label[a_id], tree_to_label[b_id]
        table = ex.all_scores(labels)
        key = (min(la, lb), max(la, lb))
        adjacent = key in table
        exact = ex.score(labels, *key) if adjacent else None
        values = np.array(sorted(table.values()))
        # ties (and the ~1e-13 disagreement between the coarse and the fine
        # route) must not be read as suboptimality, so the rank counts only
        # candidates that are *strictly* cheaper beyond a relative tolerance
        rank = -1
        if adjacent:
            tol = 1e-9 * max(abs(exact["score"]), 1.0)
            rank = int((values < exact["score"] - tol).sum()) + 1
        rec = result.merge_records_[t]
        members_a = set(np.flatnonzero(labels == la).tolist())
        members_b = set(np.flatnonzero(labels == lb).tolist())
        rows.append(
            {
                "label": label,
                "merge": t,
                "n_blocks_before": int(labels.max()) + 1,
                "pair": key,
                "adjacent": bool(adjacent),
                "exact_score": None if exact is None else float(exact["score"]),
                "exact_raw_score": None if exact is None else float(exact["raw_score"]),
                "exact_eta0": None if exact is None else float(exact["eta0"]),
                "stored_score": float(rec["score"]),
                "stored_a_sq": float(rec["a_sq"]),
                "n_candidates": len(table),
                "rank": rank,
                "best_available": float(values[0]) if values.size else float("nan"),
                "regret": (
                    float(exact["score"] - values[0]) if adjacent else float("nan")
                ),
                "ball": int(rec["ball"]),
                "role": group_role(members_a, members_b, group_sets),
                "is_completion": False,
            }
        )
    return result, rows


def audit_certificate(W, Z, tau, *, coarsen_kwargs, levels=None):
    """``eps_Q^2 <= S_n <= q eps_Q^2`` with exact quantities only."""

    n = W.shape[0]
    result = deflated_coarsen(
        W, Z, tau, rule="dual-ward", build_full_tree=True,
        track_euclidean=False, record_curve=False, **coarsen_kwargs,
    )
    children = np.asarray(result.children_, np.int64).reshape(-1, 2)
    ex = ExactDeflation(W, Z, tau)  # the certificate is always the fine route
    q = int(ex.q)
    constrained = len(result.merge_records_)

    # the theoretical cumulative score, recomputed exactly merge by merge
    exact_scores = np.zeros(constrained)
    for t in range(constrained):
        labels, tree_to_label = _tree_labels(children, n, t)
        a_id, b_id = int(children[t, 0]), int(children[t, 1])
        la, lb = tree_to_label[a_id], tree_to_label[b_id]
        exact_scores[t] = ex.score(labels, min(la, lb), max(la, lb))["score"]

    stored = np.array([r["a_sq"] for r in result.merge_records_])
    cum_exact = np.concatenate([[0.0], np.cumsum(exact_scores)])
    cum_stored = np.concatenate([[0.0], np.cumsum(stored)])

    if levels is None:
        levels = sorted({1, 2, 3, 5, 10, 20, 50, 100, constrained // 2, constrained})
    rows = []
    for t in levels:
        if not 1 <= t <= constrained:
            continue
        labels = _labels_after(children, n, t)
        eps = ex.epsilon_q(labels)
        eps_sq = eps * eps
        rows.append(
            {
                "merge_index": int(t),
                "n_coarse": int(n - t),
                "q": q,
                "epsilon_q": eps,
                "epsilon_q_squared": eps_sq,
                "S_n_exact": float(cum_exact[t]),
                "S_n_stored": float(cum_stored[t]),
                "trace_H": float(np.trace(ex.harmonic_H(labels))),
                "lower_gap_exact": float(cum_exact[t] - eps_sq),
                "upper_gap_exact": float(q * eps_sq - cum_exact[t]),
                "lower_gap_stored": float(cum_stored[t] - eps_sq),
                "upper_gap_stored": float(q * eps_sq - cum_stored[t]),
            }
        )
    return {
        "q": q,
        "constrained_merges": constrained,
        "max_abs_stored_vs_exact": float(np.max(np.abs(stored - exact_scores))),
        "mean_abs_stored_vs_exact": float(np.mean(np.abs(stored - exact_scores))),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=150)
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument("--group-size", type=int, nargs="+", default=[5, 9])
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--route", choices=("fine", "coarse"), default="fine")
    parser.add_argument(
        "--merge-stride",
        type=int,
        default=0,
        help="audit every k-th committed merge across the whole hierarchy "
        "instead of the first --max-merges consecutive ones",
    )
    parser.add_argument(
        "--settings", nargs="+", default=None, help="subset of the ablation settings"
    )
    parser.add_argument("--skip-certificate", action="store_true")
    parser.add_argument("--skip-exhaustive", action="store_true")
    parser.add_argument("--max-merges", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("results2/deflated_audit"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)

    bundle, _learner, _res, bases = frozen_representation(
        args.seed,
        num_nodes=args.num_nodes,
        num_train=1,
        num_test=0,
        num_groups=args.num_groups,
        group_size=list(args.group_size),
        num_heads=args.num_heads,
    )
    graph = bundle.train_graphs[0]
    W = _to_scipy(graph.adjacency)
    Z = bases[graph.graph_id].detach().cpu().numpy()
    tau = 0.5
    group_sets = [set(g.nodes.tolist()) for g in graph.groups]

    #: the production defaults, then one approximation removed at a time
    settings = {
        "production": dict(
            hops=2, max_ball=32, max_rescore=8, fanout=32, solve="local",
            commit_solve="local",
        ),
        "exact_solve": dict(
            hops=2, max_ball=32, max_rescore=8, fanout=32, solve="exact",
            commit_solve="exact",
        ),
        "no_fanout": dict(
            hops=2, max_ball=32, max_rescore=8, fanout=0, solve="local",
            commit_solve="local",
        ),
        "no_lazy": dict(
            hops=2, max_ball=32, max_rescore=10**9, fanout=0, solve="local",
            commit_solve="local",
        ),
        "wide_ball": dict(
            hops=6, max_ball=10**6, max_rescore=10**9, fanout=0, solve="local",
            commit_solve="local",
        ),
        "certified": dict(
            hops=2, max_ball=32, max_rescore=4096, fanout=32, solve="local",
            commit_solve="local", queue_key="certified",
        ),
        "tight_preset": dict(
            hops=3, max_ball=64, max_rescore=4096, fanout=32, solve="local",
            commit_solve="local", queue_key="certified",
        ),
        "strict_exact": dict(
            hops=2, max_ball=32, max_rescore=8, fanout=0, solve="exact",
            commit_solve="exact", strict=True,
        ),
    }

    if args.settings:
        settings = {k: v for k, v in settings.items() if k in args.settings}
    merge_indices = (
        list(range(0, int(graph.num_nodes) - 2, args.merge_stride))
        if args.merge_stride
        else None
    )
    print(f"graph {graph.graph_id}: n={graph.num_nodes}, target width {Z.shape[1]}")
    all_rows, summary = [], {}
    for name, kwargs in settings.items():
        _result, rows = audit_selection(
            W, Z, tau, max_merges=args.max_merges, coarsen_kwargs=kwargs,
            group_sets=group_sets, label=name, route=args.route,
            merge_indices=merge_indices,
        )
        all_rows.extend(rows)
        ranks = np.array([r["rank"] for r in rows])
        regret = np.array([r["regret"] for r in rows])
        err = np.array([abs(r["stored_a_sq"] - (r["exact_score"] or np.nan)) for r in rows])
        summary[name] = {
            "merges_audited": len(rows),
            "optimal_selections": int((ranks == 1).sum()),
            "optimal_fraction": float((ranks == 1).mean()),
            "median_rank": float(np.median(ranks)),
            "max_rank": int(ranks.max()),
            "mean_regret": float(np.nanmean(regret)),
            "max_regret": float(np.nanmax(regret)),
            "non_adjacent": int((ranks < 0).sum()),
            "max_stored_vs_exact_score": float(np.nanmax(err)),
        }
        s = summary[name]
        print(
            f"{name:14s} optimal {s['optimal_selections']:3d}/{s['merges_audited']:3d} "
            f"({s['optimal_fraction']:.1%})  median rank {s['median_rank']:5.1f}  "
            f"max rank {s['max_rank']:4d}  mean regret {s['mean_regret']:.3e}  "
            f"|stored-exact| <= {s['max_stored_vs_exact_score']:.2e}",
            flush=True,
        )

    if not args.skip_exhaustive:
        print("\n--- exhaustive exact greedy (reference) ---")
        _hist, recs = exact_greedy_hierarchy(W, Z, tau, max_merges=args.max_merges)
        print(f"  {len(recs)} merges, mean score {np.mean([r['score'] for r in recs]):.4e}")

    print("\n--- cumulative-score certificate, exact quantities ---")
    certs = {}
    wanted = [n for n in ("production", "exact_solve", "tight_preset") if n in settings]
    for name in ([] if args.skip_certificate else wanted):
        cert = audit_certificate(W, Z, tau, coarsen_kwargs=settings[name])
        certs[name] = cert
        print(f"{name}: q={cert['q']}, |stored - exact score| max "
              f"{cert['max_abs_stored_vs_exact']:.2e}")
        for row in cert["rows"]:
            ok_lo = row["lower_gap_exact"] >= -1e-9
            ok_hi = row["upper_gap_exact"] >= -1e-9
            print(
                f"   n={row['n_coarse']:5d}  eps_Q^2={row['epsilon_q_squared']:.6g}  "
                f"S_n={row['S_n_exact']:.6g}  q*eps^2={row['q'] * row['epsilon_q_squared']:.6g}  "
                f"tr H={row['trace_H']:.6g}  lower {'OK ' if ok_lo else 'FAIL'} "
                f"upper {'OK ' if ok_hi else 'FAIL'}"
            )

    (args.out / "selection_rows.json").write_text(json.dumps(all_rows, indent=1))
    (args.out / "selection_summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "certificate.json").write_text(json.dumps(certs, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
