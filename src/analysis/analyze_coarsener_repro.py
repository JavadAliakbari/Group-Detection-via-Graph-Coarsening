r"""Summaries for the coarsener comparison: own-cut, oracle, and matched budgets.

Every matched comparison reads the *same* level of each method's own hierarchy,
so a difference there is a difference between merge orders and cannot be a
stopping-rule artefact.  Three matched axes are reported:

``n_coarse``    the same number of supernodes
``reduction``   the same removed fraction (equal to the above when ``N`` matches)
``epsilon``     the same common Loukas Def. 2 constant -- for each method, the
                coarsest level whose exact-or-interpolated ``epsilon`` is still
                within the target
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

METHOD_ORDER = ("ward_tree", "raw_ward", "deflated_ward")
_METRICS = ("all_mean_f1", "all_mean_precision", "all_mean_recall", "all_detection_rate")


def _agg(values):
    values = np.asarray(values, float)
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0


def own_cut_table(rows, methods) -> dict:
    out = {}
    for scope in ("train", "test", "all"):
        for method in methods:
            sel = [
                r for r in rows
                if r["method"] == method and (scope == "all" or r["scope"] == scope)
            ]
            if not sel:
                continue
            entry = {"n_graphs": len(sel)}
            for key in (
                "stop_f1", "stop_precision", "stop_recall", "stop_detection_rate",
                "stop_n_coarse", "stop_epsilon",
                "oracle_f1", "oracle_precision", "oracle_recall",
                "oracle_detection_rate", "oracle_n_coarse", "oracle_epsilon",
                "pr_auc", "runtime_s", "completion_merges",
            ):
                m, s = _agg([r[key] for r in sel])
                entry[key] = m
                entry[key + "_std"] = s
            out[f"{scope}/{method}"] = entry
    return out


def _row_at_n(traj, n_target):
    """Row with ``n_coarse`` closest to (and not above) the target."""

    below = [r for r in traj if r["n_coarse"] <= n_target]
    return below[0] if below else traj[-1]


def _row_at_epsilon(traj, eps_target):
    """Coarsest row whose common epsilon is still within the target."""

    within = [r for r in traj if r["epsilon"] <= eps_target + 1e-12]
    return within[-1] if within else traj[0]


def matched_table(trajectories, methods, *, n_grid, eps_grid) -> dict:
    keys = sorted(trajectories)
    graphs = sorted({(s, g) for (s, m, g) in keys})
    out = {"n_coarse": {}, "epsilon": {}}

    for target in n_grid:
        for method in methods:
            picked = [
                _row_at_n(trajectories[(s, method, g)], target)
                for (s, g) in graphs
                if (s, method, g) in trajectories
            ]
            if not picked:
                continue
            entry = {"n_graphs": len(picked)}
            for key in _METRICS + ("n_coarse", "epsilon", "reduction"):
                m, sd = _agg([r[key] for r in picked])
                entry[key], entry[key + "_std"] = m, sd
            out["n_coarse"].setdefault(str(target), {})[method] = entry

    for target in eps_grid:
        for method in methods:
            picked = [
                _row_at_epsilon(trajectories[(s, method, g)], target)
                for (s, g) in graphs
                if (s, method, g) in trajectories
            ]
            if not picked:
                continue
            entry = {"n_graphs": len(picked)}
            for key in _METRICS + ("n_coarse", "epsilon", "reduction"):
                m, sd = _agg([r[key] for r in picked])
                entry[key], entry[key + "_std"] = m, sd
            out["epsilon"].setdefault(f"{target:g}", {})[method] = entry
    return out


def paired_test(trajectories, methods, reference, *, n_grid):
    """Paired sign counts + Wilcoxon p at every matched level, vs ``reference``."""

    from scipy.stats import wilcoxon

    keys = sorted(trajectories)
    graphs = sorted({(s, g) for (s, m, g) in keys})
    out = {}
    for target in n_grid:
        ref = np.array([
            _row_at_n(trajectories[(s, reference, g)], target)["all_mean_f1"]
            for (s, g) in graphs
        ])
        for method in methods:
            if method == reference:
                continue
            other = np.array([
                _row_at_n(trajectories[(s, method, g)], target)["all_mean_f1"]
                for (s, g) in graphs
            ])
            diff = ref - other
            try:
                p = float(wilcoxon(ref, other).pvalue) if np.any(diff) else 1.0
            except ValueError:
                p = 1.0
            out.setdefault(str(target), {})[f"{reference}_minus_{method}"] = {
                "mean_diff": float(diff.mean()),
                "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "p_wilcoxon": p,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("results2/coarsener_repro"))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER))
    parser.add_argument("--reference", default="ward_tree")
    args = parser.parse_args()

    rows = json.loads((args.dir / "rows.json").read_text())
    with open(args.dir / "trajectories.pkl", "rb") as fh:
        trajectories = pickle.load(fh)
    methods = [m for m in args.methods if any(r["method"] == m for r in rows)]

    n_grid = [2500, 2000, 1500, 1200, 1000, 800, 600, 400, 200, 100]
    eps_grid = [0.6, 0.8, 0.9, 0.95, 0.99, 1.0, 1.02, 1.05]

    own = own_cut_table(rows, methods)
    matched = matched_table(trajectories, methods, n_grid=n_grid, eps_grid=eps_grid)
    paired = paired_test(trajectories, methods, args.reference, n_grid=n_grid)

    print("=== own cut (stopping rule) and oracle, pooled over graphs ===")
    hdr = f"{'scope/method':28s} {'stopF1':>16s} {'stopDet':>8s} {'stop n':>7s} " \
          f"{'oracleF1':>16s} {'oracDet':>8s} {'orac n':>7s} {'PR-AUC':>16s}"
    print(hdr)
    for scope in ("train", "test", "all"):
        for method in methods:
            key = f"{scope}/{method}"
            if key not in own:
                continue
            e = own[key]
            print(
                f"{key:28s} {e['stop_f1']:.3f}+-{e['stop_f1_std']:.3f}   "
                f"{e['stop_detection_rate']:7.3f} {e['stop_n_coarse']:7.0f} "
                f"{e['oracle_f1']:.3f}+-{e['oracle_f1_std']:.3f}   "
                f"{e['oracle_detection_rate']:7.3f} {e['oracle_n_coarse']:7.0f} "
                f"{e['pr_auc']:.3f}+-{e['pr_auc_std']:.3f}"
            )

    print("\n=== matched number of supernodes (same level of each own hierarchy) ===")
    print(f"{'n_coarse':>9s} " + " ".join(f"{m:>22s}" for m in methods))
    for target in n_grid:
        cells = []
        for m in methods:
            e = matched["n_coarse"].get(str(target), {}).get(m)
            cells.append("n/a".rjust(22) if e is None
                         else f"{e['all_mean_f1']:.3f}+-{e['all_mean_f1_std']:.3f}".rjust(22))
        print(f"{target:9d} " + " ".join(cells))

    print("\n=== matched number of supernodes: detection rate ===")
    print(f"{'n_coarse':>9s} " + " ".join(f"{m:>22s}" for m in methods))
    for target in n_grid:
        cells = []
        for m in methods:
            e = matched["n_coarse"].get(str(target), {}).get(m)
            cells.append("n/a".rjust(22) if e is None
                         else f"{e['all_detection_rate']:.3f}".rjust(22))
        print(f"{target:9d} " + " ".join(cells))

    print("\n=== matched common epsilon (coarsest level within the target) ===")
    print(f"{'epsilon':>9s} " + " ".join(f"{m:>26s}" for m in methods))
    for target in eps_grid:
        cells = []
        for m in methods:
            e = matched["epsilon"].get(f"{target:g}", {}).get(m)
            cells.append("n/a".rjust(26) if e is None
                         else f"F1 {e['all_mean_f1']:.3f} @ n={e['n_coarse']:6.0f}".rjust(26))
        print(f"{target:9g} " + " ".join(cells))

    print(f"\n=== paired F1 difference vs {args.reference} at matched n_coarse ===")
    for target in n_grid:
        for name, d in paired.get(str(target), {}).items():
            print(
                f"n={target:5d} {name:34s} delta={d['mean_diff']:+.4f} "
                f"W/L/T={d['wins']}/{d['losses']}/{d['ties']} p={d['p_wilcoxon']:.2e}"
            )

    out = {"own_cut": own, "matched": matched, "paired": paired}
    (args.dir / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.dir / 'summary.json'}")


if __name__ == "__main__":
    main()
