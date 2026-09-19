r"""Does shaping the level ``ell`` beat shaping only ``Gamma``?

Trains the collective bank with the level-contrastive term of
:mod:`src.level_contrast` -- ``s^0`` down inside a group, up across its boundary,
and optionally down on random node pairs -- then coarsens with **raw Ward**
(whose score is exactly the quantity the term optimizes) and with
**deflated dual Ward**, and reports every arm at ``epsilon*``, the sweep level
where mean F1 peaks.

Three questions:

1. does the level term lift raw Ward at all?
2. does the **negative** term (random nodes pulled to a similar level) help or
   hurt?  It is not arbitrary -- the idealized target ``Z* = M_tau^{-1}V*`` has a
   level that is flat off the groups -- but it can also flatten the boundary jump
   along with the host, so the sign of its effect is an empirical question.
3. does a level-trained raw Ward close the gap to deflated dual Ward, which
   :mod:`src.diag_ward_score_gap` showed wins by ordering internal below boundary
   merges more often (AUC 0.71 -> 0.82)?

The training-time AUC on sampled pairs is logged next to the detection numbers,
so it is visible whether an arm that failed to help failed because the objective
did not move, or because moving it did not matter.

Example::

    python -m src.run_level_contrast --seeds 1,2,3 --epochs 600 \
        --level-weights 0,2 --negative-weights 0,0.25,0.5,1
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch

from src.collective_detector import CollectiveBankDetector, DetectorConfig
from src.level_contrast import build_level_pairs, level_scores
from src.run_collective_bank_detection import _basis_stack, _filtered_bank, theta_degree
from src.run_synthetic_modular import _coarsen_and_score, _fmt_table, _load_graph
from src.utils.utils import LOGGER, now


def _separation_auc(det, data, patterns, tau, seed):
    """``P(s^0_internal < s^0_boundary)`` under the FROZEN filter.

    The statistic the level term optimizes, measured wherever we like -- on the
    gangs the loss saw, on held-out gangs of the same graph, or on a fresh graph.
    The spread between those three is the generalization gap; closing it is the
    whole point of training on several graphs.
    """

    geo = det.geometry(data)
    prop = _basis_stack(
        geo.prop, data.X, theta_degree(det.theta_), det.config.basis, tau, geometry=geo
    )
    Z = _filtered_bank(prop, det.theta_)
    mz = geo.m_apply(Z, tau)
    pr = build_level_pairs(data.adjacency, patterns, tau=tau, budget=4096, seed=seed)
    if pr["counts"]["int"] == 0 or pr["counts"]["bnd"] == 0:
        return float("nan")
    with torch.no_grad():
        si = level_scores(Z, mz, pr["d_tilde"], pr["int_u"], pr["int_v"], pr["int_c"])
        sb = level_scores(Z, mz, pr["d_tilde"], pr["bnd_u"], pr["bnd_v"], pr["bnd_c"])
    return float((si[:, None] < sb[None, :]).double().mean())


def _arms(args):
    """The (level_weight, form, negative_weight, neg_kind) grid, deduplicated."""

    lw = [float(x) for x in args.level_weights.split(",") if x.strip()]
    nw = [float(x) for x in args.negative_weights.split(",") if x.strip()]
    fm = [x.strip() for x in args.forms.split(",") if x.strip()]
    nk = [x.strip() for x in args.neg_kinds.split(",") if x.strip()]
    out, seen = [], set()
    for w in lw:
        if w == 0.0:  # the level term is off: the other three knobs are inert
            key = ("base",)
            if key not in seen:
                seen.add(key)
                out.append({"level_weight": 0.0, "form": fm[0], "neg": 0.0,
                            "neg_kind": nk[0], "name": "baseline (Gamma only)"})
            continue
        for f in fm:
            for k in nk:
                for b in nw:
                    if b == 0.0 and k != nk[0]:
                        continue  # neg_kind is inert when the negative term is off
                    key = (w, f, k, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    nm = f"level/{f} w={w:g}" + (
                        f" neg={b:g}({k})" if b else " neg=0"
                    )
                    out.append({"level_weight": w, "form": f, "neg": b,
                                "neg_kind": k, "name": nm})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument(
        "--num-graphs-list",
        type=str,
        default="1",
        help="comma list of TRAINING-graph counts to compare.  More graphs is the "
        "direct attack on the generalization gap: the filter must separate gangs "
        "on several graphs at once, so graph-specific level structure cannot be "
        "memorized.  The evaluation set is held fixed across the list.",
    )
    ap.add_argument("--day-aggregate", type=str, default="sample")
    ap.add_argument("--transfer-graphs", type=int, default=3)
    ap.add_argument("--num-nodes", type=int, default=1500)
    ap.add_argument("--num-motifs", type=int, default=12)
    ap.add_argument("--motif-size-min", type=int, default=7)
    ap.add_argument("--motif-size-max", type=int, default=20)
    ap.add_argument("--motif-types", type=str, default="random")
    ap.add_argument("--motif-density", type=float, default=0.4)
    ap.add_argument("--motif-conductance", type=float, default=-1.0)
    ap.add_argument("--avg-degree", type=float, default=6.0)
    ap.add_argument("--feature-dim", type=int, default=32)
    ap.add_argument("--feat-shared", type=float, default=0.0)
    ap.add_argument("--feat-signature", type=float, default=0.0)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--max-train-gangs", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--conf-weight", type=float, default=50.0)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--ward-stop", type=str, default="f1")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--compare-num-cuts", type=int, default=120)
    ap.add_argument(
        "--coarseners", type=str, default="raw-ward,deflated-dual-ward,ward-tree"
    )
    ap.add_argument("--level-weights", type=str, default="0,2")
    ap.add_argument("--negative-weights", type=str, default="0,0.25,0.5,1")
    ap.add_argument("--forms", type=str, default="margin,rank")
    ap.add_argument("--neg-kinds", type=str, default="pairs,edges")
    ap.add_argument("--level-budget", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=Path(f"results/level_contrast/{now}/"))
    args = ap.parse_args()
    args.motif_types = [t.strip() for t in args.motif_types.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    coarseners = [x.strip() for x in args.coarseners.split(",") if x.strip()]
    arms = _arms(args)

    LOGGER.info("=" * 100)
    LOGGER.info("LEVEL-CONTRASTIVE TRAINING vs Gamma-ONLY   (all numbers at eps*)")
    LOGGER.info(
        f"  arms       : {len(arms)}\n"
        f"  seeds      : {seeds}  ({args.transfer_graphs} transfer graphs each)\n"
        f"  coarseners : {coarseners}\n"
        f"  graph      : N={args.num_nodes:,}  {args.num_motifs} motifs "
        f"sizes {args.motif_size_min}-{args.motif_size_max}  deg {args.avg_degree}"
    )
    LOGGER.info("=" * 100)

    rows, fits = [], []
    n_graphs_list = [int(x) for x in args.num_graphs_list.split(",") if x.strip()]
    for seed in seeds:
        torch.manual_seed(seed)
        # the training pool: graph 0 is the reported one, the rest are extra
        # training graphs.  Built once so every arm sees the SAME graphs.
        pool = [_load_graph(seed + 100 * g, args) for g in range(max(n_graphs_list))]
        data, tr, te, allp = pool[0]
        transfer = []
        for k in range(args.transfer_graphs):
            t_data, _a, _b, t_all = _load_graph(seed + 9000 + 100 * k, args)
            transfer.append((f"T{k}", t_data, t_all))
        # held FIXED across the n_graphs axis, so more training graphs cannot be
        # credited with an easier evaluation
        eval_specs = ([("train:test", data, te)] if te else []) + transfer
        LOGGER.info(
            f"\n{'=' * 100}\n[seed {seed}]  N={data.num_nodes:,}  "
            f"{len(allp)} motifs (train {len(tr)} / test {len(te)})  "
            f"{len(eval_specs)} evaluation graphs, training pool "
            f"{max(n_graphs_list)} graphs\n{'=' * 100}"
        )

        for n_graphs, arm in [(g, a) for g in n_graphs_list for a in arms]:
            cfg = DetectorConfig(
                degree=args.degree,
                tau=args.tau,
                epochs=args.epochs,
                heads=args.heads,
                conf_weight=args.conf_weight,
                capture_objective="lambda_min",
                coarsening_method="raw-ward",
                coarsening_laplacian="symmetric",
                ward_stop=args.ward_stop,
                epsilon=args.epsilon,
                ward_num_cuts=args.compare_num_cuts,
                threshold=args.threshold,
                seed=seed,
                day_aggregate=args.day_aggregate,
                level_weight=arm["level_weight"],
                level_form=arm["form"],
                level_negative_weight=arm["neg"],
                level_neg_kind=arm["neg_kind"],
                level_budget=args.level_budget,
            )
            det = CollectiveBankDetector(cfg)
            t0 = time.time()
            det.fit(
                [
                    (f"G{g}", pool[g][0], pool[g][1], pool[g][2])
                    for g in range(n_graphs)
                ]
            )
            secs = time.time() - t0
            fi = det.fit_info_
            li, lf = fi.get("level_initial"), fi.get("level_final")
            cap = det.capture(data, allp)
            auc_tr = _separation_auc(det, data, tr, args.tau, seed)
            auc_ho = _separation_auc(det, data, te, args.tau, seed + 1) if te else float("nan")
            auc_tf = float(
                np.nanmean(
                    [
                        _separation_auc(det, d, p, args.tau, seed + 7 * i)
                        for i, (_l, d, p) in enumerate(transfer)
                    ]
                )
            )
            fits.append(
                {
                    "seed": seed, "arm": arm["name"], "n_graphs": n_graphs,
                    "auc_train_gangs": auc_tr,
                    "auc_heldout_gangs": auc_ho,
                    "auc_transfer_graphs": auc_tf, **{
                        k: arm[k] for k in ("level_weight", "form", "neg", "neg_kind")
                    },
                    "fit_seconds": secs,
                    "lambda_min": float(fi.get("objective", float("nan"))),
                    "mean_capture": cap["mean_capture"],
                    "auc_init": (li or {}).get("auc", float("nan")),
                    "auc_final": (lf or {}).get("auc", float("nan")),
                    "s_int": (lf or {}).get("int", float("nan")),
                    "s_bnd": (lf or {}).get("bnd", float("nan")),
                    "s_neg": (lf or {}).get("neg", float("nan")),
                }
            )
            LOGGER.info(
                f"  [G={n_graphs}] {arm['name']:<30} "
                f"lam_min={fits[-1]['lambda_min']:.4f} "
                f"C={cap['mean_capture']:.4f}"
                + (
                    f"   level AUC {fits[-1]['auc_init']:.3f}->"
                    f"{fits[-1]['auc_final']:.3f}  "
                    f"s0 int {fits[-1]['s_int']:.4f} bnd {fits[-1]['s_bnd']:.4f} "
                    f"neg {fits[-1]['s_neg']:.4f}"
                    if arm["level_weight"]
                    else "   (level term off)"
                )
                + f"   AUC train {auc_tr:.3f} / heldout {auc_ho:.3f} / "
                f"transfer {auc_tf:.3f}"
                + f"   ({secs:.0f}s)"
            )

            bases = [(l, d, p, det.target_subspace(d, p)) for l, d, p in eval_specs]
            for method in coarseners:
                for lbl, d, p, b in bases:
                    _curve, r = _coarsen_and_score(det, d, b, p, method, args)
                    r.update(
                        {
                            "seed": seed, "arm": arm["name"], "n_graphs": n_graphs,
                            "coarsener": method, "graph": lbl,
                            **{k: arm[k] for k in
                               ("level_weight", "form", "neg", "neg_kind")},
                        }
                    )
                    rows.append(r)
                sub = [
                    x for x in rows
                    if x["arm"] == arm["name"] and x["coarsener"] == method
                    and x["seed"] == seed and x["n_graphs"] == n_graphs
                ]
                LOGGER.info(
                    f"      {method:<20} F1*={np.mean([x['f1_star'] for x in sub]):.4f}"
                    f"  det*={np.mean([x['detection_star'] for x in sub]):.1%}"
                    f"  eps*={np.mean([x['epsilon_star'] for x in sub]):.3f}"
                    f"  n*={np.mean([x['n_coarse_star'] for x in sub]):.0f}"
                )

    D = pd.DataFrame(rows)
    F = pd.DataFrame(fits)
    D.to_csv(args.out / "level_rows.csv", index=False)
    F.to_csv(args.out / "level_fits.csv", index=False)

    LOGGER.info("\n" + "=" * 100)
    LOGGER.info(
        "[1] DOES THE SEPARATION GENERALIZE?  AUC = P(s^0_int < s^0_bnd) under the "
        "frozen filter"
    )
    LOGGER.info(
        "  train gangs = the pairs the loss saw; heldout = unseen gangs on the SAME "
        "graph;\n  transfer = unseen gangs on FRESH graphs.  The train->transfer drop "
        "is the gap\n  more training graphs are meant to close."
    )
    LOGGER.info("=" * 100)
    acols = ["auc_train_gangs", "auc_heldout_gangs", "auc_transfer_graphs",
             "lambda_min", "mean_capture"]
    g = F.groupby(["n_graphs", "arm"])[acols].mean()
    hdr = f"  {'G':>3} {'arm':<32}" + "".join(f"{c.replace('auc_', ''):>20}" for c in acols)
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for (ng, a), r in g.iterrows():
        LOGGER.info(
            f"  {ng:>3} {a:<32}" + "".join(f"{r[c]:>20.4f}" for c in acols)
        )

    LOGGER.info("\n" + "=" * 100)
    LOGGER.info("[2] DETECTION AT eps*  (mean over seeds and evaluation graphs)")
    LOGGER.info("=" * 100)
    D["cell"] = "G=" + D.n_graphs.astype(str) + " " + D.arm
    F["cell"] = "G=" + F.n_graphs.astype(str) + " " + F.arm
    _fmt_table(D, "cell", "coarsener", "f1_star", title="mean F1*")
    _fmt_table(D, "cell", "coarsener", "detection_star", "{:.1%}", "detection at eps*")

    LOGGER.info("\n" + "=" * 100)
    LOGGER.info("[3] PAIRED EFFECTS (by seed x evaluation graph x coarsener)")
    LOGGER.info("=" * 100)
    base = "baseline (Gamma only)"
    piv = D.pivot_table(
        index=["seed", "graph", "coarsener"], columns="cell", values="f1_star"
    )
    ngs = sorted(set(D.n_graphs))
    for method in coarseners:
        sub = piv.xs(method, level="coarsener")
        LOGGER.info(f"\n  ---------- {method} ----------")
        # (a) more graphs, holding the arm fixed
        for a in sorted(set(D.arm)):
            for ng in ngs[1:]:
                c0, c1 = f"G={ngs[0]} {a}", f"G={ng} {a}"
                if c0 in sub and c1 in sub:
                    d = (sub[c1] - sub[c0]).dropna()
                    LOGGER.info(
                        f"    G={ngs[0]}->{ng}  {a:<30}{d.mean():+.4f}"
                        f"   wins {int((d > 0).sum())}/{len(d)}"
                        f"   ({sub[c0].mean():.4f} -> {sub[c1].mean():.4f})"
                    )
        # (b) the level term, holding the graph count fixed
        for ng in ngs:
            cb = f"G={ng} {base}"
            if cb not in sub:
                continue
            for a in sorted(set(D.arm) - {base}):
                ca = f"G={ng} {a}"
                if ca not in sub:
                    continue
                d = (sub[ca] - sub[cb]).dropna()
                LOGGER.info(
                    f"    level@G={ng}  {a:<30}{d.mean():+.4f}"
                    f"   wins {int((d > 0).sum())}/{len(d)}"
                )

    LOGGER.info(f"\n  rows -> {args.out / 'level_rows.csv'}")
    LOGGER.info(f"  fits -> {args.out / 'level_fits.csv'}")


if __name__ == "__main__":
    main()
