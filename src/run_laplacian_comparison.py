"""Symmetric vs combinatorial screened geometry, on the same Elliptic++ window.

The geometry (``--coarsening-laplacian``) is not a coarsener-only knob: it fixes
the metric ``M_tau`` the filter bank is trained in, the group indicator ``v_S``
that capture and confusability are fractions of, the operator the Chebyshev bank
propagates on, and the metric the RSA distortion is measured in (see
:mod:`src.screened_geometry`).  This driver runs :mod:`src.run_elliptic_modular`
once per cell of a small factorial design and tabulates the detection metrics,
which are the only numbers that are directly comparable across geometries --
capture, confusability and epsilon are all fractions *of the geometry's own
norm*, so a cross-geometry difference in them is partly a change of units.

The cells separate the three things that move together when the convention
flips::

    symmetric                       baseline
    combinatorial                   metric + indicator + propagation (the paper)
    combinatorial, prop=a_hat       metric + indicator only
    combinatorial, tau matched      as the paper, but screened like-for-like
                                    (tau * lambda_max/2, since M_tau = L + tau I
                                    screens relative to the Laplacian's scale)

and each is run under both Ward stop rules, because the RSA epsilon budget is
itself geometry-dependent: block averaging distorts ``D - W`` far more than
``I - A_hat`` on a heavy-tailed graph, so a fixed ``--epsilon`` can stop the
combinatorial run before it merges anything.  ``--ward-stop f1`` removes that
confound by letting each geometry pick its own best cut.

Run::

    conda activate FedStruct
    python -m src.run_laplacian_comparison --train-days 24-26 --epochs 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _cells(tau: float, tau_matched: float) -> list:
    """The factorial design: ``(tag, extra CLI args)`` per cell."""

    return [
        ("symmetric", ["--coarsening-laplacian", "symmetric", "--tau", str(tau)]),
        (
            "combinatorial",
            ["--coarsening-laplacian", "combinatorial", "--tau", str(tau)],
        ),
        (
            "combinatorial/prop=a_hat",
            [
                "--coarsening-laplacian", "combinatorial",
                "--propagation", "a_hat",
                "--tau", str(tau),
            ],
        ),
        (
            "combinatorial/tau-matched",
            [
                "--coarsening-laplacian", "combinatorial",
                "--tau", f"{tau_matched:.6g}",
            ],
        ),
    ]


def _lambda_max(data_dir: Path, lo: int, hi: int) -> float:
    """``lambda_max(D - W)`` of the evaluated window -- sets the matched ``tau``."""

    import numpy as np
    import scipy.sparse as sp
    import torch

    from src.run_elliptic_gang_conductance import build_graph
    from src.screened_geometry import _lambda_max_combinatorial

    a_unw, _a_w, _cls, _nodes = build_graph(data_dir, lo, hi)
    coo = sp.coo_matrix(a_unw)
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    w = torch.sparse_coo_tensor(
        idx, torch.tensor(coo.data, dtype=torch.float64), coo.shape
    ).coalesce()
    return _lambda_max_combinatorial(w)


def _read_report(out_dir: Path) -> dict:
    """Pull the split metrics out of a finished run's JSON report."""

    files = sorted(out_dir.glob("elliptic_modular_d*.json"))
    if not files:
        return {}
    with open(files[-1]) as fh:
        blob = json.load(fh)
    row = {}
    for split in ("train", "test", "all"):
        rep = (blob.get("report") or {}).get(split) or {}
        row[f"{split}_f1"] = rep.get("mean_f1")
        row[f"{split}_recall"] = rep.get("mean_recall")
        row[f"{split}_precision"] = rep.get("mean_precision")
        row[f"{split}_detection"] = rep.get("detection_rate")
    coarsening = blob.get("coarsening") or {}
    row["n_coarse"] = coarsening.get("n_coarse")
    row["epsilon"] = coarsening.get("epsilon")
    avg = (blob.get("transfer") or {}).get("average") or {}
    row["transfer_days"] = avg.get("n_days_scored")
    row["transfer_f1"] = avg.get("mean_f1")
    row["transfer_detection"] = avg.get("detection_rate")
    row["lambda_min_final"] = blob.get("lambda_min_final")
    captures = blob.get("captures") or {}
    row["mean_capture_all"] = (captures.get("all") or {}).get("mean_capture")
    row["min_capture_all"] = (captures.get("all") or {}).get("min_capture")
    return row


def plot_comparison(df, path: Path) -> None:
    """Grouped bars per geometry: held-out F1 / detection and the transfer pair.

    Only the *detection* metrics are plotted.  Capture, confusability and the RSA
    epsilon are all fractions of the geometry's own norm, so plotting them side by
    side would invite a comparison that is partly a change of units; they stay in
    the CSV with that caveat attached.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("test_f1", "held-out F1"),
        ("test_detection", "held-out detection"),
        ("transfer_f1", "transfer F1"),
        ("transfer_detection", "transfer detection"),
    ]
    stops = list(dict.fromkeys(df["stop"]))
    fig, axes = plt.subplots(
        1, len(stops), figsize=(7.5 * len(stops), 4.6), squeeze=False
    )
    for ax, stop in zip(axes[0], stops):
        sub_df = df[df["stop"] == stop]
        geometries = list(sub_df["geometry"])
        x = np.arange(len(metrics))
        width = 0.8 / max(len(geometries), 1)
        for i, geom in enumerate(geometries):
            row = sub_df[sub_df["geometry"] == geom].iloc[0]
            values = [float(row[key] or 0.0) for key, _ in metrics]
            ax.bar(x + i * width - 0.4 + width / 2, values, width, label=geom)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(f"Ward stop = {stop}")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Screened geometry: symmetric vs combinatorial (Elliptic++)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--train-days", default="24-26")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--transfer-days", type=int, default=3)
    ap.add_argument("--ward-num-cuts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--stop", nargs="+", default=["f1", "epsilon"], choices=["f1", "epsilon"]
    )
    ap.add_argument("--out", default=Path("results/laplacian_comparison"), type=Path)
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="skip the runs and only re-tabulate/plot an existing "
        "laplacian_comparison.csv in --out (useful while a sweep is still going)",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        df = pd.read_csv(args.out / "laplacian_comparison.csv")
        plot_comparison(df, args.out / "laplacian_comparison.png")
        with pd.option_context("display.width", 250, "display.max_columns", 60):
            print(df.to_string(index=False))
        print(f"\nplot -> {args.out / 'laplacian_comparison.png'}")
        return
    last = args.train_days.split(",")[-1].strip()
    lo, hi = (int(x) for x in (last.split("-") if "-" in last else [last, last]))
    lam = _lambda_max(args.data_dir, lo, hi)
    tau_matched = args.tau * lam / 2.0
    print(
        f"evaluated window d{lo}-{hi}: lambda_max(D - W) = {lam:.5g}  ->  "
        f"tau {args.tau:g} (symmetric) matches tau {tau_matched:.5g} (combinatorial)"
    )

    rows = []
    for stop in args.stop:
        for tag, extra in _cells(args.tau, tau_matched):
            slug = f"{tag.replace('/', '_').replace('=', '')}__{stop}"
            run_out = args.out / slug
            cmd = [
                sys.executable, "-m", "src.run_elliptic_modular",
                "--data-dir", str(args.data_dir),
                "--train-days", args.train_days,
                "--epochs", str(args.epochs),
                "--degree", str(args.degree),
                "--transfer-days", str(args.transfer_days),
                "--ward-stop", stop,
                "--ward-num-cuts", str(args.ward_num_cuts),
                "--seed", str(args.seed),
                "--no-pr-sweep", "--no-compare-coarseners",
                "--out", str(run_out),
                *extra,
            ]
            print(f"\n=== [{stop}] {tag} ===\n{' '.join(cmd)}", flush=True)
            log = run_out / "run.log"
            run_out.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            row = {"stop": stop, "geometry": tag, "returncode": proc.returncode}
            row.update(_read_report(run_out))
            rows.append(row)
            print(
                f"  -> rc={proc.returncode}  test F1={row.get('test_f1')}  "
                f"detection={row.get('all_detection')}  log={log}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    csv = args.out / "laplacian_comparison.csv"
    df.to_csv(csv, index=False)
    plot_comparison(df, args.out / "laplacian_comparison.png")
    print(f"\ncomparison table -> {csv}")
    print(f"comparison plot  -> {args.out / 'laplacian_comparison.png'}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
