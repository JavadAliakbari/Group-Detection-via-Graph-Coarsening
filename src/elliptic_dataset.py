"""Do illicit accounts form dense, low-conductance motifs in Elliptic++ Actors?

Motivation
----------
The accompanying analysis ("Graph Coarsening for Gang Detection") argues that
AML *gangs* are low-conductance vertex sets: internally connected but weakly
attached to the rest of the graph. Conductance is the single quantity that
controls whether a gang lives in the low-frequency Laplacian subspace and can
therefore be recovered by (spectral / SGC) coarsening.

For a vertex set S in a weighted graph with adjacency W, degree d = W·1:

    vol(S)  = sum_{i in S} d_i
    cut(S)  = sum_{i in S, j not in S} W_ij
    Phi(S)  = cut(S) / min(vol(S), vol(V\S))      (normalized conductance)
    phi(S)  = cut(S) / |S|                          (boundary-per-node proxy, = m1)

This script tests the hypothesis on the Elliptic++ Actors (wallet-address)
dataset: we define illicit *gangs* as the connected components (size >= 2) of
the illicit-induced subgraph, then compare their conductance against
  (a) licit connected components, and
  (b) size-matched random node sets and random *connected* sets,
all measured in the full transaction graph of the chosen day window.

Edge weights
------------
AddrAddr_edgelist.csv carries only address pairs (no per-transaction BTC
amount). The richest "transactions as weights" signal available is the
*multiplicity* of an address-address pair (how many transaction records connect
them). We therefore report two graphs in parallel:
  - unweighted  (W_ij = 1)
  - weighted    (W_ij = number of transaction records between i and j)

Run
---
    conda activate FedStruct   # python with numpy/scipy/pandas
    python -m src.GangPrediction.run_elliptic_gang_conductance \
        --day-start 24 --day-end 26 --out results/elliptic_conductance
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch


def build_graph(
    data_dir: Path, day_start: int, day_end: int
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, pd.DataFrame]:
    """Build symmetric unweighted and weighted adjacency for the day window.

    Returns
    -------
    A_unw : csr_matrix (N, N)  symmetric 0/1 adjacency
    A_w   : csr_matrix (N, N)  symmetric weighted adjacency (edge multiplicity)
    cls   : (N,) int array of class labels (1 illicit, 2 licit, 3 unknown)
    nodes_df : the per-node frame (address, Time step, class)
    """
    print(f"  Reading features (time steps {day_start}-{day_end}) …")
    feat = pd.read_csv(
        data_dir / "wallets_features.csv",
        usecols=["address", "Time step"],
        dtype={"address": str, "Time step": "int32"},
    )
    feat = feat[feat["Time step"].between(day_start, day_end)]
    feat = feat.drop_duplicates(subset=["address"], keep="last").copy()

    print("  Reading classes …")
    classes = pd.read_csv(data_dir / "wallets_classes.csv", dtype={"address": str})
    nodes_df = feat.merge(classes, on="address", how="left")
    nodes_df["class"] = nodes_df["class"].fillna(3).astype(int)

    node_to_index = {a: i for i, a in enumerate(nodes_df["address"].values)}
    N = len(nodes_df)
    print(f"  Nodes: {N:,}")

    print("  Reading edge list …")
    edg = pd.read_csv(
        data_dir / "AddrAddr_edgelist.csv",
        dtype={"input_address": str, "output_address": str},
    )
    nodeset = set(node_to_index)
    mask = edg["input_address"].isin(nodeset) & edg["output_address"].isin(nodeset)
    edg = edg[mask]
    s = edg["input_address"].map(node_to_index).to_numpy(np.int64)
    d = edg["output_address"].map(node_to_index).to_numpy(np.int64)
    print(f"  Directed edge records in window: {len(s):,}")

    # Symmetrize: stack both directions, then sum duplicate pairs -> multiplicity.
    row = np.r_[s, d]
    col = np.r_[d, s]
    val = np.ones(len(row), dtype=np.float64)
    A_w = csr_matrix((val, (row, col)), shape=(N, N))
    A_w.sum_duplicates()  # weighted: W_ij = #records between i and j (both dirs)
    A_w.setdiag(0)
    A_w.eliminate_zeros()

    A_unw = A_w.copy()
    A_unw.data[:] = 1.0  # unweighted: presence only

    cls = nodes_df["class"].to_numpy()
    print(
        f"  Undirected edges: {A_unw.nnz // 2:,}  |  "
        f"illicit={int((cls==1).sum()):,} licit={int((cls==2).sum()):,} "
        f"unknown={int((cls==3).sum()):,}"
    )
    return A_unw, A_w, cls, nodes_df


def connected_components_sets(
    A: csr_matrix, member_idx: np.ndarray, min_size: int = 2
) -> List[np.ndarray]:
    """Connected components (size >= min_size) of the subgraph induced by member_idx.

    Returned node ids are in the original (global) node space.
    """
    if len(member_idx) == 0:
        return []
    sub = A[member_idx][:, member_idx]
    n_comp, lab = connected_components(sub, directed=False)
    out = []
    for c in range(n_comp):
        m = lab == c
        if m.sum() >= min_size:
            out.append(member_idx[m])
    return out


# ---------------------------------------------------------------------------
# Feature matrix aligned to the build_graph node ordering
# ---------------------------------------------------------------------------

_NON_FEATURE_COLS = {"address", "Time step", "class", "Time_step"}


def random_structural_features(num_nodes: int, width: int, seed: int) -> torch.Tensor:
    """Isotropic random range-finder ``Omega`` (the structural feature channel)."""

    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(num_nodes, width, dtype=torch.float64, generator=gen)
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)


def load_node_features(
    data_dir: Path,
    nodes_df: pd.DataFrame,
    day_start: int,
    day_end: int,
    *,
    keep_columns: "list[str] | None" = None,
    return_columns: bool = False,
):
    """Standardised wallet-feature matrix X aligned to ``nodes_df`` row order.

    ``build_graph`` assigns node index i to ``nodes_df['address'].iloc[i]``; we
    reindex the feature rows by that address order so X[i] is node i's features.

    ``keep_columns`` pins the numeric feature columns to a fixed set (e.g. the
    columns kept on the training day) so a filter fit on one day can be applied
    to another day whose zero-variance columns differ -- the column set (hence
    ``X``'s width) stays identical while the z-score statistics are still
    recomputed on *this* day's rows.  With ``return_columns=True`` the kept
    column names are returned alongside ``X`` so the caller can reuse them.
    """

    feat = pd.read_csv(data_dir / "wallets_features.csv", dtype={"address": str})
    feat = feat[feat["Time step"].between(day_start, day_end)]
    feat = feat.drop_duplicates(subset=["address"], keep="last").set_index("address")
    feat = feat.reindex(nodes_df["address"].values)  # align to node index order

    num = feat.drop(columns=[c for c in _NON_FEATURE_COLS if c in feat.columns])
    num = num.select_dtypes(include=[np.number])
    if keep_columns is not None:
        # align to the training day's column set (missing columns -> all-NaN -> 0)
        num = num.reindex(columns=list(keep_columns))
    X = num.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    if keep_columns is None:
        # column z-score (drop zero-variance columns so they don't blow up)
        keep = sd > 1e-12
        cols = list(num.columns[keep])
        X = (X[:, keep] - mu[keep]) / sd[keep]
    else:
        # columns are pinned: keep every one, clamp the divisor so a column that
        # is constant on this day maps to 0 rather than blowing up.
        cols = list(num.columns)
        X = (X - mu) / np.clip(sd, 1e-12, None)
    # row L2-normalisation: makes each node's feature vector unit length,
    # removing inter-node magnitude differences after column standardisation.
    row_norm = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    X = X / row_norm
    print(
        f"  Feature matrix X: {X.shape[0]:,} x {X.shape[1]} "
        "(col z-scored + row L2-normalised)"
    )
    Xt = torch.from_numpy(X).to(torch.float64)
    if return_columns:
        return Xt, cols
    return Xt
