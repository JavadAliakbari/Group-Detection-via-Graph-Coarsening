"""Gang detection on the GADBench homogeneous fraud graphs (arXiv:2312.06441).

Applies the same spectral-coarsening gang detector built for Elliptic++ to the
node-level anomaly graphs from the benchmark:

  * Amazon    -- 11,944 nodes,  ~4.40M edges, 25 feats,  821 fraud (6.87%)
  * T-Finance -- 39,357 nodes, ~21.22M edges, 10 feats,        (4.58%)
  * T-Social  -- 5,781,065 nodes, ~73.1M edges, 10 feats,      (3.01%)

All three are *homogeneous* graphs with a binary anomaly label.  We define a
*gang* exactly as before -- a connected component (size >= 2) of the
anomaly-induced subgraph -- then run the **structural** and **joint** encoders
(no Laplacian: too expensive) as the Loukas RSA coarsening target and report how
many gangs collapse into a single super-node (recall>thr AND precision>thr).

NOTE on structure.  These are dense node-anomaly graphs, not sparse transaction
graphs: the anomalies tend to form *one* large connected blob rather than many
disjoint motifs (e.g. Amazon: a single 806-node component + isolated singletons).
So the connected-component gang definition yields very few gangs here -- itself
the finding that AML transaction graphs and dense node-anomaly graphs have very
different "gang" structure.  ``--gang-louvain`` optionally subdivides the blob
into communities for a finer, more informative multi-gang detection rate.

Data
----
Amazon downloads itself (DGL FraudAmazon.zip -> Amazon.mat, scipy.io).  T-Finance
and T-Social are DGL ``.bin`` graphs (Tang et al. 2022); pass ``--dgl-path`` to a
local ``tfinance``/``tsocial`` saved with ``dgl.save_graphs`` (requires dgl).

Run::

    python -m src.run_graph_fraud_gang_detection --dataset amazon \
        --coarsening-method capped --epsilon 5
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy.sparse import coo_matrix

from src.utils.utils import *

# ---------------------------------------------------------------------------
# Dataset loading -> (A_unw csr, X float64, y int{0,1})
# ---------------------------------------------------------------------------


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-12
    return (X[:, keep] - mu[keep]) / sd[keep]


def load_amazon(data_dir: Path):
    """DGL FraudAmazon ``Amazon.mat`` -- the homogeneous ('homo') graph."""

    import scipy.io as sio

    mat_path = data_dir / "Amazon.mat"
    if not mat_path.exists():
        import urllib.request
        import zipfile

        data_dir.mkdir(parents=True, exist_ok=True)
        zpath = data_dir / "FraudAmazon.zip"
        LOGGER.info("  downloading FraudAmazon.zip …")
        urllib.request.urlretrieve("https://data.dgl.ai/dataset/FraudAmazon.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data_dir)
    m = sio.loadmat(mat_path)
    A = m["homo"].tocsr().astype(np.float64)
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    X = _standardize(np.asarray(m["features"].todense(), dtype=np.float64))
    y = m["label"].ravel().astype(np.int64)
    return A, X, y


def load_dgl_bin(dgl_path: Path):
    """T-Finance / T-Social: a homogeneous DGL graph saved with save_graphs."""

    import dgl  # noqa: F401

    graphs, _ = dgl.load_graphs(str(dgl_path))
    g = graphs[0]
    src, dst = (t.numpy() for t in g.edges())
    n = g.num_nodes()
    A = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    feat_key = "feature" if "feature" in g.ndata else "feat"
    X = _standardize(g.ndata[feat_key].numpy().astype(np.float64))
    y = g.ndata["label"].numpy().astype(np.int64)
    return A, X, y


def load_snap_community(data_dir: Path, name: str):
    """Any SNAP ground-truth-community graph (amazon / dblp / youtube / lj / orkut).

    Downloads ``com-<name>.ungraph.txt`` + ``com-<name>.top5000.cmty.txt`` and
    returns ``(A, communities)`` -- a sparse 0/1 adjacency and the community node
    sets (the 'gangs').  No node features: structural-only.
    """

    import gzip
    import urllib.request

    data_dir = Path(data_dir)
    base = "https://snap.stanford.edu/data/bigdata/communities/"
    g_txt = data_dir / f"com-{name}.ungraph.txt"
    c_txt = data_dir / f"com-{name}.top5000.cmty.txt"
    for fname in (f"com-{name}.ungraph.txt", f"com-{name}.top5000.cmty.txt"):
        if not (data_dir / fname).exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"  downloading {fname} …")
            urllib.request.urlretrieve(base + fname + ".gz", data_dir / (fname + ".gz"))
            with gzip.open(data_dir / (fname + ".gz"), "rb") as fz:
                (data_dir / fname).write_bytes(fz.read())

    edges = np.loadtxt(g_txt, dtype=np.int64, comments="#")
    ids = np.unique(edges)
    remap = {int(v): i for i, v in enumerate(ids)}
    r = np.fromiter((remap[int(a)] for a in edges[:, 0]), dtype=np.int64)
    c = np.fromiter((remap[int(b)] for b in edges[:, 1]), dtype=np.int64)
    n = len(ids)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    communities = []
    for line in open(c_txt):
        nodes = [remap[int(x)] for x in line.split() if int(x) in remap]
        if len(nodes) >= 2:
            communities.append(np.asarray(nodes, dtype=np.int64))
    return A, communities


def load_dataset(args):
    if args.dataset == "amazon":
        return load_amazon(args.data_dir)
    if args.dataset in ("tfinance", "tsocial"):
        if not args.dgl_path:
            raise ValueError(
                f"{args.dataset} needs --dgl-path to a DGL .bin graph (requires dgl)"
            )
        return load_dgl_bin(Path(args.dgl_path))
    raise ValueError(args.dataset)
