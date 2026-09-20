"""Shared utilities for data prep, sparse ops, and graph algebra."""

from copy import deepcopy
import os
import random
from datetime import datetime


import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    add_self_loops,
    scatter,
)
from pygsp import graphs
import scipy as sp
from tqdm import tqdm
from src.utils.logger import getLOGGER

seed = 6
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

now = datetime.now().strftime("%Y%m%d_%H%M%S")

result_path = "results/"
save_path = f"{result_path}{now}/"

os.makedirs(save_path, exist_ok=True)
LOGGER = getLOGGER(
    name=f"{now}_Cora",
    log_on_file=True,
    save_path=save_path,
)


# NOTE: This module uses sparse COO tensors in multiple places.
# On Apple Silicon, sparse ops on MPS are still incomplete for this pipeline,
# so prefer CUDA when available, otherwise fall back to CPU (not MPS).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_global_seed(seed: int) -> None:
    """Best-effort deterministic seeding for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep algorithm choices deterministic where PyTorch provides stable kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(seed)


def create_pyg_data(features, edges_idx, labels) -> Data:
    """Build a PyG Data object from numpy feature/edge/label arrays."""
    x = torch.FloatTensor(features.astype(np.float32))
    y = torch.LongTensor(labels)
    edge_index = torch.LongTensor(edges_idx.T)  # expects (2, num_edges)
    edge_index = to_undirected(edge_index)

    return Data(x=x, edge_index=edge_index, y=y)


def create_pygsp_graph(data: Data) -> graphs.Graph:
    """Convert a PyG graph into a pygsp Graph with Laplacian and layout."""
    # adjacency matrix
    adj_matrix = sp.sparse.coo_matrix(
        (np.ones(len(data.edge_index)), (data.edge_index[0], data.edge_index[1])),
        shape=(data.num_nodes, data.num_nodes),
    )
    # adj_matrix = adj_matrix + adj_matrix.T
    # adj_matrix.data = np.ones(len(adj_matrix.data))

    # pygsp graph
    G = graphs.Graph(adj_matrix)
    G.compute_laplacian()
    G.set_coordinates()

    return G


def create_train_val_test_split(num_nodes, train_ratio=0.6, val_ratio=0.2):
    """Randomly split node indices into train/val/test."""
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return train_idx, val_idx, test_idx


def degree(edge_index, num_nodes, edge_weights=None):
    """Compute degree (optionally weighted) from edge index."""
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    if edge_weights is not None:
        for i, j, w in zip(edge_index[0], edge_index[1], edge_weights):
            deg[i] += w
            deg[j] += w
    else:
        for i, j in zip(edge_index[0], edge_index[1]):
            deg[i] += 1
            deg[j] += 1
    return deg


def sparse_eye(size):
    """Create a sparse identity matrix."""
    indices = torch.arange(size).repeat(2, 1)
    values = torch.ones(size, dtype=torch.float32, device=device)
    C = torch.sparse_coo_tensor(indices, values, (size, size), device=device)
    return C


def graph_params(G: Data):
    """Compute adjacency, Laplacian, and degree for a PyG graph."""
    num_nodes = G.num_nodes
    # edge_index, edge_weight = remove_self_loops(G.edge_index, G.edge_weight)
    edge_index, edge_weight = G.edge_index, G.edge_weight

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), dtype=torch.float32, device=edge_index.device
        )
    # Use the de-looped edge_index (not G.edge_index) for the adjacency matrix
    W = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes),
        device=edge_index.device,
    )

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([-edge_weight, deg], dim=0)

    L = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes), device=edge_index.device
    )

    return W, L, deg


def create_P(indices, n, device="cpu"):
    """Create a sparse selector matrix for a subset of nodes."""
    n_t = len(indices)
    P_indices = torch.stack(
        [
            indices,  # Column indices
            torch.arange(n_t, device=device),  # Row indices
        ]
    )
    P_values = torch.ones(n_t, dtype=torch.float32, device=device)
    P = torch.sparse_coo_tensor(P_indices, P_values, (n, n_t), device=device)
    return P


import torch


def _spmm(A, B):
    """Sparse x dense matrix multiply that works for COO/CSR/BSR."""
    if B.dim() == 1:  # (n,) -> (n,1)
        return (A @ B.unsqueeze(1)).squeeze(1)
    return A @ B


def _cg_multi(A_mv, B, tol=1e-6, maxiter=None, M=None):
    """
    Solve A X = B for multiple RHS (columns of B) with (preconditioned) CG.
    A_mv: callable(V) -> A @ V
    B: [m, d] dense
    M: optional preconditioner callable(V) ≈ A^{-1} V (e.g., Jacobi)
    """
    m, d = B.shape
    X = torch.zeros_like(B)
    R = B - A_mv(X)
    Z = M(R) if M is not None else R
    P = Z.clone()

    # Per-column inner products
    def col_dot(U, V):  # returns [d]
        return (U * V).sum(dim=0)

    rz_old = col_dot(R, Z)  # [d]
    normB = B.norm(dim=0)  # [d]
    active = normB > 0
    if maxiter is None:
        maxiter = 2 * m  # simple cap

    for _ in range(maxiter):
        AP = A_mv(P)  # [m, d]
        denom = col_dot(P, AP).clamp_min(1e-30)  # [d]
        alpha = torch.zeros_like(denom)
        alpha[active] = rz_old[active] / denom[active]
        X = X + P * alpha.unsqueeze(0)
        R = R - AP * alpha.unsqueeze(0)

        # Convergence (per column)
        done = R.norm(dim=0) <= tol * normB
        new_active = active & (~done)
        if not new_active.any():
            break
        active = new_active

        Z = M(R) if M is not None else R
        rz_new = col_dot(R, Z)
        beta = torch.zeros_like(rz_new)
        nz = rz_old.abs() > 0
        beta[nz] = rz_new[nz] / rz_old[nz]
        P = Z + P * beta.unsqueeze(0)
        rz_old = rz_new
    return X


def min_norm_lstsq_sparse_multi(
    C, Y, lam=0.0, tol=1e-6, maxiter=None, precondition=True
):
    """
    Solve min_X ||Y - C X||_F^2 with sparse C (m<n) via dual CG on (C C^T + lam I) Z = Y.
    Returns the (approx.) minimum-norm solution X = C^T Z (ridge if lam>0).

    C:   sparse tensor [m, n] (COO/CSR/BSR)
    Y:   dense tensor  [m, d]
    lam: Tikhonov damping (>=0) for stability
    """
    assert C.layout in (
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
    ), "C must be sparse"
    m, n = C.shape
    assert Y.shape[0] == m

    Ct = C.transpose(0, 1)

    def A_mv(V):  # V: [m, d]
        out = _spmm(C, _spmm(Ct, V))
        return out if lam == 0.0 else out + lam * V

    # Jacobi preconditioner M ≈ (C C^T + lam I)^{-1}
    if precondition:
        Cc = (
            C.coalesce()
            if C.layout == torch.sparse_coo
            else C.to_sparse_coo().coalesce()
        )
        rows = Cc.indices()[0]
        diag = torch.zeros(m, dtype=Cc.dtype, device=Cc.device)
        diag.scatter_add_(0, rows, Cc.values() ** 2)
        if lam != 0.0:
            diag = diag + lam
        diag = diag.clamp_min(1e-12)
        M = lambda V: V / diag.unsqueeze(1)  # row-wise scaling
    else:
        M = None

    # Solve for Z, then recover X
    Z = _cg_multi(A_mv, Y, tol=tol, maxiter=maxiter, M=M)  # [m, d]
    X = _spmm(Ct, Z)  # [n, d]
    return X


# ----------------------------
# AX (and [X, AX, A^2X, ...])
# ----------------------------
def ax_stack(
    A: torch.Tensor,
    X: torch.Tensor,  # [N, d]
    powers: int = 1,  # 1 -> AX; >1 -> [X, AX, ..., A^p X]
) -> torch.Tensor:
    """Stack powers of A times X for simple feature propagation."""
    feats = [X] if powers > 1 else []
    cur = X
    for _ in range(powers):
        cur = torch.sparse.mm(A, cur)  # [N, d]
        feats.append(cur)
    return torch.cat(feats, dim=1) if len(feats) > 1 else cur  # [N, k]


# ----------------------------
# L-orthonormal basis for span(AX)
# B = Y (Y^T L Y)^{-1/2}
# ----------------------------
def inv_sqrt_psd(M: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Compute a ridge-regularized inverse square root for PSD matrices."""
    # symmetric PSD regularized inverse sqrt
    e, U = torch.linalg.eigh(M + ridge * torch.eye(M.size(0), device=M.device))
    e_clamped = torch.clamp(e, min=1e-12)
    inv_sqrt = (e_clamped).rsqrt()
    return (U * inv_sqrt) @ U.mH  # U diag(inv_sqrt) U^T


def l_orthonormalize(
    Y: torch.Tensor,  # [N, k]
    L: torch.Tensor,  # sparse [N, N]
    K: int,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Return an L-orthonormal basis for the span of Y."""
    d, V = torch.linalg.eigh(Y + ridge * torch.eye(Y.size(0), device=Y.device))
    d = d[:K]
    V = V[:, :K]
    Yp = V @ torch.diag(d.sqrt())  # [N, K]
    LY = torch.sparse.mm(L, Y)  # [N, k]
    G = Y.mT @ LY  # [k, k]  (dense)
    G_inv_sqrt = inv_sqrt_psd(G, ridge)
    B = Y @ G_inv_sqrt  # [N, k]
    # sanity check (optional):
    # torch.testing.assert_close(B.mT @ torch.sparse.mm(L, B), torch.eye(B.size(1), device=B.device), atol=1e-3, rtol=1e-3)
    return B


def compute_class_weights(
    labels: torch.Tensor, train_idx: torch.Tensor = None, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency for handling class imbalance.

    Args:
        labels: Tensor of shape [N] with class labels
        train_idx: Optional tensor of training indices. If provided, weights are computed
                   only from training labels. Otherwise, uses all labels.
        device: Device to place the weights tensor on

    Returns:
        Tensor of shape [num_classes] with class weights (higher weight for minority class)
    """
    if train_idx is not None:
        train_labels = labels[train_idx]
    else:
        train_labels = labels

    num_classes = len(torch.unique(labels))
    class_counts = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        class_counts[c] = (train_labels == c).sum().float()

    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)

    # Inverse frequency weighting
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)

    # Normalize so weights sum to num_classes (maintains loss scale)
    class_weights = class_weights * num_classes / class_weights.sum()

    return class_weights


def create_adj(
    edge_index,
    edge_weight=None,
    normalization="normal",
    self_loop=False,
    num_nodes=None,
    nodes=None,
):
    if num_nodes is None:
        num_nodes = max(torch.flatten(edge_index)) + 1
    if nodes is None:
        nodes = torch.arange(num_nodes, device=edge_index.device)
    if self_loop:
        undirected_edges = add_self_loops(edge_index)[0]
    else:
        undirected_edges = edge_index

    edge_mask = undirected_edges[0].unsqueeze(1).eq(nodes).any(1)
    directed_edges = undirected_edges[:, edge_mask]
    # directed_edges, _ = remove_self_loops(directed_edges)

    if edge_weight is not None:
        edge_weight = edge_weight.to(edge_index.device)
    else:
        edge_weight = torch.ones(
            directed_edges.size(1), dtype=torch.float32, device=edge_index.device
        )
    row, col = directed_edges[0], directed_edges[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")

    if normalization == "rw":
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv = deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif normalization == "sym":
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        deg2 = scatter(edge_weight, col, 0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt2 = deg2.pow_(-0.5)
        deg_inv_sqrt2 = deg_inv_sqrt2.masked_fill_(deg_inv_sqrt2 == float("inf"), 0)
        edge_weight *= deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt2[edge_index[1]]
    adj = torch.sparse_coo_tensor(
        directed_edges,
        edge_weight,
        (num_nodes, num_nodes),
        dtype=torch.float32,
        device=edge_index.device,
    )

    return adj


def calc_abar(edge_index, num_nodes, num_layers, edge_weight=None):
    adj = create_adj(
        edge_index,
        edge_weight=edge_weight,
        normalization="rw",
        self_loop=True,
        num_nodes=num_nodes,
        # nodes=self.graph.node_ids,
    )
    num_nodes = adj.shape[1]

    abar = sparse_eye(num_nodes).to(device)  # Start with identity

    for i in range(num_layers):
        abar = torch.matmul(adj, abar)  # Sparse-dense matrix multiplication
    return abar


def arnoldi_iteration(A, m: int, b=None, log=True):
    local_dev = "cpu"
    # local_dev = dev
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.

    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    h : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    A = deepcopy(A).double()
    A = A.to_sparse().to(local_dev)
    if b is None:
        # b = torch.ones(A.shape[0], dtype=torch.double, device=dev)
        b = torch.randn(A.shape[0], dtype=torch.double, device=local_dev)
        if torch.sum(b) < 0:
            b = -b
    eps = 1e-12
    h = torch.zeros((m, m), dtype=torch.double, device=local_dev)
    Q = torch.zeros((A.shape[0], m), dtype=torch.double, device=local_dev)
    # Normalize the input vector
    Q[:, 0] = b / torch.norm(b, 2)  # Use it as the first Krylov vector
    if log:
        bar = tqdm(total=m - 1)
    for k in range(1, m):
        v = A @ Q[:, k - 1]  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = Q[:, j].conj() @ v
            v = v - h[j, k - 1] * Q[:, j]

        h[k, k - 1] = torch.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            return h, Q
        if log:
            bar.update()

    # h = h.to(dev)
    # Q = Q.to(dev)
    return h.float(), Q.float()


def _degrees(adjacency: torch.Tensor) -> torch.Tensor:
    degree = torch.zeros(
        adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    degree.scatter_add_(0, adjacency.indices()[0], adjacency.values())
    return degree


def normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return sparse symmetric ``A_hat = D^-1/2 (A + I) D^-1/2``.

    The AMLGenTex loader may return a directed transaction graph.  Eq. (48)
    assumes the symmetric normalized adjacency, so directed edges are mirrored.
    Existing reciprocal edges retain their original total weight; one-way edges
    receive the same weight in both directions.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    device = edge_index.device
    rows, cols = edge_index[0], edge_index[1]
    non_self = rows != cols
    rows, cols = rows[non_self], cols[non_self]
    if edge_weight is None:
        values = torch.ones(rows.numel(), dtype=dtype, device=device)
    else:
        values = edge_weight.to(device=device, dtype=dtype)[non_self]

    # A + A.T, divided by two: reciprocal entries retain their weight while a
    # one-way transaction becomes an undirected edge of half weight per side.
    symmetric_indices = torch.cat(
        [torch.stack((rows, cols)), torch.stack((cols, rows))], dim=1
    )
    symmetric_values = torch.cat((values, values)) * 0.5
    adjacency = torch.sparse_coo_tensor(
        symmetric_indices,
        symmetric_values,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    loop_nodes = torch.arange(num_nodes, device=device)
    with_loops = torch.sparse_coo_tensor(
        torch.cat((adjacency.indices(), torch.stack((loop_nodes, loop_nodes))), dim=1),
        torch.cat(
            (adjacency.values(), torch.ones(num_nodes, dtype=dtype, device=device))
        ),
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, with_loops.indices()[0], with_loops.values())
    inv_sqrt_degree = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    indices = with_loops.indices()
    normalized_values = (
        with_loops.values() * inv_sqrt_degree[indices[0]] * inv_sqrt_degree[indices[1]]
    )
    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

