"""PyTorch SGC target construction, Loukas RSA coarsening, and pattern recall.

This is the end-to-end path requested by the Graph_Coarsening note and the
Loukas paper:

1. fit ``theta`` with Eq. (48) on training patterns;
2. form ``Z = g_theta(A_hat) X`` and ``R = span(Z)``;
3. run edge-based local-variation RSA coarsening with ``R`` as its target;
4. declare a pattern detected only when its Pattern-model recall and precision
   are both strictly greater than the supplied threshold.

All graph algebra and the Loukas Algorithm 1/2 implementation below are
PyTorch based.  No NumPy/SciPy coarsening path is used, with a single
exception: ``method="ward"`` (:func:`_ward_partition`) lazily imports
scikit-learn and SciPy for connectivity-constrained Ward agglomeration --
those packages are optional and only required when that method is selected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import json
import math

import torch
import torch.nn.functional as F


from src.utils.utils import _degrees, normalized_adjacency


@dataclass
class LoukasCoarseningResult:
    """Original-node mapping and RSA diagnostics from Algorithm 1."""

    node_to_supernode: torch.Tensor
    n_original: int
    n_coarse: int
    epsilon: float
    epsilon_bound: float
    sigmas: List[float]
    sizes: List[int]

    @property
    def reduction(self) -> float:
        return 1.0 - self.n_coarse / self.n_original


@dataclass
class LoukasPatternDetection:
    """Pattern-model recall/precision after RSA coarsening."""

    pattern_id: str
    pattern_type: str
    label: str
    recall: float
    precision: float
    f1: float
    detected: bool


def _symmetric_adjacency_without_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a sparse symmetric adjacency for the combinatorial Laplacian."""

    rows, cols = edge_index[0], edge_index[1]
    non_self = rows != cols
    rows, cols = rows[non_self], cols[non_self]
    if edge_weight is None:
        values = torch.ones(rows.numel(), dtype=dtype, device=edge_index.device)
    else:
        values = edge_weight.to(device=edge_index.device, dtype=dtype)[non_self]

    indices = torch.cat((torch.stack((rows, cols)), torch.stack((cols, rows))), dim=1)
    values = torch.cat((values, values)) * 0.5
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        dtype=dtype,
        device=edge_index.device,
    ).coalesce()


def _laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Return combinatorial ``L = D - W`` without forming a dense matrix."""

    n = adjacency.shape[0]
    diagonal = torch.arange(n, device=adjacency.device)
    indices = torch.cat((torch.stack((diagonal, diagonal)), adjacency.indices()), dim=1)
    values = torch.cat((_degrees(adjacency), -adjacency.values()))
    return torch.sparse_coo_tensor(
        indices,
        values,
        (n, n),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()


def _normalized_laplacian(
    adjacency: torch.Tensor, *, add_self_loops: bool = True
) -> torch.Tensor:
    """Return the symmetric normalized Laplacian ``L_sym = I - A_hat``.

    This is the metric the collective learnable-filter objective is derived in
    (``L = I - A_hat`` of the analysis), as opposed to the combinatorial
    ``L = D - W`` of :func:`_laplacian`.  With ``add_self_loops`` (the
    renormalization trick, matching :func:`src.sgc_detection.normalized_adjacency`)
    the operator is ``I - D_tilde^{-1/2}(W + I) D_tilde^{-1/2}`` with
    ``D_tilde = D + I``; otherwise it is the classical
    ``I - D^{-1/2} W D^{-1/2}``.  Off-diagonal entries are ``-A_hat_ij`` and the
    diagonal is ``1 - A_hat_ii``, so every eigenvalue lies in ``[0, 2]``.
    """

    n = adjacency.shape[0]
    device, dtype = adjacency.device, adjacency.dtype
    indices = adjacency.indices()
    values = adjacency.values()
    off_diagonal = indices[0] != indices[1]
    indices, values = indices[:, off_diagonal], values[off_diagonal]

    if add_self_loops:
        loop = torch.arange(n, device=device)
        indices = torch.cat((indices, torch.stack((loop, loop))), dim=1)
        values = torch.cat((values, torch.ones(n, dtype=dtype, device=device)))

    degree = torch.zeros(n, dtype=dtype, device=device)
    degree.scatter_add_(0, indices[0], values)
    inv_sqrt = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    a_hat_values = values * inv_sqrt[indices[0]] * inv_sqrt[indices[1]]

    # L = I - A_hat: negate the (self-loop-augmented) A_hat entries and add the
    # identity on the diagonal.  Coalescing sums the +1 with any -A_hat_ii entry.
    loop = torch.arange(n, device=device)
    lap_indices = torch.cat((indices, torch.stack((loop, loop))), dim=1)
    lap_values = torch.cat((-a_hat_values, torch.ones(n, dtype=dtype, device=device)))
    return torch.sparse_coo_tensor(
        lap_indices, lap_values, (n, n), dtype=dtype, device=device
    ).coalesce()


def _screened_metric(operator: torch.Tensor, tau: float) -> torch.Tensor:
    """Return the screened metric ``M_tau = operator + tau I`` (Eq. 8/9).

    ``operator`` is the base Laplacian (combinatorial ``L = D - W`` or symmetric
    normalized ``L = I - A_hat``); adding ``tau`` on the diagonal turns the
    ``L``-seminorm into the positive-definite screened norm
    ``||x||^2_{M_tau} = ||x||^2_L + tau ||x||^2_2``.  ``tau = 0`` returns the
    operator unchanged, so every RSA quantity reduces to the plain-``L`` metric.
    """

    if not tau:
        return operator
    n = operator.shape[0]
    device, dtype = operator.device, operator.dtype
    loop = torch.arange(n, device=device)
    indices = torch.cat((operator.indices(), torch.stack((loop, loop))), dim=1)
    values = torch.cat(
        (
            operator.values(),
            torch.full((n,), float(tau), dtype=dtype, device=device),
        )
    )
    return torch.sparse_coo_tensor(
        indices, values, operator.shape, dtype=dtype, device=device
    ).coalesce()


def _orthonormal_range(Z: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for ``span(Z)`` via rank-revealing QR.

    The QR step only changes the basis, not the subspace Loukas preserves.
    """

    Q, R = torch.linalg.qr(Z, mode="reduced")
    diagonal = torch.abs(torch.diagonal(R))
    tolerance = torch.finfo(Z.dtype).eps * max(Z.shape) * diagonal.max().clamp_min(1.0)
    rank = int((diagonal > tolerance).sum().item())
    if rank == 0:
        raise ValueError("subspace generator has zero numerical rank")
    return Q[:, :rank]



def _l_orthonormalize(B: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Compute the paper's ``A=B(B^T L B)^(-1/2)`` on its non-null range."""

    gram = B.T @ torch.sparse.mm(laplacian, B)
    gram = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(gram)
    maximum = values.abs().max().clamp_min(1.0)
    keep = values > torch.finfo(B.dtype).eps * max(B.shape) * maximum
    if not torch.any(keep):
        raise ValueError("target subspace has no positive Laplacian-energy direction")
    return B @ (vectors[:, keep] * values[keep].rsqrt())


def _reduce_adjacency(
    adjacency: torch.Tensor,
    groups: torch.Tensor,
    *,
    keep_self_loops: bool = False,
) -> torch.Tensor:
    """Apply the Laplacian-consistent Loukas reduction to a sparse adjacency.

    This is exactly ``W_c = S^T W S`` with ``S`` the 0/1 assignment matrix
    (``S[i, r] = 1`` iff original node ``i`` is in supernode ``r``): the
    contraction sums every original edge weight into its supernode pair.

    ``keep_self_loops`` controls the diagonal of ``W_c``:

    * ``False`` (default) drops it, so ``_degrees(W_c)`` is the *cut* degree and
      the combinatorial ``L = D - W`` is unchanged (self-loops cancel there).
      This preserves the historical behaviour of every caller.
    * ``True`` keeps ``(W_c)_{rr} = sum_{i, j in C_r} W_{ij}``, i.e. the total
      internal weight of the supernode (each internal undirected edge counted
      twice).  A single merge ``u, v -> s`` then satisfies the volume-preserving
      rules ``(W_c)_{ss} = W_{uu} + W_{vv} + 2 W_{uv}`` and ``d_s = d_u + d_v``
      automatically, because the two off-diagonal ``(u, v)`` / ``(v, u)`` entries
      collapse onto ``(s, s)`` and add.  Use this with
      :func:`_weighted_normalized_laplacian`, which reads the stored diagonal.
    """

    n_new = int(groups.max().item()) + 1
    old_indices = adjacency.indices()
    new_indices = groups[old_indices]
    if keep_self_loops:
        keep = slice(None)
    else:
        keep = new_indices[0] != new_indices[1]
    return torch.sparse_coo_tensor(
        new_indices[:, keep],
        adjacency.values()[keep],
        (n_new, n_new),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()


def _weighted_normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    r"""Self-loop-aware symmetric normalized Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}``.

    Unlike :func:`_normalized_laplacian` (which discards any stored diagonal and
    applies the renormalization trick ``W + I``, ``D_tilde = D + I``), this reads
    the diagonal ``W_{ii}`` that :func:`_reduce_adjacency` (``keep_self_loops=True``)
    stores as a supernode's internal weight.  With ``D = diag(W \mathbf 1)`` the
    full row sum (diagonal included),

        (L_sym)_{ii} = 1 - W_{ii} / d_i,
        (L_sym)_{ij} = - W_{ij} / sqrt(d_i d_j)   (i != j),

    so a dense supernode with large internal weight has a *smaller* normalized
    diagonal, reflecting that more of its volume stays inside it.  On a graph
    with no self-loops (``W_{ii} = 0``) this is the classical
    ``I - D^{-1/2} W D^{-1/2}`` and eigenvalues still lie in ``[0, 2]``.
    """

    n = adjacency.shape[0]
    device, dtype = adjacency.device, adjacency.dtype
    indices = adjacency.indices()
    values = adjacency.values()

    degree = torch.zeros(n, dtype=dtype, device=device)
    degree.scatter_add_(0, indices[0], values)
    inv_sqrt = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    a_hat_values = values * inv_sqrt[indices[0]] * inv_sqrt[indices[1]]

    # L = I - A_hat: negate every A_hat entry (diagonal included) and add the
    # identity.  Coalescing sums the +1 with the -W_ii / d_i diagonal term.
    loop = torch.arange(n, device=device)
    lap_indices = torch.cat((indices, torch.stack((loop, loop))), dim=1)
    lap_values = torch.cat((-a_hat_values, torch.ones(n, dtype=dtype, device=device)))
    return torch.sparse_coo_tensor(
        lap_indices, lap_values, (n, n), dtype=dtype, device=device
    ).coalesce()


def _reduce_basis(basis: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Apply ``B_l=P_l B_{l-1}``, with P averaging each contraction set."""

    n_new = int(groups.max().item()) + 1
    reduced = torch.zeros(n_new, basis.shape[1], dtype=basis.dtype, device=basis.device)
    reduced.index_add_(0, groups, basis)
    counts = torch.bincount(groups, minlength=n_new).to(dtype=basis.dtype).unsqueeze(1)
    return reduced / counts


def _exact_rsa_epsilon(
    a0: torch.Tensor,
    laplacian: torch.Tensor,
    original_to_supernode: torch.Tensor,
) -> float:
    """Exact restricted-spectral-approximation constant of a coarsening.

    The RSA definition (Loukas 2019, Def. 2) is the smallest ``epsilon`` with
    ``||x - Pi x||_L <= epsilon ||x||_L`` for every ``x`` in the target subspace
    ``R``, where ``Pi = P^+ P`` is the block-averaging projection onto vectors
    that are constant on each supernode.  This is the *exact* worst-case
    distortion of the cumulative coarsening -- not the looser per-level product
    bound ``prod_l (1 + sigma_l) - 1``.

    With ``a0`` an ``L``-orthonormal basis of ``R`` (``a0^T L a0 = I``), any
    ``x = a0 c`` has ``||x||_L = ||c||``, so

        epsilon^2 = max_c (c^T Y^T L Y c) / (c^T c) = lambda_max(Y^T L Y),

    with ``Y = (I - Pi) a0`` -- each row of ``a0`` minus its supernode mean.
    ``laplacian`` and ``a0`` are those of the *original* graph; only the
    partition ``original_to_supernode`` changes across levels.
    """

    n_super = int(original_to_supernode.max().item()) + 1
    counts = (
        torch.bincount(original_to_supernode, minlength=n_super)
        .to(dtype=a0.dtype)
        .clamp_min(1.0)
        .unsqueeze(1)
    )
    sums = torch.zeros(n_super, a0.shape[1], dtype=a0.dtype, device=a0.device)
    sums.index_add_(0, original_to_supernode, a0)
    residual = a0 - (sums / counts)[original_to_supernode]  # (I - Pi) a0
    gram = residual.T @ torch.sparse.mm(laplacian, residual)
    gram = 0.5 * (gram + gram.T)
    top = torch.linalg.eigvalsh(gram)[-1].clamp_min(0.0)
    return float(top.sqrt())

def evaluate_loukas_patterns(
    patterns: Sequence[Any],
    node_to_supernode: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.51,
) -> tuple[List[LoukasPatternDetection], Dict[str, Dict[str, Any]]]:
    """Evaluate Pattern.compute_detection_metrics at the final coarsening level."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if labels.numel() != node_to_supernode.numel():
        raise ValueError(
            "labels and node_to_supernode must refer to original graph nodes"
        )

    classes = max(2, int(labels.max().item()) + 1)
    pseudo_labels = F.one_hot(labels.to(torch.long), num_classes=classes).to(
        torch.float32
    )
    results: List[LoukasPatternDetection] = []
    grouped: Dict[str, Dict[str, List[LoukasPatternDetection]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pattern in patterns:
        # Test patterns are fresh loader objects, but clear this explicitly when
        # callers evaluate the same objects more than once.
        pattern.level_data.clear()
        metrics = pattern.capture_level(
            node_to_supernode=node_to_supernode,
            pseudo_labels=pseudo_labels,
        )
        recall, precision = float(metrics["recall"]), float(metrics["precision"])
        result = LoukasPatternDetection(
            pattern_id=str(pattern.id),
            pattern_type=str(pattern.pattern_type),
            label=str(pattern.label),
            recall=recall,
            precision=precision,
            f1=float(metrics["f1"]),
            detected=recall > threshold and precision > threshold,
        )
        results.append(result)
        grouped[result.label][result.pattern_type].append(result)

    by_label: Dict[str, Dict[str, Any]] = {}
    for label, by_type in sorted(grouped.items()):
        type_metrics: Dict[str, Dict[str, float]] = {}
        all_entries: List[LoukasPatternDetection] = []
        for pattern_type, entries in sorted(by_type.items()):
            detected = sum(entry.detected for entry in entries)
            type_metrics[pattern_type] = {
                "detected": detected,
                "total": len(entries),
                "detection_rate": detected / len(entries),
                "mean_recall": sum(entry.recall for entry in entries) / len(entries),
                "mean_precision": sum(entry.precision for entry in entries)
                / len(entries),
            }
            all_entries.extend(entries)
        detected = sum(entry.detected for entry in all_entries)
        by_label[label] = {
            "detected": detected,
            "total": len(all_entries),
            "detection_rate": detected / len(all_entries),
            "mean_recall": sum(entry.recall for entry in all_entries)
            / len(all_entries),
            "mean_precision": sum(entry.precision for entry in all_entries)
            / len(all_entries),
            "by_pattern_type": type_metrics,
        }
    return results, by_label


def graph_operators(graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the SGC and Loukas sparse graph operators from a loader graph."""

    normalized = normalized_adjacency(
        graph.edge_index, int(graph.num_nodes), getattr(graph, "edge_weight", None)
    )
    adjacency = _symmetric_adjacency_without_loops(
        graph.edge_index, int(graph.num_nodes), getattr(graph, "edge_weight", None)
    )
    return normalized, adjacency
