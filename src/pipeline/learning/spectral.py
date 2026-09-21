r"""The unsupervised static spectral target: ``R = span(U_q)``.

This is the baseline the paper names in *Graph coarsening and Loukas' RSA* --
"the unsupervised choice :math:`\mathcal R = \mathrm{span}(\bm U_K)`" -- made
available to the pipeline as an ordinary :class:`~src.pipeline.learning.Learning`
architecture, so it travels through exactly the same data, coarsening,
evaluation, group-label and logging path as a learned bank and the *only*
variable that changes is the subspace handed to the coarsener.

The convention, fixed by Sec. "Preliminaries" of the paper
-----------------------------------------------------------
``W~ = W + I``, ``D~ = diag(W~ 1)``, and

    A_hat = D~^{-1/2} W~ D~^{-1/2},      L = I - A_hat = U Lambda U^T,

whose eigenvalues are ordered **smallest first**,
``0 = lambda_0 <= ... <= lambda_{N-1} = lambda_max < 2``, with
``U_K = [u_0, ..., u_{K-1}]`` the first ``K``.  So "top ``q`` eigenvectors"
means the ``q`` **lowest-frequency** eigenvectors of the *symmetric normalized,
self-loop-augmented* Laplacian -- not of ``W``, not of ``A_hat`` read the other
way round, and not of the combinatorial ``D - W``.  It is the same operator
:func:`src.loukas_sgc_detection._normalized_laplacian` and
:func:`src.smooth_dual_ward.screened_operators` build, so the target and the
metric ``M_tau = L + tau I`` the coarsener scores in are consistent by
construction.

``u_0`` is kept.  It is the constant direction ``D~^{1/2} 1 / ||.||``, it carries
no group information, and it is the first of "the first ``K``" -- dropping it
would be a different convention, so it stays and is logged.

No ``M_tau``-orthonormalization happens here.  The coarsener applies its own
(``m_orthonormal_basis`` / ``_l_orthonormalize``) to whatever basis it is given,
and giving the static arm the same treatment as the learned one is the point.
``U_q`` is already Euclidean-orthonormal, so that step is a well-conditioned
rescaling by ``(lambda_i + tau)^{-1/2}``.

Eigensolver
-----------
Shift-invert Lanczos on ``L`` itself: ``eigsh(L, k=q + 16, sigma=-1e-3,
which="LM")`` factorizes ``L - sigma I`` once (sparse LU; ``sigma < 0`` keeps it
definite although ``L`` is singular) and iterates on ``(L - sigma I)^{-1}``,
whose largest eigenvalues are ``L``'s smallest and are well separated.  The
``16`` extra pairs are a buffer for a cluster straddling the ``q``-th
eigenvalue; the lowest ``q`` are kept, and every kept pair must satisfy
``||L u - lambda u|| < 1e-8`` or the call raises.

The obvious alternative -- ``eigsh(A_hat, k=q, which="LA")``, using
``lambda_L = 1 - lambda_A`` -- is **not** safe here and was measured failing:
on a 3,000-node synthetic graph with ``q = 256`` it returned a set whose
eigenvalues were off by up to 0.10 from the dense ground truth (it missed part
of a cluster), and a different set on a second call.  Shift-invert matched the
dense spectrum to 2e-14 on every graph tried.  A dense ``eigh`` is used
up to ``_DENSE_MAX_NODES`` nodes; which path ran is recorded.

Reproducibility: the dense path is bitwise reproducible.  The sparse path is not,
even with the pinned start vector: ARPACK keeps internal Fortran state between
calls (including the generator it uses for restart vectors), so a later call in
the same process can return a different basis of a degenerate eigenspace than a
fresh process would.  The *span* agrees to machine precision, which is all the
coarsener uses in exact arithmetic -- but when the target is so narrow that the
deflated scores are ~0 (q = 8 on a 2,000-node graph) the greedy breaks its ties
on that last-bit difference and the tree changes.  Measured, not hypothetical;
it is why the dense path is the default wherever it is affordable.

Nothing here reads a group, a label or a split: the subspace is a function of
the graph alone, computed independently per graph, never transferred.
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from src.pipeline.learning.base import Learning
from src.utils.utils import LOGGER

__all__ = ["StaticSpectralLearning", "spectral_basis"]

#: up to this many nodes the dense ``eigh`` is used: it is exact, it is
#: bitwise reproducible (LAPACK has no hidden state), and at these sizes it is also
#: the faster route (~2 s at N = 3,000).  Above it, shift-invert Lanczos.
_DENSE_MAX_NODES = 5000


class _NoModel(nn.Module):
    """A parameterless stand-in, so the shared ``Learning.run`` path is unchanged."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError("the static spectral target has no forward pass")


def _sparse_a_hat(graph) -> sp.csr_matrix:
    operator = graph.a_hat.coalesce()
    index = operator.indices().cpu().numpy()
    values = operator.values().detach().cpu().numpy().astype(np.float64)
    n = int(operator.shape[0])
    return sp.coo_matrix((values, (index[0], index[1])), shape=(n, n)).tocsr()


def spectral_basis(graph, q: int) -> dict:
    """``U_q`` and its provenance: the ``q`` lowest eigenpairs of ``L = I - A_hat``."""

    n = int(graph.num_nodes)
    requested = int(q)
    k = max(1, min(requested, n - 1))
    started = time.time()
    A_hat = _sparse_a_hat(graph)

    if n <= _DENSE_MAX_NODES or k >= n - 2:
        solver = "dense_eigh"
        values, vectors = np.linalg.eigh(A_hat.toarray())
        order = np.argsort(-values)[:k]  # largest of A_hat = smallest of L
        a_values, U = values[order], vectors[:, order]
    else:
        solver = "sparse_shift_invert"
        from scipy.sparse.linalg import eigsh

        L = (sp.identity(n, format="csc") - A_hat).tocsc()
        extra = max(0, min(16, n - 2 - k))
        # pinned start: ARPACK's default is random, and with degenerate
        # eigenspaces (each extra component adds a lambda = 0) that returns a
        # different basis of the same span on every call.  Not D~^{1/2} 1: it is
        # itself an eigenvector, and a Krylov space started there is invariant.
        v0 = np.random.default_rng(0).standard_normal(n)
        values, vectors = eigsh(L, k=k + extra, sigma=-1e-3, which="LM", v0=v0)
        order = np.argsort(values)[:k]
        l_values, U = values[order], vectors[:, order]
        residual = np.abs(L @ U - U * l_values[None, :]).max()
        if not residual < 1e-8:
            raise RuntimeError(
                f"[{graph.graph_id}] shift-invert Lanczos did not converge: "
                f"max ||L u - lambda u|| = {residual:.2e}"
            )
        a_values = 1.0 - l_values

    lambda_L = 1.0 - a_values  # exact: L = I - A_hat, same eigenvectors
    return {
        "basis": torch.from_numpy(np.ascontiguousarray(U)).to(torch.float64),
        "requested_q": requested,
        "returned_q": int(U.shape[1]),
        "eigenvalues": [float(v) for v in lambda_L],
        "eigenvalue_min": float(lambda_L.min()),
        "eigenvalue_max": float(lambda_L.max()),
        "solver": solver,
        "operator": "L_sym = I - D~^{-1/2}(W+I)D~^{-1/2}",
        "ordering": "ascending in lambda(L) (= descending in lambda(A_hat))",
        "includes_constant_eigenvector": True,
        "seconds": time.time() - started,
    }


class StaticSpectralLearning(Learning):
    """``R = span(U_q)``: a graph-dependent subspace with nothing learned.

    ``target_widths`` maps a graph id to the ``q`` to use for that graph -- the
    density sweep fills it with the *learned* arm's effective target rank on the
    same graph, so the two subspaces are dimension-matched.  Graphs missing from
    the map fall back to ``LearningConfig.num_heads * feature_dim``, the width the
    polynomial bank would have produced.
    """

    def __init__(self, config, target_widths: "dict | None" = None):
        super().__init__(config)
        self.target_widths = dict(target_widths or {})
        self.default_width: "int | None" = None
        #: graph id -> the provenance dict of that graph's basis
        self.spectra_: dict = {}

    # -- subclass contract -------------------------------------------------- #
    def build_model(self, feature_dim: int, dtype: torch.dtype) -> nn.Module:
        if self.default_width is None:
            self.default_width = int(self.config.num_heads) * int(feature_dim)
        return _NoModel()

    def encode(self, model: nn.Module, graph, geometry) -> torch.Tensor:
        cached = self.spectra_.get(graph.graph_id)
        if cached is None:
            q = int(
                self.target_widths.get(
                    graph.graph_id, self.default_width or self.config.num_heads
                )
            )
            cached = spectral_basis(graph, q)
            self.spectra_[graph.graph_id] = cached
            LOGGER.info(
                f"  [{graph.graph_id}] static spectral target: q={cached['returned_q']} "
                f"of {cached['requested_q']} requested via {cached['solver']}, "
                f"{cached['operator']}, {cached['ordering']}, lambda in "
                f"[{cached['eigenvalue_min']:.4g}, {cached['eigenvalue_max']:.4g}], "
                f"constant eigenvector kept, {cached['seconds']:.1f}s"
            )
        return cached["basis"].to(graph.features.dtype)

    def fit_closed_form(self, contexts) -> dict:
        """Nothing is fitted; the report records what the subspace is instead.

        Returning a report rather than raising keeps ``training_mode`` and the
        frozen label head shared with the learned arm: both fit the *same* head,
        by the same procedure, on their own level.
        """

        started = time.time()
        for ctx in contexts:
            self.encode(self.model_, ctx.graph, ctx.geometry)
        return {
            "solver": "static_spectral",
            "learned_parameters": 0,
            "seconds": time.time() - started,
            "per_graph": {
                graph_id: {
                    k: v for k, v in record.items() if k not in ("basis", "eigenvalues")
                }
                for graph_id, record in self.spectra_.items()
            },
        }
