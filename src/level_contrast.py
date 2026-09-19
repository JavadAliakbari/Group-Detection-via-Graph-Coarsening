r"""Level-contrastive training term: shape ``ell`` directly, not just ``Gamma``.

The coarsener never sees ``Gamma``.  What it sees is the **level**

    ell = D_tilde^{-1/2} M_tau Z,      M_tau = L_sym + tau I,      G = Z^T M_tau Z,

through the rank-``q`` raw-Ward score of an adjacent pair (:mod:`src.raw_ward`)

    s^0(u, v) = m(u,v)^2 || ell_u - ell_v ||^2_{G^-1} / || h_{u,v} ||^2_{M_tau} ,

whose graph half ``c_{uv} := m(u,v)^2 / (ell(u,v) + tau)`` is **independent of
Theta**.  So the only thing training can move is the whitened level difference
``|| ell_u - ell_v ||^2_{G^-1}``, and ``s^0`` is a differentiable function of
``Theta`` that costs one ``q x q`` Cholesky solve per sampled pair.

That makes the following losses available directly:

* **within**  -- ``s^0`` small on edges *inside* a training group: the level should
  be flat across a group, so no internal merge is ever expensive;
* **boundary** -- ``s^0`` large on edges from a group to a neighbouring
  non-member: the level should jump at the group boundary, so those merges are
  the expensive ones;
* **negative** -- ``s^0`` small on *random* node pairs: the level should be flat
  on the host.  This is the term to be tested, and it is not arbitrary: the
  idealized screened-dual target ``Z* = M_tau^{-1} V*`` has level
  ``ell* = D_tilde^{-1/2} V*``, which is constant inside each group and flat
  (zero) everywhere else.  Pulling random pairs together pushes ``ell`` toward
  exactly that shape.  It can also backfire, by flattening the level so far that
  the boundary jump goes with it -- which is what the sweep measures.

Two forms are provided.  ``form="margin"`` is the literal reading,
``L = mean s^0_int - w_b * mean s^0_bnd + w_n * mean s^0_neg``; every term is in
``[0, 1]`` because ``s^0`` is a normalized projection, so the loss cannot run
away.  ``form="rank"`` replaces the first two by the pairwise logistic surrogate
``mean softplus((s^0_int - s^0_bnd)/T)``, which is a direct surrogate for
``AUC = P(s^0_int < s^0_bnd)`` -- the statistic that actually decides whether the
greedy merges a group before it leaks (see :mod:`src.diag_ward_score_gap`).

Symmetric geometry only: the level, the volumes and ``c_{uv}`` are all stated in
``M_tau = L_sym + tau I``.

Run ``python -m src.level_contrast`` for the validation suite (the loss's ``s^0``
must equal :mod:`src.raw_ward`'s to machine precision).
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "build_level_pairs",
    "level_scores",
    "level_contrast_loss",
    "run_validation_suite",
]


def _graph_constants(adjacency: torch.Tensor):
    """``(deg, d_tilde)`` for the symmetric convention (``W`` has no self-loops)."""

    a = adjacency.coalesce()
    idx, val = a.indices(), a.values().to(torch.float64)
    n = int(a.shape[0])
    off = idx[0] != idx[1]
    deg = torch.zeros(n, dtype=torch.float64, device=a.device)
    deg.scatter_add_(0, idx[0][off], val[off])
    return deg, deg + 1.0


def _pair_coeff(u, v, deg, d_tilde, w_uv, tau):
    r"""``c_{uv} = m(u,v)^2 / (ell(u,v) + tau)`` -- the Theta-free half of ``s^0``.

    ``ell(u,v) = [d~_v Phi_u + d~_u Phi_v + 2 w~(u,v)] / (d~_u + d~_v)`` with
    ``Phi_v = deg_v / d~_v`` at the singleton partition, so ``ell + tau =
    ||g_{u,v}||^2_{M_tau}`` exactly (:mod:`src.raw_ward`).
    """

    da, db = d_tilde[u], d_tilde[v]
    s = da + db
    m2 = da * db / s
    ell = (db * (deg[u] / da) + da * (deg[v] / db) + 2.0 * w_uv) / s
    return m2 / (ell.clamp_min(0.0) + tau)


def build_level_pairs(
    adjacency: torch.Tensor,
    patterns: list,
    *,
    tau: float,
    budget: int = 4096,
    seed: int = 0,
    neg_kind: str = "pairs",
    neg_scope: str = "all",
) -> dict:
    """Sample the internal / boundary / negative index sets and their coefficients.

    ``neg_kind="pairs"`` draws uniformly random *node pairs* (the literal "random
    nodes"); ``neg_kind="edges"`` draws random *edges* with neither endpoint in a
    group -- the host-host merges the coarsener actually scores.  ``neg_scope``
    restricts the node pool to ``"host"`` (non-group nodes) or leaves it at
    ``"all"``.

    Returns tensors on ``adjacency``'s device; empty groups simply yield empty
    index sets, which the loss skips.
    """

    a = adjacency.coalesce()
    n = int(a.shape[0])
    dev = a.device
    idx = a.indices()
    val = a.values().to(torch.float64)
    keep = idx[0] < idx[1]  # each undirected edge once
    eu, ev, ew = idx[0][keep], idx[1][keep], val[keep]
    deg, d_tilde = _graph_constants(adjacency)

    gang_of = torch.full((n,), -1, dtype=torch.long, device=dev)
    for j, p in enumerate(patterns):
        gang_of[torch.as_tensor(list(p.node_indices), dtype=torch.long, device=dev)] = j

    gu, gv = gang_of[eu], gang_of[ev]
    is_int = (gu >= 0) & (gu == gv)
    is_bnd = ((gu >= 0) & (gv < 0)) | ((gu < 0) & (gv >= 0))
    is_host = (gu < 0) & (gv < 0)

    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def _take(mask):
        pos = torch.nonzero(mask, as_tuple=False).ravel()
        if pos.numel() > budget:
            sel = torch.randperm(pos.numel(), generator=g)[:budget].to(dev)
            pos = pos[sel]
        return pos

    out: dict = {}
    for name, mask in (("int", is_int), ("bnd", is_bnd)):
        pos = _take(mask)
        u, v, w = eu[pos], ev[pos], ew[pos]
        out[f"{name}_u"], out[f"{name}_v"] = u, v
        out[f"{name}_c"] = _pair_coeff(u, v, deg, d_tilde, w, tau)

    if neg_kind == "edges":
        pos = _take(is_host)
        u, v, w = eu[pos], ev[pos], ew[pos]
    else:  # random node pairs -- which may still happen to be adjacent
        pool = (
            torch.nonzero(gang_of < 0, as_tuple=False).ravel()
            if neg_scope == "host"
            else torch.arange(n, device=dev)
        )
        if pool.numel() < 2:
            pool = torch.arange(n, device=dev)
        k = min(budget, max(pool.numel(), 2))
        iu = pool[torch.randint(pool.numel(), (k,), generator=g).to(dev)]
        iv = pool[torch.randint(pool.numel(), (k,), generator=g).to(dev)]
        ok = iu != iv
        u, v = torch.minimum(iu[ok], iv[ok]), torch.maximum(iu[ok], iv[ok])
        # A random pair is USUALLY non-adjacent, but on a dense graph it often is
        # not, and w~(u,v) enters c_{uv}.  Look the weight up exactly rather than
        # assuming zero: sorted edge keys + binary search.
        key = (eu * n + ev).contiguous()
        order = torch.argsort(key)
        key_sorted, w_sorted = key[order], ew[order]
        probe = u * n + v
        j = torch.searchsorted(key_sorted, probe.contiguous())
        j_clamped = j.clamp(max=max(key_sorted.numel() - 1, 0))
        hit = (key_sorted.numel() > 0) & (key_sorted[j_clamped] == probe)
        w = torch.where(hit, w_sorted[j_clamped], torch.zeros_like(ew[:1].expand(u.numel())))
    out["neg_u"], out["neg_v"] = u, v
    out["neg_c"] = _pair_coeff(u, v, deg, d_tilde, w, tau)
    out["d_tilde"] = d_tilde
    out["counts"] = {
        "int": int(out["int_u"].numel()),
        "bnd": int(out["bnd_u"].numel()),
        "neg": int(out["neg_u"].numel()),
    }
    return out


def level_scores(Z, m_z, d_tilde, u, v, c, ridge: float = 1e-9):
    r"""``s^0(u, v)`` for the sampled pairs -- differentiable in ``Z``.

    ``ell = D_t^{-1/2} m_z`` with ``m_z = M_tau Z``; the ``G^{-1}`` norm is taken
    by one Cholesky solve on ``G = Z^T M_tau Z`` rather than a matrix square root,
    so nothing but ``q x q`` linear algebra enters the graph.
    """

    G = Z.T @ m_z
    G = 0.5 * (G + G.T)
    scale = torch.diagonal(G).mean().clamp_min(torch.finfo(G.dtype).eps)
    chol = torch.linalg.cholesky(
        G + (ridge * scale) * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    )
    ell = m_z / d_tilde.to(m_z.dtype).sqrt().unsqueeze(1)
    D = ell[u] - ell[v]  # (P, q)
    X = torch.cholesky_solve(D.T, chol)  # G^{-1} D^T
    quad = (D.T * X).sum(dim=0)
    return c.to(quad.dtype) * quad


def level_contrast_loss(
    Z,
    m_z,
    pairs: dict,
    *,
    ridge: float = 1e-9,
    form: str = "margin",
    boundary_weight: float = 1.0,
    negative_weight: float = 0.0,
    temperature: float = 0.05,
):
    """``(loss, parts)`` -- the level-contrastive loss to be **minimized**.

    ``parts`` carries the three mean scores so the caller can log what actually
    moved (a falling ``int`` with a falling ``bnd`` is the level collapsing, not
    the objective working).
    """

    d_t = pairs["d_tilde"]
    zero = Z.new_zeros(())

    def _mean(tag):
        u = pairs[f"{tag}_u"]
        if u.numel() == 0:
            return None
        return level_scores(
            Z, m_z, d_t, u, pairs[f"{tag}_v"], pairs[f"{tag}_c"], ridge
        )

    s_int, s_bnd, s_neg = _mean("int"), _mean("bnd"), _mean("neg")
    parts = {
        "int": float(s_int.mean().detach()) if s_int is not None else float("nan"),
        "bnd": float(s_bnd.mean().detach()) if s_bnd is not None else float("nan"),
        "neg": float(s_neg.mean().detach()) if s_neg is not None else float("nan"),
    }
    if s_int is not None and s_bnd is not None:
        with torch.no_grad():
            parts["auc"] = float(
                (s_int.detach()[:, None] < s_bnd.detach()[None, :]).double().mean()
            )

    if form == "rank":
        if s_int is None or s_bnd is None:
            loss = zero
        else:
            loss = torch.nn.functional.softplus(
                (s_int[:, None] - s_bnd[None, :]) / temperature
            ).mean()
    elif form == "margin":
        loss = zero
        if s_int is not None:
            loss = loss + s_int.mean()
        if s_bnd is not None:
            loss = loss - boundary_weight * s_bnd.mean()
    else:
        raise ValueError("form must be 'margin' or 'rank'")
    if negative_weight != 0.0 and s_neg is not None:
        loss = loss + negative_weight * s_neg.mean()
    return loss, parts


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def run_validation_suite(seed: int = 0, verbose: bool = True) -> None:
    """The loss's ``s^0`` must be exactly :mod:`src.raw_ward`'s."""

    import scipy.sparse as sp

    from src.raw_ward import rank_q_level, screened_operators_kappa

    rng = np.random.default_rng(seed)
    n, q, tau = 60, 5, 0.4
    Wd = np.triu((rng.random((n, n)) < 0.2) * rng.uniform(0.5, 2.0, (n, n)), 1)
    Wd = Wd + Wd.T
    W = sp.csr_matrix(Wd)
    A, kappa, L, M = screened_operators_kappa(W, tau, "symmetric")
    Zn = rng.standard_normal((n, q))
    _ell, _G, level_r, _q = rank_q_level(Zn, M, kappa)

    coo = sp.coo_matrix(W).tocoo()
    ii = torch.from_numpy(np.stack([coo.row, coo.col]).astype(np.int64))
    adjacency = torch.sparse_coo_tensor(
        ii, torch.from_numpy(coo.data.astype(np.float64)), (n, n)
    ).coalesce()

    class _P:
        def __init__(self, nodes):
            self.node_indices = nodes

    pats = [_P(list(range(0, 8))), _P(list(range(8, 15)))]
    pairs = build_level_pairs(adjacency, pats, tau=tau, budget=10_000, seed=1)

    Z = torch.from_numpy(Zn)
    m_z = torch.from_numpy(np.asarray(M @ Zn))
    worst = 0.0
    for tag in ("int", "bnd", "neg"):
        u, v = pairs[f"{tag}_u"], pairs[f"{tag}_v"]
        if u.numel() == 0:
            continue
        got = level_scores(
            Z, m_z, pairs["d_tilde"], u, v, pairs[f"{tag}_c"], ridge=0.0
        ).numpy()
        un, vn = u.numpy(), v.numpy()
        ka, kb = kappa[un], kappa[vn]
        m2 = ka * kb / (ka + kb)
        want = np.empty(un.size)
        for t in range(un.size):
            g = np.zeros(n)
            g[un[t]] = np.sqrt(ka[t]) / ka[t]
            g[vn[t]] = -np.sqrt(kb[t]) / kb[t]
            g *= np.sqrt(m2[t])
            d = level_r[un[t]] - level_r[vn[t]]
            want[t] = m2[t] * float(d @ d) / float(g @ (M @ g))
        worst = max(worst, float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)))
    assert worst < 1e-9, worst
    if verbose:
        print(f"[1] level_scores == raw_ward s^0        : rel err {worst:.2e}")
        print(f"    pair counts                         : {pairs['counts']}")

    # the score is a normalized projection, so every term of the loss is bounded
    for tag in ("int", "bnd", "neg"):
        u = pairs[f"{tag}_u"]
        if u.numel() == 0:
            continue
        s = level_scores(
            Z, m_z, pairs["d_tilde"], u, pairs[f"{tag}_v"], pairs[f"{tag}_c"], 0.0
        )
        assert float(s.min()) >= -1e-9 and float(s.max()) <= 1 + 1e-9, (
            tag, float(s.min()), float(s.max())
        )
    if verbose:
        print("[2] every s^0 in [0, 1]                 : ok")

    # gradients flow and the loss is finite in both forms
    Zg = torch.from_numpy(Zn).requires_grad_(True)
    for form in ("margin", "rank"):
        loss, parts = level_contrast_loss(
            Zg, torch.from_numpy(np.asarray(M @ Zn)), pairs,
            form=form, negative_weight=0.5,
        )
        gr = torch.autograd.grad(loss, Zg, retain_graph=False)[0]
        assert torch.isfinite(loss) and torch.isfinite(gr).all()
        if verbose:
            print(
                f"[3] form={form:<7} loss {float(loss):+.6f}  "
                f"int {parts['int']:.4f} bnd {parts['bnd']:.4f} "
                f"neg {parts['neg']:.4f} auc {parts.get('auc', float('nan')):.4f}"
            )
    if verbose:
        print("\nALL LEVEL-CONTRAST CHECKS PASSED")


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    run_validation_suite()
