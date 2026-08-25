"""Verify the two screened geometries against their closed forms.

Run this after touching :mod:`src.screened_geometry` or any of the geometry-aware
call sites in :mod:`src.run_collective_bank_detection`.  It checks two things
that are easy to break silently and impossible to notice in a detection score:

1. **The symmetric path is unchanged.**  Passing ``geometry=None`` (or an
   explicit symmetric geometry) must reproduce the old hard-coded
   ``L = I - A_hat`` / degree-weighted-indicator arithmetic *exactly* -- same
   indicators, same Gram kernel, same confusability tables.

2. **The combinatorial path matches the paper.**  ``L = D - W``,
   ``v_S = 1_S/sqrt(|S|)`` with ``||v_S||_L^2 = cut(S)/|S| = Phi(S)`` (eq. 1-2),
   ``M_tau = L + tau I`` applied exactly, ``||vhat_S||_{M_tau} = 1``, the local
   confusability form ``Q_S`` equal to the principal submatrix ``M_tau[S, S]``,
   the fluctuation constraint ``sum_i w_i = 0``, and -- the end-to-end one --
   the table-driven confusability ``chi`` equal to a dense evaluation of its
   definition, in both geometries.

Run::

    conda activate FedStruct
    python -m src.check_screened_geometry
"""

from __future__ import annotations

import sys

import torch

from src.pattern_models import make_patterns
from src.run_collective_bank_detection import (
    _basis_stack,
    _filtered_bank,
    _gang_confusability,
    _graph_bundle,
    _l_apply,
    build_confusability_tables,
    degree_weighted_indicators,
)
from src.screened_geometry import build_geometry, symmetric_geometry

FAILURES: list = []


def check(name: str, ok: bool) -> None:
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        FAILURES.append(name)


def _toy_graph(n: int = 120, radius: float = 0.16, seed: int = 3):
    """A connected geometric graph plus its two operators and a feature matrix."""

    torch.manual_seed(seed)
    pos = torch.rand(n, 2)
    dist = torch.cdist(pos, pos)
    adj = ((dist < radius) & (dist > 0)).double()
    adj = torch.triu(adj, 1)
    adj = adj + adj.T
    for i in range(n - 1):  # a spanning path keeps it connected
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    w = adj.to_sparse_coo().coalesce()
    deg = torch.sparse.sum(w, 1).to_dense()
    d_tilde = deg + 1.0
    a_hat_dense = (adj + torch.eye(n)) / d_tilde.sqrt()[:, None] / d_tilde.sqrt()[None, :]
    return adj, w, deg, a_hat_dense.to_sparse_coo().coalesce()


def main() -> None:
    n, degree, tau = 120, 6, 0.4
    adj, w, deg, a_hat = _toy_graph(n)
    X = torch.randn(n, 6, dtype=torch.float64)
    patterns = make_patterns(
        [list(range(0, 8)), list(range(20, 27)), list(range(60, 70))],
        "alert", "clique", "g",
    )
    bundle_kw = dict(
        degree=degree, tau=tau, basis="chebyshev", conf_active=True,
        conf_delta=0.0, conf_halo_hops=1, keep_dense=False,
    )

    print("\n1. symmetric geometry reproduces the pre-refactor arithmetic")
    g_sym = symmetric_geometry(a_hat, w)
    v_old = degree_weighted_indicators(w, patterns)
    check("indicator identical", torch.equal(v_old, g_sym.indicators(patterns)))
    check("L x identical", torch.equal(_l_apply(a_hat, v_old), g_sym.l_apply(v_old)))
    b_default = _graph_bundle("d", a_hat, w, patterns, X, None, **bundle_kw)
    b_sym = _graph_bundle("d", a_hat, w, patterns, X, None, geometry=g_sym, **bundle_kw)
    check("Gram kernel identical",
          torch.equal(b_default["gram_kernel"], b_sym["gram_kernel"]))
    check("RHS kernel identical",
          torch.equal(b_default["rhs_kernel"], b_sym["rhs_kernel"]))
    for key in ("Q", "Y", "dtilde"):
        check(f"confusability table {key} identical", all(
            torch.equal(x[key], y[key])
            for x, y in zip(b_default["conf_tables"], b_sym["conf_tables"])
        ))

    print("\n2. combinatorial geometry matches the paper's Section 3")
    g_comb = build_geometry(a_hat, w, "combinatorial")
    lap = torch.diag(deg) - adj
    ones = torch.ones(len(patterns), dtype=torch.float64)
    v_comb = g_comb.indicators(patterns)
    check("||v_S||_2 = 1", torch.allclose(v_comb.norm(dim=0), ones))
    phi = (v_comb * g_comb.l_apply(v_comb)).sum(0)
    cuts = torch.tensor([
        sum(adj[i, j].item() for i in set(p.node_indices)
            for j in range(n) if j not in set(p.node_indices)) / len(p.node_indices)
        for p in patterns
    ], dtype=torch.float64)
    check("Phi(S) = cut(S)/|S|  (eq. 1-2)", torch.allclose(phi, cuts))
    check("L x = (D - W) x exactly", torch.allclose(g_comb.l_apply(X), lap @ X))
    check("M_tau x = (L + tau I) x exactly",
          torch.allclose(g_comb.m_apply(X, tau), lap @ X + tau * X))
    m_vhat = (g_comb.l_apply(v_comb) + tau * v_comb) / (phi + tau).sqrt()
    v_hat = v_comb / (phi + tau).sqrt()
    check("||vhat_S||_{M_tau} = 1", torch.allclose((v_hat * m_vhat).sum(0), ones))
    check("spec(propagation) in [-1, 1]", bool(
        torch.linalg.eigvalsh(g_comb.prop.to_dense()).abs().max() <= 1.0 + 1e-9
    ))

    print("\n3. per-gang local forms are exact restrictions of M_tau")
    b_comb = _graph_bundle("d", a_hat, w, patterns, X, None, geometry=g_comb, **bundle_kw)
    m_dense = lap + tau * torch.eye(n, dtype=torch.float64)
    check("combinatorial Q_S = M_tau[S, S]", all(
        torch.allclose(t["Q"], m_dense[torch.tensor(p.node_indices)][:, torch.tensor(p.node_indices)])
        for p, t in zip(patterns, b_comb["conf_tables"])
    ))
    check("combinatorial fluctuation constraint is sum_i w_i = 0", all(
        torch.allclose(t["dtilde"], torch.ones_like(t["dtilde"]))
        for t in b_comb["conf_tables"]
    ))
    d_half = torch.diag((deg + 1.0).sqrt())
    m_sym_z = d_half @ (
        torch.eye(n, dtype=torch.float64) - a_hat.to_dense() + tau * torch.eye(n, dtype=torch.float64)
    ) @ d_half
    check("symmetric Q_S = (D^1/2 M_sym D^1/2)[S, S]", all(
        torch.allclose(t["Q"], m_sym_z[torch.tensor(p.node_indices)][:, torch.tensor(p.node_indices)])
        for p, t in zip(patterns, b_sym["conf_tables"])
    ))

    print("\n4. table-driven chi equals the dense definition")
    for name, geo in (("symmetric", g_sym), ("combinatorial", g_comb)):
        torch.manual_seed(11)
        theta = torch.randn(1, degree + 1, X.shape[1], dtype=torch.float64)
        theta = theta / theta.norm(dim=1, keepdim=True)
        prop = _basis_stack(geo.prop, X, degree, "chebyshev", tau, geometry=geo)
        Z = _filtered_bank(prop, theta)
        m_z = geo.m_apply(Z, tau)
        gram = 0.5 * (Z.T @ m_z + (Z.T @ m_z).T)
        ridge = 1e-9 * geo.ridge_scale
        chol = torch.linalg.cholesky(
            gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
        )
        tables = build_confusability_tables(
            a_hat, w, patterns, X, tau=tau, degree=degree, basis="chebyshev",
            geometry=geo,
        )
        metric = 0.5 * (geo.m_apply(torch.eye(n, dtype=torch.float64), tau)
                        + geo.m_apply(torch.eye(n, dtype=torch.float64), tau).T)
        indicators = geo.indicators(patterns)
        for j, p in enumerate(patterns):
            nodes = torch.tensor(p.node_indices)
            size = nodes.numel()
            embed = torch.zeros(n, size, dtype=torch.float64)
            embed[nodes, torch.arange(size)] = 1.0
            constraint = indicators[:, j] @ embed
            # orthonormal basis of F_S = {supp in S, <w, v_S>_2 = 0}
            basis_f = embed @ torch.linalg.svd(
                (constraint / constraint.norm()).reshape(size, 1), full_matrices=True
            ).U[:, 1:]
            num = basis_f.T @ m_z @ torch.cholesky_solve(m_z.T @ basis_f, chol)
            den = basis_f.T @ metric @ basis_f
            chol_den = torch.linalg.cholesky(0.5 * (den + den.T))
            whitened = torch.linalg.solve_triangular(
                chol_den,
                torch.linalg.solve_triangular(
                    chol_den, 0.5 * (num + num.T), upper=False
                ).T,
                upper=False,
            ).T
            reference = float(torch.linalg.eigvalsh(0.5 * (whitened + whitened.T))[-1])
            table_chi = float(_gang_confusability(theta, chol, tables[j], 1e-300))
            check(
                f"{name} gang {j}: chi {table_chi:.8f} == dense {reference:.8f}, in [0, 1]",
                abs(table_chi - reference) < 1e-7 and table_chi <= 1.0 + 1e-8,
            )

    print("\n5. the block-averaging projector Pi_P vs each geometry's zero mode")
    # src.loukas_sgc_detection._exact_rsa_epsilon (and _reduce_basis) average each
    # supernode UNIFORMLY.  That projector fixes the constant vector, which IS the
    # kernel of L = D - W but is NOT the kernel of L_sym = I - A_hat (whose zero
    # mode is D_tilde^{1/2} 1).  So Pi_P u0 = u0 -- the property the paper asks of
    # the coarsening -- holds under the combinatorial convention and fails under
    # the symmetric one, for any partition that mixes degrees within a block.
    torch.manual_seed(5)
    labels = torch.randint(0, n // 4, (n,))

    def block_average(vec: torch.Tensor) -> torch.Tensor:
        n_blocks = int(labels.max()) + 1
        sums = torch.zeros(n_blocks, dtype=vec.dtype).index_add_(0, labels, vec)
        counts = torch.bincount(labels, minlength=n_blocks).to(vec.dtype).clamp_min(1.0)
        return (sums / counts)[labels]

    u0_comb = torch.ones(n, dtype=torch.float64) / (n ** 0.5)
    u0_sym = (deg + 1.0).sqrt()
    u0_sym = u0_sym / u0_sym.norm()
    comb_err = float((block_average(u0_comb) - u0_comb).norm())
    sym_err = float((block_average(u0_sym) - u0_sym).norm())
    print(f"        ||Pi_P u0 - u0||: combinatorial {comb_err:.2e}   symmetric {sym_err:.2e}")
    check("uniform Pi_P preserves the combinatorial zero mode", comb_err < 1e-12)
    check(
        "uniform Pi_P does NOT preserve the symmetric zero mode "
        "(pre-existing mismatch, reported not fixed)",
        sym_err > 1e-6,
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("all screened-geometry checks passed")


if __name__ == "__main__":
    main()
