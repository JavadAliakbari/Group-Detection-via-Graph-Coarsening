"""Dataset-agnostic collective filter-bank gang detector.

This is the *algorithm*, separated from any particular graph.  The learnable
Chebyshev filter bank, the collective ``lambda_min(Gamma)`` capture objective,
the confusability margin (eq. 40), the Riemannian/projected optimizers and the
RSA / Ward-tree coarsening all live (and stay tested) in
:mod:`src.run_collective_bank_detection`; this module wraps those *primitives*
into a small reusable API with the graph data factored out:

* :class:`DetectorConfig` -- every hyperparameter (no argparse, no globals).
* :class:`GraphData`      -- the only dataset-specific object: the normalized
  adjacency ``a_hat``, raw adjacency, node features ``X`` and node labels ``y``.
  Build it from any ``torch_geometric``-style graph via :meth:`GraphData.from_graph`.
* :class:`CollectiveBankDetector` -- ``fit`` -> ``target_subspace`` -> ``coarsen``
  -> ``evaluate``, or the one-shot :meth:`run`.  Nothing here knows or cares
  whether the gangs came from a synthetic planter or from Elliptic++.

Apply to a new dataset in three lines::

    data = GraphData.from_graph(my_graph)            # any graph
    det = CollectiveBankDetector(DetectorConfig(tau=0.3, conf_weight=5.0))
    result = det.run(data, train_patterns, test_patterns, all_patterns=gangs)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.loukas_sgc_detection import (
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.run_collective_bank_detection import (
    _basis_stack,
    _m_apply,
    _train_gang_m_vhat,
    build_bank_subspace,
    channel_gram_cond,
    fit_collective_bank,
    make_negative_sampler,
    retained_energy,
    ward_tree_coarsen,
)


# --------------------------------------------------------------------------- #
# configuration (all algorithm knobs; no argparse, no module globals)
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """Every hyperparameter of the collective-bank detector."""

    # --- learnable filter bank ------------------------------------------------
    degree: int = 10  # Chebyshev/monomial degree K
    basis: str = "chebyshev"  # "chebyshev" | "monomial" | "lanczos"
    tau: float = 0.3  # screened metric M_tau = L + tau I
    epochs: int = 800
    learning_rate: float = 0.02
    ridge: float = 1e-4
    optimizer: str = "projected"  # "projected" | "riemannian" | "lbfgs"
    softmin_temperature: float = 0.2  # 0 = hard lambda_min
    # "ones" = flat low-pass init; "closed_form" = best single filter consistent
    # with the per-gang Theorem 6.2 optima (per-channel top singular vector)
    warm_start: str = "ones"
    # >1 starts the soft-min at softmin_temperature * softmin_anneal and anneals
    # geometrically down to it (warm early spreads gradient over the spectrum)
    softmin_anneal: float = 1.0
    # what the bank ascends: "lambda_min" (capture + cross-gang separation, carries
    # the m>d capacity wall) | "trace" (mean per-gang capture, no separation term,
    # no capacity wall) | "softmin_diag" (worst gang's capture, no separation).
    # trace/softmin_diag rest on the connectivity constraint: a local-variation
    # coarsener never merges non-adjacent gangs, so cross-gang separation is free
    # (Prop 8.5) and only neighbour separation (the confusability chi) is needed.
    capture_objective: str = "lambda_min"
    # how the collective target is obtained:
    #   "gradient"    -- ascend capture_objective (soft-min lambda_min etc.) by
    #                    Adam/L-BFGS on the filter bank  (the classic path)
    #   "closed-form" -- Theta_beta = (G + beta*W_all)^{-1} Bhat: one shared
    #                    factorization + m linear solves, no eigen-ascent.  At
    #                    beta=0 this is the EXACT maximizer of lambda_min(Gamma)
    #                    over every target in the dictionary (Theorem A), and for
    #                    beta>0 it comes with the certified sandwich
    #                    N_beta <= Gamma <= N_0 plus chi <= (r_j-1)/beta.
    #   "trace-ratio"  -- Dinkelbach/trace-ratio iteration on the SAME bank
    #                     class: each step is d small (K+1)x(K+1) eigenproblems,
    #                     heads come out orthogonal per channel by construction.
    #                     Tens of eigensolves instead of hundreds of epochs.
    #   "aeq"         -- the signal-to-confusion generalized eigenproblem
    #                    A_eq w = lambda (H + rho I) w with A_eq = Bbar(Bbar^T
    #                    Bbar + alpha I)^{-1} Bbar^T and H the group-size
    #                    normalized mean confusability.  One eigendecomposition,
    #                    global optimum, still informative when m > d (where
    #                    lambda_min is identically 0), and every retained
    #                    direction carries an interpretable capture-per-
    #                    confusability eigenvalue.
    collective_solver: str = "gradient"
    pencil_beta: float = 0.0  # beta of the closed-form solver
    # --- A_eq pencil (collective_solver="aeq") -------------------------------
    aeq_alpha: float = 1e-3  # ridge inside A_eq (0+ = projector onto span Bbar)
    aeq_rho: float = 1e-3  # ridge on H; keeps the ratio finite on ker H
    aeq_width: int = 0  # target width d (0 = m, one per training group)
    aeq_lambda_floor: float = 0.0  # drop directions with lambda_k below this
    aeq_reweight_iters: int = 0  # >0: reweight H towards the worst-case chi
    aeq_kappa: float = 5.0  # sharpness of the reweighting softmax
    # "relative": rho is a multiple of the mean eigenvalue tr(H)/r (scale-free
    # across graphs); "absolute": rho as written
    aeq_rho_scale: str = "relative"
    trace_ratio_iters: int = 40  # iterations of the trace-ratio solver
    # multi-graph training: "sample" (one graph per epoch, stochastic), "mean"
    # (average over all graphs, deterministic) or "min" (worst graph, maximin)
    day_aggregate: str = "sample"
    # number of filter heads H: the bank is Theta (H, K+1, d) and the target is
    # the concatenated span of the H heads (H*d columns).  heads=1 is the single
    # shared filter (identical to the pre-multi-head behaviour); H>1 lets gangs
    # stop competing for one hop-profile per channel.  Same objective, epochs and
    # optimizer either way.
    heads: int = 1
    # >0 subtracts head_diversity * mean squared cosine between distinct heads,
    # pushing the bank off the degenerate all-heads-equal critical point
    head_diversity: float = 0.0
    # optional supervised head on the same embedding, trained jointly with theta
    label_weight: float = 0.0

    # --- confusability margin (eq. 40) ---------------------------------------
    conf_weight: float = 0.0  # beta; 0 = pure detect-all objective
    conf_reduce: str = "max"  # "max" (eq. 40) | "mean"
    conf_delta: float = 0.0  # 0 = hard chi^tau; >0 = delta-leaky cone
    conf_halo_hops: int = 1

    # --- negative "repeller" sets --------------------------------------------
    num_neg: int = 0
    neg_weight: float = 0.0
    neg_temperature: float = 0.1
    neg_size_min: int = 3
    neg_size_max: int = 10

    # --- target subspace handed to the coarsener -----------------------------
    # "bank"            span(Z) of the learned bank (heads*d cols, inductive)
    # "indicators"      v_hat projected onto span(Z) (capped at bank capture)
    # "dictionary"      Theorem 6.2 closed-form projection onto the FULL
    #                   dictionary (per-gang capture ceiling; needs node sets)
    # "bank+dictionary" both concatenated
    coarsen_target: str = "bank"
    structural_width: int = 0
    # gang signal the REPORTED capture is a fraction of.  "geometry" takes it from
    # coarsening_laplacian, so the reported number is always the one the fit
    # ascended (degree-weighted under symmetric, uniform under combinatorial);
    # "degree_weighted"/"plain" pin it explicitly for cross-geometry reporting.
    indicator: str = "geometry"

    # --- coarsening + detection ----------------------------------------------
    coarsening_method: str = "ward-tree"
    # --- screened-consistent (deflated) agglomeration ------------------------
    # coarsening_method "deflated-dual-ward" / "deflated-minimax"
    # (:mod:`src.deflated_coarsen`): merges are scored against the M_tau-ORTHOGONAL
    # (harmonic) block projector Q_P^tau instead of the Euclidean block average, so
    # the merge calculus is exactly rank-one PSD and the reported eps_Q is the exact
    # screened RSA of the harmonic reconstruction.  The realized coarsening is still
    # the Euclidean block averaging; the two are tied by the RSA sandwich
    # eps_Q <= eps_Pi <= mu_P^tau eps_Q, all three of which are reported.
    deflated_hops: int = 2  # r-hop coarse ball of the truncated deflation solve
    deflated_max_ball: int = 32  # cap on that ball (hub blocks)
    deflated_max_rescore: int = 8  # lazy-queue re-evaluations per merge
    deflated_fanout: int = 32  # queue entries pushed per merge (0 = all)
    deflated_max_cluster_size: int = 0  # 0 = uncapped
    # which RSA constant the --epsilon budget is spent in: "epsilon_pi" is the
    # realized Euclidean one every other method reports (comparable), "epsilon_q"
    # is the intrinsic harmonic one the algorithm optimizes (monotone).
    deflated_epsilon_key: str = "epsilon_pi"
    deflated_certify: bool = True  # exact eps_Q + mu at the chosen cut
    # THE geometry switch: it selects the screened metric M_tau the filter bank is
    # trained in, the group indicator v_S the capture/confusability are fractions
    # of, the operator the Chebyshev bank propagates on, AND the metric the RSA
    # distortion of the coarsening is measured in -- see src.screened_geometry.
    # (The name is historical: it used to reach the coarsener only.)
    #   "symmetric"     L = I - A_hat,  v_S = D_tilde^{1/2} 1_S / sqrt(vol(S)),
    #                   Phi(S) = cut/vol  (conductance)
    #   "combinatorial" L = D - W,      v_S = 1_S / sqrt(|S|),
    #                   Phi(S) = cut/|S|  (the paper's Section 3)
    coarsening_laplacian: str = "symmetric"  # "symmetric" | "combinatorial"
    # operator the bank filters on: "auto" follows coarsening_laplacian (the
    # paper's pairing), "a_hat" pins propagation to A_hat so a combinatorial run
    # differs from a symmetric one ONLY in the metric and the indicator.
    propagation: str = "auto"  # "auto" | "a_hat"
    reduction: float = 0.7
    epsilon: float | None = 0.5  # RSA distortion budget (None -> reduction only)
    epsilon_ramp_levels: int = 5
    max_levels: int = 1000
    ward_stop: str = "epsilon"  # "epsilon" | "f1"
    ward_num_cuts: int = 200
    threshold: float = 0.51
    # Smooth Dual Ward (coarsening_method="dual-ward"; see src.smooth_dual_ward)
    dual_ward_alpha: float = 0.0  # 0 = normalized sigma_DW, 1 = raw Delta_DW
    dual_ward_tau: float = 0.1  # screening level of M_tau = L_sym + tau I
    dual_ward_max_size: int = 0  # super-node cardinality cap (0 = uncapped)
    dual_ward_embedding: str = "dual"  # "dual" (M_tau U_tau) | "primal" (U_tau)

    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# graph data (the only dataset-specific object)
# --------------------------------------------------------------------------- #
@dataclass
class GraphData:
    """Everything the algorithm needs from a graph, and nothing else.

    * ``a_hat``     -- symmetric-normalized adjacency (self-loops), the operator
      the filter bank propagates on.
    * ``adjacency`` -- raw sparse weight matrix ``W`` (for degrees / coarsening).
    * ``X``         -- node feature matrix ``(N, d)`` (the filter's input signal).
    * ``y``         -- node class labels ``(N,)`` (gang nodes marked ``1``); used
      only for evaluation and, optionally, negative sampling.
    """

    edge_index: torch.Tensor
    a_hat: torch.Tensor
    adjacency: torch.Tensor
    X: torch.Tensor
    y: torch.Tensor

    @classmethod
    def from_graph(
        cls, graph, *, features: "torch.Tensor | None" = None
    ) -> "GraphData":
        """Build from any ``torch_geometric``-style graph (``edge_index``, ``x``, ``y``).

        ``features`` overrides ``graph.x`` (e.g. to swap real node features for a
        random structural range-finder).  The features are cast to the operator's
        dtype/device so the bank stays numerically consistent.
        """
        edge_index = graph.edge_index
        a_hat, adjacency = graph_operators(graph)
        X = graph.x if features is None else features
        X = X.to(device=a_hat.device, dtype=a_hat.dtype)
        y = graph.y.to(a_hat.device)
        return cls(edge_index=edge_index, a_hat=a_hat, adjacency=adjacency, X=X, y=y)

    @property
    def num_nodes(self) -> int:
        return int(self.a_hat.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.X.shape[1])


# --------------------------------------------------------------------------- #
# the algorithm
# --------------------------------------------------------------------------- #
class CollectiveBankDetector:
    """Collective learnable filter-bank gang detector (dataset-agnostic).

    Usage::

        det = CollectiveBankDetector(DetectorConfig(...))
        det.fit(data, train_patterns)               # learn Theta*
        basis = det.target_subspace(data, train_patterns)   # R = span(Z)
        coarsening, traj = det.coarsen(data, basis, train_patterns)
        report = det.evaluate(data, coarsening, {"train": ..., "test": ...})

    or the one-shot :meth:`run`.  After :meth:`fit`, ``theta_`` and ``fit_info_``
    hold the learned filter and its training diagnostics.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.theta_: torch.Tensor | None = None
        self.fit_info_: dict | None = None
        # closed-form solver: dictionary coefficients (P, m), frozen and reused
        # on transfer days exactly like ``theta_``
        self.pencil_theta_: torch.Tensor | None = None
        # one ScreenedGeometry per GraphData (see ``geometry``)
        self._geometry_cache: dict = {}

    # -- internals ---------------------------------------------------------- #
    def geometry(self, data: GraphData):
        """The :class:`~src.screened_geometry.ScreenedGeometry` for one graph.

        Built from ``config.coarsening_laplacian`` / ``config.propagation`` and
        cached, because the combinatorial constructor estimates
        ``lambda_max(D - W)`` and every stage -- fit, target subspace, capture,
        coarsening -- must be handed the *same* object or they would silently
        measure in slightly different metrics.

        The cache is keyed on the *adjacency* rather than the ``GraphData``: the
        geometry already holds the adjacency, so caching adds no retention beyond
        a sparse ``W``, whereas keying on the ``GraphData`` would pin every
        transfer day's feature matrix alive for the whole run.  ``id`` reuse is
        ruled out by re-checking identity against the cached geometry.
        """

        from src.screened_geometry import build_geometry

        key = id(data.adjacency)
        cached = self._geometry_cache.get(key)
        if cached is not None and cached.adjacency is data.adjacency:
            return cached
        geo = build_geometry(
            data.a_hat,
            data.adjacency,
            self.config.coarsening_laplacian,
            propagation=self.config.propagation,
        )
        self._geometry_cache[key] = geo
        return geo

    def _negative_sampler(self, data: GraphData, train_patterns: list):
        c = self.config
        if c.num_neg <= 0 or c.neg_weight <= 0.0:
            return None
        # avoid every known gang node so negatives are genuine background sets
        avoid = torch.nonzero(data.y == 1, as_tuple=False).flatten().tolist()
        for p in train_patterns:
            avoid.extend(int(v) for v in p.node_indices)
        edge_index = data.adjacency.coalesce().indices()
        return make_negative_sampler(
            edge_index,
            data.num_nodes,
            num_sets=c.num_neg,
            size_min=c.neg_size_min,
            size_max=c.neg_size_max,
            avoid=sorted(set(avoid)),
            rng=np.random.default_rng(c.seed + 1),
        )

    @staticmethod
    def _as_day_specs(days, train_patterns) -> list:
        """Normalize the two call shapes into one list of graph specs.

        ``fit`` is day-list-native: ``[(label, GraphData, train_patterns,
        test_patterns), ...]``.  Because a single graph is just a one-element
        list, the convenience form ``fit(data, train_patterns)`` is accepted and
        wrapped here -- this only reshapes the *arguments*; every solver below
        then runs the same code for one graph and for many.
        """

        if isinstance(days, GraphData):
            if train_patterns is None:
                raise TypeError("fit(data, train_patterns) needs train_patterns")
            return [("train", days, train_patterns, None)]
        if train_patterns is not None:
            raise TypeError(
                "pass either fit(day_specs) or fit(data, train_patterns), not both"
            )
        specs = [(s[0], s[1], s[2], s[3] if len(s) > 3 else None) for s in days]
        if not specs:
            raise ValueError("fit() needs at least one training graph")
        return specs

    # -- steps -------------------------------------------------------------- #
    def fit(
        self,
        days,
        train_patterns: "list | None" = None,
        *,
        label_y=None,
        label_idx=None,
        signal_patterns=None,
    ) -> "CollectiveBankDetector":
        """Learn the filter bank ``Theta*`` on one or more training graphs.

        ``days`` is ``[(label, GraphData, train_patterns, test_patterns), ...]``
        -- each entry one graph (a day, or a merged window).  A single graph is a
        one-element list, so ``fit(data, train_patterns)`` is accepted as sugar
        for exactly that; there is no separate single-graph code path.  Every
        solver takes the whole list:

        * ``gradient`` and ``trace-ratio`` draw one graph per step/iteration
          (see ``config.day_aggregate`` for the mean/min alternatives);
        * ``closed-form`` concatenates each graph's dictionary coefficients,
          since ``Theta`` lives in the (graph-independent) dictionary basis and
          the union of the graphs' optimal directions is a valid target on any
          of them.

        ``signal_patterns`` (``collective_solver="aeq"`` only) are extra groups
        whose indicators widen ``A_eq``'s signal span -- a list for a single
        graph, or a ``{day_label: list}`` mapping for several.  ``A_eq``'s rank
        is the number of signal groups, so without them the target can be
        neither wider than ``m`` nor able to hold a group it never saw.

        ``label_y`` / ``label_idx`` optionally attach a supervised node head
        (config ``label_weight`` > 0) trained jointly with the filter; like the
        negative sampler it is a single-graph feature.
        """

        c = self.config
        specs = self._as_day_specs(days, train_patterns)
        geometries = [self.geometry(d) for _lbl, d, _p, _te in specs]

        if c.collective_solver in ("closed-form", "aeq"):
            self._require_symmetric(c.collective_solver)
            from src.margin_pencil import aeq_pencil_theta, collective_pencil_theta

            thetas, reports = [], {}
            for lbl, d, pats, _te in specs:
                if c.collective_solver == "aeq":
                    sig = (signal_patterns.get(lbl)
                           if isinstance(signal_patterns, dict) else signal_patterns)
                    th, rep = aeq_pencil_theta(
                        d.a_hat,
                        d.adjacency,
                        d.X,
                        pats,
                        signal_patterns=sig,
                        degree=c.degree,
                        tau=c.tau,
                        alpha=c.aeq_alpha,
                        rho=c.aeq_rho,
                        width=c.aeq_width,
                        lambda_floor=c.aeq_lambda_floor,
                        reweight_iters=c.aeq_reweight_iters,
                        kappa=c.aeq_kappa,
                        rho_scale=c.aeq_rho_scale,
                        beta=c.pencil_beta,
                        basis=c.basis,
                    )
                else:
                    th, rep = collective_pencil_theta(
                        d.a_hat,
                        d.adjacency,
                        d.X,
                        pats,
                        degree=c.degree,
                        tau=c.tau,
                        beta=c.pencil_beta,
                        basis=c.basis,
                    )
                thetas.append(th)
                reports[lbl] = rep
            # one graph -> the single solve's own coefficients
            self.pencil_theta_ = (
                thetas[0] if len(thetas) == 1 else torch.cat(thetas, dim=1)
            )
            self.theta_ = None  # no filter bank is trained on this path
            base = dict(reports[specs[-1][0]])
            base["per_group"] = reports
            base["train_days"] = [s[0] for s in specs]
            self.fit_info_ = base
            return self

        if c.collective_solver == "trace-ratio":
            self._require_symmetric(c.collective_solver)
            from src.trace_ratio_bank import fit_trace_ratio_bank

            self.fit_info_ = fit_trace_ratio_bank(
                [(lbl, d.a_hat, d.adjacency, pats, d.X) for lbl, d, pats, _te in specs],
                degree=c.degree,
                heads=c.heads,
                iters=c.trace_ratio_iters,
                tau=c.tau,
                ridge=c.ridge,
                basis=c.basis,
                softmin_temperature=c.softmin_temperature,
                conf_weight=c.conf_weight,
            )
            self.theta_ = self.fit_info_["theta"]
            return self

        # The negative sampler is built from ONE graph's edges and node labels.
        # It is passed through as-is: with several graphs ``fit_collective_bank``
        # raises rather than applying one graph's negatives to whichever graph an
        # epoch happened to draw.
        neg_sampler = self._negative_sampler(specs[0][1], specs[0][2])
        self.fit_info_ = fit_collective_bank(
            [(lbl, d.a_hat, d.adjacency, pats, d.X, te) for lbl, d, pats, te in specs],
            degree=c.degree,
            epochs=c.epochs,
            learning_rate=c.learning_rate,
            ridge=c.ridge,
            fit_seed=c.seed,
            tau=c.tau,
            neg_sampler=neg_sampler,
            neg_weight=c.neg_weight,
            neg_temperature=c.neg_temperature,
            softmin_temperature=c.softmin_temperature,
            basis=c.basis,
            conf_weight=c.conf_weight,
            conf_reduce=c.conf_reduce,
            conf_delta=c.conf_delta,
            conf_halo_hops=c.conf_halo_hops,
            optimizer_kind=c.optimizer,
            heads=c.heads,
            head_diversity=c.head_diversity,
            warm_start=c.warm_start,
            softmin_anneal=c.softmin_anneal,
            capture_objective=c.capture_objective,
            label_weight=c.label_weight,
            label_y=label_y,
            label_idx=label_idx,
            day_aggregate=c.day_aggregate,
            geometries=geometries,
        )
        self.theta_ = self.fit_info_["theta"]
        self.fit_info_["geometry"] = geometries[0].describe()
        return self

    def target_subspace(self, data: GraphData, train_patterns: list) -> torch.Tensor:
        """Coarsening target ``R = span(Z)`` from the learned filter (needs :meth:`fit`)."""

        c = self.config
        geo = self.geometry(data)
        if c.collective_solver in ("closed-form", "aeq"):
            from src.margin_pencil import apply_dictionary_theta

            if self.pencil_theta_ is None:
                raise RuntimeError("call fit(...) before target_subspace()")
            return apply_dictionary_theta(
                data.a_hat,
                data.X,
                self.pencil_theta_,
                degree=c.degree,
                tau=c.tau,
                basis=c.basis,
            )
        self._require_fit()
        return build_bank_subspace(
            data.a_hat,
            data.adjacency,
            data.X,
            self.theta_,
            c.ridge,
            train_patterns,
            c.tau,
            structural_width=c.structural_width,
            seed=c.seed,
            coarsen_target=c.coarsen_target,
            basis=c.basis,
            geometry=geo,
        )

    def capture(self, data: GraphData, patterns: list) -> dict:
        """Per-gang retained ``M_tau``-energy (capture) of ``patterns``."""

        c = self.config
        geo = self.geometry(data)
        if c.collective_solver in ("closed-form", "aeq"):
            # no filter bank exists; measure the target subspace itself
            from src.run_collective_bank_detection import _basis_retained_energy

            basis = self.target_subspace(data, patterns)
            e = _basis_retained_energy(
                data.a_hat,
                data.adjacency,
                patterns,
                basis,
                c.ridge,
                c.tau,
                indicator=c.indicator,
                geometry=geo,
            )
            m_v = _train_gang_m_vhat(
                data.a_hat, data.adjacency, patterns, c.tau, geometry=geo
            )
            from src.run_collective_bank_detection import _collective_gamma

            g = _collective_gamma(
                data.a_hat, basis, m_v, c.ridge, c.tau, geometry=geo
            )
            diag = torch.diagonal(g).clamp(0.0, 1.0)
            return {
                "per_gang_capture": [float(v) for v in diag],
                "min_capture": float(diag.min()),
                "mean_capture": float(diag.mean()),
                "lambda_min_gamma": float(torch.linalg.eigvalsh(g)[0]),
            }
        self._require_fit()
        return retained_energy(
            data.a_hat,
            data.adjacency,
            patterns,
            data.X,
            self.theta_,
            c.ridge,
            c.tau,
            indicator=c.indicator,
            basis=c.basis,
            geometry=geo,
        )

    def gram_condition(self, data: GraphData) -> dict:
        """Channel-Gram conditioning in each basis at the learned filter (Prop 6.3)."""

        self._require_fit()
        c = self.config
        geo = self.geometry(data)
        return {
            "chebyshev": channel_gram_cond(
                data.a_hat, data.X, self.theta_, c.tau, "chebyshev", geometry=geo
            ),
            "monomial": channel_gram_cond(
                data.a_hat, data.X, self.theta_, c.tau, "monomial", geometry=geo
            ),
        }

    def coarsen(self, data: GraphData, basis: torch.Tensor, train_patterns: list):
        """Coarsen with target ``R = span(basis)``; returns ``(coarsening, trajectory)``.

        ``trajectory`` is the fine->coarse Ward sweep (``None`` for the greedy
        methods).  ``coarsening_method="ward-tree"`` uses the tree cut chosen by
        ``ward_stop`` (RSA-epsilon budget or best training F1); the other methods
        run the Loukas local-variation greedy under ``reduction``/``epsilon``.
        """

        c = self.config
        if c.coarsening_method == "dual-ward" and c.coarsening_laplacian in (
            "combinatorial",
            "comb",
        ):
            # src.smooth_dual_ward scores merges in M_tau = L_sym + tau I only;
            # pairing it with a combinatorial target would coarsen in a different
            # metric than the one the target and the RSA budget are stated in.
            raise NotImplementedError(
                "coarsening_method='dual-ward' is symmetric-only (it scores merges "
                "in L_sym + tau I); use 'ward-tree', 'ward' or the greedy methods "
                "with coarsening_laplacian='combinatorial'."
            )
        if c.coarsening_method in ("deflated-dual-ward", "deflated-minimax"):
            from src.deflated_coarsen import deflated_tree_coarsen

            return deflated_tree_coarsen(
                data.adjacency,
                basis,
                train_patterns,
                data.y,
                tau=c.tau,
                rule=(
                    "dual-ward"
                    if c.coarsening_method == "deflated-dual-ward"
                    else "minimax"
                ),
                laplacian=c.coarsening_laplacian,
                threshold=c.threshold,
                stop=c.ward_stop,
                epsilon_budget=(c.epsilon if c.epsilon is not None else math.inf),
                epsilon_key=c.deflated_epsilon_key,
                num_cuts=c.ward_num_cuts,
                hops=c.deflated_hops,
                max_ball=c.deflated_max_ball,
                max_rescore=c.deflated_max_rescore,
                fanout=c.deflated_fanout,
                max_cluster_size=c.deflated_max_cluster_size,
                certify=c.deflated_certify,
            )
        if c.coarsening_method == "ward-tree":
            return ward_tree_coarsen(
                data.adjacency,
                basis,
                train_patterns,
                data.y,
                tau=c.tau,
                laplacian=c.coarsening_laplacian,
                threshold=c.threshold,
                stop=c.ward_stop,
                epsilon_budget=(c.epsilon if c.epsilon is not None else math.inf),
                num_cuts=c.ward_num_cuts,
            )
        if c.epsilon is not None:
            budget = dict(
                reduction=c.reduction,
                epsilon=c.epsilon,
                epsilon_ramp_levels=c.epsilon_ramp_levels,
            )
        else:
            budget = dict(reduction=c.reduction)
        extra = {}
        if c.coarsening_method == "dual-ward":
            extra = dict(
                dual_ward_alpha=c.dual_ward_alpha,
                dual_ward_tau=c.dual_ward_tau,
                dual_ward_max_size=c.dual_ward_max_size,
                dual_ward_embedding=c.dual_ward_embedding,
            )
        coarsening = loukas_coarsen_pytorch(
            data.adjacency,
            basis,
            method=c.coarsening_method,
            laplacian=c.coarsening_laplacian,
            max_levels=c.max_levels,
            tau=c.tau,
            **budget,
            **extra,
        )
        return coarsening, None

    def evaluate(self, data: GraphData, coarsening, splits: dict) -> dict:
        """Alert recall / precision / F1 / detection rate per named split of patterns."""

        report = {}
        for name, patterns in splits.items():
            if not patterns:
                continue
            results, by_label = evaluate_loukas_patterns(
                patterns,
                coarsening.node_to_supernode,
                data.y,
                threshold=self.config.threshold,
            )
            alert = by_label.get("alert", {})
            f1 = float(np.mean([r.f1 for r in results])) if results else 0.0
            report[name] = {
                "detection_rate": alert.get("detection_rate", 0.0),
                "mean_recall": alert.get("mean_recall", 0.0),
                "mean_precision": alert.get("mean_precision", 0.0),
                "mean_f1": f1,
                "detected": int(alert.get("detected", 0)),
                "total": int(alert.get("total", 0)),
            }
        return report

    def run(
        self,
        data: GraphData,
        train_patterns: list,
        test_patterns: list,
        *,
        all_patterns: "list | None" = None,
        label_y=None,
        label_idx=None,
    ) -> dict:
        """Full pipeline: fit -> target -> coarsen -> evaluate, returning a results dict.

        ``label_y`` / ``label_idx`` optionally attach the joint supervised node head
        (active when ``config.label_weight > 0``) during the fit.
        """

        self.fit(data, train_patterns, label_y=label_y, label_idx=label_idx)
        basis = self.target_subspace(data, train_patterns)
        coarsening, trajectory = self.coarsen(data, basis, train_patterns)
        splits = {
            "train": train_patterns,
            "test": test_patterns,
            "all": (
                all_patterns
                if all_patterns is not None
                else train_patterns + test_patterns
            ),
        }
        report = self.evaluate(data, coarsening, splits)
        captures = {
            name: self.capture(data, pats) for name, pats in splits.items() if pats
        }
        return {
            "config": self.config.to_dict(),
            "theta": self.theta_,
            "fit": self.fit_info_,
            "basis": basis,
            "coarsening": coarsening,
            "trajectory": trajectory,
            "report": report,
            "captures": captures,
        }

    # -- misc --------------------------------------------------------------- #
    def _require_symmetric(self, what: str) -> None:
        """Refuse the code paths whose geometry is still hard-coded symmetric.

        ``src.margin_pencil`` (the ``closed-form`` and ``aeq`` solvers) and
        ``src.trace_ratio_bank`` build their own degree-weighted indicators and
        local ``M_tau`` forms internally.  Running them under
        ``coarsening_laplacian="combinatorial"`` would train in one geometry and
        coarsen/report in another -- silently, and with a plausible-looking
        number at the end.  Failing loudly is the only honest option until those
        modules take a geometry too.
        """

        if self.config.coarsening_laplacian in ("combinatorial", "comb"):
            raise NotImplementedError(
                f"collective_solver={what!r} is not geometry-aware yet: it builds "
                "degree-weighted indicators internally and would train in the "
                "symmetric metric while the coarsening and the reported capture "
                "use the combinatorial one.  Use collective_solver='gradient' "
                "with coarsening_laplacian='combinatorial', or keep "
                "coarsening_laplacian='symmetric' for this solver."
            )

    def _require_fit(self) -> None:
        if self.theta_ is None:
            raise RuntimeError("call fit(data, train_patterns) before this step")
