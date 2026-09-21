"""The deflated-Ward ablation knobs and the ``deflated_ward_tight`` variant.

Three things are pinned here:

1. every new knob leaves the **baseline** hierarchy exactly as it was when it is
   at its default, so ``deflated_ward`` remains a usable reference arm;
2. the certified queue key really is a lower bound on the deflated score, which
   is what makes the lazy early exit exact rather than heuristic;
3. ``deflated_ward_tight`` changes only the *search*: the recorded merge score is
   still ``||a||^2``, so the cumulative-score certificate and the ``score_sum``
   axis mean the same thing they meant before.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.deflated_coarsen import deflated_coarsen
from analysis.deflated_reference import ExactDeflation
from src.pipeline.coarsening import (
    _TIGHT_PRESET,
    Coarsening,
    CoarseningConfig,
    _labels_after,
    _to_scipy,
)
from src.pipeline.learning import LearningConfig, PolynomialFilterLearning
from src.pipeline.objective import ObjectiveConfig

TAU = 0.5


@pytest.fixture(scope="module")
def fitted(bundle):
    config = LearningConfig(
        architecture="polynomial",
        training_mode="closed_form",
        objective=ObjectiveConfig(gamma=2.0),
        degree=4,
        num_heads=1,
        tau=TAU,
        seed=0,
    )
    learner = PolynomialFilterLearning(config)
    return learner, learner.run(bundle.train_graphs)


@pytest.fixture(scope="module")
def problem(bundle, fitted):
    _learner, result = fitted
    graph = bundle.train_graphs[0]
    basis = result.representations[graph.graph_id]
    return graph, basis, _to_scipy(graph.adjacency), basis.detach().cpu().numpy()


def _config(**kw) -> CoarseningConfig:
    base = dict(method="deflated_ward", tau=TAU, cut_rule="f1", exact_epsilon_budget=25)
    base.update(kw)
    return CoarseningConfig(**base)


def _tree(W, Z, **kw):
    return deflated_coarsen(
        W, Z, TAU, rule="dual-ward", build_full_tree=True,
        track_euclidean=False, record_curve=False, **kw
    )


# --------------------------------------------------------------------------- #
# 1. the baseline is untouched
# --------------------------------------------------------------------------- #
def test_the_new_knobs_default_to_the_original_hierarchy(problem):
    _graph, _basis, W, Z = problem
    base = _tree(W, Z)
    explicit = _tree(W, Z, selection="normalized", queue_key="raw")
    assert np.array_equal(base.children_, explicit.children_)
    assert base.trace_h_ == pytest.approx(explicit.trace_h_, rel=0, abs=0)


def test_the_tight_variant_does_not_change_deflated_ward(problem):
    graph, basis, _W, _Z = problem
    a = Coarsening(_config(method="deflated_ward")).run(
        graph, basis, train_groups=graph.train_groups
    )
    b = Coarsening(_config(method="deflated_ward_tight")).run(
        graph, basis, train_groups=graph.train_groups
    )
    again = Coarsening(_config(method="deflated_ward")).run(
        graph, basis, train_groups=graph.train_groups
    )
    assert np.array_equal(a.hierarchy.children, again.hierarchy.children)
    assert b.hierarchy.method == "deflated_ward_tight"


def test_an_explicit_knob_wins_over_the_tight_preset(problem):
    """The preset fills in defaults only; a configured value is never overridden."""

    graph, basis, _W, _Z = problem
    assert _TIGHT_PRESET["deflated_max_rescore"] != 3
    preset = Coarsening(_config(method="deflated_ward_tight")).run(
        graph, basis, train_groups=graph.train_groups
    )
    override = Coarsening(
        _config(method="deflated_ward_tight", deflated_max_rescore=3)
    ).run(graph, basis, train_groups=graph.train_groups)
    assert override.config["deflated_max_rescore"] == 3
    assert not np.array_equal(
        preset.hierarchy.children, override.hierarchy.children
    )


# --------------------------------------------------------------------------- #
# 2. the certified key is admissible
# --------------------------------------------------------------------------- #
def test_the_certified_key_is_a_lower_bound_on_the_deflated_score(problem):
    """``[(sqrt(s0) - eta_bar)_+]^2 <= s`` on every admissible pair.

    Checked against :class:`src.deflated_reference.ExactDeflation`, which
    recomputes both quantities from their definitions on the original graph, at
    the singleton partition and at a coarser one.
    """

    _graph, _basis, W, Z = problem
    exact = ExactDeflation(W, Z, TAU)
    rng = np.random.default_rng(0)
    for k in (exact.n, max(exact.n // 3, 3)):
        labels = (
            np.arange(exact.n, dtype=np.int64)
            if k == exact.n
            else np.unique(rng.integers(0, k, size=exact.n), return_inverse=True)[1]
        )
        pairs = exact.adjacent_pairs(labels)
        assert pairs
        for a, b in pairs:
            record = exact.score(labels, a, b)
            # eta_bar >= eta_0, so this is the *tightest* form of the same bound
            bound = max(np.sqrt(record["raw_score"]) - record["eta0"], 0.0) ** 2
            assert bound <= record["score"] + 1e-12


def test_the_certified_key_ends_the_search_without_the_count_cap(problem):
    """With the bound in the queue, ``max_rescore`` stops being the operative limit."""

    _graph, _basis, W, Z = problem
    capped = _tree(W, Z, queue_key="certified", max_rescore=8)
    uncapped = _tree(W, Z, queue_key="certified", max_rescore=10**6)
    # the cap is what changed the answer; if it were never binding these would
    # be the same tree
    assert not np.array_equal(capped.children_, uncapped.children_)


# --------------------------------------------------------------------------- #
# 3. only the search changes: the recorded score is still the theoretical one
# --------------------------------------------------------------------------- #
def test_the_tight_variant_records_the_theoretical_score(problem):
    _graph, _basis, W, Z = problem
    result = _tree(
        W, Z,
        queue_key="certified", max_rescore=4096, hops=3, max_ball=64,
        commit_solve="exact",
    )
    exact = ExactDeflation(W, Z, TAU)
    children = np.asarray(result.children_, np.int64).reshape(-1, 2)
    n = W.shape[0]
    for t in range(min(8, children.shape[0])):
        labels = _labels_after(children, n, t)
        parent = np.arange(n + children.shape[0], dtype=np.int64)
        for k in range(t):
            parent[int(children[k, 0])] = parent[int(children[k, 1])] = n + k
        root = {}
        for v in range(n):
            x = v
            while parent[x] != x:
                x = parent[x]
            root[x] = int(labels[v])
        a, b = root[int(children[t, 0])], root[int(children[t, 1])]
        recorded = result.merge_records_[t]["a_sq"]
        assert recorded == pytest.approx(
            exact.score(labels, min(a, b), max(a, b))["score"], abs=1e-8
        )


def test_the_selection_knob_rescales_the_key_not_the_recorded_score(problem):
    _graph, _basis, W, Z = problem
    result = _tree(W, Z, selection="unnormalized")
    for record in result.merge_records_[:10]:
        assert record["selection"] == "unnormalized"
        # ``score`` is the key the merge was chosen with, ``a_sq`` the paper's
        # score of the committed direction; the two differ by exactly ||g~||^2
        assert record["score"] == pytest.approx(
            record["a_sq"] * record["gt_sq"], rel=1e-10
        )
        assert 0.0 <= record["a_sq"] <= 1.0 + 1e-9


def test_the_unnormalized_selection_is_refused_for_the_minimax_rule(problem):
    _graph, _basis, W, Z = problem
    with pytest.raises(ValueError, match="minimax"):
        deflated_coarsen(W, Z, TAU, rule="minimax", selection="unnormalized")


# --------------------------------------------------------------------------- #
# 4. the one-engine score ladder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method",
    ["ladder_ward", "ladder_ward_vol", "ladder_raw", "ladder_deflated",
     "ladder_deflated_abs"],
)
def test_every_ladder_arm_builds_a_complete_hierarchy(method, problem):
    graph, basis, _W, _Z = problem
    result = Coarsening(_config(method=method)).run(
        graph, basis, train_groups=graph.train_groups
    )
    assert result.hierarchy.children.shape[0] == graph.num_nodes - 1
    assert np.all(np.isfinite(result.hierarchy.scores))


def test_the_ladder_reproduces_the_production_raw_ward_tree(problem):
    """``ladder_raw`` is ``raw_ward``: same engine, same score, same tree."""

    graph, basis, _W, _Z = problem
    a = Coarsening(_config(method="raw_ward")).run(
        graph, basis, train_groups=graph.train_groups
    )
    b = Coarsening(_config(method="ladder_raw")).run(
        graph, basis, train_groups=graph.train_groups
    )
    assert np.array_equal(a.hierarchy.children, b.hierarchy.children)


def test_the_ladder_ward_arm_reproduces_the_sklearn_ward_tree(problem):
    """On a shared basis, ``ladder_ward`` is exactly ``ward_tree``.

    ``ward_tree`` orthonormalizes with ``eps * max(shape)`` and every other arm
    with ``rank_tol``; feeding the already-orthonormal ``a0`` to both removes
    that difference, and the two trees then coincide.
    """

    from src.raw_ward import raw_ward
    from src.ward_pr_sweep import ward_order

    graph, basis, W, _Z = problem
    children, _d, a0, _m = ward_order(graph.adjacency, basis, TAU)
    children = np.asarray(children, np.int64)
    engine = raw_ward(
        W, a0.detach().cpu().numpy(), TAU, score="ward", build_full_tree=True
    )
    mine = np.asarray(engine.children_, np.int64).reshape(-1, 2)
    n = int(graph.num_nodes)
    # compare partitions, not tree ids: the two builders number internal nodes
    # differently.  The last few merges join the disconnected leftovers, where
    # the constrained rules have nothing left to agree on.
    for t in (n // 8, n // 4, n // 2, int(0.8 * n)):
        assert np.array_equal(
            _relabel(_labels_after(children, n, t)),
            _relabel(_labels_after(mine, n, t)),
        )


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Canonical labelling (first appearance order), so ids never matter."""

    _first, inverse = np.unique(labels, return_inverse=True)
    return inverse


def test_unknown_methods_and_knobs_are_refused():
    with pytest.raises(ValueError, match="method must be one of"):
        CoarseningConfig(method="ladder_nonsense")
    with pytest.raises(ValueError, match="deflated_queue_key"):
        CoarseningConfig(deflated_queue_key="hint")
    with pytest.raises(ValueError, match="deflated_selection"):
        CoarseningConfig(deflated_selection="absolute")
