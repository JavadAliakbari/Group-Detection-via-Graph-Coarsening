r"""Shared harness for the ward_tree / raw_ward / deflated_ward comparison.

One learned representation per (seed, graph) is fitted **once** and frozen, then
handed unchanged to every coarsening variant, so nothing but the merge rule (and
whatever single component an ablation changes) differs between arms.

The variants are declared in :data:`VARIANTS` as ``(name, CoarseningConfig
overrides, taxonomy)``; the taxonomy records, for every ablation, which of the
four things it touches::

    score      the quantity being minimized
    search     how candidate pairs are found and ranked
    update     how the hierarchy/linkage state is updated after a merge
    cut        only the operating-point selection

so a result can never be read as "a better hierarchy" when it is only a better
stopping rule.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.pipeline.coarsening import Coarsening, CoarseningConfig  # noqa: E402
from src.pipeline.data import Data, DataConfig  # noqa: E402
from src.pipeline.learning import LearningConfig, build_learner  # noqa: E402
from src.pipeline.objective import ObjectiveConfig  # noqa: E402

TAU = 0.5

#: the ``polynomial-closed-form`` example of ``run_modular_pipeline``, verbatim
BASE_DATA = DataConfig(
    source="synthetic",
    num_train_graphs=3,
    num_test_graphs=5,
    num_nodes=3000,
    group_size=[7, 20],
    num_groups=12,
    group_density=0.4,
    avg_degree=4.0,
    feature_dim=32,
    group_types=("random",),
    train_ratio=0.5,
    seed=3,
)

BASE_LEARNING = LearningConfig(
    architecture="polynomial",
    training_mode="closed_form",
    objective=ObjectiveConfig(
        gamma=None,
        host_mode="none",
        host_weight=0.25,
        host_count=100,
        label_enabled=True,
        label_weight=1.0,
        seed=0,
    ),
    tau=TAU,
    degree=32,
    num_heads=8,
    shared_filters=True,
    label_head_epochs=500,
    label_head_learning_rate=0.05,
    seed=0,
)

BASE_COARSENING = CoarseningConfig(
    method="ward_tree",
    tau=TAU,
    cut_rule="score_sum",
    transfer_cut_rule="score_sum",
    epsilon_budget=0.2,
    reduction=0.8,
    exact_epsilon_budget=200,
    deflated_commit_solve="local",
)


#: friendly aliases for the ``DataConfig`` fields the harness overrides most
_DATA_ALIASES = {
    "num_nodes": "num_nodes",
    "num_train": "num_train_graphs",
    "num_test": "num_test_graphs",
    "num_groups": "num_groups",
    "group_size": "group_size",
    "avg_degree": "avg_degree",
}


#: learning knobs the harness may narrow (only to keep a *small* audit graph's
#: target-to-node ratio comparable to the production one -- never in a
#: performance comparison)
_LEARNING_KEYS = ("num_heads", "degree")


def make_configs(seed: int, **overrides):
    learning_over = {
        k: overrides.pop(k) for k in _LEARNING_KEYS if overrides.get(k) is not None
    }
    fields = {
        _DATA_ALIASES.get(k, k): v for k, v in overrides.items() if v is not None
    }
    data = replace(BASE_DATA, seed=seed, **fields)
    learning = replace(BASE_LEARNING, seed=seed, **learning_over)
    learning.objective = replace(BASE_LEARNING.objective, seed=seed)
    return data, learning


def frozen_representation(seed: int, **data_overrides):
    """Bundle + one frozen basis per graph.  Fitted once, reused by every arm."""

    data_cfg, learning_cfg = make_configs(seed, **data_overrides)
    torch.manual_seed(seed)
    np.random.seed(seed)
    bundle = Data(data_cfg).run()
    learner = build_learner(learning_cfg)
    result = learner.run(bundle.train_graphs)
    bases = {g.graph_id: result.representations[g.graph_id] for g in bundle.train_graphs}
    for g in bundle.test_graphs:
        bases[g.graph_id] = learner.represent(g)
    return bundle, learner, result, bases
