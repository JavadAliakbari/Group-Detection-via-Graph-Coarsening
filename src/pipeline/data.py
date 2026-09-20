r"""Data component: one graph container, one group container, one dataset bundle.

The pipeline never sees a dataset-specific object.  :class:`Data` turns a
validated :class:`DataConfig` into a :class:`DatasetBundle` of
:class:`Graph` objects, each carrying its own :class:`Group` list split into
``"train"`` and ``"test"``.

Splits
------
Every *training* graph owns both training groups and held-out groups; every
*test/transfer* graph owns held-out groups only.  Test-group labels are visible
to nothing but evaluation: the learners receive ``graph.train_groups`` and the
coarsener's F1 cut rule is fitted on training groups of training graphs alone.

Overlapping groups
------------------
Groups may overlap (the SNAP ground-truth communities do).  Each group's
internal and boundary edges are evaluated independently, so an edge can be
internal to one group and boundary to another; nothing uses a
one-group-per-node array.  What is *rejected* is a node that two overlapping
groups label differently -- :meth:`Graph.node_group_labels` raises rather than
silently picking one.

Feature and label conventions for datasets that provide neither are documented
on the adapters below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.loukas_sgc_detection import graph_operators

__all__ = [
    "Group",
    "Graph",
    "DatasetBundle",
    "DataConfig",
    "Data",
]

_SPLITS = ("train", "test")


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Group:
    """One target group: its identifier, member nodes, label and split."""

    group_id: str
    nodes: torch.Tensor
    label: int
    split: str
    group_type: str = "group"

    def __post_init__(self) -> None:
        nodes = torch.as_tensor(self.nodes, dtype=torch.long).flatten()
        if nodes.numel() == 0:
            raise ValueError(f"group {self.group_id!r} is empty")
        unique = torch.unique(nodes)
        if unique.numel() != nodes.numel():
            raise ValueError(f"group {self.group_id!r} repeats a node index")
        if int(self.label) < 0:
            raise ValueError(f"group {self.group_id!r} has a negative label")
        if self.split not in _SPLITS:
            raise ValueError(
                f"group {self.group_id!r} split must be one of {_SPLITS}, "
                f"got {self.split!r}"
            )
        object.__setattr__(self, "nodes", unique)
        object.__setattr__(self, "label", int(self.label))

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.numel())

    @property
    def node_indices(self) -> torch.Tensor:
        """Alias used by the legacy coarseners, which duck-type on this name."""

        return self.nodes

    def to_pattern(self):
        """A legacy :class:`~src.pattern_models.Pattern` view of this group.

        The three coarseners score their own training F1 through
        :func:`src.loukas_sgc_detection.evaluate_loukas_patterns`, which needs a
        mutable ``Pattern``.  Building one on demand keeps :class:`Group`
        immutable and keeps the adapter in exactly one place.
        """

        from src.pattern_models import create_pattern

        return create_pattern(
            pattern_id=self.group_id,
            nodes=[int(v) for v in self.nodes],
            pattern_type=self.group_type,
            label="alert",
        )


@dataclass
class Graph:
    """A single graph plus its groups; the only object the pipeline passes around."""

    graph_id: str
    num_nodes: int
    edge_index: torch.Tensor
    features: torch.Tensor
    node_labels: torch.Tensor
    groups: "list[Group]" = field(default_factory=list)
    edge_weight: "torch.Tensor | None" = None
    node_id_map: "np.ndarray | None" = None

    _a_hat: "torch.Tensor | None" = field(default=None, repr=False, compare=False)
    _adjacency: "torch.Tensor | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation ------------------------------------------------------- #
    def validate(self) -> "Graph":
        n = int(self.num_nodes)
        if n <= 0:
            raise ValueError(f"graph {self.graph_id!r}: num_nodes must be positive")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"graph {self.graph_id!r}: edge_index must be (2, E)")
        self.edge_index = self.edge_index.to(torch.long)
        if self.edge_index.numel() and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= n
        ):
            raise ValueError(
                f"graph {self.graph_id!r}: edge_index holds out-of-range node ids"
            )
        if self.edge_weight is not None:
            if self.edge_weight.shape != (self.edge_index.shape[1],):
                raise ValueError(
                    f"graph {self.graph_id!r}: edge_weight must have one entry per edge"
                )
            if float(self.edge_weight.min()) < 0.0:
                raise ValueError(f"graph {self.graph_id!r}: negative edge weight")
        if self.features.ndim != 2 or self.features.shape[0] != n:
            raise ValueError(
                f"graph {self.graph_id!r}: features must be (num_nodes, feature_dim), "
                f"got {tuple(self.features.shape)} for {n} nodes"
            )
        if self.features.shape[1] == 0:
            raise ValueError(f"graph {self.graph_id!r}: feature dimension is zero")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError(
                f"graph {self.graph_id!r}: features contain non-finite values"
            )
        self.node_labels = self.node_labels.to(torch.long).flatten()
        if self.node_labels.numel() != n:
            raise ValueError(
                f"graph {self.graph_id!r}: node_labels must have one entry per node"
            )
        if n and int(self.node_labels.min()) < 0:
            raise ValueError(f"graph {self.graph_id!r}: negative node label")
        seen: set = set()
        for group in self.groups:
            if group.group_id in seen:
                raise ValueError(
                    f"graph {self.graph_id!r}: duplicate group id {group.group_id!r}"
                )
            seen.add(group.group_id)
            if int(group.nodes.max()) >= n or int(group.nodes.min()) < 0:
                raise ValueError(
                    f"graph {self.graph_id!r}: group {group.group_id!r} references "
                    "an out-of-range node"
                )
        if self.node_id_map is not None and len(self.node_id_map) != n:
            raise ValueError(f"graph {self.graph_id!r}: node_id_map length mismatch")
        return self

    # -- derived operators ------------------------------------------------- #
    def _build_operators(self) -> None:
        view = SimpleNamespace(
            edge_index=self.edge_index,
            num_nodes=int(self.num_nodes),
            edge_weight=self.edge_weight,
        )
        self._a_hat, self._adjacency = graph_operators(view)

    @property
    def a_hat(self) -> torch.Tensor:
        """Self-loop-augmented symmetric normalized adjacency (the bank's operator)."""

        if self._a_hat is None:
            self._build_operators()
        return self._a_hat

    @property
    def adjacency(self) -> torch.Tensor:
        """Symmetric sparse weight matrix ``W`` without self-loops."""

        if self._adjacency is None:
            self._build_operators()
        return self._adjacency

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    # -- groups ------------------------------------------------------------ #
    @property
    def train_groups(self) -> "list[Group]":
        return [g for g in self.groups if g.split == "train"]

    @property
    def test_groups(self) -> "list[Group]":
        return [g for g in self.groups if g.split == "test"]

    def groups_of(self, split: "str | None") -> "list[Group]":
        return (
            list(self.groups)
            if split is None
            else [g for g in self.groups if g.split == split]
        )

    def node_group_labels(
        self, split: str = "train"
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """``(labels, mask)`` from the groups of ``split``; conflicts raise.

        A node in several same-split groups is fine as long as they agree on its
        label.  Two groups that disagree are a data error, never resolved
        silently -- the paper's supervision is *the group's* label, so a node
        cannot carry two.
        """

        labels = torch.full((int(self.num_nodes),), -1, dtype=torch.long)
        owner: dict = {}
        for group in self.groups_of(split):
            for node in group.nodes.tolist():
                previous = labels[node].item()
                if previous >= 0 and previous != group.label:
                    raise ValueError(
                        f"graph {self.graph_id!r}: node {node} is labelled "
                        f"{previous} by group {owner[node]!r} and {group.label} by "
                        f"group {group.group_id!r}; resolve the conflict in the data"
                    )
                labels[node] = group.label
                owner[node] = group.group_id
        mask = labels >= 0
        return labels.clamp_min(0), mask


@dataclass
class DatasetBundle:
    """Training graphs (train + held-out groups) and test/transfer graphs."""

    train_graphs: "list[Graph]"
    test_graphs: "list[Graph]" = field(default_factory=list)
    name: str = "dataset"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train_graphs:
            raise ValueError("a dataset bundle needs at least one training graph")
        dims = {g.feature_dim for g in self.train_graphs + self.test_graphs}
        if len(dims) > 1:
            raise ValueError(
                f"graphs disagree on the feature dimension: {sorted(dims)}; one "
                "shared model cannot be trained on them"
            )
        for graph in self.test_graphs:
            if graph.train_groups:
                raise ValueError(
                    f"test graph {graph.graph_id!r} carries training groups; "
                    "test graphs hold out every group"
                )

    @property
    def feature_dim(self) -> int:
        return self.train_graphs[0].feature_dim

    @property
    def num_classes(self) -> int:
        """Number of label classes, inferred from every group of every graph."""

        labels = [
            g.label
            for graph in self.train_graphs + self.test_graphs
            for g in graph.groups
        ]
        return int(max(labels)) + 1 if labels else 1

    @property
    def graphs(self) -> "list[Graph]":
        return list(self.train_graphs) + list(self.test_graphs)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def _resolve_range(value, name: str, rng: np.random.Generator) -> int:
    """A fixed ``int`` or an inclusive ``[min, max]`` range sampled per draw."""

    if isinstance(value, (int, np.integer)):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive, got {value}")
        return int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = int(value[0]), int(value[1])
        if lo < 1 or hi < lo:
            raise ValueError(f"{name} range must satisfy 1 <= min <= max, got {value}")
        return int(rng.integers(lo, hi + 1))
    raise ValueError(
        f"{name} must be an int or an inclusive [min, max] pair, got {value!r}"
    )


@dataclass
class DataConfig:
    """Validated configuration of the :class:`Data` component."""

    source: str = "synthetic"

    # --- shared ---
    seed: int = 0
    train_ratio: float = 0.5
    num_train_graphs: int = 1
    num_test_graphs: int = 0

    # --- synthetic ---
    num_groups: int = 8
    num_nodes: "int | list[int]" = 1000
    group_size: "int | list[int]" = 10
    group_density: float = 0.4
    avg_degree: float = 6.0
    feature_dim: int = 32
    group_types: "tuple[str, ...]" = ("random",)

    # --- loaded datasets ---
    data_dir: "Path | None" = None
    elliptic_train_days: "tuple[int, ...]" = (25,)
    elliptic_test_days: "tuple[int, ...]" = ()
    elliptic_feature_mode: str = "wallet+random"
    min_group_size: int = 2
    snap_name: str = "amazon"
    snap_max_groups: int = 200
    structural_feature_dim: int = 32

    def __post_init__(self) -> None:
        if self.source not in ("synthetic", "elliptic", "snap"):
            raise ValueError(
                f"source must be 'synthetic', 'elliptic' or 'snap', got {self.source!r}"
            )
        if not 0.0 < self.train_ratio <= 1.0:
            raise ValueError("train_ratio must lie in (0, 1]")
        if self.num_train_graphs < 1:
            raise ValueError("num_train_graphs must be at least 1")
        if self.num_test_graphs < 0:
            raise ValueError("num_test_graphs must be non-negative")
        if self.source == "synthetic":
            valid = {"clique", "cycle", "star", "random"}
            bad = set(self.group_types) - valid
            if bad:
                raise ValueError(
                    f"group_types must be a subset of {sorted(valid)}, got {sorted(bad)}"
                )
            if not 0.0 <= self.group_density <= 1.0:
                raise ValueError("group_density must lie in [0, 1]")
            if self.feature_dim < 1:
                raise ValueError("feature_dim must be positive")
            rng = np.random.default_rng(0)
            _resolve_range(self.num_nodes, "num_nodes", rng)
            _resolve_range(self.group_size, "group_size", rng)
        else:
            if self.data_dir is None:
                raise ValueError(f"source={self.source!r} requires data_dir")
            self.data_dir = Path(self.data_dir)
        if self.source == "elliptic" and not self.elliptic_train_days:
            raise ValueError("elliptic source needs at least one training day")
        if self.source == "elliptic" and self.elliptic_feature_mode not in (
            "wallet",
            "random",
            "wallet+random",
        ):
            raise ValueError(
                "elliptic_feature_mode must be 'wallet', 'random' or 'wallet+random'"
            )


# --------------------------------------------------------------------------- #
# synthetic generator
# --------------------------------------------------------------------------- #
def _motif_edges(
    nodes: "list[int]", kind: str, density: float, rng
) -> "list[tuple[int, int]]":
    """Undirected edge list of one planted group (reused from the legacy planter)."""

    from src.run_collective_bank_detection import _motif_edges as legacy

    return legacy(nodes, kind, density=density, rng=rng)


def _standardized_features(num_nodes: int, width: int, seed: int) -> torch.Tensor:
    """Isotropic standardized Gaussian channels -- the repo's structural features."""

    gen = torch.Generator().manual_seed(int(seed))
    X = torch.randn(num_nodes, width, dtype=torch.float64, generator=gen)
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)


def build_synthetic_graph(config: DataConfig, graph_id: str, seed: int) -> Graph:
    """Erdos-Renyi background with ``num_groups`` disjoint planted groups.

    Node count and group size are each either a fixed integer or an inclusive
    ``[min, max]`` range drawn independently for this graph (sizes are drawn per
    group).  Every node carries a feature row; ``node_labels`` marks group
    members class ``1``.
    """

    rng = np.random.default_rng(seed)
    num_nodes = _resolve_range(config.num_nodes, "num_nodes", rng)
    sizes = [
        _resolve_range(config.group_size, "group_size", rng)
        for _ in range(config.num_groups)
    ]
    kinds = [
        config.group_types[int(i)]
        for i in rng.integers(0, len(config.group_types), config.num_groups)
    ]
    if sum(sizes) > num_nodes:
        raise ValueError(
            f"planted group sizes sum to {sum(sizes)} but the graph has {num_nodes} "
            "nodes; lower num_groups/group_size or raise num_nodes"
        )

    perm = rng.permutation(num_nodes)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    blocks = [
        perm[offsets[i] : offsets[i + 1]].tolist() for i in range(config.num_groups)
    ]

    edges: set = set()
    for u, v in zip(
        rng.integers(
            0, num_nodes, size=int(num_nodes * config.avg_degree / 2)
        ).tolist(),
        rng.integers(
            0, num_nodes, size=int(num_nodes * config.avg_degree / 2)
        ).tolist(),
    ):
        if u != v:
            edges.add((min(u, v), max(u, v)))
    labels = np.zeros(num_nodes, dtype=np.int64)
    for block, kind in zip(blocks, kinds):
        for u, v in _motif_edges(block, kind, config.group_density, rng):
            edges.add((min(u, v), max(u, v)))
        labels[np.asarray(block)] = 1

    pairs = np.array(sorted(edges), dtype=np.int64).T
    edge_index = torch.from_numpy(np.concatenate([pairs, pairs[::-1]], axis=1)).long()
    groups = [
        Group(
            group_id=f"{graph_id}:{kind}{i}",
            nodes=torch.as_tensor(block),
            label=1,
            split="train",
            group_type=kind,
        )
        for i, (block, kind) in enumerate(zip(blocks, kinds))
    ]
    return Graph(
        graph_id=graph_id,
        num_nodes=num_nodes,
        edge_index=edge_index,
        features=_standardized_features(num_nodes, config.feature_dim, seed),
        node_labels=torch.from_numpy(labels),
        groups=groups,
    )


# --------------------------------------------------------------------------- #
# the component
# --------------------------------------------------------------------------- #
class Data:
    """Builds a :class:`DatasetBundle` from a validated :class:`DataConfig`."""

    def __init__(self, config: DataConfig):
        self.config = config

    def run(self) -> DatasetBundle:
        if self.config.source == "synthetic":
            return self._synthetic()
        if self.config.source == "elliptic":
            return self._elliptic()
        return self._snap()

    # -- splitting --------------------------------------------------------- #
    def _split_groups(self, graph: Graph, seed: int, *, all_test: bool) -> Graph:
        """Relabel a graph's groups into train/test without touching the graph."""

        if all_test:
            graph.groups = [
                Group(g.group_id, g.nodes, g.label, "test", g.group_type)
                for g in graph.groups
            ]
            return graph
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(graph.groups))
        n_train = max(1, int(round(self.config.train_ratio * len(graph.groups))))
        split = {}
        for rank, index in enumerate(order.tolist()):
            split[index] = "train" if rank < n_train else "test"
        graph.groups = [
            Group(g.group_id, g.nodes, g.label, split[i], g.group_type)
            for i, g in enumerate(graph.groups)
        ]
        return graph

    # -- sources ----------------------------------------------------------- #
    def _synthetic(self) -> DatasetBundle:
        c = self.config
        train, test = [], []
        for i in range(c.num_train_graphs):
            seed = c.seed + 100 * i
            train.append(
                self._split_groups(
                    build_synthetic_graph(c, f"G{i}", seed), seed + 1, all_test=False
                )
            )
        for k in range(c.num_test_graphs):
            seed = c.seed + 9000 + 100 * k
            test.append(
                self._split_groups(
                    build_synthetic_graph(c, f"T{k}", seed), seed + 1, all_test=True
                )
            )
        return DatasetBundle(
            train, test, name="synthetic", metadata={"config": c.__dict__}
        )

    def _elliptic(self) -> DatasetBundle:
        """Elliptic++ Actors wallet graph, one graph per day.

        Groups are the connected components (size ``>= min_group_size``) of the
        illicit-induced subgraph -- the repository's gang definition.  Features
        follow ``elliptic_feature_mode``: the z-scored/row-normalized wallet
        features, an isotropic random structural range finder, or both
        concatenated.  The feature *columns* are pinned to the first training
        day's set so a filter fit on one day applies to every other.
        """

        from src.run_elliptic_gang_conductance import (
            build_graph,
            connected_components_sets,
        )
        from src.run_elliptic_gang_detection import load_node_features

        c = self.config
        columns: "list[str] | None" = None

        def one_day(day: int, graph_id: str) -> Graph:
            nonlocal columns
            a_unw, _a_w, cls, nodes_df = build_graph(c.data_dir, day, day)
            n = int(a_unw.shape[0])
            parts = []
            if c.elliptic_feature_mode in ("wallet", "wallet+random"):
                if columns is None:
                    wallet, columns = load_node_features(
                        c.data_dir, nodes_df, day, day, return_columns=True
                    )
                else:
                    wallet = load_node_features(
                        c.data_dir, nodes_df, day, day, keep_columns=columns
                    )
                parts.append(wallet)
            if c.elliptic_feature_mode in ("random", "wallet+random"):
                parts.append(
                    _standardized_features(n, c.structural_feature_dim, c.seed + day)
                )
            features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

            coo = a_unw.tocoo()
            edge_index = torch.from_numpy(
                np.vstack([coo.row, coo.col]).astype(np.int64)
            )
            sets = connected_components_sets(
                a_unw, np.where(cls == 1)[0], c.min_group_size
            )
            groups = [
                Group(
                    f"{graph_id}:g{i}",
                    torch.as_tensor(np.asarray(s, dtype=np.int64)),
                    1,
                    "train",
                    "gang",
                )
                for i, s in enumerate(sets)
            ]
            return Graph(
                graph_id=graph_id,
                num_nodes=n,
                edge_index=edge_index,
                features=features,
                node_labels=torch.from_numpy((cls == 1).astype(np.int64)),
                groups=groups,
                node_id_map=nodes_df["address"].to_numpy(),
            )

        train = [
            self._split_groups(one_day(d, f"d{d}"), c.seed + d, all_test=False)
            for d in c.elliptic_train_days
        ]
        test = [
            self._split_groups(one_day(d, f"d{d}"), c.seed + d, all_test=True)
            for d in c.elliptic_test_days
        ]
        return DatasetBundle(
            train,
            test,
            name="elliptic++",
            metadata={
                "train_days": list(c.elliptic_train_days),
                "test_days": list(c.elliptic_test_days),
            },
        )

    def _snap(self) -> DatasetBundle:
        """A SNAP ground-truth-community graph (com-amazon / com-dblp / com-youtube).

        The dataset has neither node features nor node labels, so the repository's
        conventions are applied explicitly:

        * **features** -- an isotropic standardized Gaussian range finder of
          ``structural_feature_dim`` columns (the structural channel used
          throughout the repo when a graph carries no attributes);
        * **node labels** -- ``1`` for a node in at least one retained community,
          ``0`` otherwise.

        The top-5000 communities overlap; they are kept overlapping.  There is one
        graph, split into train and held-out communities.
        """

        from src.run_graph_fraud_gang_detection import load_snap_community

        c = self.config
        adjacency, communities = load_snap_community(c.data_dir, c.snap_name)
        communities = sorted(communities, key=len, reverse=True)[: c.snap_max_groups]
        communities = [s for s in communities if len(s) >= c.min_group_size]
        n = int(adjacency.shape[0])
        coo = adjacency.tocoo()
        edge_index = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        labels = np.zeros(n, dtype=np.int64)
        for nodes in communities:
            labels[np.asarray(nodes)] = 1
        groups = [
            Group(
                f"snap:c{i}",
                torch.as_tensor(np.asarray(s, dtype=np.int64)),
                1,
                "train",
                "community",
            )
            for i, s in enumerate(communities)
        ]
        graph = Graph(
            graph_id=f"com-{c.snap_name}",
            num_nodes=n,
            edge_index=edge_index,
            features=_standardized_features(n, c.structural_feature_dim, c.seed),
            node_labels=torch.from_numpy(labels),
            groups=groups,
        )
        return DatasetBundle(
            [self._split_groups(graph, c.seed, all_test=False)],
            [],
            name=f"snap:com-{c.snap_name}",
            metadata={"n_communities": len(groups)},
        )


def sample_training_graph(
    bundle: DatasetBundle, mode: str, rng: np.random.Generator
) -> "Graph | None":
    """One training graph per epoch under ``mode='sample'``; ``None`` aggregates.

    ``'mean'`` and ``'min'`` are handled by the learner (it needs every graph's
    objective), so they return ``None`` here.
    """

    if mode == "sample":
        return bundle.train_graphs[int(rng.integers(len(bundle.train_graphs)))]
    if mode in ("mean", "min"):
        return None
    raise ValueError(
        f"multi-graph mode must be 'sample', 'mean' or 'min', got {mode!r}"
    )
