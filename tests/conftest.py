"""Shared fixtures for the modular pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_default_dtype(torch.float64)

from src.pipeline.data import Data, DataConfig  # noqa: E402
from src.pipeline.geometry import screened_geometry, screened_level  # noqa: E402


@pytest.fixture(scope="session")
def small_config() -> DataConfig:
    return DataConfig(
        source="synthetic",
        num_train_graphs=2,
        num_test_graphs=1,
        num_nodes=200,
        group_size=[5, 9],
        num_groups=5,
        feature_dim=8,
        avg_degree=6.0,
        seed=11,
    )


@pytest.fixture(scope="session")
def bundle(small_config):
    return Data(small_config).run()


@pytest.fixture(scope="session")
def graph(bundle):
    return bundle.train_graphs[0]


@pytest.fixture(scope="session")
def geometry(graph):
    return screened_geometry(graph.a_hat, graph.adjacency)


@pytest.fixture()
def level(graph, geometry):
    generator = torch.Generator().manual_seed(0)
    Z = torch.randn(graph.num_nodes, 6, dtype=torch.float64, generator=generator)
    return screened_level(Z, geometry, 0.5)
