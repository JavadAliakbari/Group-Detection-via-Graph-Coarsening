"""Learning component: one interface, three architectures."""

from src.pipeline.learning.base import (
    GraphContext,
    Learning,
    LearningConfig,
    LearningResult,
)
from src.pipeline.learning.nonlinear import GCNLearning, GraphSAGELearning
from src.pipeline.learning.polynomial import PolynomialFilterLearning
from src.pipeline.learning.spectral import StaticSpectralLearning

__all__ = [
    "Learning",
    "LearningConfig",
    "LearningResult",
    "GraphContext",
    "PolynomialFilterLearning",
    "StaticSpectralLearning",
    "GCNLearning",
    "GraphSAGELearning",
    "build_learner",
]

_REGISTRY = {
    "polynomial": PolynomialFilterLearning,
    "gcn": GCNLearning,
    "graphsage": GraphSAGELearning,
    "static_spectral": StaticSpectralLearning,
}


def build_learner(config: LearningConfig) -> Learning:
    """The learner named by ``config.architecture``."""

    return _REGISTRY[config.architecture](config)
