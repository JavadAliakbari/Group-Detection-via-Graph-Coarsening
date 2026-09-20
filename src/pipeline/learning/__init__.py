"""Learning component: one interface, three architectures."""

from src.pipeline.learning.base import (
    GraphContext,
    Learning,
    LearningConfig,
    LearningResult,
)
from src.pipeline.learning.nonlinear import GCNLearning, GraphSAGELearning
from src.pipeline.learning.polynomial import PolynomialFilterLearning

__all__ = [
    "Learning",
    "LearningConfig",
    "LearningResult",
    "GraphContext",
    "PolynomialFilterLearning",
    "GCNLearning",
    "GraphSAGELearning",
    "build_learner",
]

_REGISTRY = {
    "polynomial": PolynomialFilterLearning,
    "gcn": GCNLearning,
    "graphsage": GraphSAGELearning,
}


def build_learner(config: LearningConfig) -> Learning:
    """The learner named by ``config.architecture``."""

    return _REGISTRY[config.architecture](config)
