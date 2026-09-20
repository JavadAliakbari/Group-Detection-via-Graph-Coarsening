"""Modular group-detection pipeline: Data, Learning, Coarsening, LoggingVisualization."""

from src.pipeline.coarsening import (
    Coarsening,
    CoarseningConfig,
    CoarseningResult,
    CutEvaluation,
    CutRule,
)
from src.pipeline.data import Data, DataConfig, DatasetBundle, Graph, Group
from src.pipeline.geometry import ScreenedLevel, screened_geometry, screened_level
from src.pipeline.group_labels import GroupLabelReport, evaluate_group_labels
from src.pipeline.learning import (
    GCNLearning,
    GraphSAGELearning,
    Learning,
    LearningConfig,
    LearningResult,
    PolynomialFilterLearning,
    build_learner,
)
from src.pipeline.logging_visualization import (
    GraphEvaluation,
    LoggingVisualization,
    LoggingVisualizationConfig,
)
from src.pipeline.objective import LabelHead, ObjectiveConfig
from src.pipeline.pipeline import PipelineConfig, PipelineResult, run_pipeline

__all__ = [
    "Data",
    "DataConfig",
    "DatasetBundle",
    "Graph",
    "Group",
    "ScreenedLevel",
    "screened_geometry",
    "screened_level",
    "ObjectiveConfig",
    "LabelHead",
    "Learning",
    "LearningConfig",
    "LearningResult",
    "PolynomialFilterLearning",
    "GCNLearning",
    "GraphSAGELearning",
    "build_learner",
    "Coarsening",
    "CoarseningConfig",
    "CoarseningResult",
    "CutEvaluation",
    "CutRule",
    "GroupLabelReport",
    "evaluate_group_labels",
    "GraphEvaluation",
    "LoggingVisualization",
    "LoggingVisualizationConfig",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]
