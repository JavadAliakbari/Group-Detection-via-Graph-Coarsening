r"""Pipeline entry point: coordination only, no algorithmic logic.

``run_pipeline`` wires the four components together in the only order that
respects the leakage rules:

1. :class:`~src.pipeline.data.Data` builds the bundle.
2. :class:`~src.pipeline.learning.Learning` fits on the **training groups of the
   training graphs** only.
3. :class:`~src.pipeline.coarsening.Coarsening` coarsens each training graph.
   The stopping rule is chosen from training groups only; the resulting
   :class:`~src.pipeline.coarsening.CutRule` (``epsilon*`` and ``reduction*``) is
   recorded and transferred to every held-out graph.  Each graph additionally
   gets an *oracle* cut -- the F1-optimal level of the same hierarchy -- which is
   reported as a post-hoc upper bound and feeds back into nothing.
4. When label prediction is enabled, group labels are evaluated by majority vote
   at both cuts, conditional on detection and always reported with coverage.
5. :class:`~src.pipeline.logging_visualization.LoggingVisualization` reports and
   serializes everything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.pipeline.coarsening import Coarsening, CoarseningConfig, CutRule
from src.pipeline.data import Data, DataConfig
from src.pipeline.group_labels import evaluate_group_labels
from src.pipeline.learning import LearningConfig, build_learner
from src.pipeline.logging_visualization import (
    MODES,
    GraphEvaluation,
    LoggingVisualization,
    LoggingVisualizationConfig,
    relocate_log_file,
)

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline"]


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    coarsening: CoarseningConfig = field(default_factory=CoarseningConfig)
    logging: LoggingVisualizationConfig = field(
        default_factory=LoggingVisualizationConfig
    )

    def __post_init__(self) -> None:
        if abs(self.learning.tau - self.coarsening.tau) > 0:
            raise ValueError(
                f"learning.tau ({self.learning.tau}) and coarsening.tau "
                f"({self.coarsening.tau}) must agree: they are the same screened metric"
            )

    def to_dict(self) -> dict:
        return {
            "data": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self.data).items()
            },
            "learning": self.learning.to_dict(),
            "coarsening": asdict(self.coarsening),
            "logging": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self.logging).items()
            },
            "seeds": {
                "data": self.data.seed,
                "learning": self.learning.seed,
                "coarsening": self.coarsening.seed,
                "objective": self.learning.objective.seed,
            },
        }


@dataclass
class PipelineResult:
    bundle: object
    learner: object
    learning: object
    evaluations: list
    cut_rule: "CutRule | None"
    summaries: dict
    config: dict

    @property
    def coarsenings(self) -> dict:
        return {e.graph_id: e.coarsening for e in self.evaluations}

    def to_serializable(self) -> dict:
        return {
            "config": self.config,
            "dataset": {
                "name": self.bundle.name,
                "train_graphs": [g.graph_id for g in self.bundle.train_graphs],
                "test_graphs": [g.graph_id for g in self.bundle.test_graphs],
                "feature_dim": self.bundle.feature_dim,
                "num_classes": self.bundle.num_classes,
            },
            "learning": self.learning.to_serializable(),
            "cut_rule": None if self.cut_rule is None else self.cut_rule.to_dict(),
            "coarsening": {
                e.graph_id: e.coarsening.to_serializable() for e in self.evaluations
            },
            "summaries": self.summaries,
            "group_labels": {
                mode: {
                    e.graph_id: e.label_reports[mode].metrics
                    for e in self.evaluations
                    if e.label_reports.get(mode)
                }
                for mode in MODES
            },
        }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Data -> Learning -> Coarsening -> evaluation -> reporting."""

    reporter = LoggingVisualization(config.logging)
    reporter.log_configuration(config.to_dict())

    bundle = Data(config.data).run()
    reporter.log_dataset(bundle)

    learner = build_learner(config.learning)
    learning_result = learner.run(bundle.train_graphs)
    reporter.log_learning(learning_result)

    labels_enabled = (
        config.learning.objective.label_enabled and learning_result.label_head is not None
    )
    coarsener = Coarsening(config.coarsening)
    evaluations: list = []
    cut_rule: "CutRule | None" = None

    for graph in bundle.train_graphs:
        result = coarsener.run(
            graph,
            learning_result.representations[graph.graph_id],
            train_groups=graph.train_groups,
        )
        evaluations.append(_evaluate(learner, graph, result, "train", labels_enabled))
        reporter.log_coarsening(evaluations[-1])
        if cut_rule is None and result.cut_rule is not None:
            cut_rule = result.cut_rule
            reporter.log_cut_rule(cut_rule)

    for graph in bundle.test_graphs:
        result = coarsener.run(
            graph,
            learner.represent(graph),
            cut_rule=cut_rule,  # None -> the configured epsilon/reduction rule
        )
        evaluations.append(_evaluate(learner, graph, result, "test", labels_enabled))
        reporter.log_coarsening(evaluations[-1])

    summaries = reporter.log_summaries(evaluations, labels_enabled=labels_enabled)
    if labels_enabled:
        reporter.log_group_labels(evaluations)

    pipeline_result = PipelineResult(
        bundle=bundle,
        learner=learner,
        learning=learning_result,
        evaluations=evaluations,
        cut_rule=cut_rule,
        summaries=summaries,
        config=config.to_dict(),
    )
    reporter.write_results(pipeline_result.to_serializable())
    relocate_log_file(config.logging.output_dir)
    return pipeline_result


def _evaluate(learner, graph, result, scope: str, labels_enabled: bool) -> GraphEvaluation:
    """Attach the screened level and the majority-vote group labels to one graph."""

    level = learner.level_of(graph)
    reports = {}
    if labels_enabled:
        probabilities = learner.predict_proba(graph)
        reports = {
            mode: evaluate_group_labels(graph, probabilities, result.cut(mode))
            for mode in MODES
        }
    return GraphEvaluation(
        graph=graph,
        coarsening=result,
        level=level.values.detach(),
        scope=scope,
        label_reports=reports,
    )
