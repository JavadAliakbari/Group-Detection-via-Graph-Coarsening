"""Entry point and example configurations for the modular pipeline.

Run one of the named examples::

    python -m src.run_modular_pipeline --example polynomial-closed-form
    python -m src.run_modular_pipeline --example gcn --epochs 500 --out results/gcn

``--example`` picks a complete :class:`~src.pipeline.pipeline.PipelineConfig`;
the remaining flags override the few knobs worth sweeping from the command line.
Everything else is configured in code, through the validated config objects.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import torch

from src.pipeline.coarsening import CoarseningConfig
from src.pipeline.data import DataConfig
from src.pipeline.learning import LearningConfig
from src.pipeline.logging_visualization import LoggingVisualizationConfig
from src.pipeline.objective import ObjectiveConfig
from src.pipeline.pipeline import PipelineConfig, run_pipeline
from src.utils.utils import now

TAU = 0.5

SYNTHETIC = DataConfig(
    source="synthetic",
    num_train_graphs=3,
    num_test_graphs=5,
    num_nodes=3000,
    group_size=[7, 20],
    num_groups=12,
    group_density=0.4,
    avg_degree=4.0,
    feature_dim=16,
    group_types=("random",),
    train_ratio=0.5,
    seed=3,
)

ELLIPTIC = DataConfig(
    source="elliptic",
    data_dir=Path("data/elliptic_actors"),
    elliptic_train_days=(24, 25, 26),
    elliptic_test_days=(27, 28, 29),
    elliptic_feature_mode="wallet+random",
    structural_feature_dim=64,
    min_group_size=2,
    train_ratio=0.6,
    seed=1,
)


def _coarsening(method: str = "deflated_ward") -> CoarseningConfig:
    return CoarseningConfig(
        method=method,
        tau=TAU,
        cut_rule="score_sum",
        transfer_cut_rule="score_sum",
        epsilon_budget=0.45,
        reduction=0.8,
        exact_epsilon_budget=200,
        detection_threshold=0.51,
        deflated_commit_solve="local",
        seed=0,
    )


#: One complete configuration per supported architecture and training mode.
EXAMPLES: dict = {
    # polynomial filter bank, gradient training, difference objective
    "polynomial-gradient": PipelineConfig(
        data=SYNTHETIC,
        learning=LearningConfig(
            architecture="polynomial",
            training_mode="gradient",
            objective=ObjectiveConfig(
                gamma=1.0, host_mode="none", host_weight=0.25, host_count=100, seed=0
            ),
            tau=TAU,
            degree=32,
            basis="chebyshev",
            num_heads=8,
            shared_filters=True,
            epochs=2000,
            learning_rate=0.05,
            optimizer="adam",
            multi_graph_mode="sample",
            seed=0,
        ),
        coarsening=_coarsening(),
        logging=LoggingVisualizationConfig(Path(f"results2/poly_gradient/{now}/")),
    ),
    # polynomial filter bank, gradient training, ratio objective (Dinkelbach)
    "polynomial-gradient-ratio": PipelineConfig(
        data=SYNTHETIC,
        learning=LearningConfig(
            architecture="polynomial",
            training_mode="gradient",
            objective=ObjectiveConfig(gamma=None, seed=0),
            tau=TAU,
            degree=32,
            num_heads=8,
            epochs=2000,
            learning_rate=0.05,
            dinkelbach_iters=10,
            multi_graph_mode="sample",
            seed=0,
        ),
        coarsening=_coarsening(),
        logging=LoggingVisualizationConfig(Path(f"results2/poly_ratio/{now}/")),
    ),
    # polynomial filter bank, closed form, with the frozen-level label head.
    # The ratio form solves for the penalty, so it cannot be configured above the
    # pencil cliff lambda_max(N, P) the way a fixed gamma can.
    "polynomial-closed-form": PipelineConfig(
        data=SYNTHETIC,
        learning=LearningConfig(
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
            num_heads=4,
            shared_filters=True,
            label_head_epochs=500,
            label_head_learning_rate=0.05,
            seed=0,
        ),
        coarsening=CoarseningConfig(
            method="ward_tree",
            tau=TAU,
            cut_rule="score_sum",
            transfer_cut_rule="score_sum",
            epsilon_budget=0.2,
            reduction=0.8,
            exact_epsilon_budget=200,
            deflated_commit_solve="local",
        ),
        logging=LoggingVisualizationConfig(Path(f"results2/poly_closed_form/{now}/")),
    ),
    # nonlinear GCN, gradient training, with the jointly trained label head
    "gcn": PipelineConfig(
        data=SYNTHETIC,
        learning=LearningConfig(
            architecture="gcn",
            training_mode="gradient",
            objective=ObjectiveConfig(
                gamma=1.0, label_enabled=True, label_weight=1.0, seed=0
            ),
            tau=TAU,
            num_heads=4,
            num_layers=2,
            hidden_dim=32,
            activation="relu",
            dropout=0.1,
            epochs=1000,
            learning_rate=0.01,
            multi_graph_mode="sample",
            seed=0,
        ),
        coarsening=_coarsening(),
        logging=LoggingVisualizationConfig(Path(f"results2/gcn/{now}/")),
    ),
    # nonlinear GraphSAGE, gradient training, with the jointly trained label head.
    # GraphSAGE's self path (W_self h) is not a graph propagation, so under the
    # ratio form it collapses the *training* groups onto the origin of the level:
    # a perfect boundary/internal ratio that captures nothing and transfers
    # nowhere.  The difference form bounds that trade and the host term pushes the
    # background to the origin instead; the capture trace shows whether it worked.
    "graphsage": PipelineConfig(
        data=SYNTHETIC,
        learning=LearningConfig(
            architecture="graphsage",
            training_mode="gradient",
            objective=ObjectiveConfig(
                gamma=1.0,
                host_mode="neighbours",
                host_weight=0.5,
                label_enabled=True,
                label_weight=1.0,
                seed=0,
            ),
            tau=TAU,
            num_heads=4,
            num_layers=2,
            hidden_dim=32,
            activation="relu",
            dropout=0.1,
            aggregation="mean",
            epochs=1000,
            learning_rate=0.01,
            weight_decay=1e-3,
            multi_graph_mode="sample",
            seed=0,
        ),
        coarsening=_coarsening(),
        logging=LoggingVisualizationConfig(Path(f"results2/graphsage/{now}/")),
    ),
    # real data: Elliptic++ with the closed-form bank and transfer days
    "elliptic-closed-form": PipelineConfig(
        data=ELLIPTIC,
        learning=LearningConfig(
            architecture="polynomial",
            training_mode="closed_form",
            objective=ObjectiveConfig(
                gamma=None, host_mode="none", host_weight=0.25, host_count=100, seed=1
            ),
            tau=TAU,
            degree=32,
            num_heads=16,
            shared_filters=False,
            seed=1,
        ),
        coarsening=_coarsening("deflated_ward"),
        logging=LoggingVisualizationConfig(Path(f"results2/elliptic/{now}/")),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        choices=sorted(EXAMPLES),
        # default="elliptic-closed-form",
        # default="gcn",
        # default="polynomial-gradient-ratio",
        default="polynomial-closed-form",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        # default=f"results2/synthetic_modular/{now}/",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--coarsening-method",
        choices=["ward_tree", "raw_ward", "deflated_ward", "deflated_ward_tight"],
        default="deflated_ward_tight",
    )
    parser.add_argument(
        "--cut-rule",
        choices=["epsilon", "epsilon_q", "score_sum", "reduction", "f1"],
        default="score_sum",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    config = EXAMPLES[args.example]
    if args.out is not None:
        config.logging.output_dir = Path(args.out)
    if args.epochs is not None:
        config.learning.epochs = args.epochs
    if args.seed is not None:
        config.data.seed = config.learning.seed = config.coarsening.seed = args.seed
    if args.coarsening_method is not None:
        config.coarsening.method = args.coarsening_method
    if args.cut_rule is not None:
        config.coarsening.cut_rule = args.cut_rule
    run_pipeline(config)


if __name__ == "__main__":
    main()
