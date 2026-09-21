"""End-to-end orchestration and command-line interface for the experiment."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch

from src.compare_solutions import (
    ComparisonResult,
    ConvergenceRecord,
    ExportedResultPaths,
    compare_methods,
    export_results,
    finite_difference_convergence,
)
from src.config import DEFAULT_PROBLEM, ODEProblem
from src.pinn import PINNConfig, PINNResult, train_pinn
from src.plotting import FigurePaths, generate_figures


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one complete numerical experiment."""

    pinn: PINNConfig = field(default_factory=PINNConfig)
    fdm_intervals: int = 100
    convergence_intervals: tuple[int, ...] = (10, 20, 40, 80)
    output_root: Path = Path("outputs")

    def __post_init__(self) -> None:
        if isinstance(self.fdm_intervals, bool) or self.fdm_intervals < 2:
            raise ValueError("fdm_intervals must be an integer of at least 2.")
        if len(self.convergence_intervals) < 2:
            raise ValueError("at least two convergence interval counts are required.")
        if any(
            fine <= coarse
            for coarse, fine in zip(
                self.convergence_intervals[:-1],
                self.convergence_intervals[1:],
            )
        ):
            raise ValueError("convergence interval counts must be strictly increasing.")


@dataclass(frozen=True)
class ExperimentResult:
    """In-memory results and artifact paths from one complete run."""

    pinn: PINNResult
    comparison: ComparisonResult
    convergence: tuple[ConvergenceRecord, ...]
    exported_data: ExportedResultPaths
    figures: FigurePaths
    training_seconds: float


def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> ExperimentResult:
    """Train, evaluate, export, and plot one reproducible experiment."""

    training_start = perf_counter()
    pinn_result = train_pinn(config.pinn, problem)
    training_seconds = perf_counter() - training_start

    comparison = compare_methods(pinn_result.model, config.fdm_intervals, problem)
    convergence = finite_difference_convergence(
        config.convergence_intervals,
        problem,
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "problem": asdict(problem),
        "pinn_config": asdict(config.pinn),
        "fdm_intervals": config.fdm_intervals,
        "convergence_intervals": config.convergence_intervals,
        "training_seconds": training_seconds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "device": "cpu",
            "precision": "float64",
        },
    }
    exported_data = export_results(
        comparison,
        convergence,
        config.output_root / "data",
        pinn_loss_history=pinn_result.loss_history,
        metadata=metadata,
    )
    figures = generate_figures(
        comparison,
        convergence,
        pinn_result.loss_history,
        config.output_root / "figures",
    )

    return ExperimentResult(
        pinn=pinn_result,
        comparison=comparison,
        convergence=convergence,
        exported_data=exported_data,
        figures=figures,
        training_seconds=training_seconds,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _integer_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without executing an experiment."""

    parser = argparse.ArgumentParser(
        description="Solve and compare the ODE using exact, FDM, and PINN methods.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=5_000)
    parser.add_argument("--collocation-points", type=_positive_int, default=64)
    parser.add_argument("--hidden-layers", type=_integer_tuple, default=(32, 32))
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fdm-intervals", type=_positive_int, default=100)
    parser.add_argument(
        "--convergence-intervals",
        type=_integer_tuple,
        default=(10, 20, 40, 80),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser


def _print_summary(result: ExperimentResult) -> None:
    comparison = result.comparison
    print("\nExperiment complete")
    print(f"PINN training time: {result.training_seconds:.3f} s")
    print(
        "FDM errors: "
        f"L_inf={comparison.finite_difference_metrics.max_absolute_error:.6e}, "
        f"RMSE={comparison.finite_difference_metrics.rmse:.6e}"
    )
    print(
        "PINN errors: "
        f"L_inf={comparison.pinn_metrics.max_absolute_error:.6e}, "
        f"RMSE={comparison.pinn_metrics.rmse:.6e}, "
        f"residual RMSE={comparison.pinn_residual_rmse:.6e}"
    )
    print(f"Data directory: {result.exported_data.summary_json.parent.resolve()}")
    print(f"Figure directory: {result.figures.solutions.parent.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line experiment and return a process exit status."""

    arguments = build_argument_parser().parse_args(argv)
    config = ExperimentConfig(
        pinn=PINNConfig(
            hidden_layers=arguments.hidden_layers,
            n_collocation=arguments.collocation_points,
            learning_rate=arguments.learning_rate,
            epochs=arguments.epochs,
            seed=arguments.seed,
        ),
        fdm_intervals=arguments.fdm_intervals,
        convergence_intervals=arguments.convergence_intervals,
        output_root=arguments.output_root,
    )

    print(
        f"Training PINN for {config.pinn.epochs} epochs "
        f"with seed {config.pinn.seed}...",
        flush=True,
    )
    result = run_experiment(config)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
