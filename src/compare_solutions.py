"""Metrics and common-grid comparison for all solution methods."""

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn

from src.config import DEFAULT_PROBLEM, ODEProblem
from src.exact_solution import exact_solution
from src.finite_difference import solve_finite_difference
from src.pinn import ode_residual, predict


@dataclass(frozen=True)
class ErrorMetrics:
    """Error norms evaluated against a reference solution."""

    max_absolute_error: float
    rmse: float
    relative_l2_error: float


@dataclass(frozen=True)
class BoundaryErrors:
    """Absolute errors at the left and right boundaries."""

    left: float
    right: float


@dataclass(frozen=True)
class ConvergenceRecord:
    """One row in a finite-difference grid-refinement study."""

    n_intervals: int
    step_size: float
    max_absolute_error: float
    observed_order: float | None


@dataclass(frozen=True)
class ComparisonResult:
    """Solutions and metrics evaluated on the finite-difference grid."""

    grid: NDArray[np.float64]
    exact_values: NDArray[np.float64]
    finite_difference_values: NDArray[np.float64]
    pinn_values: NDArray[np.float64]
    pinn_residual_values: NDArray[np.float64]
    finite_difference_metrics: ErrorMetrics
    pinn_metrics: ErrorMetrics
    pinn_residual_rmse: float
    pinn_boundary_errors: BoundaryErrors


@dataclass(frozen=True)
class ExportedResultPaths:
    """Locations of machine-readable experiment artifacts."""

    solutions_csv: Path
    convergence_csv: Path
    summary_json: Path
    pinn_loss_csv: Path | None


def compute_error_metrics(
    reference: ArrayLike,
    approximation: ArrayLike,
) -> ErrorMetrics:
    """Compute maximum, RMS, and relative L2 errors."""

    reference_values = np.asarray(reference, dtype=np.float64)
    approximation_values = np.asarray(approximation, dtype=np.float64)

    if reference_values.shape != approximation_values.shape:
        raise ValueError("reference and approximation must have the same shape.")
    if reference_values.size == 0:
        raise ValueError("error metrics require at least one value.")
    if not np.isfinite(reference_values).all() or not np.isfinite(
        approximation_values
    ).all():
        raise ValueError("error metrics require finite values.")

    reference_norm = float(np.linalg.norm(reference_values.reshape(-1), ord=2))
    if reference_norm == 0.0:
        raise ValueError("relative L2 error is undefined for a zero reference.")

    error = approximation_values - reference_values
    return ErrorMetrics(
        max_absolute_error=float(np.max(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        relative_l2_error=float(
            np.linalg.norm(error.reshape(-1), ord=2) / reference_norm
        ),
    )


def finite_difference_convergence(
    interval_counts: Sequence[int] = (10, 20, 40, 80),
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> tuple[ConvergenceRecord, ...]:
    """Run a refinement study and estimate order from consecutive grids."""

    counts = tuple(interval_counts)
    if len(counts) < 2:
        raise ValueError("at least two interval counts are required.")
    if any(fine <= coarse for coarse, fine in zip(counts[:-1], counts[1:])):
        raise ValueError("interval counts must be strictly increasing.")

    records: list[ConvergenceRecord] = []
    previous_error: float | None = None
    previous_step: float | None = None

    for n_intervals in counts:
        result = solve_finite_difference(n_intervals, problem)
        reference = np.asarray(exact_solution(result.grid, problem))
        error = compute_error_metrics(reference, result.values).max_absolute_error

        observed_order = None
        if previous_error is not None and previous_step is not None:
            observed_order = math.log(previous_error / error) / math.log(
                previous_step / result.system.step_size
            )

        records.append(
            ConvergenceRecord(
                n_intervals=n_intervals,
                step_size=result.system.step_size,
                max_absolute_error=error,
                observed_order=observed_order,
            )
        )
        previous_error = error
        previous_step = result.system.step_size

    return tuple(records)


def compare_methods(
    model: nn.Module,
    n_intervals: int = 100,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> ComparisonResult:
    """Compare FDM and PINN predictions against the exact solution."""

    finite_difference = solve_finite_difference(n_intervals, problem)
    grid = finite_difference.grid
    reference = np.asarray(exact_solution(grid, problem), dtype=np.float64)
    pinn_values = np.asarray(predict(model, grid, problem), dtype=np.float64)

    residual_points = torch.as_tensor(
        grid.reshape(-1, 1),
        dtype=torch.float64,
    ).requires_grad_(True)
    was_training = model.training
    model.eval()
    residual = (
        ode_residual(model, residual_points, problem)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    model.train(was_training)

    return ComparisonResult(
        grid=grid,
        exact_values=reference,
        finite_difference_values=finite_difference.values,
        pinn_values=pinn_values,
        pinn_residual_values=residual,
        finite_difference_metrics=compute_error_metrics(
            reference, finite_difference.values
        ),
        pinn_metrics=compute_error_metrics(reference, pinn_values),
        pinn_residual_rmse=float(np.sqrt(np.mean(residual**2))),
        pinn_boundary_errors=BoundaryErrors(
            left=abs(float(pinn_values[0]) - problem.y_left),
            right=abs(float(pinn_values[-1]) - problem.y_right),
        ),
    )


def export_results(
    comparison: ComparisonResult,
    convergence: Sequence[ConvergenceRecord],
    output_directory: str | Path = "outputs/data",
    *,
    pinn_loss_history: ArrayLike | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ExportedResultPaths:
    """Export solution values, convergence data, metrics, and metadata."""

    records = tuple(convergence)
    if not records:
        raise ValueError("at least one convergence record is required.")

    losses: NDArray[np.float64] | None = None
    if pinn_loss_history is not None:
        losses = np.asarray(pinn_loss_history, dtype=np.float64)
        if losses.ndim != 1 or losses.size == 0:
            raise ValueError("PINN loss history must be a non-empty one-dimensional array.")
        if not np.isfinite(losses).all():
            raise ValueError("PINN loss history must contain only finite values.")

    summary = {
        "metrics": {
            "finite_difference": asdict(comparison.finite_difference_metrics),
            "pinn": {
                **asdict(comparison.pinn_metrics),
                "residual_rmse": comparison.pinn_residual_rmse,
                "boundary_errors": asdict(comparison.pinn_boundary_errors),
            },
        },
        "convergence": [asdict(record) for record in records],
        "metadata": dict(metadata or {}),
    }
    summary_text = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    solutions_path = output_path / "solutions.csv"
    convergence_path = output_path / "fdm_convergence.csv"
    summary_path = output_path / "summary.json"
    loss_path = output_path / "pinn_loss.csv" if losses is not None else None

    with solutions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "x",
                "exact",
                "finite_difference",
                "pinn",
                "pinn_residual",
                "finite_difference_absolute_error",
                "pinn_absolute_error",
            )
        )
        for x, exact, finite_difference, pinn, residual in zip(
            comparison.grid,
            comparison.exact_values,
            comparison.finite_difference_values,
            comparison.pinn_values,
            comparison.pinn_residual_values,
            strict=True,
        ):
            writer.writerow(
                (
                    x,
                    exact,
                    finite_difference,
                    pinn,
                    residual,
                    abs(finite_difference - exact),
                    abs(pinn - exact),
                )
            )

    with convergence_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "n_intervals",
                "step_size",
                "max_absolute_error",
                "observed_order",
            ),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    if losses is not None and loss_path is not None:
        with loss_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("epoch", "physics_loss"))
            writer.writerows(enumerate(losses, start=1))

    return ExportedResultPaths(
        solutions_csv=solutions_path,
        convergence_csv=convergence_path,
        summary_json=summary_path,
        pinn_loss_csv=loss_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate the command-line workflow to the experiment runner."""

    from src.experiment import main as experiment_main

    return experiment_main(argv)


if __name__ == "__main__":
    sys.exit(main())
