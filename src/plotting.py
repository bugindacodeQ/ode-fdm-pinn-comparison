"""Publication-oriented plots for the numerical experiment."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from src.compare_solutions import ComparisonResult, ConvergenceRecord


@dataclass(frozen=True)
class FigurePaths:
    """Locations of figures generated for one experiment."""

    solutions: Path
    absolute_errors: Path
    pinn_residual: Path
    pinn_loss: Path
    finite_difference_convergence: Path


_STYLE = {
    "axes.grid": True,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "font.size": 10,
    "grid.alpha": 0.25,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
}


def _finish_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def generate_figures(
    comparison: ComparisonResult,
    convergence: Sequence[ConvergenceRecord],
    pinn_loss_history: ArrayLike,
    output_directory: str | Path = "outputs/figures",
) -> FigurePaths:
    """Generate and save all standard comparison and diagnostic figures."""

    records = tuple(convergence)
    if len(records) < 2:
        raise ValueError("at least two convergence records are required.")

    losses = np.asarray(pinn_loss_history, dtype=np.float64)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("PINN loss history must be a non-empty one-dimensional array.")
    if not np.isfinite(losses).all() or np.any(losses < 0.0):
        raise ValueError("PINN loss history must contain finite non-negative values.")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = FigurePaths(
        solutions=output_path / "solutions_comparison.png",
        absolute_errors=output_path / "absolute_errors.png",
        pinn_residual=output_path / "pinn_residual.png",
        pinn_loss=output_path / "pinn_loss.png",
        finite_difference_convergence=output_path / "fdm_convergence.png",
    )

    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        marker_interval = max(1, comparison.grid.size // 12)
        axis.plot(
            comparison.grid,
            comparison.exact_values,
            color="#161616",
            linewidth=2.2,
            label="Exact",
        )
        axis.plot(
            comparison.grid,
            comparison.finite_difference_values,
            color="#2774ae",
            linestyle="--",
            marker="o",
            markersize=3.5,
            markevery=marker_interval,
            linewidth=1.5,
            label="Finite difference",
        )
        axis.plot(
            comparison.grid,
            comparison.pinn_values,
            color="#d95f02",
            linestyle="-.",
            linewidth=1.7,
            label="PINN",
        )
        axis.set(title="Solution comparison", xlabel="$x$", ylabel="$y(x)$")
        axis.legend(frameon=False)
        _finish_figure(figure, paths.solutions)

        figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        floor = np.finfo(np.float64).eps
        finite_difference_error = np.maximum(
            np.abs(comparison.finite_difference_values - comparison.exact_values),
            floor,
        )
        pinn_error = np.maximum(
            np.abs(comparison.pinn_values - comparison.exact_values),
            floor,
        )
        axis.semilogy(
            comparison.grid,
            finite_difference_error,
            color="#2774ae",
            linewidth=1.7,
            label="Finite difference",
        )
        axis.semilogy(
            comparison.grid,
            pinn_error,
            color="#d95f02",
            linewidth=1.7,
            label="PINN",
        )
        axis.set(
            title="Pointwise absolute error",
            xlabel="$x$",
            ylabel="Absolute error",
        )
        axis.legend(frameon=False)
        _finish_figure(figure, paths.absolute_errors)

        figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        axis.axhline(0.0, color="#777777", linewidth=1.0)
        axis.plot(
            comparison.grid,
            comparison.pinn_residual_values,
            color="#b2182b",
            linewidth=1.6,
        )
        axis.set(
            title="PINN differential-equation residual",
            xlabel="$x$",
            ylabel="$r_\\theta(x)$",
        )
        _finish_figure(figure, paths.pinn_residual)

        figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        epochs = np.arange(1, losses.size + 1)
        axis.semilogy(
            epochs,
            np.maximum(losses, np.finfo(np.float64).tiny),
            color="#6a3d9a",
        )
        axis.set(
            title="PINN training history",
            xlabel="Epoch",
            ylabel="Physics loss",
        )
        _finish_figure(figure, paths.pinn_loss)

        figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        step_sizes = np.asarray([record.step_size for record in records])
        errors = np.asarray([record.max_absolute_error for record in records])
        reference_order = errors[0] * (step_sizes / step_sizes[0]) ** 2
        axis.loglog(
            step_sizes,
            errors,
            color="#1b7837",
            marker="o",
            linewidth=1.7,
            label="Observed error",
        )
        axis.loglog(
            step_sizes,
            reference_order,
            color="#555555",
            linestyle="--",
            linewidth=1.3,
            label="$\\mathcal{O}(h^2)$",
        )
        axis.set(
            title="Finite-difference grid convergence",
            xlabel="Grid spacing $h$",
            ylabel="Maximum absolute error",
        )
        axis.legend(frameon=False)
        _finish_figure(figure, paths.finite_difference_convergence)

    return paths
