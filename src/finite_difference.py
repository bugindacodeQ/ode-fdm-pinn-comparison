"""Second-order finite-difference solver for the boundary-value problem."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from src.config import DEFAULT_PROBLEM, ODEProblem


@dataclass(frozen=True)
class FiniteDifferenceSystem:
    """Linear system produced by the centered finite-difference stencil."""

    grid: NDArray[np.float64]
    matrix: NDArray[np.float64]
    right_hand_side: NDArray[np.float64]
    step_size: float

    @property
    def interior_grid(self) -> NDArray[np.float64]:
        """Return grid points whose solution values are unknown."""

        return self.grid[1:-1]


@dataclass(frozen=True)
class FiniteDifferenceResult:
    """Finite-difference solution and the system used to compute it."""

    grid: NDArray[np.float64]
    values: NDArray[np.float64]
    system: FiniteDifferenceSystem


def _validate_interval_count(n_intervals: int) -> int:
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, Integral):
        raise TypeError("n_intervals must be an integer.")
    if n_intervals < 2:
        raise ValueError("n_intervals must be at least 2.")
    return int(n_intervals)


def build_uniform_grid(
    n_intervals: int,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> NDArray[np.float64]:
    """Construct the N + 1 nodes of a uniform grid on the domain."""

    count = _validate_interval_count(n_intervals)
    return np.linspace(problem.x_left, problem.x_right, count + 1, dtype=np.float64)


def assemble_system(
    n_intervals: int,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> FiniteDifferenceSystem:
    """Assemble the positive-definite tridiagonal system for interior nodes."""

    grid = build_uniform_grid(n_intervals, problem)
    count = len(grid) - 1
    step_size = problem.domain_length / count
    n_interior = count - 1

    matrix = 2.0 * np.eye(n_interior, dtype=np.float64)
    if n_interior > 1:
        off_diagonal = -np.ones(n_interior - 1, dtype=np.float64)
        matrix += np.diag(off_diagonal, k=-1)
        matrix += np.diag(off_diagonal, k=1)

    interior_grid = grid[1:-1]
    right_hand_side = step_size**2 * interior_grid**2
    right_hand_side[0] += problem.y_left
    right_hand_side[-1] += problem.y_right

    return FiniteDifferenceSystem(
        grid=grid,
        matrix=matrix,
        right_hand_side=right_hand_side,
        step_size=step_size,
    )


def solve_finite_difference(
    n_intervals: int,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> FiniteDifferenceResult:
    """Solve the boundary-value problem on a uniform finite-difference grid."""

    system = assemble_system(n_intervals, problem)
    interior_values = np.linalg.solve(system.matrix, system.right_hand_side)

    values = np.empty_like(system.grid)
    values[0] = problem.y_left
    values[-1] = problem.y_right
    values[1:-1] = interior_values

    return FiniteDifferenceResult(grid=system.grid, values=values, system=system)
