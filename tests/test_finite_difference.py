"""Tests for the centered finite-difference method."""

import numpy as np
import pytest

from src.config import DEFAULT_PROBLEM, ODEProblem
from src.exact_solution import exact_solution
from src.finite_difference import (
    assemble_system,
    build_uniform_grid,
    solve_finite_difference,
)


def test_uniform_grid_matches_hand_derived_example() -> None:
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(build_uniform_grid(4), expected, rtol=0.0, atol=0.0)


def test_system_assembly_for_four_subintervals() -> None:
    system = assemble_system(4)
    expected_matrix = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )
    expected_rhs = system.step_size**2 * system.interior_grid**2
    expected_rhs[0] += DEFAULT_PROBLEM.y_left
    expected_rhs[-1] += DEFAULT_PROBLEM.y_right

    np.testing.assert_array_equal(system.matrix, expected_matrix)
    np.testing.assert_allclose(system.right_hand_side, expected_rhs)
    assert system.step_size == pytest.approx(0.25)


def test_solution_enforces_boundary_conditions() -> None:
    result = solve_finite_difference(16)

    assert result.values[0] == pytest.approx(DEFAULT_PROBLEM.y_left)
    assert result.values[-1] == pytest.approx(DEFAULT_PROBLEM.y_right)


def test_solution_satisfies_assembled_linear_system() -> None:
    result = solve_finite_difference(16)
    discrete_residual = (
        result.system.matrix @ result.values[1:-1]
        - result.system.right_hand_side
    )

    np.testing.assert_allclose(discrete_residual, 0.0, rtol=0.0, atol=1e-13)


def test_solution_supports_non_default_domain_and_boundaries() -> None:
    problem = ODEProblem(x_left=-1.0, x_right=2.0, y_left=3.0, y_right=-2.0)
    result = solve_finite_difference(32, problem)
    expected = exact_solution(result.grid, problem)

    assert result.values[0] == pytest.approx(problem.y_left)
    assert result.values[-1] == pytest.approx(problem.y_right)
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=2e-3)


def test_grid_refinement_has_second_order_convergence() -> None:
    errors = []
    for n_intervals in (10, 20, 40):
        result = solve_finite_difference(n_intervals)
        expected = exact_solution(result.grid)
        errors.append(float(np.max(np.abs(result.values - expected))))

    observed_orders = [
        np.log2(coarse_error / fine_error)
        for coarse_error, fine_error in zip(errors[:-1], errors[1:])
    ]

    assert errors[0] > errors[1] > errors[2]
    np.testing.assert_allclose(observed_orders, 2.0, rtol=0.0, atol=0.02)


@pytest.mark.parametrize("n_intervals", [0, 1, -4])
def test_interval_count_must_be_at_least_two(n_intervals: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_uniform_grid(n_intervals)


@pytest.mark.parametrize("n_intervals", [4.0, "4", True])
def test_interval_count_must_be_an_integer(n_intervals: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        build_uniform_grid(n_intervals)  # type: ignore[arg-type]
