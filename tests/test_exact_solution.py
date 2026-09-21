"""Tests for the analytical reference solution."""

import numpy as np
import pytest

from src.config import DEFAULT_PROBLEM, ODEProblem
from src.exact_solution import (
    exact_first_derivative,
    exact_residual,
    exact_second_derivative,
    exact_solution,
)


def test_exact_solution_satisfies_boundary_conditions() -> None:
    assert exact_solution(DEFAULT_PROBLEM.x_left) == pytest.approx(DEFAULT_PROBLEM.y_left)
    assert exact_solution(DEFAULT_PROBLEM.x_right) == pytest.approx(DEFAULT_PROBLEM.y_right)


def test_exact_solution_matches_closed_form() -> None:
    x = np.linspace(0.0, 1.0, 11)
    expected = -(x**4) / 12.0 + (25.0 / 12.0) * x + 2.0

    np.testing.assert_allclose(exact_solution(x), expected, rtol=0.0, atol=1e-14)


def test_exact_derivatives_match_closed_form() -> None:
    x = np.linspace(0.0, 1.0, 11)

    np.testing.assert_allclose(
        exact_first_derivative(x),
        -(x**3) / 3.0 + 25.0 / 12.0,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        exact_second_derivative(x),
        -(x**2),
        rtol=0.0,
        atol=1e-14,
    )


def test_exact_solution_has_zero_ode_residual() -> None:
    x = np.linspace(DEFAULT_PROBLEM.x_left, DEFAULT_PROBLEM.x_right, 101)
    np.testing.assert_allclose(exact_residual(x), 0.0, rtol=0.0, atol=1e-14)


def test_exact_solution_supports_non_default_boundary_data() -> None:
    problem = ODEProblem(x_left=-1.0, x_right=2.0, y_left=3.0, y_right=-2.0)

    assert exact_solution(problem.x_left, problem) == pytest.approx(problem.y_left)
    assert exact_solution(problem.x_right, problem) == pytest.approx(problem.y_right)


@pytest.mark.parametrize(
    "problem",
    [
        ODEProblem(x_left=0.0, x_right=1.0),
        ODEProblem(x_left=-1.0, x_right=2.0),
    ],
)
def test_problem_domain_length(problem: ODEProblem) -> None:
    assert problem.domain_length == pytest.approx(problem.x_right - problem.x_left)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_left": 1.0, "x_right": 1.0},
        {"x_left": 2.0, "x_right": 1.0},
        {"y_left": float("nan")},
    ],
)
def test_problem_rejects_invalid_configuration(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ODEProblem(**kwargs)
