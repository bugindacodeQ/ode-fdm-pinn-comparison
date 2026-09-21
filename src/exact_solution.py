"""Analytical reference solution for the boundary-value problem."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.config import DEFAULT_PROBLEM, ODEProblem

ScalarOrArray = float | NDArray[np.float64]


def _return_like_input(x: ArrayLike, values: NDArray[np.float64]) -> ScalarOrArray:
    """Return a float for scalar input and an array otherwise."""

    if np.asarray(x).ndim == 0:
        return float(values)
    return values


def _integration_constants(problem: ODEProblem) -> tuple[float, float]:
    """Compute the constants in y(x) = -x^4 / 12 + c1*x + c0."""

    def particular(x: float) -> float:
        return -(x**4) / 12.0

    c1 = (
        problem.y_right
        - problem.y_left
        - particular(problem.x_right)
        + particular(problem.x_left)
    ) / problem.domain_length
    c0 = problem.y_left - particular(problem.x_left) - c1 * problem.x_left
    return c1, c0


def _evaluate(
    x: ArrayLike,
    expression: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> ScalarOrArray:
    coordinates = np.asarray(x, dtype=np.float64)
    return _return_like_input(x, expression(coordinates))


def exact_solution(
    x: ArrayLike,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> ScalarOrArray:
    """Evaluate the exact solution at one or more coordinates."""

    c1, c0 = _integration_constants(problem)
    return _evaluate(x, lambda coordinates: -(coordinates**4) / 12.0 + c1 * coordinates + c0)


def exact_first_derivative(
    x: ArrayLike,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> ScalarOrArray:
    """Evaluate the first derivative of the exact solution."""

    c1, _ = _integration_constants(problem)
    return _evaluate(x, lambda coordinates: -(coordinates**3) / 3.0 + c1)


def exact_second_derivative(x: ArrayLike) -> ScalarOrArray:
    """Evaluate the second derivative of the exact solution."""

    return _evaluate(x, lambda coordinates: -(coordinates**2))


def exact_residual(x: ArrayLike) -> ScalarOrArray:
    """Evaluate y''(x) + x^2, which is identically zero."""

    coordinates = np.asarray(x, dtype=np.float64)
    second_derivative = np.asarray(exact_second_derivative(coordinates))
    return _return_like_input(x, second_derivative + coordinates**2)
