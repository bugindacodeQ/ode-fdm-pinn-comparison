"""Shared definition of the boundary-value problem."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ODEProblem:
    """Domain and Dirichlet boundary data for the model problem."""

    x_left: float = 0.0
    x_right: float = 1.0
    y_left: float = 2.0
    y_right: float = 4.0

    def __post_init__(self) -> None:
        values = (self.x_left, self.x_right, self.y_left, self.y_right)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Problem bounds and boundary values must be finite.")
        if self.x_left >= self.x_right:
            raise ValueError("x_left must be strictly smaller than x_right.")

    @property
    def domain_length(self) -> float:
        """Return the length of the spatial interval."""

        return self.x_right - self.x_left


DEFAULT_PROBLEM = ODEProblem()
