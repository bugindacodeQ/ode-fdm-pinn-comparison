"""Physics-informed neural network for the boundary-value problem."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor, nn

from src.config import DEFAULT_PROBLEM, ODEProblem


@dataclass(frozen=True)
class PINNConfig:
    """Architecture and optimization settings for PINN training."""

    hidden_layers: tuple[int, ...] = (32, 32)
    n_collocation: int = 64
    learning_rate: float = 1e-3
    epochs: int = 5_000
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths.")
        if self.n_collocation < 1:
            raise ValueError("n_collocation must be positive.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")


class PINN(nn.Module):
    """Fully connected network representing the unconstrained correction."""

    def __init__(self, hidden_layers: tuple[int, ...] = (32, 32)) -> None:
        super().__init__()
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive widths.")

        layers: list[nn.Module] = []
        input_width = 1
        for output_width in hidden_layers:
            linear = nn.Linear(input_width, output_width)
            nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            input_width = output_width

        output_layer = nn.Linear(input_width, 1)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)
        self.to(dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


@dataclass(frozen=True)
class PINNResult:
    """Trained model and loss values recorded after every epoch."""

    model: PINN
    loss_history: NDArray[np.float64]
    collocation_points: NDArray[np.float64]


def make_collocation_points(
    n_collocation: int,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> Tensor:
    """Create uniformly spaced interior points with gradient tracking enabled."""

    if isinstance(n_collocation, bool) or not isinstance(n_collocation, int):
        raise TypeError("n_collocation must be an integer.")
    if n_collocation < 1:
        raise ValueError("n_collocation must be positive.")

    points = torch.linspace(
        problem.x_left,
        problem.x_right,
        n_collocation + 2,
        dtype=torch.float64,
    )[1:-1]
    return points.reshape(-1, 1).requires_grad_(True)


def trial_solution(
    model: nn.Module,
    x: Tensor,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> Tensor:
    """Apply a transformation that satisfies both boundaries exactly."""

    normalized_x = (x - problem.x_left) / problem.domain_length
    boundary_interpolant = problem.y_left + normalized_x * (
        problem.y_right - problem.y_left
    )
    boundary_vanishing_factor = (x - problem.x_left) * (problem.x_right - x)
    return boundary_interpolant + boundary_vanishing_factor * model(x)


def _derivative(outputs: Tensor, inputs: Tensor) -> Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]


def ode_residual(
    model: nn.Module,
    x: Tensor,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> Tensor:
    """Evaluate y_theta''(x) + x^2 using automatic differentiation."""

    coordinates = x
    if not coordinates.requires_grad:
        coordinates = coordinates.detach().clone().requires_grad_(True)

    values = trial_solution(model, coordinates, problem)
    first_derivative = _derivative(values, coordinates)
    second_derivative = _derivative(first_derivative, coordinates)
    return second_derivative + coordinates**2


def physics_loss(
    model: nn.Module,
    x: Tensor,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> Tensor:
    """Return the mean-squared differential-equation residual."""

    residual = ode_residual(model, x, problem)
    return torch.mean(residual.square())


def train_pinn(
    config: PINNConfig = PINNConfig(),
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> PINNResult:
    """Train a hard-constrained PINN with the Adam optimizer."""

    torch.manual_seed(config.seed)
    model = PINN(config.hidden_layers)
    collocation_points = make_collocation_points(config.n_collocation, problem)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_history = np.empty(config.epochs, dtype=np.float64)

    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = physics_loss(model, collocation_points, problem)
        loss.backward()
        optimizer.step()
        loss_history[epoch] = loss.detach().item()

    return PINNResult(
        model=model,
        loss_history=loss_history,
        collocation_points=collocation_points.detach().cpu().numpy().reshape(-1),
    )


def predict(
    model: nn.Module,
    x: ArrayLike,
    problem: ODEProblem = DEFAULT_PROBLEM,
) -> float | NDArray[np.float64]:
    """Evaluate a trained PINN while preserving scalar or array input shape."""

    coordinates = np.asarray(x, dtype=np.float64)
    tensor = torch.as_tensor(coordinates.reshape(-1, 1), dtype=torch.float64)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        values = trial_solution(model, tensor, problem).cpu().numpy().reshape(coordinates.shape)
    model.train(was_training)

    if coordinates.ndim == 0:
        return float(values)
    return values
