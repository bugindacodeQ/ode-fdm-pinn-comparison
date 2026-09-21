"""Tests for the physics-informed neural network."""

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.config import DEFAULT_PROBLEM, ODEProblem
from src.pinn import (
    PINN,
    PINNConfig,
    make_collocation_points,
    ode_residual,
    physics_loss,
    predict,
    train_pinn,
    trial_solution,
)


class ExactCorrection(nn.Module):
    """Correction that makes the default constrained trial solution exact."""

    def forward(self, x: Tensor) -> Tensor:
        return (1.0 + x + x**2) / 12.0


def test_network_has_expected_shape_and_precision() -> None:
    model = PINN(hidden_layers=(8, 8))
    x = torch.zeros((5, 1), dtype=torch.float64)

    output = model(x)

    assert output.shape == (5, 1)
    assert output.dtype == torch.float64


def test_trial_solution_enforces_default_boundaries() -> None:
    model = PINN(hidden_layers=(8,))
    boundaries = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    values = trial_solution(model, boundaries).detach().numpy().reshape(-1)

    np.testing.assert_allclose(values, [2.0, 4.0], rtol=0.0, atol=1e-14)


def test_trial_solution_enforces_non_default_boundaries() -> None:
    problem = ODEProblem(x_left=-2.0, x_right=3.0, y_left=1.5, y_right=-0.5)
    model = PINN(hidden_layers=(8,))
    boundaries = torch.tensor(
        [[problem.x_left], [problem.x_right]],
        dtype=torch.float64,
    )
    values = trial_solution(model, boundaries, problem).detach().numpy().reshape(-1)

    np.testing.assert_allclose(
        values,
        [problem.y_left, problem.y_right],
        rtol=0.0,
        atol=1e-14,
    )


def test_exact_correction_has_zero_residual() -> None:
    points = make_collocation_points(25)
    residual = ode_residual(ExactCorrection(), points)

    np.testing.assert_allclose(
        residual.detach().numpy(),
        0.0,
        rtol=0.0,
        atol=1e-13,
    )
    assert physics_loss(ExactCorrection(), points).item() == pytest.approx(0.0, abs=1e-26)


def test_training_reduces_physics_loss() -> None:
    config = PINNConfig(
        hidden_layers=(12, 12),
        n_collocation=24,
        learning_rate=2e-3,
        epochs=300,
        seed=7,
    )
    result = train_pinn(config)

    assert np.isfinite(result.loss_history).all()
    assert result.loss_history[-1] < result.loss_history[0] * 0.02
    assert result.collocation_points.shape == (config.n_collocation,)


def test_training_is_reproducible_for_fixed_seed() -> None:
    config = PINNConfig(
        hidden_layers=(6,),
        n_collocation=8,
        learning_rate=1e-3,
        epochs=5,
        seed=11,
    )

    first = train_pinn(config)
    second = train_pinn(config)

    np.testing.assert_array_equal(first.loss_history, second.loss_history)
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters()
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0.0, atol=0.0)


def test_prediction_preserves_input_shape_and_boundaries() -> None:
    model = PINN(hidden_layers=(8,))
    x = np.linspace(0.0, 1.0, 7)

    values = predict(model, x)

    assert isinstance(values, np.ndarray)
    assert values.shape == x.shape
    assert predict(model, 0.0) == pytest.approx(DEFAULT_PROBLEM.y_left)
    assert predict(model, 1.0) == pytest.approx(DEFAULT_PROBLEM.y_right)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_layers": ()},
        {"hidden_layers": (8, 0)},
        {"n_collocation": 0},
        {"learning_rate": 0.0},
        {"epochs": 0},
    ],
)
def test_invalid_training_configuration_is_rejected(kwargs: object) -> None:
    with pytest.raises(ValueError):
        PINNConfig(**kwargs)  # type: ignore[arg-type]
