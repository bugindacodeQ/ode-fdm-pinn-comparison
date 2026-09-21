"""Tests for publication figure generation."""

from collections.abc import Iterator
from pathlib import Path
import shutil
import uuid

import matplotlib.pyplot as plt
import pytest
from torch import Tensor, nn

from src.compare_solutions import compare_methods, finite_difference_convergence
from src.plotting import generate_figures


class ExactCorrection(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (1.0 + x + x**2) / 12.0


@pytest.fixture
def figure_directory() -> Iterator[Path]:
    directory = Path("outputs") / f".test-figures-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def test_generate_figures_writes_five_valid_png_files(
    figure_directory: Path,
) -> None:
    comparison = compare_methods(ExactCorrection(), n_intervals=20)
    convergence = finite_difference_convergence((10, 20, 40))

    paths = generate_figures(
        comparison,
        convergence,
        pinn_loss_history=[1.0, 0.1, 0.01],
        output_directory=figure_directory,
    )

    for path in (
        paths.solutions,
        paths.absolute_errors,
        paths.pinn_residual,
        paths.pinn_loss,
        paths.finite_difference_convergence,
    ):
        assert path.exists()
        assert path.stat().st_size > 10_000
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "losses",
    [[], [[1.0, 0.1]], [1.0, float("nan")], [1.0, -0.1]],
)
def test_generate_figures_rejects_invalid_loss_history(
    figure_directory: Path,
    losses: list[object],
) -> None:
    comparison = compare_methods(ExactCorrection(), n_intervals=4)
    convergence = finite_difference_convergence((4, 8))

    with pytest.raises(ValueError, match="loss history"):
        generate_figures(comparison, convergence, losses, figure_directory)
