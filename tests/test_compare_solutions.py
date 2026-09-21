"""Tests for solution metrics and cross-method comparison."""

import csv
from collections.abc import Iterator
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
from torch import Tensor, nn

from src.compare_solutions import (
    compare_methods,
    compute_error_metrics,
    export_results,
    finite_difference_convergence,
)


class ExactCorrection(nn.Module):
    """Network-shaped function that reproduces the analytical solution."""

    def forward(self, x: Tensor) -> Tensor:
        return (1.0 + x + x**2) / 12.0


@pytest.fixture
def export_directory() -> Iterator[Path]:
    directory = Path("outputs") / f".test-export-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def test_error_metrics_have_expected_values() -> None:
    metrics = compute_error_metrics([1.0, 2.0], [2.0, 0.0])

    assert metrics.max_absolute_error == pytest.approx(2.0)
    assert metrics.rmse == pytest.approx(math.sqrt(2.5))
    assert metrics.relative_l2_error == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("reference", "approximation", "message"),
    [
        ([], [], "at least one"),
        ([1.0], [1.0, 2.0], "same shape"),
        ([0.0, 0.0], [0.0, 0.0], "zero reference"),
        ([1.0, float("nan")], [1.0, 2.0], "finite values"),
    ],
)
def test_error_metrics_reject_invalid_inputs(
    reference: list[float],
    approximation: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_error_metrics(reference, approximation)


def test_finite_difference_convergence_is_second_order() -> None:
    records = finite_difference_convergence((10, 20, 40, 80))

    assert records[0].observed_order is None
    assert all(
        record.observed_order == pytest.approx(2.0, abs=0.02)
        for record in records[1:]
    )
    assert all(
        fine.max_absolute_error < coarse.max_absolute_error
        for coarse, fine in zip(records[:-1], records[1:])
    )


@pytest.mark.parametrize("counts", [(10,), (20, 10), (10, 10)])
def test_convergence_requires_increasing_grids(counts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        finite_difference_convergence(counts)


def test_comparison_uses_a_common_grid_and_reports_all_metrics() -> None:
    result = compare_methods(ExactCorrection(), n_intervals=20)

    assert result.grid.shape == (21,)
    assert result.exact_values.shape == result.grid.shape
    assert result.finite_difference_values.shape == result.grid.shape
    assert result.pinn_values.shape == result.grid.shape
    assert result.pinn_residual_values.shape == result.grid.shape
    assert result.finite_difference_metrics.max_absolute_error > 0.0
    assert result.finite_difference_metrics.max_absolute_error < 1e-4
    assert result.pinn_metrics.max_absolute_error < 1e-14
    assert result.pinn_residual_rmse < 1e-13
    assert result.pinn_boundary_errors.left == pytest.approx(0.0, abs=1e-14)
    assert result.pinn_boundary_errors.right == pytest.approx(0.0, abs=1e-14)


def test_export_results_writes_structured_artifacts(export_directory: Path) -> None:
    comparison = compare_methods(ExactCorrection(), n_intervals=20)
    convergence = finite_difference_convergence((10, 20, 40))
    losses = np.array([1.0, 0.25, 0.0625])

    paths = export_results(
        comparison,
        convergence,
        export_directory,
        pinn_loss_history=losses,
        metadata={"seed": 42, "precision": "float64"},
    )

    assert paths.solutions_csv.exists()
    assert paths.convergence_csv.exists()
    assert paths.summary_json.exists()
    assert paths.pinn_loss_csv is not None
    assert paths.pinn_loss_csv.exists()

    with paths.solutions_csv.open(newline="", encoding="utf-8") as stream:
        solution_rows = list(csv.DictReader(stream))
    assert len(solution_rows) == 21
    assert set(solution_rows[0]) == {
        "x",
        "exact",
        "finite_difference",
        "pinn",
        "pinn_residual",
        "finite_difference_absolute_error",
        "pinn_absolute_error",
    }

    with paths.convergence_csv.open(newline="", encoding="utf-8") as stream:
        convergence_rows = list(csv.DictReader(stream))
    assert len(convergence_rows) == 3
    assert float(convergence_rows[-1]["observed_order"]) == pytest.approx(2.0)

    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["metadata"] == {"precision": "float64", "seed": 42}
    assert summary["metrics"]["pinn"]["max_absolute_error"] < 1e-14

    with paths.pinn_loss_csv.open(newline="", encoding="utf-8") as stream:
        loss_rows = list(csv.DictReader(stream))
    assert loss_rows[-1] == {"epoch": "3", "physics_loss": "0.0625"}


@pytest.mark.parametrize(
    "losses",
    [[], [[1.0, 0.5]], [1.0, float("inf")]],
)
def test_export_rejects_invalid_loss_history(
    export_directory: Path,
    losses: list[object],
) -> None:
    comparison = compare_methods(ExactCorrection(), n_intervals=4)
    convergence = finite_difference_convergence((4, 8))

    with pytest.raises(ValueError, match="loss history"):
        export_results(
            comparison,
            convergence,
            export_directory,
            pinn_loss_history=losses,
        )
