"""Tests for end-to-end experiment orchestration."""

from collections.abc import Iterator
import json
from pathlib import Path
import shutil
import uuid

import pytest

from src.experiment import ExperimentConfig, build_argument_parser, run_experiment
from src.pinn import PINNConfig


@pytest.fixture
def experiment_directory() -> Iterator[Path]:
    directory = Path("outputs") / f".test-experiment-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def test_run_experiment_creates_complete_artifact_set(
    experiment_directory: Path,
) -> None:
    config = ExperimentConfig(
        pinn=PINNConfig(
            hidden_layers=(6,),
            n_collocation=8,
            learning_rate=1e-3,
            epochs=3,
            seed=5,
        ),
        fdm_intervals=8,
        convergence_intervals=(4, 8),
        output_root=experiment_directory,
    )

    result = run_experiment(config)

    assert result.pinn.loss_history.shape == (3,)
    assert result.comparison.grid.shape == (9,)
    assert len(result.convergence) == 2
    assert result.training_seconds > 0.0

    data_paths = (
        result.exported_data.solutions_csv,
        result.exported_data.convergence_csv,
        result.exported_data.summary_json,
        result.exported_data.pinn_loss_csv,
    )
    assert all(path is not None and path.exists() for path in data_paths)
    assert all(path.exists() for path in vars(result.figures).values())

    metadata = json.loads(
        result.exported_data.summary_json.read_text(encoding="utf-8")
    )["metadata"]
    assert metadata["pinn_config"]["seed"] == 5
    assert metadata["environment"]["precision"] == "float64"
    assert metadata["fdm_intervals"] == 8


def test_argument_parser_accepts_experiment_overrides() -> None:
    arguments = build_argument_parser().parse_args(
        [
            "--epochs",
            "25",
            "--hidden-layers",
            "8,16",
            "--convergence-intervals",
            "5,10,20",
            "--seed",
            "9",
        ]
    )

    assert arguments.epochs == 25
    assert arguments.hidden_layers == (8, 16)
    assert arguments.convergence_intervals == (5, 10, 20)
    assert arguments.seed == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fdm_intervals": 1},
        {"convergence_intervals": (10,)},
        {"convergence_intervals": (10, 5)},
    ],
)
def test_experiment_configuration_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)  # type: ignore[arg-type]
