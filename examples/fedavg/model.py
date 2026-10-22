"""A minimal linear model with local SGD, and its federated wrapper.

The model is plain Python (no torch) to keep the example dependency-free and
model-agnostic: FLYPSE core ships no model, so the example defines its own and
exposes its parameters as numpy arrays through a ``FederatedModel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from flypse.federation import FederatedModel

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from flypse.data.view import DataView
    from flypse.federation import ModelVersion


@dataclass
class LinearModel:
    """A linear model with one weight per feature and a bias."""

    weights: list[float]
    bias: float

    def predict(self, features: Sequence[float]) -> float:
        """Return the prediction of the model for the given features.

        Args:
            features (Sequence[float]): The input features.

        Returns:
            float: The predicted value.
        """
        return sum(w * x for w, x in zip(self.weights, features, strict=True)) + self.bias

    def copy(self) -> LinearModel:
        """Return a copy of the model.

        Returns:
            LinearModel: A copy of the model.
        """
        return LinearModel(list(self.weights), self.bias)


def local_train(
    model: LinearModel,
    data: DataView,
    *,
    epochs: int,
    lr: float,
) -> LinearModel:
    """Train a copy of the model on the entity data with SGD.

    Args:
        model (LinearModel): The model to start from.
        data (DataView): The local data of the entity.
        epochs (int): The number of local passes over the data.
        lr (float): The learning rate.

    Returns:
        LinearModel: The locally trained model.
    """
    trained = model.copy()
    for _ in range(epochs):
        for features, target in data:
            error = trained.predict(features) - target
            for j, x in enumerate(features):
                trained.weights[j] -= lr * error * x
            trained.bias -= lr * error
    return trained


def mean_squared_error(model: LinearModel, data: DataView) -> float:
    """Return the mean squared error of the model on the data.

    Args:
        model (LinearModel): The model to evaluate.
        data (DataView): The data to evaluate the model on.

    Returns:
        float: The mean squared error, or 0.0 when the data is empty.
    """
    if len(data) == 0:
        return 0.0
    total = 0.0
    for features, target in data:
        total += (model.predict(features) - target) ** 2
    return total / len(data)


class FedAvgModel(FederatedModel):
    """The federated wrapper of a linear model: its weights and its bias."""

    def _read_parameters(self) -> Iterator[tuple[str, np.ndarray]]:
        yield "weights", np.array(self.model.weights)
        yield "bias", np.array([self.model.bias])

    def _write_parameter(self, name: str, value: np.ndarray) -> None:
        if name == "weights":
            self.model.weights = [float(w) for w in value]
        else:
            self.model.bias = float(value[0])

    def snapshot(self, record: ModelVersion) -> LinearModel:
        """Return a linear model with the parameters of a recorded version.

        Args:
            record (ModelVersion): The recorded version.

        Returns:
            LinearModel: A new model, independent of the live one.
        """
        params = self.get_parameters(flattened=False, version=record.version)
        return LinearModel(
            [float(w) for w in params["weights"]], float(params["bias"][0])
        )
