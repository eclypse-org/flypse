"""A minimal linear model with local SGD and federated averaging.

The model is plain Python (no numpy/torch) to keep the example dependency-free
and model-agnostic: FLYPSE core ships no model, so the example defines its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flypse.data.view import DataView


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
        for index in range(len(data)):
            features, target = data[index]
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
    for index in range(len(data)):
        features, target = data[index]
        total += (model.predict(features) - target) ** 2
    return total / len(data)


def federated_average(updates: Sequence[tuple[LinearModel, int]]) -> LinearModel:
    """Average the models weighted by their number of samples.

    This is the FedAvg aggregation ``Θ = Σ_c (n_c / n) Θ_c``.

    Args:
        updates (Sequence[tuple[LinearModel, int]]): The models to average,
            paired with the number of samples they were trained on.

    Returns:
        LinearModel: The aggregated model.
    """
    n_total = sum(n for _, n in updates)
    if n_total == 0:
        return updates[0][0].copy()
    n_features = len(updates[0][0].weights)
    weights = [
        sum(model.weights[j] * n for model, n in updates) / n_total
        for j in range(n_features)
    ]
    bias = sum(model.bias * n for model, n in updates) / n_total
    return LinearModel(weights, bias)
