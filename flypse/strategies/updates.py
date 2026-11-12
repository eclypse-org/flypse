"""Module for the update strategies.

An update is how the result of an aggregation changes the entity's own model.
Replacing the model is :meth:`~flypse.federation.model.FederatedModel.update_version`
itself; the strategies here compute the new parameters to pass it.

The optimisers follow one sign convention: the update Δ is the result minus the
entity's own model (e.g., the mean of the pseudogradients θ_c - θ), and every
step goes along Δ. An optimiser keeps its state between steps; it lives on the
entity, and comes back with it when the federation is redeployed.

To leave a group of coordinates out (e.g., batch normalisation statistics),
step on the others only, with the same mask at every step:

.. code-block:: python

    keep = ~self.model.mask("statistics")
    new = self.model.get_parameters()
    new[keep] = self.optimizer.step(new[keep], update[keep])
"""

from __future__ import annotations

import numpy as np


def interpolate(own: np.ndarray, other: np.ndarray, alpha: float) -> np.ndarray:
    """Move the entity's model towards another one, as (1 - alpha) own + alpha other.

    It is the mixing of asynchronous federated optimisation (FedAsync), where alpha
    is often scaled by the staleness of ``other``.

    Args:
        own (np.ndarray): The entity's own parameters.
        other (np.ndarray): The parameters to move towards.
        alpha (float): How far to move, in [0, 1].

    Returns:
        np.ndarray: The interpolated parameters.

    Raises:
        ValueError: If ``alpha`` is not in [0, 1].
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return (1.0 - alpha) * own + alpha * other


class SGD:
    """Step along the update, with momentum: v ← β v + Δ, θ ← θ + η v.

    With η = 1 and β = 0 the step adopts the result: averaging the
    pseudogradients gives federated averaging. β > 0 is FedAvgM; η < 1 slows
    the server down, as in FedBuff.
    """

    def __init__(self, lr: float = 1.0, momentum: float = 0.0):
        """Initialise the optimiser.

        Args:
            lr (float): The learning rate η. Defaults to 1.0.
            momentum (float): The momentum β, in [0, 1). Defaults to 0.0.

        Raises:
            ValueError: If ``lr`` is not positive, or ``momentum`` is not in
                [0, 1).
        """
        if lr <= 0 or not 0 <= momentum < 1:
            raise ValueError(
                f"lr must be positive and momentum in [0, 1), got {lr}, {momentum}"
            )
        self.lr = lr
        self.momentum = momentum
        self.velocity: np.ndarray | None = None
        """The momentum v, None before the first step."""

    def step(self, own: np.ndarray, update: np.ndarray) -> np.ndarray:
        """Step from the entity's parameters along the update.

        Args:
            own (np.ndarray): The entity's own parameters.
            update (np.ndarray): The update Δ.

        Returns:
            np.ndarray: The new parameters.
        """
        if self.velocity is None:
            self.velocity = np.zeros(np.shape(update))
        self.velocity = self.momentum * self.velocity + update
        return own + self.lr * self.velocity


class Adam:
    """Step along the update, scaled by its moments (FedAdam).

    It is the server optimiser of adaptive federated optimisation, in the form
    of the paper (Reddi et al., 2021): m ← β1 m + (1 - β1) Δ, v ← β2 v +
    (1 - β2) Δ², θ ← θ + η m / (√v + τ), with v starting from τ² and no bias
    correction.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ):
        """Initialise the optimiser.

        Args:
            lr (float): The learning rate η. Defaults to 1e-2.
            beta1 (float): The decay β1 of the first moment, in [0, 1).
                Defaults to 0.9.
            beta2 (float): The decay β2 of the second moment, in [0, 1).
                Defaults to 0.99.
            tau (float): The adaptivity τ, positive. Defaults to 1e-3.

        Raises:
            ValueError: If a value is out of its range.
        """
        if lr <= 0 or tau <= 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError(
                "lr and tau must be positive, beta1 and beta2 in [0, 1), got "
                f"{lr}, {tau}, {beta1}, {beta2}"
            )
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m: np.ndarray | None = None
        """The first moment, None before the first step."""
        self.v: np.ndarray | None = None
        """The second moment, None before the first step."""

    def step(self, own: np.ndarray, update: np.ndarray) -> np.ndarray:
        """Step from the entity's parameters along the update.

        Args:
            own (np.ndarray): The entity's own parameters.
            update (np.ndarray): The update Δ.

        Returns:
            np.ndarray: The new parameters.
        """
        if self.m is None or self.v is None:
            self.m = np.zeros(np.shape(update))
            self.v = np.full(np.shape(update), self.tau**2)
        self.m = self.beta1 * self.m + (1 - self.beta1) * update
        self.v = self._second_moment(self.v, np.square(update))
        return own + self.lr * self.m / (np.sqrt(self.v) + self.tau)

    def _second_moment(self, v: np.ndarray, squared: np.ndarray) -> np.ndarray:
        """Return the new second moment.

        Args:
            v (np.ndarray): The second moment.
            squared (np.ndarray): The square of the update.

        Returns:
            np.ndarray: The second moment after the update.
        """
        return self.beta2 * v + (1 - self.beta2) * squared


class Yogi(Adam):
    """Step along the update, scaled by its moments (FedYogi).

    As :class:`Adam`, with the second moment of Yogi: v ← v - (1 - β2) Δ²
    sign(v - Δ²), which grows no faster than the squared update does.
    """

    def _second_moment(self, v: np.ndarray, squared: np.ndarray) -> np.ndarray:
        """Return the new second moment.

        Args:
            v (np.ndarray): The second moment.
            squared (np.ndarray): The square of the update.

        Returns:
            np.ndarray: The second moment after the update.
        """
        return v - (1 - self.beta2) * squared * np.sign(v - squared)
