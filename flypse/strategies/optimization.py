"""Module for the optimisation strategies.

Optimisation decides how the result changes the entity's own model. The entity can adopt
it (federated averaging), move part of the way towards it (FedAsync), or treat its
difference from the model as a gradient and take an optimiser step (FedAvgM, FedAdam).
Adopting the result is
[update_version][flypse.federation.model.FederatedModel.update_version] itself, and
adopting only part of a received model (e.g., keeping local layers, as FedPer and FedBN
do) is [load_version][flypse.federation.model.FederatedModel.load_version] with a
``mask``.

The optimisers follow one sign convention: the update Δ is the result minus the
entity's own model (e.g., the mean of the pseudogradients ``θ_c - θ``), and every
step goes along Δ. An optimiser keeps its state between steps; it lives on the
entity, and comes back with it when the federation is redeployed.

To leave a group of coordinates out (e.g., batch normalisation statistics),
step on the others only, with the same mask at every step:

```python
keep = ~self.model.mask("statistics")
new = self.model.get_parameters()
new[keep] = self.optimizer.step(new[keep], update[keep])
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def interpolate(own: np.ndarray, other: np.ndarray, alpha: float) -> np.ndarray:
    """Move the entity's model towards another one.

    The result is ``(1 - alpha) own + alpha other``. It is the mixing of asynchronous
    federated optimisation (FedAsync), where alpha is often scaled by the staleness of
    ``other``.

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
    """Step along the update, with momentum.

    The step is ``v ← β v + Δ``, ``θ ← θ + η v``. With ``η = 1`` and ``β = 0`` it adopts
    the result: averaging the pseudogradients gives federated averaging. ``β > 0`` is
    FedAvgM; η is the server learning rate, as in FedBuff. With Nesterov momentum the
    step is ``θ ← θ + η (Δ + β v)``: the outer step of DiLoCo.
    """

    def __init__(self, lr: float = 1.0, momentum: float = 0.0, nesterov: bool = False):
        """Initialise the optimiser.

        Args:
            lr (float): The learning rate η. Defaults to 1.0.
            momentum (float): The momentum β, in [0, 1). Defaults to 0.0.
            nesterov (bool): Whether to use Nesterov momentum. Defaults to
                False.

        Raises:
            ValueError: If ``lr`` is not positive, or ``momentum`` is not in
                [0, 1).
        """
        if lr <= 0 or not 0 <= momentum < 1:
            raise ValueError(
                f"lr must be positive and momentum in [0, 1), got {lr}, {momentum}"
            )
        self.lr = lr
        """The learning rate η."""
        self.momentum = momentum
        """The momentum β."""
        self.nesterov = nesterov
        """Whether to use Nesterov momentum."""
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
        if self.nesterov:
            return own + self.lr * (update + self.momentum * self.velocity)
        return own + self.lr * self.velocity


class Adam:
    """Step along the update, scaled by its moments (FedAdam).

    It is the server optimiser of adaptive federated optimisation, in the form
    of the paper (Reddi et al., 2021): ``m ← β1 m + (1 - β1) Δ``,
    ``v ← β2 v + (1 - β2) Δ²``, ``θ ← θ + η m / (√v + τ)``, with v starting from
    ``τ²`` and no bias correction.
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
        """The learning rate η."""
        self.beta1 = beta1
        """The decay β1 of the first moment."""
        self.beta2 = beta2
        """The decay β2 of the second moment."""
        self.tau = tau
        """The adaptivity τ."""
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

    As [Adam][flypse.strategies.optimization.Adam], with the second moment of Yogi:
    ``v ← v - (1 - β2) Δ² sign(v - Δ²)``, which moves by at most ``(1 - β2) Δ²``: small
    updates do not make it shrink, and hence the steps grow, abruptly.
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


class Adagrad(Adam):
    """Step along the update, scaled by its moments (FedAdagrad).

    As [Adam][flypse.strategies.optimization.Adam], with the second moment of Adagrad:
    ``v ← v + Δ²``, which only grows, so the steps shrink over time. ``beta2`` is not
    used.
    """

    def _second_moment(self, v: np.ndarray, squared: np.ndarray) -> np.ndarray:
        """Return the new second moment.

        Args:
            v (np.ndarray): The second moment.
            squared (np.ndarray): The square of the update.

        Returns:
            np.ndarray: The second moment after the update.
        """
        return v + squared


def fedexp(
    own: np.ndarray, updates: Sequence[np.ndarray], epsilon: float = 1e-3
) -> np.ndarray:
    """Step along the mean update, extrapolating when the updates disagree (FedExP).

    The step is ``η = max(1, Σ_i ‖Δ_i‖² / (2 M (‖Δ̄‖² + ε)))``, with Δ̄ the plain
    mean of the M updates: when they point in different directions their mean
    is short, and the step grows (Jhunjhunwala et al., 2023). The paper
    evaluates the average of the last two models, as the steps oscillate.

    Args:
        own (np.ndarray): The entity's own parameters.
        updates (Sequence[np.ndarray]): The update of each contribution.
        epsilon (float): The term ε keeping the step finite. Defaults to
            1e-3.

    Returns:
        np.ndarray: The new parameters.

    Raises:
        ValueError: If ``epsilon`` is not positive.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    stacked = np.vstack(updates).astype(float)
    mean = stacked.mean(axis=0)
    spread = np.sum(stacked**2) / (2 * len(stacked) * (np.sum(mean**2) + epsilon))
    return own + max(1.0, float(spread)) * mean
