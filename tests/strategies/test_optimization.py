"""Tests for the optimisation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from flypse.strategies.optimization import (
    SGD,
    Adam,
    Yogi,
    interpolate,
)


def test_interpolate_moves_part_of_the_way():
    """alpha = 0 keeps the model, alpha = 1 adopts the other."""
    own, other = np.array([0.0, 4.0]), np.array([2.0, 0.0])

    assert interpolate(own, other, 0.25).tolist() == [0.5, 3.0]
    assert interpolate(own, other, 1.0).tolist() == other.tolist()


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_interpolate_refuses_alpha_out_of_range(alpha):
    """Beyond [0, 1] it would extrapolate."""
    with pytest.raises(ValueError, match="alpha"):
        interpolate(np.zeros(1), np.ones(1), alpha)


def test_sgd_without_momentum_on_the_mean_update_is_federated_averaging():
    """η = 1, β = 0: θ + mean(θ_c - θ) = mean(θ_c)."""
    rng = np.random.default_rng(0)
    own = rng.normal(size=5)
    clients = [rng.normal(size=5) for _ in range(3)]

    new = SGD().step(own, np.mean([c - own for c in clients], axis=0))

    assert np.allclose(new, np.mean(clients, axis=0))


def test_sgd_momentum_accumulates_the_updates():
    """v ← β v + Δ, θ ← θ + η v."""
    sgd = SGD(lr=0.5, momentum=0.9)
    update = np.array([1.0])

    sgd.step(np.zeros(1), update)

    assert sgd.step(np.zeros(1), update).tolist() == [0.5 * 1.9]


@pytest.mark.parametrize(("lr", "momentum"), [(0.0, 0.0), (1.0, 1.0), (1.0, -0.1)])
def test_sgd_refuses_out_of_range_values(lr, momentum):
    """η > 0 and β in [0, 1)."""
    with pytest.raises(ValueError):
        SGD(lr, momentum)


def test_adam_takes_the_form_of_the_paper():
    """v starts from τ², with no bias correction."""
    lr, beta1, beta2, tau = 0.1, 0.9, 0.99, 1e-3
    update = np.array([0.5, -2.0])

    new = Adam(lr, beta1, beta2, tau).step(np.zeros(2), update)

    m = (1 - beta1) * update
    v = beta2 * tau**2 + (1 - beta2) * update**2
    assert np.allclose(new, lr * m / (np.sqrt(v) + tau))


@pytest.mark.parametrize("optimizer", [SGD, Adam, Yogi])
def test_optimisers_step_towards_the_contributions(optimizer):
    """A step goes along Δ, never away from it, on every coordinate."""
    own = np.array([1.0, -1.0, 0.0])
    update = np.array([0.3, -2.0, 0.01])

    new = optimizer().step(own, update)

    assert np.all(np.sign(new - own) == np.sign(update))


def test_yogi_second_moment_moves_towards_the_squared_update():
    """v ← v - (1 - β2) Δ² sign(v - Δ²): it decreases when v exceeds Δ²."""
    yogi = Yogi(beta2=0.5, tau=1.0)

    yogi.step(np.zeros(1), np.array([0.5]))

    assert yogi.v is not None
    assert yogi.v.tolist() == [1.0 - 0.5 * 0.25]


@pytest.mark.parametrize(
    "values",
    [{"lr": 0.0}, {"tau": 0.0}, {"beta1": 1.0}, {"beta2": -0.1}],
)
def test_adam_refuses_out_of_range_values(values):
    """η and τ positive, β1 and β2 in [0, 1)."""
    with pytest.raises(ValueError):
        Adam(**values)


def test_adam_keeps_its_moments_between_steps():
    """The second step decays the moments of the first."""
    adam = Adam(lr=0.1, beta1=0.5, beta2=0.5, tau=1.0)
    update = np.array([2.0])

    adam.step(np.zeros(1), update)
    adam.step(np.zeros(1), update)

    assert adam.m is not None
    assert adam.v is not None
    assert adam.m.tolist() == [0.5 * 1.0 + 0.5 * 2.0]
    assert adam.v.tolist() == [0.5 * (0.5 * 1.0 + 0.5 * 4.0) + 0.5 * 4.0]
