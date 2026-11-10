"""Tests for the Device class."""

from __future__ import annotations

from eclypse.remote.node import RemoteNode

from flypse.data import UniformDataset
from flypse.system import Device


def _make_device(data=None) -> Device:
    """Return a Device whose node setup has run with the given data."""
    device = Device.__new__(Device)
    device.build(data=data)
    return device


def test_device_is_a_remote_node():
    """A device is an ECLYPSE node."""
    assert issubclass(Device, RemoteNode)


def test_device_build_stores_the_dataset():
    """The dataset given to the node setup is held by the device."""
    dataset = UniformDataset([1, 2, 3])

    device = _make_device(dataset)

    assert device.data is dataset


def test_device_data_defaults_to_none():
    """A device built without a dataset holds no data."""
    device = _make_device()

    assert device.data is None


def test_device_build_sets_up_its_slots():
    """The limit given to the node setup is that of the device."""
    device = Device.__new__(Device)
    device.build(max_concurrency=2)

    assert device.concurrency.max_concurrency == 2


def test_device_has_no_limit_by_default():
    """A device built without a limit admits every entity at once."""
    device = _make_device()

    assert device.concurrency.max_concurrency is None
