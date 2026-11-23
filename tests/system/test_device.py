"""Tests for the Device class."""

from __future__ import annotations

from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np
from eclypse.remote.communication.mpi import EclypseMPI
from eclypse.remote.communication.rest import EclypseREST
from eclypse.remote.node import RemoteNode

from flypse.data import UniformDataset
from flypse.system import Device

if TYPE_CHECKING:
    from eclypse.remote.service import Service


def _hosting(device: Device, interface: str) -> list[dict]:
    """Host an entity ``a`` on the device, keeping what reaches its interface."""
    delivered: list[dict] = []

    async def handle(route, **body):  # noqa: ARG001
        delivered.append(body)

    service = SimpleNamespace(**{interface: SimpleNamespace(_handle_request=handle)})
    device._services = {"a": cast("Service", service)}
    return delivered


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


async def test_device_stamps_the_messages_it_delivers(monkeypatch):
    """A message gets when it reached the device and its size on the wire."""
    device = _make_device()
    delivered = _hosting(device, "mpi")
    monkeypatch.setattr("flypse.system.device.time", lambda: 3.0)
    parameters = np.ones(1000, dtype=np.float32)

    await device.service_comm_entrypoint(
        SimpleNamespace(recipient_id="a"),
        EclypseMPI,
        record=parameters,
        data={},
        sent_at=1.0,
    )

    [body] = delivered
    assert body["received_at"] == 3.0
    assert parameters.nbytes < body["size"] < parameters.nbytes + 500


async def test_device_leaves_other_interfaces_alone():
    """Requests of other interfaces reach the service as they are."""
    device = _make_device()
    delivered = _hosting(device, "rest")

    await device.service_comm_entrypoint(
        SimpleNamespace(recipient_id="a"), EclypseREST, method="GET"
    )

    assert delivered == [{"method": "GET"}]
