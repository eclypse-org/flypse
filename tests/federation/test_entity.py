"""Tests for the Entity class."""

from __future__ import annotations

from flypse.data import UniformDataset
from flypse.federation import Entity
from flypse.system import Device


def _deploy(entity: Entity, data=None) -> Entity:
    """Attach the entity to a device holding the given data and deploy it."""
    device = Device.__new__(Device)
    device.build(data=data)
    entity.attach_node(device)
    entity.on_deploy()
    return entity


def _split_dataset() -> UniformDataset:
    """Return a 6-item dataset split across two entities."""
    dataset = UniformDataset(list(range(6)))
    dataset._split(["a", "b"])
    return dataset


def test_entity_has_no_data_before_deploy():
    """An entity that is not deployed holds no data."""
    assert Entity("a").data is None


async def test_entity_step_has_no_effect():
    """The entity step is a no-op."""
    assert await Entity("a").step() is None


def test_entity_deploy_binds_its_view():
    """Deploying binds the entity's own portion of the device dataset."""
    entity = _deploy(Entity("b"), _split_dataset())

    assert [entity.data[i] for i in range(len(entity.data))] == [3, 4, 5]


def test_entity_deploy_without_device_data():
    """Deploying on a device without data leaves the entity without a view."""
    entity = _deploy(Entity("a"), None)

    assert entity.data is None


def test_entity_undeploy_releases_the_view():
    """Undeploying drops the bound view."""
    entity = _deploy(Entity("b"), _split_dataset())

    entity.on_undeploy()

    assert entity.data is None


def test_entity_redeploy_rebinds():
    """An undeployed entity can be deployed again."""
    entity = _deploy(Entity("a"), _split_dataset())
    entity.on_undeploy()

    entity.on_deploy()

    assert [entity.data[i] for i in range(len(entity.data))] == [0, 1, 2]
