"""Tests for the FederatedModel base class."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from flypse.federation import (
    FederatedModel,
    ModelVersion,
)


class VectorModel(FederatedModel):
    """A federated model over a dict of numpy parameters."""

    def _read_parameters(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self.model.items()

    def _write_parameter(self, name: str, value: np.ndarray) -> None:
        self.model[name] = value


class CountedModel(VectorModel):
    """A model whose integer counter never enters the parameters."""

    def _read_parameters(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from (
            (name, value) for name, value in self.model.items() if name != "counter"
        )


def make_model(
    parameters: dict[str, np.ndarray] | None = None, **kwargs
) -> VectorModel:
    """Build a model over a small deterministic parameter dict."""
    return VectorModel(
        parameters
        if parameters is not None
        else {
            "w": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "b": np.array([5.0]),
        },
        **kwargs,
    )


def test_init_fixes_the_scheme_and_records_the_zero_version():
    """The initial state is the zero version of every level."""
    model = make_model()

    flat = model.get_parameters()
    as_dict = model.get_parameters(flattened=False)

    np.testing.assert_allclose(flat, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert flat.dtype == np.float32
    assert set(as_dict) == {"w", "b"}
    assert as_dict["w"].shape == (2, 2)
    assert model.version_levels == ("global", "local")
    assert model.version == (0, 0)
    assert model.versions[0].version == (0, 0)
    assert isinstance(model.versions[0], ModelVersion)
    assert isinstance(model.versions[0].timestamp, float)


def test_init_with_custom_levels():
    """Any non-empty sequence of unique levels is accepted."""
    model = make_model(
        {"w": np.array([1.0])}, version_levels=("global", "group", "local")
    )

    assert model.version == (0, 0, 0)


def test_init_with_invalid_levels_raises():
    """Empty or duplicated levels are rejected."""
    with pytest.raises(ValueError, match="levels"):
        make_model({"w": np.array([1.0])}, version_levels=())
    with pytest.raises(ValueError, match="levels"):
        make_model({"w": np.array([1.0])}, version_levels=("global", "global"))


def test_max_versions_drops_the_oldest_records():
    """A capped history keeps only the latest records."""
    model = make_model({"w": np.zeros(1)}, max_versions=2)

    model.update_version(np.array([1.0]), level="local")
    model.update_version(np.array([2.0]), level="local")

    assert [record.version for record in model.versions] == [(0, 1), (0, 2)]
    with pytest.raises(ValueError, match="unknown version"):
        model.get_parameters(version=(0, 0))


def test_max_versions_below_one_raises():
    """A history that cannot hold the current version is rejected."""
    with pytest.raises(ValueError, match="max_versions"):
        make_model({"w": np.zeros(1)}, max_versions=0)


def test_init_with_dtype_override():
    """A float64 model is produced when requested."""
    model = VectorModel({"w": np.array([1.0])}, dtype=np.float64)

    assert model.get_parameters().dtype == np.float64


def test_init_without_parameters_raises():
    """Models with nothing to exchange fail at initialization."""
    with pytest.raises(ValueError, match="no parameters"):
        make_model({})


def test_get_parameters_returns_copies():
    """Mutating what is returned never touches the model."""
    model = make_model({"w": np.array([1.0])})

    model.get_parameters(flattened=False)["w"] *= 5.0

    np.testing.assert_allclose(model.model["w"], [1.0])


def test_get_parameters_is_typed_by_its_form() -> None:
    """The requested form fixes the static type (checked by mypy)."""
    model = make_model({"w": np.array([1.0])})

    flat: np.ndarray = model.get_parameters()
    named: dict[str, np.ndarray] = model.get_parameters(flattened=False)

    assert flat.shape == (1,)
    assert set(named) == {"w"}


def test_get_parameters_reads_the_live_model():
    """No caching: local training is always visible on the next read."""
    model = make_model({"w": np.array([1.0])})
    initial = model.get_parameters()

    model.model["w"] = np.array([3.0])  # local training stand-in

    np.testing.assert_allclose(model.get_parameters() - initial, [2.0])


def test_update_version_bumps_the_level():
    """A local update increments the local counter only."""
    model = make_model({"w": np.zeros(2)})

    model.update_version(np.array([1.5, 2.5]), level="local")

    np.testing.assert_allclose(model.model["w"], [1.5, 2.5])
    assert model.version == (0, 1)
    assert model.versions[-1].version == (0, 1)
    np.testing.assert_allclose(model.get_parameters(version=(0, 1)), [1.5, 2.5])


def test_update_version_resets_the_finer_levels():
    """A coarse bump starts a fresh finer lineage."""
    model = make_model({"w": np.zeros(2)}, version_levels=("global", "group", "local"))
    model.load_version(ModelVersion((2, 5, 4), 0.0, np.zeros(2)))

    model.update_version(np.ones(2), level="group")

    assert model.version == (2, 6, 0)
    model.update_version(np.ones(2), level="global")
    assert model.version == (3, 0, 0)


def test_update_version_unknown_level_raises():
    """Unknown levels are rejected."""
    model = make_model({"w": np.zeros(2)})

    with pytest.raises(ValueError, match="unknown level"):
        model.update_version(np.ones(2), level="group")

    np.testing.assert_allclose(model.model["w"], np.zeros(2))
    assert model.version == (0, 0)


def test_history_is_isolated_from_the_model_and_the_caller():
    """In-place training or caller mutations never alter recorded versions."""
    model = make_model({"w": np.zeros(2)})
    sent = np.ones(2, dtype=np.float32)

    model.update_version(sent, level="local")
    sent *= 7.0
    model.model["w"] += 10.0  # in-place local training stand-in

    np.testing.assert_allclose(model.get_parameters(version=(0, 1)), [1.0, 1.0])
    with pytest.raises(ValueError, match="read-only"):
        model.versions[-1].parameters[0] = 5.0


def test_update_version_timestamp():
    """The generator stamps the wall clock unless told otherwise."""
    model = make_model({"w": np.zeros(2)})

    model.update_version(np.ones(2), level="local", timestamp=123.45)

    assert model.versions[-1].timestamp == 123.45
    model.update_version(np.ones(2), level="local")
    assert model.versions[-1].timestamp > 0


def test_update_version_accepts_a_dict():
    """The write also takes parameters shaped as the scheme."""
    model = make_model({"w": np.zeros((2, 2)), "b": np.zeros(1)})

    model.update_version({"w": np.ones((2, 2)), "b": np.array([9.0])}, level="local")

    np.testing.assert_allclose(model.model["w"], np.ones((2, 2)))
    np.testing.assert_allclose(model.model["b"], [9.0])


def test_update_version_validation():
    """Wrong sizes, names and shapes are rejected."""
    model = make_model({"w": np.zeros(3)})

    with pytest.raises(ValueError, match="expected 3 parameters, got 2"):
        model.update_version(np.zeros(2), level="local")
    with pytest.raises(ValueError, match="do not match the scheme"):
        model.update_version({"bias": np.zeros(3)}, level="local")
    with pytest.raises(ValueError, match="does not match the scheme shape"):
        model.update_version({"w": np.zeros(4)}, level="local")


def test_load_version_adopts_the_received_lineage():
    """Loaded versions are adopted verbatim and recorded locally."""
    model = make_model(
        {"w": np.array([1.0])}, version_levels=("global", "group", "local")
    )

    model.load_version(ModelVersion((2, 6, 0), 42.0, np.array([2.0])))

    np.testing.assert_allclose(model.model["w"], [2.0])
    assert model.version == (2, 6, 0)

    model.update_version(np.array([3.0]), level="local")

    assert model.version == (2, 6, 1)
    assert model.versions[-2].timestamp == 42.0  # stamped by the generator


def test_load_version_records_a_copy_in_the_history():
    """The loaded record stays in the history, parameters included."""
    model = make_model({"w": np.array([1.0])})
    model.load_version(ModelVersion((1, 0), 7.5, np.array([2.0])))
    model.model["w"] = np.array([9.0])  # local training stand-in

    np.testing.assert_allclose(model.get_parameters(version=(1, 0)), [2.0])
    assert model.versions[-1].timestamp == 7.5


def test_load_version_wrong_counters_raise():
    """Versions not matching the levels are rejected."""
    model = make_model({"w": np.zeros(2)})

    with pytest.raises(ValueError, match="2 version counters, got 3"):
        model.load_version(ModelVersion((1, 0, 0), 0.0, np.ones(2)))


def test_get_parameters_unknown_version_raises():
    """Unknown version identifiers are rejected."""
    model = make_model({"w": np.array([1.0])})

    with pytest.raises(ValueError, match="unknown version"):
        model.get_parameters(version=(3, 3))


def test_get_parameters_latest_record_wins():
    """On duplicate versions, the latest record answers."""
    model = make_model({"w": np.array([1.0])})
    model.load_version(ModelVersion((0, 0), 1.0, np.array([2.0])))
    model.load_version(ModelVersion((0, 0), 2.0, np.array([3.0])))

    np.testing.assert_allclose(model.get_parameters(version=(0, 0)), [3.0])


def test_non_combinable_entries_never_travel():
    """Hook-skipped entries stay out of the parameters and untouched."""
    model = CountedModel({"w": np.array([1.0]), "counter": np.array([7])})

    model.update_version(np.array([4.0]), level="local")

    np.testing.assert_allclose(model.get_parameters(), [4.0])
    np.testing.assert_allclose(model.model["counter"], [7])


def test_hook_drift_after_init_raises():
    """The read hook must keep yielding the initialization entries."""
    model = make_model({"w": np.array([1.0])})
    model.model["extra"] = np.array([9.0])

    with pytest.raises(ValueError, match="same entries"):
        model.get_parameters()


def test_hook_shape_drift_after_init_raises():
    """A managed entry changing shape is caught on the live read."""
    model = make_model({"w": np.zeros(2)})
    model.model["w"] = np.zeros(5)

    with pytest.raises(ValueError, match="shapes"):
        model.get_parameters()


def test_integer_entries_are_rejected():
    """Integer state yielded by the hook fails loudly at initialization."""
    with pytest.raises(ValueError, match="'n' is not floating point"):
        make_model({"w": np.zeros(1), "n": np.array([3])})


def test_duplicate_names_are_rejected():
    """A name yielded twice fails instead of silently collapsing."""

    class TwiceModel(VectorModel):
        def _read_parameters(self) -> Iterator[tuple[str, np.ndarray]]:
            yield "w", self.model["a"]
            yield "w", self.model["b"]

    with pytest.raises(ValueError, match="'w' is yielded twice"):
        TwiceModel({"a": np.zeros(2), "b": np.ones(3)})


def test_roundtrip_between_models():
    """A vector moves between structurally identical models."""
    source = make_model({"w": np.array([[1.0, 2.0], [3.0, 4.0]])})
    target = make_model({"w": np.zeros((2, 2))})

    record = source.update_version(source.get_parameters() + 1.0, level="local")

    loaded = target.load_version(record)

    assert record is source.versions[-1]
    assert loaded is target.versions[-1]
    np.testing.assert_allclose(target.model["w"], source.model["w"])
    assert target.version == source.version
    np.testing.assert_allclose(
        target.get_parameters(version=(0, 1), flattened=False)["w"],
        [[2.0, 3.0], [4.0, 5.0]],
    )


def test_records_own_a_flat_vector():
    """Records hold one flat vector owning its data, so it is sized right."""
    model = make_model({"w": np.zeros((2, 2)), "b": np.zeros(1)})

    model.update_version(np.arange(5.0), level="local")
    model.update_version({"w": np.ones((2, 2)), "b": np.ones(1)}, level="local")

    for record in model.versions:
        assert record.parameters.shape == (5,)
        assert record.parameters.base is None
        assert not record.parameters.flags.writeable
