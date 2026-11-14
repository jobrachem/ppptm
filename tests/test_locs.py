import jax.numpy as jnp
import numpy as np
import pytest

import ppptm as gptm


def test_new_from_accepts_numpy():
    locs = gptm.LocationVars.new_from(
        np.array([[0.0, 0.0], [90.0, 0.0], [0.0, 90.0]]),
        n_subset=2,
        from_2d_to_3d=True,
    )

    assert locs.sample_locs.value.shape == (3, 3)
    assert locs.inducing_locs.value.shape == (2, 3)


def test_location_metadata_access():
    locs = gptm.unit_grid_vars(ngrid=2)
    assert locs.locs.nloc == 4

    without_metadata = gptm.LocationVars(locs.ordered, locs.ordered_subset)
    with pytest.raises(RuntimeError, match="Location metadata is unavailable"):
        _ = without_metadata.locs


def test_from_2d_to_3d_stacks_coordinates_along_a_new_last_axis():
    lon = jnp.array([[0.0, 90.0, 180.0], [0.0, 90.0, 180.0]])
    lat = jnp.zeros((2, 3))

    xyz = gptm.Locations.from_2d_to_3d(lon, lat)
    assert xyz.shape == (2, 3, 3)
    assert jnp.allclose(xyz[0, 1], jnp.array([0.0, 1.0, 0.0]), atol=1e-6)

    assert gptm.Locations.from_2d_to_3d(jnp.array(0.0), jnp.array(90.0)).shape == (3,)
    assert gptm.Locations.from_2d_to_3d(jnp.array([0.0, 90.0]), 0.0).shape == (2, 3)


def test_new_from_defaults_and_caps_the_inducing_subset():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    every = gptm.LocationVars.new_from(coords)
    assert every.inducing_locs.value.shape == (4, 2)

    capped = gptm.LocationVars.new_from(coords, n_subset=10)
    assert capped.locs.n_subset == 4
    assert capped.inducing_locs.value.shape == (4, 2)
