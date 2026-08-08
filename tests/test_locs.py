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
