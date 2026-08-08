import pytest

import ppptm as gptm


def test_location_metadata_access():
    locs = gptm.unit_grid_vars(ngrid=2)
    assert locs.locs.nloc == 4

    without_metadata = gptm.LocationVars(locs.ordered, locs.ordered_subset)
    with pytest.raises(RuntimeError, match="Location metadata is unavailable"):
        _ = without_metadata.locs
