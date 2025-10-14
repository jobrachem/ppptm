from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from veccs.orderings2 import farthest_first_ordering as maxmin


@dataclass
class Locations:
    unordered: ArrayLike
    ordering: ArrayLike
    n_subset: int = -1

    @property
    def ordered(self):
        return self.unordered[self.ordering, ...]

    @property
    def ordered_subset(self):
        return self.ordered[: self.n_subset, ...]

    @property
    def nloc(self):
        return jnp.shape(self.unordered)[0]


@dataclass
class LocationVars:
    locs: Locations

    def __post_init__(self):
        self.ordered = lsl.Var.new_value(
            jnp.asarray(self.locs.ordered), name="locs_ordered"
        )
        self.ordered_subset = lsl.Var.new_calc(
            lambda ordered: ordered[: self.locs.n_subset, ...],
            ordered=self.ordered,
            name="locs_ordered_subset",
        )

    @classmethod
    def new_from(cls, unordered: ArrayLike, n_subset: int = -1) -> LocationVars:
        i = jnp.asarray(maxmin(np.asarray(unordered))[0])
        locs = Locations(unordered=unordered, ordering=i, n_subset=n_subset)
        return cls(locs)


def expand_grid(*arrays: ArrayLike) -> Array:
    grids = jnp.meshgrid(*[jnp.asarray(ar) for ar in arrays], indexing="ij")
    stacked = jnp.stack(grids, axis=-1)
    return stacked.reshape(-1, len(arrays))


def long_lat_grid(lon: ArrayLike, lat: ArrayLike, n_subset: int = -1) -> Locations:
    locs = expand_grid(lon, lat)
    i = jnp.asarray(maxmin(np.asarray(locs))[0])
    return Locations(unordered=locs, ordering=i, n_subset=n_subset)


def unit_grid(ngrid: int = 10, n_subset: int = -1) -> Locations:
    lon = jnp.linspace(0.0, 1.0, ngrid)
    lat = jnp.linspace(0.0, 1.0, ngrid)
    return long_lat_grid(lon, lat, n_subset)


def unit_grid_vars(ngrid: int = 10, n_subset: int = -1) -> LocationVars:
    lon = jnp.linspace(0.0, 1.0, ngrid)
    lat = jnp.linspace(0.0, 1.0, ngrid)
    return LocationVars(long_lat_grid(lon, lat, n_subset))
