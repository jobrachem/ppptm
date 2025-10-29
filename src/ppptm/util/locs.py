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
    unordered: Array
    ordering: Array
    n_subset: int | None = None

    def __post_init__(self):
        if self.n_subset is None:
            self.n_subset = self.unordered.shape[0]

        if not self.n_subset >= 1:
            raise ValueError(f"n_subset must be >= 1, got {self.n_subset}")

    @classmethod
    def new_from_unordered(
        cls, unordered: Array, n_subset: int | None = None
    ) -> Locations:
        ordering = jnp.asarray(maxmin(np.asarray(unordered))[0])
        return cls(unordered, ordering, n_subset)

    @property
    def ordered(self):
        return self.unordered[self.ordering, ...]

    @property
    def ordered_subset(self):
        return self.ordered[: self.n_subset, ...]

    @property
    def nloc(self):
        return jnp.shape(self.unordered)[0]

    @staticmethod
    def from_2d_to_3d(lon: ArrayLike, lat: ArrayLike) -> Array:
        lon = jnp.asarray(lon)
        lat = jnp.asarray(lat)

        lon_ = lon / 360 * 2 * jnp.pi
        lat_ = lat / 360 * 2 * jnp.pi

        loc1 = jnp.cos(lat_) * jnp.cos(lon_)
        loc2 = jnp.cos(lat_) * jnp.sin(lon_)
        loc3 = jnp.sin(lat_)

        locs_3d = jnp.c_[loc1, loc2, loc3]
        return locs_3d


@dataclass
class LocationVars:
    ordered: lsl.Var
    ordered_subset: lsl.Var
    locs: Locations | None = None  # kept for backwards compatibility

    def __post_init__(self):
        self._validate_shape(self.ordered, self.ordered_subset)

    @staticmethod
    def _validate_shape(locs1: lsl.Var, locs2: lsl.Var):
        shape_locs1 = locs1.value.shape
        shape_locs2 = locs2.value.shape
        if not shape_locs1[1:] == shape_locs2[1:]:
            raise ValueError(
                f"Shapes of locations are incompatible: {shape_locs1} and "
                f"{shape_locs2}. All dimensions except the first one need to be equal."
            )

    @property
    def inducing_locs(self) -> lsl.Var:
        return self.ordered_subset

    @inducing_locs.setter
    def inducing_locs(self, value: lsl.Var):
        self._validate_shape(self.ordered, value)
        self.ordered_subset = value

    @property
    def sample_locs(self) -> lsl.Var:
        return self.ordered

    @sample_locs.setter
    def sample_locs(self, value: lsl.Var):
        self._validate_shape(value, self.ordered_subset)
        self.ordered = value

    @classmethod
    def new_from_ordered(cls, locs: Locations) -> LocationVars:
        ordered = lsl.Var.new_value(jnp.asarray(locs.ordered), name="sample_locs")
        ordered_subset = lsl.Var.new_value(
            value=ordered.value[: locs.n_subset, ...],
            name="inducing_locs",
        )
        return cls(ordered, ordered_subset, locs=locs)

    @classmethod
    def new_from(
        cls, unordered: Array, n_subset: int = -1, from_2d_to_3d: bool = False
    ) -> LocationVars:
        if from_2d_to_3d:
            lon = unordered[..., 0]
            lat = unordered[..., 1]
            unordered = Locations.from_2d_to_3d(lon, lat)

        i = jnp.asarray(maxmin(np.asarray(unordered))[0])
        locs = Locations(unordered=unordered, ordering=i, n_subset=n_subset)

        ordered = lsl.Var.new_value(locs.ordered, name="sample_locs")
        subset = lsl.Var.new_value(locs.ordered_subset, name="inducing_locs")
        return cls(ordered, subset, locs=locs)


def expand_grid(*arrays: ArrayLike) -> Array:
    grids = jnp.meshgrid(*[jnp.asarray(ar) for ar in arrays], indexing="ij")
    stacked = jnp.stack(grids, axis=-1)
    return stacked.reshape(-1, len(arrays))


def long_lat_grid(
    lon: ArrayLike, lat: ArrayLike, n_subset: int | None = None
) -> Locations:
    locs = expand_grid(lon, lat)
    i = jnp.asarray(maxmin(np.asarray(locs))[0])
    return Locations(unordered=locs, ordering=i, n_subset=n_subset)


def unit_grid(ngrid: int = 10, n_subset: int | None = None) -> Locations:
    lon = jnp.linspace(0.0, 1.0, ngrid)
    lat = jnp.linspace(0.0, 1.0, ngrid)
    return long_lat_grid(lon, lat, n_subset)


def unit_grid_vars(ngrid: int = 10, n_subset: int | None = None) -> LocationVars:
    lon = jnp.linspace(0.0, 1.0, ngrid)
    lat = jnp.linspace(0.0, 1.0, ngrid)
    return LocationVars.new_from_ordered(long_lat_grid(lon, lat, n_subset))
