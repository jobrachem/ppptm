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
    """Location coordinates together with their maximin ordering.

    Parameters
    ----------
    unordered
        Coordinates with shape ``(nloc, ndim)``. The first axis always indexes
        locations.
    ordering
        Permutation that orders ``unordered`` for the predictive process.
    n_subset
        Number of leading ordered locations used as inducing locations. Defaults to
        all locations and must be at least one; larger values are capped at the
        number of locations. Predictive-process cost grows with this value, while
        approximation error generally decreases.

    Attributes
    ----------
    unordered
        Input coordinates with shape ``(nloc, ndim)``.
    ordering
        Maximin ordering of the locations.
    n_subset
        Number of inducing locations.

    Notes
    -----
    The ordering is farthest-first (maximin): early locations are spread across the
    domain. This makes ``ordered_subset`` useful as a space-filling inducing set.

    Examples
    --------
    ```pycon
    >>> from ppptm import unit_grid
    >>> unit_grid(2).ordered.shape
    (4, 2)

    ```
    """

    unordered: Array
    ordering: Array
    n_subset: int | None = None

    def __post_init__(self):
        nloc = self.unordered.shape[0]
        if self.n_subset is None:
            self.n_subset = nloc

        if not self.n_subset >= 1:
            raise ValueError(f"n_subset must be >= 1, got {self.n_subset}")
        # More inducing locations than locations are not available; cap the request
        # so that n_subset matches the actual number of inducing locations.
        self.n_subset = min(int(self.n_subset), int(nloc))

    @classmethod
    def new_from_unordered(
        cls, unordered: Array, n_subset: int | None = None
    ) -> Locations:
        """Order coordinates with the maximin ordering.

        Parameters
        ----------
        unordered
            Coordinates with locations along the first axis.
        n_subset
            Number of leading ordered locations to retain as inducing locations.
            ``None`` uses all locations; values below one raise ``ValueError``.

        Returns
        -------
        locations :
            Metadata whose ``ordered`` values can be used as sample locations and
            whose ``ordered_subset`` values can be used as inducing locations.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Locations
        >>> locs = Locations.new_from_unordered(jnp.array([[0.0], [1.0], [0.5]]), n_subset=2)
        >>> assert locs.ordered.shape == (3, 1)
        >>> assert locs.ordered_subset.shape == (2, 1)

        ```
        """
        ordering = jnp.asarray(maxmin(np.asarray(unordered))[0])
        return cls(unordered, ordering, n_subset)

    @property
    def ordered(self):
        """Coordinates ordered for predictive-process construction.

        The result has shape ``(nloc, ndim)`` and is useful wherever a consistent
        sample-location order is required.

        Examples
        --------
        ```pycon
        >>> from ppptm import unit_grid
        >>> unit_grid(2).ordered.shape
        (4, 2)

        ```
        """
        return self.unordered[self.ordering, ...]

    @property
    def ordered_subset(self):
        """Leading ordered coordinates used as inducing locations.

        The result has shape ``(n_subset, ndim)`` and provides the low-rank support
        for predictive GPs.

        Examples
        --------
        ```pycon
        >>> from ppptm import unit_grid
        >>> unit_grid(3, n_subset=4).ordered_subset.shape
        (4, 2)

        ```
        """
        return self.ordered[: self.n_subset, ...]

    @property
    def nloc(self):
        """Number of locations along the first coordinate axis.

        Examples
        --------
        ```pycon
        >>> from ppptm import unit_grid
        >>> unit_grid(3).nloc
        9

        ```
        """
        return jnp.shape(self.unordered)[0]

    @staticmethod
    def from_2d_to_3d(lon: ArrayLike, lat: ArrayLike) -> Array:
        """Embed longitude/latitude coordinates on the unit sphere.

        Parameters
        ----------
        lon
            Longitudes in degrees.
        lat
            Latitudes in degrees.

        Returns
        -------
        coordinates :
            Cartesian unit-sphere coordinates with shape ``broadcast(lon, lat) +
            (3,)``. Euclidean distances between these rows approximate great-circle
            proximity without a longitude discontinuity.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Locations
        >>> xyz = Locations.from_2d_to_3d(jnp.array([0.0, 90.0]), jnp.array([0.0, 0.0]))
        >>> assert xyz.shape == (2, 3)
        >>> assert jnp.allclose(jnp.linalg.norm(xyz, axis=1), 1.0)

        ```
        """
        lon = jnp.asarray(lon)
        lat = jnp.asarray(lat)

        lon_ = lon / 360 * 2 * jnp.pi
        lat_ = lat / 360 * 2 * jnp.pi

        loc1 = jnp.cos(lat_) * jnp.cos(lon_)
        loc2 = jnp.cos(lat_) * jnp.sin(lon_)
        loc3 = jnp.sin(lat_)

        return jnp.stack(jnp.broadcast_arrays(loc1, loc2, loc3), axis=-1)


def lon_lat_columns(coordinates: ArrayLike) -> tuple[Array, Array]:
    """Return longitude/latitude-style columns for tabular summaries.

    Unit-sphere coordinates with three columns, as produced by
    ``Locations.from_2d_to_3d``, are converted back to degrees. For other
    coordinates the first two columns are returned unchanged.
    """
    coordinates = jnp.asarray(coordinates)
    if coordinates.shape[-1] == 3:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        lon = jnp.degrees(jnp.arctan2(y, x))
        lat = jnp.degrees(jnp.arcsin(jnp.clip(z, -1.0, 1.0)))
        return lon, lat
    return coordinates[..., 0], coordinates[..., 1]


@dataclass(init=False)
class LocationVars:
    """Liesel variables for ordered sample and inducing locations.

    Parameters
    ----------
    ordered
        Liesel variable with value shape ``(nloc, ndim)`` containing all sample
        locations.
    ordered_subset
        Liesel variable with value shape ``(ninducing, ndim)`` containing inducing
        locations. Its trailing dimensions must match ``ordered``.
    locs
        Optional location metadata retained for reconstruction and inspection.

    Attributes
    ----------
    sample_locs
        Alias for all ordered sample locations.
    inducing_locs
        Alias for the ordered inducing-location subset.

    Examples
    --------
    ```pycon
    >>> from ppptm import unit_grid_vars
    >>> locs = unit_grid_vars(3, n_subset=4)
    >>> assert locs.sample_locs.value.shape == (9, 2)
    >>> assert locs.inducing_locs.value.shape == (4, 2)

    ```
    """

    ordered: lsl.Var
    ordered_subset: lsl.Var
    _locs: Locations | None

    def __init__(
        self,
        ordered: lsl.Var,
        ordered_subset: lsl.Var,
        locs: Locations | None = None,
    ) -> None:
        self.ordered = ordered
        self.ordered_subset = ordered_subset
        self._locs = locs
        self.__post_init__()

    def __post_init__(self):
        self._validate_shape(self.ordered, self.ordered_subset)

    @property
    def locs(self) -> Locations:
        """Return retained ordering and inducing-subset metadata.

        Use this to recover the original unordered coordinates, permutation, and
        requested subset size. Instances built directly from Liesel variables do
        not have this metadata.

        Raises
        ------
        RuntimeError
            If this instance was created without metadata.

        Examples
        --------
        ```pycon
        >>> from ppptm import unit_grid_vars
        >>> unit_grid_vars(2).locs.nloc
        4

        ```
        """
        if self._locs is None:
            raise RuntimeError(
                "Location metadata is unavailable; use sample_locs and inducing_locs."
            )
        return self._locs

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
        """Liesel variable containing the inducing locations.

        Its value has shape ``(ninducing, ndim)``. Assigning a replacement validates
        that all dimensions after the location axis match ``sample_locs``.

        Examples
        --------
        ```pycon
        >>> from ppptm import unit_grid_vars
        >>> unit_grid_vars(3, n_subset=2).inducing_locs.value.shape
        (2, 2)

        ```
        """
        return self.ordered_subset

    @inducing_locs.setter
    def inducing_locs(self, value: lsl.Var):
        self._validate_shape(self.ordered, value)
        self.ordered_subset = value

    @property
    def sample_locs(self) -> lsl.Var:
        """Liesel variable containing all sample locations.

        Its value has shape ``(nloc, ndim)``. Assigning a replacement validates
        that all dimensions after the location axis match ``inducing_locs``.

        Examples
        --------
        ```pycon
        >>> from ppptm import unit_grid_vars
        >>> unit_grid_vars(3).sample_locs.value.shape
        (9, 2)

        ```
        """
        return self.ordered

    @sample_locs.setter
    def sample_locs(self, value: lsl.Var):
        self._validate_shape(value, self.ordered_subset)
        self.ordered = value

    def copy(self) -> LocationVars:
        """Return new location variables with the same values, names and metadata.

        Liesel variables can belong to only one model, so each ``Model`` needs its
        own ``LocationVars``. Use a copy to build a second model on the same
        locations, for example a ``new_G`` baseline next to a ``new_HG`` model.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> locs = unit_grid_vars(2)
        >>> baseline = Model.new_G(jnp.arange(8.0).reshape(2, 4), locs)
        >>> flexible = Model.new_HG(jnp.arange(8.0).reshape(2, 4), locs.copy(), nparam=4)
        >>> assert (flexible.locs.sample_locs.value == locs.sample_locs.value).all()

        ```
        """
        ordered = lsl.Var.new_value(
            jnp.asarray(self.ordered.value), name=self.ordered.name
        )
        ordered_subset = lsl.Var.new_value(
            jnp.asarray(self.ordered_subset.value), name=self.ordered_subset.name
        )
        return type(self)(ordered, ordered_subset, locs=self._locs)

    @classmethod
    def new_from_ordered(cls, locs: Locations) -> LocationVars:
        """Create Liesel variables from ordered location metadata.

        Parameters
        ----------
        locs
            Ordered coordinates and inducing-subset metadata.

        Returns
        -------
        location_vars :
            Variables named ``sample_locs`` and ``inducing_locs`` that can be fed
            directly to the spatial GP constructors.

        Examples
        --------
        ```pycon
        >>> from ppptm import LocationVars, unit_grid
        >>> variables = LocationVars.new_from_ordered(unit_grid(3, n_subset=2))
        >>> variables.inducing_locs.value.shape
        (2, 2)

        ```
        """
        ordered = lsl.Var.new_value(jnp.asarray(locs.ordered), name="sample_locs")
        ordered_subset = lsl.Var.new_value(
            value=ordered.value[: locs.n_subset, ...],
            name="inducing_locs",
        )
        return cls(ordered, ordered_subset, locs=locs)

    @classmethod
    def new_from(
        cls,
        unordered: ArrayLike,
        n_subset: int | None = None,
        from_2d_to_3d: bool = False,
    ) -> LocationVars:
        """Create ordered Liesel location variables from raw coordinates.

        Parameters
        ----------
        unordered
            Coordinates with locations along the first axis.
        n_subset
            Number of locations to use as inducing locations; ``None`` (default)
            uses all locations. Values above the number of locations are capped.
        from_2d_to_3d
            Whether longitude/latitude inputs should be embedded on the unit sphere.

        Returns
        -------
        location_vars :
            Ordered variables ready for spatial GP construction. Values have shapes
            ``(nloc, ndim)`` and ``(n_subset, ndim)``; spherical conversion changes
            ``ndim`` from two to three.

        Raises
        ------
        ValueError
            If ``n_subset < 1``.

        Examples
        --------
        ```pycon
        >>> import numpy as np
        >>> from ppptm import LocationVars
        >>> locs = LocationVars.new_from(np.array([[0.0, 0.0], [90.0, 0.0]]), n_subset=1, from_2d_to_3d=True)
        >>> assert locs.sample_locs.value.shape == (2, 3)
        >>> assert locs.inducing_locs.value.shape == (1, 3)

        ```
        """
        unordered = jnp.asarray(unordered)
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
    """Return the Cartesian product of one-dimensional coordinate arrays.

    Parameters
    ----------
    *arrays
        One-dimensional coordinate arrays.

    Returns
    -------
    grid :
        Array with shape ``(prod(len(array)), len(arrays))``. Each input's order is
        preserved, making the result suitable for constructing regular location
        grids.

    Examples
    --------
    ```pycon
    >>> from ppptm.util.locs import expand_grid
    >>> expand_grid([0, 1], [10, 20]).shape
    (4, 2)

    ```
    """
    grids = jnp.meshgrid(*[jnp.asarray(ar) for ar in arrays], indexing="ij")
    stacked = jnp.stack(grids, axis=-1)
    return stacked.reshape(-1, len(arrays))


def long_lat_grid(
    lon: ArrayLike, lat: ArrayLike, n_subset: int | None = None
) -> Locations:
    """Build ordered locations from longitude and latitude coordinate arrays.

    Parameters
    ----------
    lon
        Longitudinal grid coordinates.
    lat
        Latitudinal grid coordinates.
    n_subset
        Number of ordered locations to use as inducing locations. ``None`` uses all
        grid points.

    Returns
    -------
    locations :
        Ordered Cartesian longitude/latitude grid with
        ``len(lon) * len(lat)`` rows, ready for ``LocationVars.new_from_ordered``.

    Examples
    --------
    ```pycon
    >>> from ppptm.util.locs import long_lat_grid
    >>> grid = long_lat_grid([0.0, 1.0], [10.0, 20.0, 30.0], n_subset=2)
    >>> assert grid.ordered.shape == (6, 2)
    >>> assert grid.ordered_subset.shape == (2, 2)

    ```
    """
    locs = expand_grid(lon, lat)
    i = jnp.asarray(maxmin(np.asarray(locs))[0])
    return Locations(unordered=locs, ordering=i, n_subset=n_subset)


def unit_grid(ngrid: int = 10, n_subset: int | None = None) -> Locations:
    """Build an ordered square grid on the unit square.

    Parameters
    ----------
    ngrid
        Grid resolution per coordinate axis; ``10`` by default.
    n_subset
        Number of ordered grid points to use as inducing locations. ``None`` uses
        all ``ngrid**2`` points.

    Returns
    -------
    locations :
        ``ngrid * ngrid`` ordered two-dimensional locations, useful for examples,
        simulations, and regular spatial predictions.

    Examples
    --------
    ```pycon
    >>> from ppptm import unit_grid
    >>> unit_grid(3).nloc
    9

    ```
    """
    lon = jnp.linspace(0.0, 1.0, ngrid)
    lat = jnp.linspace(0.0, 1.0, ngrid)
    return long_lat_grid(lon, lat, n_subset)


def unit_grid_vars(ngrid: int = 10, n_subset: int | None = None) -> LocationVars:
    """Build Liesel location variables for an ordered unit-square grid.

    Parameters
    ----------
    ngrid
        Grid resolution per coordinate axis; ``10`` by default.
    n_subset
        Number of ordered grid points to use as inducing locations. ``None`` uses
        all grid points.

    Returns
    -------
    location_vars :
        Variables for all sample locations and their inducing subset, ready for
        ``G``, ``H``, and GP node constructors.

    Examples
    --------
    ```pycon
    >>> from ppptm import unit_grid_vars
    >>> unit_grid_vars(2).sample_locs.value.shape
    (4, 2)

    ```
    """
    lon = jnp.linspace(0.0, 1.0, ngrid)
    lat = jnp.linspace(0.0, 1.0, ngrid)
    return LocationVars.new_from_ordered(long_lat_grid(lon, lat, n_subset))
