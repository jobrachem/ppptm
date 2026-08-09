import zipfile
from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np
import pandas as pd


def load_americas_locs() -> np.ndarray:
    """Load bundled longitude/latitude coordinates for the Americas example.

    Returns
    -------
    locs :
        Array with shape ``(nloc, 2)`` whose columns are longitude and latitude.
        Pair it with ``load_americas_prec`` or use it to construct ``LocationVars``.

    Examples
    --------
    ```pycon
    >>> from ppptm.io import load_americas_locs
    >>> load_americas_locs().shape[1]
    2

    ```
    """
    resource = files("ppptm.data").joinpath("locs.csv.zip")
    with as_file(resource) as p, zipfile.ZipFile(p) as zf, zf.open("locs.csv") as f:
        return np.loadtxt(f, delimiter=",", skiprows=1)


def load_americas_prec() -> np.ndarray:
    """Load bundled log-precipitation observations for the Americas example.

    Returns
    -------
    obs :
        Array with shape ``(nobs, nloc)`` aligned with ``load_americas_locs()``.
        Rows are independent fields and columns are spatial locations, matching the
        response layout expected by ``Model``.

    Examples
    --------
    ```pycon
    >>> from ppptm.io import load_americas_prec
    >>> load_americas_prec().ndim
    2

    ```
    """
    resource = files("ppptm.data").joinpath("log_prec.csv.zip")
    with as_file(resource) as p, zipfile.ZipFile(p) as zf, zf.open("log_prec.csv") as f:
        return np.loadtxt(f, delimiter=",").T


@dataclass
class AmericasData:
    """Bundled Americas observations and their longitude/latitude locations.

    Attributes
    ----------
    locs
        Longitude/latitude coordinates with shape ``(nloc, 2)``.
    obs
        Log-precipitation observations with shape ``(nobs, nloc)``.

    Examples
    --------
    ```pycon
    >>> from ppptm import load_americas
    >>> data = load_americas()
    >>> assert data.locs.ndim == 2
    >>> assert data.obs.shape[1] == data.locs.shape[0]

    ```
    """

    locs: np.ndarray
    obs: np.ndarray

    def as_df(self) -> pd.DataFrame:
        """Return observations in long tabular form.

        Returns
        -------
        data :
            One row per observation/location pair, with location, longitude, and
            latitude columns. This representation is convenient for plotting and
            grouped summaries; it contains ``nobs * nloc`` rows.

        Examples
        --------
        ```pycon
        >>> from ppptm import load_americas
        >>> data = load_americas()
        >>> frame = data.as_df()
        >>> assert len(frame) == data.obs.size
        >>> assert {"loc", "lon", "lat", "obs", "value"} <= set(frame)

        ```
        """
        df = pd.DataFrame(self.obs.T).reset_index(names="loc")
        df["lon"] = self.locs[:, 0]
        df["lat"] = self.locs[:, 1]
        return df.melt(id_vars=["loc", "lon", "lat"], var_name="obs")


def load_americas() -> AmericasData:
    """Load the bundled Americas example dataset.

    Returns
    -------
    data :
        Coordinates and log-precipitation observations ready for model construction,
        visualization, or conversion to long form with ``data.as_df()``.

    Examples
    --------
    ```pycon
    >>> from ppptm import load_americas
    >>> data = load_americas()
    >>> data.obs.shape[1] == data.locs.shape[0]
    True

    ```
    """
    return AmericasData(locs=load_americas_locs(), obs=load_americas_prec())
