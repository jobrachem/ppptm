import numpy as np
import pandas as pd
import plotnine as p9
from numpy.typing import ArrayLike
from scipy import stats


def summary_at_2d_locs(x, locs) -> pd.DataFrame:
    """Summarize samples and normality diagnostics at two-dimensional locations.

    Parameters
    ----------
    x
        Samples with shape ``(nsample, nloc)``. Each column must have nonzero
        variance; Shapiro-Wilk requires at least three observations.
    locs
        Longitude/latitude coordinates with shape ``(nloc, 2)``.

    Returns
    -------
    summary :
        Per-location means, standard deviations, Shapiro-Wilk statistics, and
        coordinates. Use ``w`` and ``p`` to locate departures from marginal
        normality after standardization.

    Notes
    -----
    The function standardizes each location independently before running SciPy's
    Shapiro-Wilk test. The loop is intended for diagnostics, not large-scale model
    fitting.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> from ppptm import summary_at_2d_locs
    >>> summary_at_2d_locs(np.arange(12).reshape(4, 3), np.zeros((3, 2))).shape
    (3, 6)

    ```
    """
    nlocs = locs.shape[0]
    means = x.mean(axis=0)
    sds = x.std(axis=0)

    x = (x - means) / sds

    shapiro_w = np.zeros(nlocs)
    shapiro_p = np.zeros(nlocs)
    for loci in range(nlocs):
        shapiro_w[loci], shapiro_p[loci] = stats.shapiro(x[:, loci])

    df = pd.DataFrame(
        {
            "w": shapiro_w,
            "p": shapiro_p,
            "mean": means,
            "sd": sds,
            "lon": locs[:, 0],
            "lat": locs[:, 1],
        }
    )
    return df


def long_df(locs: ArrayLike, obs: ArrayLike, value_name: str = "value") -> pd.DataFrame:
    """Return observations with shape ``(nobs, nloc)`` in long form.

    Parameters
    ----------
    locs
        Longitude/latitude coordinates with shape ``(nloc, 2)``.
    obs
        Observations with shape ``(nobs, nloc)``.
    value_name
        Name for the returned observation-value column; ``"value"`` by default.

    Returns
    -------
    data :
        One row per observation/location pair and columns ``loc``, ``lon``, ``lat``,
        ``obs``, and ``value_name``. This is the shape expected by ``plot_df`` and
        other tidy-data plotting operations.

    Notes
    -----
    Location column ``j`` is paired with row ``j`` of ``locs``. The function relies
    on pandas to report incompatible location counts.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> from ppptm import long_df
    >>> long_df(np.zeros((2, 2)), np.ones((3, 2))).shape
    (6, 5)

    ```
    """
    locs = np.asarray(locs)
    obs = np.asarray(obs)
    df = pd.DataFrame(obs.T).reset_index(names="loc")
    df["lon"] = locs[:, 0]
    df["lat"] = locs[:, 1]
    return df.melt(id_vars=["loc", "lon", "lat"], var_name="obs", value_name=value_name)


def long_df_multiple(locs: ArrayLike, **kwargs: ArrayLike) -> pd.DataFrame:
    """Combine named observation arrays into one long-form data frame.

    Parameters
    ----------
    locs
        Longitude/latitude coordinates with shape ``(nloc, 2)``.
    **kwargs
        At least two named observation arrays, each with shape ``(nobs, nloc)`` and
        the same observation/location indexing.

    Returns
    -------
    data :
        Long-form data frame with one value column for each named array. This makes
        aligned fields easy to compare or map in one table.

    Notes
    -----
    Tables are inner-joined on location coordinates and observation index. Mismatched
    coordinates or dimensions therefore remove unmatched rows.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> from ppptm import long_df_multiple
    >>> long_df_multiple(np.zeros((2, 2)), a=np.ones((3, 2)), b=np.zeros((3, 2))).shape
    (6, 6)

    ```
    """
    dfs = [long_df(locs, obs=v, value_name=k) for k, v in kwargs.items()]
    df = pd.merge(dfs[0], dfs[1], on=["loc", "lon", "lat", "obs"])
    for i in range(2, len(dfs)):
        df = pd.merge(df, dfs[i], on=["loc", "lon", "lat", "obs"])
    return df


def plot_df(df: pd.DataFrame, fill: str = "value") -> p9.ggplot:
    """Create a fixed-aspect tile plot from a long-form location data frame.

    Parameters
    ----------
    df
        Data frame produced by ``long_df`` or an equivalent table containing
        ``lon``, ``lat``, and ``fill`` columns.
    fill
        Column mapped to tile fill colour; ``"value"`` by default.

    Returns
    -------
    plot :
        Plotnine tile plot with fixed coordinate aspect. Add scales, facets, or a
        theme to this object before drawing or saving it.

    Examples
    --------
    ```pycon
    >>> import pandas as pd
    >>> from ppptm import plot_df
    >>> plot_df(pd.DataFrame({"lon": [0], "lat": [0], "value": [1]})).__class__.__name__
    'ggplot'

    ```
    """
    p = p9.ggplot(df) + p9.geom_tile(p9.aes("lon", "lat", fill=fill)) + p9.coord_fixed()
    return p
