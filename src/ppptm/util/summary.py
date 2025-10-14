import numpy as np
import pandas as pd
import plotnine as p9
from numpy.typing import ArrayLike
from scipy import stats


def summary_at_2d_locs(x, locs) -> pd.DataFrame:
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
    """
    Assumptions:
    obs: (nobs, nlocs)
    locs: (nlocs, 2)
    """
    locs = np.asarray(locs)
    obs = np.asarray(obs)
    df = pd.DataFrame(obs.T).reset_index(names="loc")
    df["lon"] = locs[:, 0]
    df["lat"] = locs[:, 1]
    return df.melt(id_vars=["loc", "lon", "lat"], var_name="obs", value_name=value_name)


def long_df_multiple(locs: ArrayLike, **kwargs: ArrayLike) -> pd.DataFrame:
    dfs = [long_df(locs, obs=v, value_name=k) for k, v in kwargs.items()]
    df = pd.merge(dfs[0], dfs[1], on=["loc", "lon", "lat", "obs"])
    for i in range(2, len(dfs)):
        df = pd.merge(df, dfs[i], on=["loc", "lon", "lat", "obs"])
    return df


def plot_df(df: pd.DataFrame, fill: str = "value") -> p9.ggplot:
    p = p9.ggplot(df) + p9.geom_tile(p9.aes("lon", "lat", fill=fill)) + p9.coord_fixed()
    return p
