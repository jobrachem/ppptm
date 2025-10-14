import zipfile
from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np
import pandas as pd


def load_americas_locs() -> np.ndarray:
    resource = files("ppptm.data").joinpath("locs.csv.zip")
    with as_file(resource) as p:
        with zipfile.ZipFile(p) as zf:
            with zf.open("locs.csv") as f:
                return np.loadtxt(f, delimiter=",", skiprows=1)


def load_americas_prec() -> np.ndarray:
    resource = files("ppptm.data").joinpath("log_prec.csv.zip")
    with as_file(resource) as p:
        with zipfile.ZipFile(p) as zf:
            with zf.open("log_prec.csv") as f:
                return np.loadtxt(f, delimiter=",").T


@dataclass
class AmericasData:
    locs: np.ndarray
    obs: np.ndarray

    def as_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.obs.T).reset_index(names="loc")
        df["lon"] = self.locs[:, 0]
        df["lat"] = self.locs[:, 1]
        return df.melt(id_vars=["loc", "lon", "lat"], var_name="obs")


def load_americas() -> AmericasData:
    return AmericasData(locs=load_americas_locs(), obs=load_americas_prec())
