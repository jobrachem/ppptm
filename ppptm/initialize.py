import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist


def spatially_smoothed_mean_and_var(
    y: ArrayLike, loc: ArrayLike, sub: ArrayLike, bandwidth: float = 1.0
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute spatially smoothed means and variances for locations in `sub` based on
    nearby `loc` points.

    Parameters:
    - y: array of shape (nobs, nloc), values at known locations
    - loc: array of shape (nloc, 2), known locations
    - sub: array of shape (nsub, 2), target locations for smoothing
    - bandwidth: float, controls the smoothing (higher -> more smoothing)

    Returns:
    - smoothed_means: array of shape (nobs, nsub), smoothed means at `sub` locations
    - smoothed_vars: array of shape (nobs, nsub), smoothed variances at `sub` locations
    """
    y = np.asarray(y)
    loc = np.asarray(loc)
    sub = np.asarray(sub)

    # Compute pairwise distances between each sub location and all loc points
    dists = cdist(sub, loc)  # Shape (nsub, nloc)

    # Convert distances to weights using a Gaussian kernel
    weights = np.exp(-(dists**2) / (2 * bandwidth**2))  # Shape (nsub, nloc)

    # Normalize weights to sum to 1 for each sub locationranks
    weights /= weights.sum(axis=1, keepdims=True)  # Shape (nsub, nloc)

    # Compute weighted mean over locations for each observation
    smoothed_values = (weights @ y.T).T  # Shape (nobs, nsub)

    # Compute mean and variance over observations
    smoothed_means = smoothed_values.mean(axis=0)  # Shape (nsub,)
    smoothed_vars = smoothed_values.var(axis=0)  # Shape (nsub,)

    return smoothed_means, smoothed_vars
