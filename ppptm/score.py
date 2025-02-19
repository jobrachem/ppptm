from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

try:
    from rpy2 import robjects
except ImportError:
    pass


def tw_mv_score(
    y: ArrayLike,
    dat: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    scoring_rule: Literal["es", "vs", "mmds"] = "es",
) -> ArrayLike:
    """
    Evaluates a threshold-weighted multivariate scoring rule with chaining function
    based on the normal CDF.

    Expected shapes:
        - y: (nloc,)
        - dat: (nsamples, nloc)
        - mu: (nloc,)
        - sigma: (nloc,)

    Example R code::

        get_weight_func <- function(name, mu, sigma, weight) {
            # Placeholder for actual implementation
            return(function(x) pnorm(x, mean=mu, sd=sigma))
        }

        twes_sample <- function(y, dat1, chain_func) {
            # Placeholder for actual implementation
            return(list(y = y, dat1 = dat1, chain_func = chain_func))
        }
    """

    y = np.asarray(y)
    dat = np.asarray(dat).T
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    y_n = y.shape[0]
    dat_n = dat.shape[0]
    mu_n = mu.shape[0]
    sigma_n = sigma.shape[0]

    if not len({y_n, dat_n, mu_n, sigma_n}) == 1:
        raise ValueError("All inputs must have the same leading dimension.")

    # Convert inputs to R objects
    y_r = robjects.FloatVector(y)
    robjects.vectors.Matrix()
    dat_r = robjects.r["matrix"](
        robjects.FloatVector(dat.flatten()),
        nrow=dat.shape[0],
        ncol=dat.shape[1],
        byrow=True,
    )
    mu_r = robjects.FloatVector(mu)
    sigma_r = robjects.FloatVector(sigma)

    # Get the chain function from R
    chain_func = robjects.r("scoringRules::get_weight_func")(
        name="norm_cdf", mu=mu_r, sigma=sigma_r, weight=False
    )

    # Call the twes_sample function in R
    result = robjects.r(f"scoringRules::tw{scoring_rule}_sample")(
        y=y_r, dat=dat_r, chain_func=chain_func
    )

    return np.asarray(result).squeeze()


def vectorized_tw_mv_score(
    y: ArrayLike,
    dat: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    scoring_rule: Literal["es", "vs", "mmds"] = "es",
) -> ArrayLike:
    """
    Expected shapes:
        - y: (nsamples, nloc)
        - dat: (nsamples, nloc)
        - mu: (nloc,)
        - sigma: (nloc,)
    """

    fn = np.vectorize(
        tw_mv_score,
        excluded=["dat", "mu", "sigma", "scoring_rule"],
        signature="(1)->()",
    )

    return fn(y, dat=dat, mu=mu, sigma=sigma, scoring_rule=scoring_rule).mean()
