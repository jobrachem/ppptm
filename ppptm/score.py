from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import conversion, default_converter
    robjects.conversion.set_conversion(default_converter)
except ImportError:
    pass


def tw_score(
    y: ArrayLike,
    dat: ArrayLike,
    a: float,
    b: float,
    scoring_rule: Literal["crps"] = "crps",
    weighting: Literal["ow", "tw"] = "tw"
) -> ArrayLike:
    """
    Evaluates a threshold-weighted scoring rule.

    Expected shapes:
        - y: (nloc,)
        - dat: (nsamples, nloc)
        - a: ()
        - b: ()
    
    """

    y = np.asarray(y)
    dat = np.asarray(dat).T
    a = np.atleast_1d(np.asarray(a))
    b = np.atleast_1d(np.asarray(b))

    y_n = y.shape[0]
    dat_n = dat.shape[0]

    if not len({y_n, dat_n}) == 1:
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
    a_r = robjects.FloatVector(a)
    b_r = robjects.FloatVector(b)

    # Call the twes_sample function in R
    result = robjects.r(f"scoringRules::{weighting}{scoring_rule}_sample")(
        y=y_r, dat=dat_r, a=a_r, b=b_r
    )
 
    return np.asarray(result).squeeze().mean()


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

    with conversion.localconverter(default_converter):
        out = fn(y, dat=dat, mu=mu, sigma=sigma, scoring_rule=scoring_rule).mean()
    
    return out


def mv_score(
    y: ArrayLike,
    dat: ArrayLike,
    scoring_rule: Literal["es", "vs", "mmds"] = "es",
) -> ArrayLike:
    """
    Evaluates a threshold-weighted multivariate scoring rule with chaining function
    based on the normal CDF.

    Expected shapes:
        - y: (nloc,)
        - dat: (nsamples, nloc)
    
    """

    y = np.asarray(y)
    dat = np.asarray(dat).T

    y_n = y.shape[0]
    dat_n = dat.shape[0]

    if not len({y_n, dat_n}) == 1:
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

    # Call the twes_sample function in R
    result = robjects.r(f"scoringRules::{scoring_rule}_sample")(y=y_r, dat=dat_r)
 
    return np.asarray(result).squeeze()


def vectorized_mv_score(
    y: ArrayLike,
    dat: ArrayLike,
    scoring_rule: Literal["es", "vs", "mmds"] = "es",
) -> ArrayLike:
    """
    Expected shapes:
        - y: (nsamples, nloc)
        - dat: (nsamples, nloc)
    """

    fn = np.vectorize(
        mv_score,
        excluded=["dat", "scoring_rule"],
        signature="(1)->()",
    )

    with conversion.localconverter(default_converter):
        out = fn(y, dat=dat, scoring_rule=scoring_rule).mean()
    
    return out

def vectorized_tw_score(
    y: ArrayLike,
    dat: ArrayLike,
    a: float,
    b: float,
    scoring_rule: Literal["crps"] = "crps",
    weighting: Literal["ow", "tw"] = "tw"
) -> ArrayLike:
    """
    Expected shapes:
        - y: (nsamples, nloc)
        - dat: (nsamples, nloc)
        - a: ()
        - b: ()
    """

    fn = np.vectorize(
        tw_score,
        excluded=["dat", "a", "b", "scoring_rule", "weighting"],
        signature="(1)->()",
    )

    with conversion.localconverter(default_converter):
        out = fn(y, dat=dat, a=a, b=b, scoring_rule=scoring_rule, weighting=weighting).mean()
    
    return out
