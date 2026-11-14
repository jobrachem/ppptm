from __future__ import annotations

from copy import deepcopy

import jax.numpy as jnp
import liesel.model as lsl
import pandas as pd
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax import Array
from jax.random import key
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from ..util.locs import LocationVars, lon_lat_columns
from .kernel import GPKernel

_IDENTITY_BIJECTOR = tfb.Identity()


def brownian_motion_mat(nrows: int, ncols: int):
    """Construct a Brownian-motion covariance matrix.

    Parameters
    ----------
    nrows, ncols
        Matrix dimensions.

    Returns
    -------
    matrix :
        Matrix whose entry ``(i, j)`` is ``min(i + 1, j + 1)``.

    Examples
    --------
    ```pycon
    >>> from ppptm.nodes.ppvar_rw import brownian_motion_mat
    >>> brownian_motion_mat(2, 2).tolist()
    [[1, 1], [1, 2]]

    ```
    """
    r = jnp.arange(nrows)[:, None] + 1
    c = jnp.arange(ncols)[None, :] + 1
    return jnp.minimum(r, c)


def rw_weight_matrix(D: int):
    """Construct the centered random-walk weight matrix for ``D`` coefficients.

    Parameters
    ----------
    D
        Number of transformation coefficients.

    Returns
    -------
    weights :
        Centered random-walk weight matrix with shape ``(D - 1, D - 2)``. Multiplying
        unconstrained increments by this matrix produces sum-to-zero coefficients.

    Notes
    -----
    ``D`` must be at least three for the Brownian covariance and Cholesky factor to
    be nonempty.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from ppptm.nodes.ppvar_rw import rw_weight_matrix
    >>> weights = rw_weight_matrix(5)
    >>> assert weights.shape == (4, 3)
    >>> assert jnp.allclose(weights.sum(axis=0), 0.0, atol=1e-6)

    ```
    """
    C = jnp.eye(D - 1) - jnp.ones(D - 1) / (D - 1)
    B = brownian_motion_mat(D - 2, D - 2)
    L = jnp.linalg.cholesky(B, upper=False)
    W = C @ jnp.r_[jnp.zeros((1, D - 2)), L]
    return W


class RandomWalkParamPredictiveProcessGP(lsl.Var):
    """Spatial predictive-process coefficients with random-walk structure.

    Parameters
    ----------
    gp_kernel
        Kernel whose amplitude and length scale define spatial dependence.
    locs
        Ordered sample and inducing-location variables with shapes ``(N, d)`` and
        ``(M, d)``.
    nparam
        Number ``D`` of random-walk coefficients at each location. ``D`` must be at
        least three.
    bijector
        Transformation applied elementwise to the scaled coefficient process;
        identity by default.
    name
        Liesel node name.
    salt
        Diagonal jitter added before Cholesky decomposition; ``1e-6`` by default.

    Attributes
    ----------
    latent_var
        Standard-normal latent variables at inducing locations.
    Kuu, Kdu
        Inducing-to-inducing and sample-to-inducing covariance nodes.

    Notes
    -----
    Spatial projection and coefficient smoothing are separable: a predictive GP
    projects ``M * D`` latent values to ``N`` locations, while the Cholesky factor
    ``W`` of the Brownian-motion covariance ``S[i, j] = min(i + 1, j + 1)`` couples
    the ``D`` coefficients. Here ``W`` is lower triangular with entries of one,
    so ``W @ W.T = S``. The resulting node has shape ``(N, D)``.

    Examples
    --------
    ```pycon
    >>> import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
    >>> from ppptm import GPKernel, RandomWalkParamPredictiveProcessGP, unit_grid_vars
    >>> gp = RandomWalkParamPredictiveProcessGP(GPKernel(tfk.ExponentiatedQuadratic), unit_grid_vars(2), 3, name="gp")
    >>> gp.value.shape
    (4, 3)

    ```
    """

    def __init__(
        self,
        gp_kernel: GPKernel,
        locs: LocationVars,
        nparam: int,
        bijector: tfb.Bijector = _IDENTITY_BIJECTOR,
        name: str = "",
        salt: float = 1e-6,
    ) -> None:
        K = gp_kernel
        D = nparam
        Kuu = K.new(
            x1=locs.ordered_subset,
            x2=locs.ordered_subset,
            amplitude=jnp.array(1.0),
            name=f"{name}_Kuu",
        ).update()

        Kdu = K.new(
            x1=locs.ordered,
            x2=locs.ordered_subset,
            amplitude=jnp.array(1.0),
            name=f"{name}_Kdu",
        ).update()

        W = jnp.tril(jnp.ones((D, D)))  # Exact Cholesky factor of Brownian covariance.

        n_inducing_locs = Kuu.value.shape[0]
        M = n_inducing_locs

        self.latent_var = lsl.Var.new_param(
            jnp.zeros((n_inducing_locs * (W.shape[1]),)),
            distribution=lsl.Dist(tfd.Normal, loc=jnp.array(0.0), scale=jnp.array(1.0)),
            name=f"{name}_latent",
        )

        def _compute_param(latent_var, Kuu, Kdu, amplitude):
            """
            latent_var: (M*D,)
            Kuu: (M, M)
            Kdu: (N, M)

            Returns: (N, D)
            """
            Kuu = Kuu.at[jnp.diag_indices(M)].add(salt)
            L = jnp.linalg.cholesky(Kuu)  # (M x M)

            Bt = jnp.reshape(latent_var, shape=(n_inducing_locs, W.shape[1]))
            B = solve_triangular(L, Bt, trans="T", lower=True)  # (M, D)
            delta_mat = (W @ B.T) @ Kdu.T  # (D, D) @ (D, M) @ (M, N) -> (D, N)
            delta_mat = delta_mat * jnp.expand_dims(amplitude, 0)
            return bijector.forward(delta_mat.T)  # (N, D)

        super().__init__(
            lsl.Calc(
                _compute_param,
                self.latent_var,
                Kuu,
                Kdu,
                K.amplitude,
            ),
            name=name,
        )

        self.bijector = bijector
        self.amplitude = K.amplitude
        self.length_scale = K.length_scale
        self.kernel = K.kernel
        self.Kuu = Kuu
        self.Kdu = Kdu
        self.locs = locs
        self.parameter_names = [self.latent_var.name]

    @staticmethod
    def sample_df(
        locs: LocationVars,
        nparam: int = 5,
        n: int = 1,
        seed: int = 1,
        salt: float = 1e-6,
        amplitude: ArrayLike = 0.1,
        length_scale: ArrayLike = 0.1,
    ) -> pd.DataFrame:
        """Sample random-walk predictive-process draws in long tabular form.

        Parameters
        ----------
        locs
            Ordered sample and inducing-location variables.
        nparam
            Number of coefficients at each location; ``5`` by default and at least
            three.
        n
            Number of draws; ``1`` by default.
        seed
            JAX random seed; ``1`` by default, making repeated calls deterministic.
        salt
            Diagonal covariance jitter; ``1e-6`` by default.
        amplitude
            Initial GP amplitude.
        length_scale
            Initial GP length scale.

        Returns
        -------
        samples :
            Long data frame with ``n * N * nparam`` rows and columns ``lon``, ``lat``,
            ``loc``, ``D``, ``sample``, and ``value``. It is useful for plotting
            coefficient surfaces by index. Unit-sphere locations are reported as
            degrees in ``lon`` and ``lat``.

        Examples
        --------
        ```pycon
        >>> from ppptm import GPTMCoef, unit_grid_vars
        >>> draws = GPTMCoef.sample_df(unit_grid_vars(2), nparam=3, n=2, seed=7)
        >>> assert draws.shape == (24, 6)
        >>> assert draws["D"].nunique() == 3

        ```
        """
        amplitude = jnp.asarray(amplitude)
        length_scale = jnp.asarray(length_scale)

        K = GPKernel(
            tfk.ExponentiatedQuadratic, amplitude=amplitude, length_scale=length_scale
        )
        var = RandomWalkParamPredictiveProcessGP(
            locs=deepcopy(locs),
            nparam=nparam,
            gp_kernel=K,
            name="v",
            salt=salt,
        )

        model = lsl.Model([var], to_float32=False)
        samples = model.sample(seed=key(seed), shape=(n, 1))
        var_samples = var.predict(samples).squeeze(1)

        n, nloc, D = var_samples.shape
        var_samples_long = jnp.reshape(var_samples, (n, nloc * D))
        df = pd.DataFrame(var_samples_long.T)
        df["D"] = jnp.tile(jnp.arange(D), nloc)
        df["loc"] = jnp.repeat(jnp.arange(nloc), D)
        df["lon"], df["lat"] = lon_lat_columns(locs.ordered.value[df["loc"].to_numpy()])

        return df.melt(id_vars=["lon", "lat", "loc", "D"], var_name="sample")

    @staticmethod
    def sample_array(
        locs: LocationVars,
        nparam: int = 5,
        n: int = 1,
        seed: int = 1,
        salt: float = 1e-6,
        amplitude: ArrayLike = 0.1,
        length_scale: ArrayLike = 0.1,
    ) -> Array:
        """Sample random-walk predictive-process draws as an array.

        Parameters
        ----------
        locs
            Ordered sample and inducing-location variables.
        nparam
            Number of coefficients at each location; ``5`` by default and at least
            three.
        n
            Number of draws; ``1`` by default.
        seed
            JAX random seed; ``1`` by default, making repeated calls deterministic.
        salt
            Diagonal covariance jitter; ``1e-6`` by default.
        amplitude
            Initial GP amplitude.
        length_scale
            Initial GP length scale.

        Returns
        -------
        samples :
            Array with shape ``(n, N, nparam)``. Use it for simulation or numerical
            summaries; its location axis follows ``locs.ordered``.

        Examples
        --------
        ```pycon
        >>> from ppptm import GPTMCoef, unit_grid_vars
        >>> draws = GPTMCoef.sample_array(unit_grid_vars(2), nparam=3, n=2, seed=7)
        >>> assert draws.shape == (2, 4, 3)
        >>> assert (draws == GPTMCoef.sample_array(unit_grid_vars(2), nparam=3, n=2, seed=7)).all()

        ```
        """
        amplitude = jnp.asarray(amplitude)
        length_scale = jnp.asarray(length_scale)

        K = GPKernel(
            tfk.ExponentiatedQuadratic, amplitude=amplitude, length_scale=length_scale
        )
        var = RandomWalkParamPredictiveProcessGP(
            locs=deepcopy(locs),
            nparam=nparam,
            gp_kernel=K,
            name="v",
            salt=salt,
        )

        model = lsl.Model([var], to_float32=False)
        samples = model.sample(seed=key(seed), shape=(n, 1))
        var_samples = var.predict(samples).squeeze(1)
        return var_samples


SpatPTMCoef = RandomWalkParamPredictiveProcessGP
GPTMCoef = RandomWalkParamPredictiveProcessGP
