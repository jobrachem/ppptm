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


class ParamPredictiveProcessGP(lsl.Var):
    """Predictive-process Gaussian-process parameter over spatial locations.

    Parameters
    ----------
    locs
        Ordered sample and inducing-location variables with shapes ``(N, d)`` and
        ``(M, d)``.
    gp_kernel
        Kernel whose amplitude and length scale define the process. The amplitude
        is the standard deviation of the spatial field before ``bijector``.
    bijector
        Transformation applied to the GP mean plus the scaled spatial field.
    name
        Liesel node name.
    salt
        Diagonal jitter added before Cholesky decomposition; ``1e-6`` by default.
        Increase it if the inducing covariance is numerically singular.

    Attributes
    ----------
    latent_var
        Standard-normal latent variables at inducing locations.
    mean
        Scalar GP mean parameter.
    Kuu, Kdu
        Inducing-to-inducing and sample-to-inducing covariance nodes.

    Notes
    -----
    The node value is ``bijector(mean + amplitude * f)`` with shape ``(N,)``. Here
    ``f`` is a unit-variance predictive-process field: the process stores ``M``
    standard-normal latent values and projects them to ``N`` sample locations by a
    triangular solve with the Cholesky factor of ``Kuu``, which reduces the latent
    dimension when ``M < N``. Because the amplitude scales ``f`` before the
    bijector, an amplitude of zero gives the constant ``bijector(mean)`` and larger
    amplitudes give more spatial variation, for constrained parameters as well.

    Examples
    --------
    ```pycon
    >>> import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
    >>> from ppptm import GPKernel, ParamPredictiveProcessGP, unit_grid_vars
    >>> gp = ParamPredictiveProcessGP(unit_grid_vars(2), GPKernel(tfk.ExponentiatedQuadratic), name="gp")
    >>> gp.value.shape
    (4,)

    ```
    """

    def __init__(
        self,
        locs: LocationVars,
        gp_kernel: GPKernel,
        bijector: tfb.Bijector = _IDENTITY_BIJECTOR,
        name: str = "",
        salt: float = 1e-6,
    ) -> None:
        kernel_uu = gp_kernel.new(
            x1=locs.ordered_subset,
            x2=locs.ordered_subset,
            amplitude=jnp.array(1.0),
            name=f"{name}_Kuu",
        ).update()

        kernel_du = gp_kernel.new(
            x1=locs.ordered,
            x2=locs.ordered_subset,
            amplitude=jnp.array(1.0),
            name=f"{name}_Kdu",
        ).update()

        n_inducing_locs = kernel_uu.value.shape[0]
        M = n_inducing_locs

        self.latent_var = lsl.Var.new_param(
            jnp.zeros((n_inducing_locs,)),
            distribution=lsl.Dist(tfd.Normal, loc=jnp.array(0.0), scale=jnp.array(1.0)),
            name=f"{name}_latent",
        )

        self.mean = lsl.Var.new_param(jnp.array(0.0), name=f"{name}_mean")

        def _compute_param(latent_var, Kuu, Kdu, mean, amplitude):
            """
            latent_var: (M,)
            Kuu: (M, M)
            Kdu: (N, M)
            mean: (,)

            Returns: (N,)
            """
            Kuu = Kuu.at[jnp.diag_indices(M)].add(salt)

            L = jnp.linalg.cholesky(Kuu)  # (M x M)
            field = Kdu @ solve_triangular(L, latent_var, trans="T", lower=True)
            return bijector.forward(mean + amplitude * field)

        super().__init__(
            lsl.Calc(
                _compute_param,
                self.latent_var,
                kernel_uu,
                kernel_du,
                self.mean,
                gp_kernel.amplitude,
            ),
            name=name,
        )

        self.Kuu = kernel_uu
        self.Kdu = kernel_du

        self.bijector = bijector
        self.amplitude = gp_kernel.amplitude
        self.length_scale = gp_kernel.length_scale
        self.kernel = gp_kernel.kernel
        self.locs = locs
        self.parameter_names = [self.latent_var.name, self.mean.name]

    @staticmethod
    def sample_df(
        locs: LocationVars,
        n: int = 1,
        seed: int = 1,
        salt: float = 1e-6,
        amplitude: ArrayLike = 0.1,
        length_scale: ArrayLike = 0.1,
    ) -> pd.DataFrame:
        """Sample predictive-process draws in long tabular form.

        Parameters
        ----------
        locs
            Ordered sample and inducing-location variables.
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
            Long data frame with ``n * N`` rows and columns ``lon``, ``lat``,
            ``loc``, ``sample``, and ``value``. It is ready for plotting or grouped
            summaries. Unit-sphere locations are reported as degrees in ``lon``
            and ``lat``.

        Examples
        --------
        ```pycon
        >>> from ppptm import GPVar, unit_grid_vars
        >>> draws = GPVar.sample_df(unit_grid_vars(2), n=2, seed=7)
        >>> assert draws.shape == (8, 5)
        >>> assert draws["sample"].nunique() == 2

        ```
        """
        amplitude = jnp.asarray(amplitude)
        length_scale = jnp.asarray(length_scale)

        K = GPKernel(
            tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )
        var = ParamPredictiveProcessGP(
            locs=deepcopy(locs),
            gp_kernel=K,
            name="v",
            salt=salt,
        )

        model = lsl.Model([var], to_float32=False)
        samples = model.sample(seed=key(seed), shape=(n, 1))
        var_samples = var.predict(samples).squeeze(1)

        df = pd.DataFrame(var_samples.T)
        df["lon"], df["lat"] = lon_lat_columns(locs.ordered.value)
        df = df.reset_index(names="loc")

        return df.melt(id_vars=["lon", "lat", "loc"], var_name="sample")

    @staticmethod
    def sample_array(
        locs: LocationVars,
        n: int = 1,
        seed: int = 1,
        salt: float = 1e-6,
        amplitude: ArrayLike = 0.1,
        length_scale: ArrayLike = 0.1,
    ) -> Array:
        """Sample predictive-process draws as an array.

        Parameters
        ----------
        locs
            Ordered sample and inducing-location variables.
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
            Array with shape ``(n, N)`` for downstream simulation or numerical
            summaries. Rows are independent draws and columns follow ``locs.ordered``.

        Examples
        --------
        ```pycon
        >>> from ppptm import GPVar, unit_grid_vars
        >>> draws = GPVar.sample_array(unit_grid_vars(2), n=2, seed=7)
        >>> assert draws.shape == (2, 4)
        >>> assert (draws == GPVar.sample_array(unit_grid_vars(2), n=2, seed=7)).all()

        ```
        """
        amplitude = jnp.asarray(amplitude)
        length_scale = jnp.asarray(length_scale)

        K = GPKernel(
            tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )
        var = ParamPredictiveProcessGP(
            locs=deepcopy(locs),
            gp_kernel=K,
            name="v",
            salt=salt,
        )

        model = lsl.Model([var], to_float32=False)
        samples = model.sample(seed=key(seed), shape=(n, 1))
        var_samples = var.predict(samples).squeeze(1)
        return var_samples


GPVar = ParamPredictiveProcessGP
