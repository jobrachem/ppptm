from __future__ import annotations

from copy import deepcopy

import jax.numpy as jnp
import liesel.model as lsl
import pandas as pd
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax.random import key
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from ..util.locs import LocationVars
from .kernel import GPKernel


class ParamPredictiveProcessGP(lsl.Var):
    def __init__(
        self,
        locs: LocationVars,
        gp_kernel: GPKernel,
        bijector: tfb.Bijector = tfb.Identity(),
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
            value = Kdu @ solve_triangular(L, latent_var, trans="T", lower=True)
            value = bijector.forward(value + mean)

            return amplitude * value

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
        df["lon"] = locs.ordered.value[:, 0]
        df["lat"] = locs.ordered.value[:, 1]
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
    ) -> pd.DataFrame:
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
